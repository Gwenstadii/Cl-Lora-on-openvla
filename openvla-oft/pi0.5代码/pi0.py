# Build complete Pi0/05 model.
import logging
import dataclasses
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# 生成注意力掩码矩阵
def make_attn_mask(input_mask, mask_ar):  # 由[B,N]的input_mask和[?B,N]的mask_ar生成[B,N,N]的注意力掩码矩阵，mark_ar若为[,N]则表示所有batch共用同一个mask_ar
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:  

      [[1 1 1 1 1 1]]: pure causal attention.  (第i个token只能关注前i个token)

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.  (前3个token互相关注但看不了后面的,后3个token只能关注前面包括自己的token)

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.  (每个block的开始处都是1,因此后面的block的mask_ar值都大于前面的block,所以后面的block能看前面的block,而自己block内全是0,因此block内token能互相关注)

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.  (input_mask里1则表示该token是有效输入,0表示padding)
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.  (mask_ar第一维不一定是B,因为可能不是所有批次共用同一个mask_ar)
    """
    # 即token只能关注那些累计mask_ar值小于等于自己的有效输入token
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)  # 把mask_ar广播成和input_mask一样的形状[B,N]
    cumsum = jnp.cumsum(mask_ar, axis=1)  # 沿token维度累加mask_ar值，形状仍为[B,N],得到的其实是每个token的“块”编号，某个块只能关注前面编号小于等于自己的块
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # 生成注意力掩码矩阵[B,N,N] （[:,None,:]把分组号放到列维度，所以是被关注的，而[:,:,None]把分组号放到行维度，所以是关注者,所以关注者组号需大于等于被关注者组号）
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]  # 提取出有效token之间的关注关系矩阵[B,N,N]
    return jnp.logical_and(attn_mask, valid_mask)  # 返回最终的注意力掩码矩阵[B,N,N]，代表有效token之间第i行个能否关注第j列个token

"""
   区分Rope位置编码和sincos位置编码: 
   原理不同,而且Rope是在gemma.py的Attention块里为q、k头加入位置信息的,其输入类型离散位置索引如这个token是第一个第二个等,让模型理解上下文序列顺序
   而sincos位置编码在pi0.py的embed_suffix函数里,为流匹配去噪的时间步t加入位置信息,输入时连续浮点数t,让模型深刻理解扩散时间步信息,即现在去噪到哪一步了
   t经过sincos位置编码后会变成一个编码向量,再经过相关特征映射得到adarms_cond条件向量,用于后续gemma条件输入
"""

# 生成正弦余弦位置编码  
@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:  # 把[b,]时间步pos嵌入成形状为[b,embedding_dim]的编码向量
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")  # embedding_dim必须是2的倍数

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)  # 把sin和cos拼接起来作为最终的位置编码向量，形状为[b,embedding_dim]

# Pi0模型整体类
class Pi0(_model.BaseModel):  
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)  # 调用BaseModel的初始化方法，设置动作维度、动作时间步和最大token长度
        self.pi05 = config.pi05  # 检查是否是Pi05模型
        
        # 根据变体和get_config函数获取PaliGemma（gemma_2b）和动作专家（gemma_300m）的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        if getattr(paligemma_config, "use_block_weight", False):
            # 检查 attn 的 lora 配置
            attn_lora = paligemma_config.lora_configs.get("attn")
            if attn_lora and not attn_lora.use_orthogonal_init:
                logger.warning(
                    "⚠️ Warning: CL-LoRA variant (use_block_weight=True) detected, "
                    "but 'attn' LoRA is NOT using orthogonal initialization! "
                    "Please check gemma.py config."
                ) # 两个专家若开了cl-lora，即块权重打开了，则必须开正交初始化（attn和ffn都是）
                

        # 实例化两个Gemma模块分别处理视觉语言和输出动作
        # TODO: rewrite gemma in NNX. For now, use bridge.  （因为gemma.py沿用的是Flax.linen，而此处沿用的是Flax.NNX）
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  
                embed_dtype=config.dtype,
                adarms=config.pi05,  # dropout等参数用默认值
            )
        )
        
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])  # 调用gemma.Modul.init初始化，若pi0.5启用则动作专家开启adarms时间条件输入，若不是则不开启（第一个专家始终不开启时间条件输入）
        # linen还是得先传假参初始化，然后bridge才能转换成NNX模块，参数才能变成NNX的状态
        
        # 实例化SigLIP视觉编码器
        img = nnx_bridge.ToNNX(
            _siglip.Module(  
                num_classes=paligemma_config.width,  # 输出的嵌入维度跟Gemma_2b的主干宽度一致
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)  # 同理假图像初始化（因为siglip也是linen逻辑）
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)  # 把专家实例（注意，这是一个实例整体，只是其内部由两个专家并行组成，其中两个专家并行各自处理prefix和suffix是通过gemma.block对xs的for循环实现的，单个那个embedder层是为专家0建立的）和视觉编码器打包成一个字典模块
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # 把动作维度映射到Gemma_300m的主干宽度
        
        # pi0.5则
        if config.pi05:  # 没有状态输入是因为状态被离散编码成token输入给第一个专家了
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  # 对进行了sincos位置编码的时间步t再做MLP映射，得到时间条件输入adarms_cond,代表时间信息的嵌入
        # pi0则
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)  # 此时把action_embed和time_embed拼接使得维度翻倍变成2*width实现动作和时间的融合，后面再做MLP映射 
        
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)  # 把Gemma_300m的主干宽度输出映射回动作维度

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    # prefix：image、prompt(language and state)
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:  # 输入当前的Observation，输出嵌入好整体特征向量[b,s,emb]以及对应的输入掩码[b,s]和掩码分组累积标记[,s]，没有b维因为所有样本共用同一个掩码分组
        
        input_mask = []  # 存放所有嵌入好的token对应的输入掩码 （决定某个batch的所有token是否是有效输入）
        ar_mask = []  # 存放掩码分组累积标记（决定token之间能否互相关注）
        tokens = []  # 存放所有嵌入好的token
        
        # embed images
        for name in obs.images:  # 遍历所有摄像头的图像，比如base_0_rgb、wrist_0_rgb等
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  # 调用SigLIP处理Observation里的每张图像，输入[b,h,w,c]输出[b,s,d]的图像token序列,s是图像token数，d是嵌入维度

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )  # 把每张图像对应的形状为[*b]的mask广播成和图像token数一样的形状[b,s]，并拼接到列表input_mask里
            )      # 图像掩码每个batch的全部token要么全部是有效输入，要么全无效，因为image_masks本身形状是[*b]的
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]  # ar_mask是一个s个0组成的列表，说明每一个batch里所有token都是同一组的，可互相关注，即双向注意力

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            # 调用专家实例里的embed方法处理语言指令，因为gemma.Module.embedder类只传入了第一个专家的config，所以实际这嵌入层是专家1专用的
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # [b,l] -> [b,l,d]
            tokens.append(tokenized_inputs)  
            input_mask.append(obs.tokenized_prompt_mask)  # 同理把[b,l]形状的语言掩码加入input_mask列表（l决定每个b里每个token是否有效）
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]  # 也是添加l个0进入ar_mask列表，说明语言token和图像token之间也是同一组的，可互相关注
        
        tokens = jnp.concatenate(tokens, axis=1)  # 沿token维度concat所有图像token和语言token，得到形状为[b,s+l,d]的整体嵌入
        input_mask = jnp.concatenate(input_mask, axis=1)  # 同理token维度拼接掩码
        ar_mask = jnp.array(ar_mask)  # 把ar_mask列表转换成数组，形状为[s+l]，说明所有batch共用同一个掩码分组，且每个batch里的image和语言token都在同一组，可互相关注
        return tokens, input_mask, ar_mask

    # suffix：noisy actions (timestep变成adarms_cond条件输入,而不直接拼接到token里)
    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]  # noisy_actions形状为[b,ah,ad]，即每ah个动作token组成一个动作块，这个动作块去噪和加噪都是同时进行的，因此timestep维度直接是[b,]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],  # 嵌入好后的整体特征向量[b,s,emb]
        at.Bool[at.Array, "b s"],  # 对应的输入掩码[b,s]，代表哪些token是有效输入
        at.Bool[at.Array, " s"],  # 掩码分组累积标记[,s]，代表哪些token之间能互相关注，每个batch公用
        at.Float[at.Array, "b emb"] | None,  # pi0.5专属的时间条件输入[b,emb]，pi0则为None
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)  # 把形状为[b,ah,ad]含噪动作映射为[b,ah,width]的动作特征，width是Gemma_300m的主干宽度
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)  # [,b] -> [b,width],width同上
        if self.pi05:
            # time MLP (for adaRMS) - 对加入sincos位置编码后的时间步t做MLP映射，得到形状为[b,width]时间条件输入adarms_cond
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)  
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb  # 后续这个adarms_cond会传给Gemma_300m每一个Block的RMSNorm层做condition，以transformer架构做扩散时间条件输入，是典型的DiT式设计
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  # 生成形状为[b,ah]的全1掩码，说明动作token全是有效输入
        # image/language/state inputs do not attend to action tokens
        # ar_mask加一个1后跟ah-1个0，这个1是用来标记新分组的，说明动作token和之前的image/language/state不在同一组
        # 即action能看到前缀内容，前缀看不到action。而这ah个token在同一组，因此动作token间是双向注意力）
        ar_mask += [True] + ([False] * (self.action_horizon - 1))  
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)  # 同理token维度concat
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
    
    @at.typecheck
    def forward_denoise(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        noise: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, "b"],
    ) -> at.Float[at.Array, "b ah ad"]: 
        """
        确定性的前向去噪函数，用于 CL-LoRA 的蒸馏对齐。
        输入确定的 noise 和 time,输出模型预测的向量场 v_t。
        """
        # 1. 构造含噪动作 x_t (确定性计算，没有随机化噪声和时间步的过程了)
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 2. Embedding (前缀 + 后缀)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        # 3. 构造 Attention Mask (确定性计算)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 4. 构造 Positions
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # 5. Transformer 前向传播
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], 
            mask=attn_mask, 
            positions=positions, 
            adarms_cond=[None, adarms_cond]
        )
        # 6. 输出投影 (取后缀的最后 ah 个 token)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return v_t



    # loss fn
    @override
    def compute_loss(  # 单次前向并计算损失（因为是训练，所以是在某个随机的time step上求loss）
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)  # 把随机数种子分成三份分别用于预处理、加噪、时间步
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # 因为是训练，因此开启图像增强

        batch_shape = actions.shape[:-2]  # 取出batch维度
        noise = jax.random.normal(noise_rng, actions.shape)  # 生成形状为[b,ah,ad]的正态噪声
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # 从Beta分布采样时间步t，形状为[b,]，范围在[0.001,1.0)，避免取到0
        u_t = noise - actions  # 目标向量场，真实数据指向噪声
        v_t = self.forward_denoise(observation, actions, noise, time)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)  # 在动作维度上计算MSE，得到形状为[b,ah]的loss

    # inference fn
    @override
    def sample_actions(  # 因为是推理，所以是完整的去噪过程，每次对ah个动作循环去噪num_steps个time step，直到从纯噪声变成干净的动作输出
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,  # 基于当前的Observation(prompt+images)，对当前batch的ah个动作token同时走num_steps步去噪，才得到ah个干净动作token输出
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        
        observation = _model.preprocess_observation(None, observation, train=False)  # 不需要数据增强
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps  # 每一步的时间步长，负号表示时间步t是从1递减到0的，1是满噪声，0是GT
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))  # 生成[b,ah,ad]的随机噪声

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # 只对prefix_tokens做一次前向，把KV_cache存起来，接下来gemma_300m对当前ah个动作token的num_steps步去噪都参考这个KV_cache（即后缀的Q会attend到这些前缀的KV上来），避免每一步去噪都把prefix_tokens过一遍浪费计算资源
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)  # 前缀的参考作用在这里体现

        # 单个去噪step (每ah个动作token同时需要去噪num_steps步才得到最终动作)
        def step(carry):
            x_t, time = carry # x_t是当前的含噪动作[b,ah,ad]，time是当前的时间步
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)  # 输入当前观察、含噪动作和[,b]的时间步做后缀嵌入
            )
            
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # 先生成后缀的掩码阵，决定后缀内部关注情况

            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])  # 把[b,prefix_len]拓展成[b,suffix_len,prefix_len]，在关注者维度（第一维）扩出suffix_len,说明每个suffix_token都能按前缀内部关系看到所有前缀token
            
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)  # concat出[b,suffix_len,prefix_len+suffix_len]的完整掩码阵(此时关注者只有suffix)，代表后缀能同时关注前缀和自己
            
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )  # 检查形状
            
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1  # 得到形状为[b,suffix_len]的后缀token的绝对位置索引，前缀token的绝对位置索引在之前已经计算过了并存储在KV_cache里了，这里后缀token的绝对位置索引是接在前缀token后面的，即前缀token数加上后缀token的相对位置索引

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )  # 因为prefix之前提前做了一次前向存下了KV_cache，所以这里prefix_tokens输入None，由于整个llm有效输入长度也是suffix_len,所以注意力掩码阵第一维就是suffix_len
            assert prefix_out is None  # 确认prefix输出是None，因为prefix_tokens输入是None了，prefix_out自然也是None了
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # suffix_out的最后ah个token是动作专家输出的动作特征（同理是防御性写法），映射回动作维度得到[v_t]，形状为[b,ah,ad]

            return x_t + dt * v_t, time + dt  # 返回本次去噪后的动作和更新后的时间步

        def cond(carry):  # 用于while循环的条件函数，根据更新后的时间步判定是否继续去噪（因为可能存在浮点误差，所以加个dt/2的余量）
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  # 从时间步1的纯噪声noise开始，每步调用step，直到cond返回False
        return x_0
