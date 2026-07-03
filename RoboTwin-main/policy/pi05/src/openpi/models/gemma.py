# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Gemma_list_for_Pi0/05
"""Gemma adaptation for Pi, taken from big_vision.  # Gemma是Big Vision里的一个大型transformer模型

We follow this einsum axis naming convention:
  B: batch
  T: query length  (当前作为query的token序列长度)
  S: k/v length  (当前作为key/value的token序列长度)
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head  (每个k/v头对应的query头数)
  H: head dim
  D: d_model ("features") 也是 “width”
"""
# 区分：此处Config是单个Gemma专家的配置，configs是多个专家的配置列表，config是具体configs中某个专家的配置,而/src/openpi/training/config.py中是训练的相关配置
from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:  # Gemma模型的总配置，包括transformer结构参数和LoRA和CL-LoRA相关配置
    width: int  # 每个token经过embedding后的维度，跟features和d_model是同义词,属于是整个模型的主干宽度，因为经过不同块处理后维度依旧会变回这个width
    depth: int  # 层数，即同样的transformer block堆叠多少次
    mlp_dim: int  # FFN里面的隐藏层维度
    num_heads: int  # Query头数，而num_heads×head_dim=width，即一开始分头处理，最后则把处理结果拼接回width维度
    num_kv_heads: int  # Key/Value头数
    head_dim: int # 每个头的维度
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)  #Gemma模型层级上的lora配置字典，dict的key为是字符串，这里主要有att和ffn两个key，对应是注意力模块里的lora和前馈网络里的lora。value则为对应的LoRAConfig实例

    ###add3: new parms for CL-LoRA
    shared_pos: list[int] = dataclasses.field(default_factory=list)
    specific_pos: list[int] = dataclasses.field(default_factory=list)  # 这两个pos动态阶段通过__call__影响冻结和块权重
    use_block_weight: bool = False
    task_bank_size: int = 1
    use_task_lora_b_bank: bool = False
    use_task_block_weight_bank: bool = False
    freeze_shared_lora: bool = False
    freeze_lora_b: bool = False
    freeze_lora_a: bool = False
    ###end_add3

Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_cl_lora", "gemma_300m_cl_lora"]  # 定义了一个类型别名Variant，限定了它只能取上述字符串中的一个值，代表不同的Gemma变体类型


def get_config(variant: Variant) -> Config:  #根据variant变体类型返回对应的Gemma配置，即定义好的Config实例
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":  # 视觉语言含lora微调分支
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":  # 动作头含lora微调分支
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    if variant == "gemma_2b_cl_lora":
        depth = 18
        split = 9 # 默认前 9 层作为 Shared Layers
        return Config(
            width=2048,
            depth=depth,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            # 定义 LoRA 配置
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0, use_orthogonal_init=True), "ffn": lora.LoRAConfig(rank=16, alpha=16.0, use_orthogonal_init=True)},
            shared_pos=list(range(split)),
            specific_pos=list(range(split, depth)),
            use_block_weight=True,
        )
    if variant == "gemma_300m_cl_lora":
        depth = 18
        split = 9
        return Config(
            width=1024,
            depth=depth,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0, use_orthogonal_init=True), "ffn": lora.LoRAConfig(rank=32, alpha=32.0, use_orthogonal_init=True)},
            shared_pos=list(range(split)),
            specific_pos=list(range(split, depth)),
            use_block_weight=True,
        )
    raise ValueError(f"Unknown variant: {variant}")


def with_cl_lora_options(
    config: Config,
    *,
    task_bank_size: int,
    use_task_lora_b_bank: bool,
    use_task_block_weight_bank: bool,
    freeze_shared_lora: bool,
    freeze_lora_b: bool,
    freeze_lora_a: bool,
) -> Config:
    """Return a Gemma config with CL-LoRA v2 task-bank controls applied."""
    lora_configs = {
        name: dataclasses.replace(
            lora_config,
            task_bank_size=task_bank_size,
            use_task_lora_b_bank=use_task_lora_b_bank,
            freeze_lora_a=freeze_lora_a,
        )
        for name, lora_config in config.lora_configs.items()
    }
    return dataclasses.replace(
        config,
        lora_configs=lora_configs,
        task_bank_size=task_bank_size,
        use_task_lora_b_bank=use_task_lora_b_bank,
        use_task_block_weight_bank=use_task_block_weight_bank,
        freeze_shared_lora=freeze_shared_lora,
        freeze_lora_b=freeze_lora_b,
        freeze_lora_a=freeze_lora_a,
    )


@at.typecheck
class RMSNorm(nn.Module):  # RMSNorm归一化模块
    @nn.compact
    def __call__(self, x, cond):  # x是输入张量(B,T,D)。cond是条件，形状一般为（B,D）或None，一般代表流匹配的时间步信息
        # 计算归一化的输入
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # 在特征维度D上计算均方值，keepdim所以var形状为(B,T,1)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # reciprocal是取倒数，sqrt是开根号，asarray保证输出是JAX张量，由于广播var会扩展成(B,T,D)，因此normed_inputs形状为(B,T,D)

        if cond is None:
            # regular RMSNorm (普通)
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))  # 初始化一个可学习的缩放参数scale，形状为(D,)
            normed_inputs = normed_inputs * (
                1 + scale  # 缩放，由于广播scale会变成(B,T,D)
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm （自适应）
        # nn.Dense用法是（features, kernel_init，dtype），生成全连接层
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)  # 条件cond通过一个全连接层，形状从(B,D)映射到(B,3D)，3D是因为后面要拆分成scale,shift,gate三个部分
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)  # 在特征维度拆分成三个形状为(B,1,D)的张量，方便后续广播
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32  （做缩放且偏移）
        return normed_inputs.astype(dtype), gate  # 门控值会用于后续的残差连接


@at.typecheck
class Embedder(nn.Module):  # 语言信息的嵌入模块（根据token id做编码，或者解码回词表维度）
    """Embedder module."""

    vocab_size: int  # 词表大小
    embed_dim: int  # 嵌入维度

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )  # 初始化嵌入矩阵

    def encode(self, x):
        x = self.input_embedding_table[(x,)]  # 根据输入的token id在嵌入矩阵里取出对应的行向量
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 乘一个缩放因子
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)  # 用嵌入矩阵的转置和x做点积映射回词表维度


@at.typecheck
class Attention(nn.Module):  # Attention块(因为调用了lora.Einsum,所以已含lora)
    """Attention module."""

    configs: Sequence[Config]  # configs是一个包含多个Config对象的列表，对应多个不同Gemma专家。config则是对应单个Config对象的实例

    @nn.compact
    def __call__(
        self,
        xs,
        positions,
        attn_mask,
        kv_cache,
        freeze_as: Sequence[bool] | None = None,
        freeze_bs: Sequence[bool] | None = None,
        use_task_banks: Sequence[bool] | None = None,
        block_scales: Sequence[float] | None = None,
        task_id=None,
    ):  # xs是一个包含多个输入张量的列表，对应多个不同Gemma专家的输入
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work  （不同专家的token最后会被拼接到一起做自注意力，因此头维头数kv头数等必须一致）
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision，统一精度

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):  # 根据每个专家的输入x和配置Config创建对应的QKV计算einsum方式,索引为i
            # 单个Attention块里按专家i进行区分的原因是因为这两个专家的注意力信息会融合
            # attention_2_gemma2b 丨  attention_2_gemma300m
            # attention_1_gemma2b 丨  attention_1_gemma300m
            #    x1(专家1的输入)           x2(专家2的输入)

            if x is None:
                continue

            do_freeze = freeze_as[i] if freeze_as is not None else False
            do_freeze_b = freeze_bs[i] if freeze_bs is not None else False
            b_scale = block_scales[i] if block_scales is not None else None
            use_task_bank = use_task_banks[i] if use_task_banks is not None else False

            # 第一种情况：KV头数等于Q头数，说明每个Q头都有对应的KV头，是全注意力
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),  # 由于此时QKV头数相等，因此线性变换阵可一起初始化，形状为3KDH,3代表QKV的，K为头数，DH才是真正跟输入做点积的维度，把原来每个token的D维映射到每个头的H维
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),  # 3和头数都是并行维度
                    lora_config=config.lora_configs.get("attn"),  # 取出当前专家的config里的lora配置字典中key为"attn"的LoRAConfig实例 （第一个PaliGemma专家QKV权重也可以微调）
                )
                qkvs.append(qkv_einsum(
                    "BSD,3KDH->3BSKH",
                    x,
                    freeze_a=do_freeze,
                    block_scale=b_scale,
                    task_id=task_id,
                    use_task_bank=use_task_bank,
                    freeze_b=do_freeze_b,
                ))  # 对输入x按照字符串方式计算qkv并加入列表中（x形状是BSD，因为此时S跟T是相等的，既是查询者也是被查询者）
                                                               # 实际是BSD的D跟3KDH的DH做映射，D变成H，调整顺序后得到3BSKH,3代表QKV，B是batch，S是token长度，K是头数，H是头维(相当于每个token的width被映射到K个H里)
                                                               # append进列表里的是一个形状为3BSKH的张量

            # 第二种情况：KV头数少于Q头数，说明多个Q头共享同一个KV头
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),  # Q头的Einsum实例化
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum(
                    "BTD,NDH->BTNH",
                    x,
                    freeze_a=do_freeze,
                    block_scale=b_scale,
                    task_id=task_id,
                    use_task_bank=use_task_bank,
                    freeze_b=do_freeze_b,
                )  # 同理对形状BTD的输入x计算q，得到BTNH形状的q,同理把输出p维度排列好，Batch里每个token的D维映射到N个H维里
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),  # KV头的Einsum实例化
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),  # 2和kv头数都是并行维度，2代表K和V
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum(
                    "BSD,2KDH->2BSKH",
                    x,
                    freeze_a=do_freeze,
                    block_scale=b_scale,
                    task_id=task_id,
                    use_task_bank=use_task_bank,
                    freeze_b=do_freeze_b,
                )  # 同理对形状BSD的输入x计算k和v，得到2BSKH形状的k和v,同理把输出p维度排列好，Batch里每个token的D维映射到K个H维里（2是因为KV一起算）
                qkvs.append((q, k, v))  # append进列表里的是一个包含q,k,v三个张量的元组，形状分别是BTNH,BSKH,BSKH

        # 不同专家之间的注意力融合
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 第一种情况：KV头等于Q头  （zip对张量则默认第一个维度是可迭代位置，即此处的3）
        # *qkvs同理拆解列表得到专家1的(3,B,S1,K,H)和专家2的(3,B,S2,K,H)，3代表QKV，zip则取出来得到两个专家的Q的BSKH并同理在axis=1即S上拼接，KV同理

        # 第二种情况：KV头少于Q头 （zip对元组直接并行取出并打包）
        # *qkvs先把qkvs里两个专家的qkv分别取出来，即qkvs[0]、qkv[1]得到(q1,k1,v1)和(q2,k2,v2)，zip则并行取出来得到(q1,q2),(k1,k2),(v1,v2)这三个元组
        # 然后对这三个y1y2y3在元素的axis=1即token长度维度上进行concat，比如q1形状为(B,T1,N,H),q2形状为(B,T2,N,H)，concat后得到的q形状为(B,T1+T2,N,H)，k和v同理，得到所有专家共享的qkv

        # 给q和k应用rope位置编码
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5  # 给Q做缩放防止数值异常

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype  # 验证统一精度

        # KV缓存机制 （之前定义的Q长度T和KV长度S只有第一次预填充时是相等的，推理时S会随着积累信息的增加而增长，而T则保持不变，如果每次都要重新计算过去的kv则效率低下，因此引入kv cache）
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)  # kv cache不为空则把新的k和v按token维度拼接到原有缓存的后面，才是完整的k和v

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)  # 把q的头数N拆分成K×G的形式，G是每个kv头对应的query头数,K是kv头数
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)  #BTKGH的q和BSKH的k做点积得到BKGTS形状的logits，代表本Batch里第K组的第G个query头的第T个token对第S个历史token的注意力分数，这是初始注意力分数
        # einsum是不带lora的，Einsum才带

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )  # 确保注意力掩码形状为(B, 1, T, S)

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py  （设置一个很小的负数，用于掩码）

        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        # 把attn_mask扩容成[B,1,1,T,S],跟形状为[B,K,G,T,S]的logits，由于广播机制，attn_mask形状会变成[B,K,G,T,S]，然后对应位置上如果mask值为True则保留logits的值，否则替换成big_neg，从而实现掩码功能
        # 无论是G=1或者G＞1都适用，因为K×G始终是q头的个数，即所有q头都要符合掩码规则

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)  # 最后一轴做softmax得到归一化后的第T个token对第S个token的权重

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)  # 按照注意力分数加权v得到BTKGH形状的结果，代表q提取到的信息
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")  # 把K和G轴合并回N轴，得到BTNH形状

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:

                do_freeze = freeze_as[i] if freeze_as is not None else False
                do_freeze_b = freeze_bs[i] if freeze_bs is not None else False
                b_scale = block_scales[i] if block_scales is not None else None
                use_task_bank = jnp.logical_not(jnp.asarray(do_freeze, dtype=jnp.bool_))

                end = start + x.shape[1]  # x.shape[1]是当前专家输入的token长度
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )  # 实例化把num_heads×head_dim映射回width的形状为[N,H,D]的线性矩阵
                out.append(out_einsum(
                    "BTNH,NHD->BTD",
                    encoded[:, start:end],
                    freeze_a=do_freeze,
                    block_scale=b_scale,
                    task_id=task_id,
                    use_task_bank=use_task_bank,
                    freeze_b=do_freeze_b,
                ))  # [:,start:end]等价[:, start:end, :, :],在token维度拆出当前专家提到的特征，并映射回BTD形状并加入输出列表
                start = end  # 下一次的start位置更新
            else:
                out.append(None)

        return out, (k, v)  # 返回每个专家对应的输出列表，以及更新后的kv cache


@at.typecheck
class FeedForward(nn.Module):  # FFN块（没有插入lora，因此实际用的还是lora.Feedforward）
    """Feed forward module."""

    features: int  # 本身的width
    hidden_dim: int  # FFN的隐藏层维度

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)  # 初始化两个线性变换矩阵，形状均为(features,hidden_dim)，分别用于计算gate和ff1
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)  # 形状为(B,T,hidden_dim)，经过gelu激活函数得到门控值

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1  # 门控值和ff1逐元素相乘，得到激活值

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)  # 把最后一维映射回(features)维度
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):  # Transformer块
    """Transformer block."""
    "Structure: input -> RMSNorm -> Attention -> Dropout -> Residual -> RMSNorm -> FFN -> Residual -> output"
    configs: tuple[Config, ...]  # 各专家的配置列表

    dropout: float = 0.0  # dropout率
    dropout_bdims: tuple[int, ...] = ()  # 代表dropout的维度

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        deterministic=True,
        freeze_as_vec=None,
        freeze_bs_vec=None,
        use_task_banks_vec=None,
        block_scales_vec=None,
        task_id=None,
    ):  # noqa: FBT002，deterministic代表是否在推理模式,训练时dropout，推理时不dropout
        xs = sharding.activation_sharding_constraint(xs)  # 强制把各专家的输入x按策略划分到多个GPU上面
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x  # 作用是根据dropout率实例化dropout模块，若dropout率为0则不做dropout

        attn = Attention(configs=self.configs, name="attn")  # 实例化Attention块，此时传入的configs是多专家的配置，因为Attention块里会融合各专家的注意力信息

        pre_attn = []  # 存放每个专家经过注意力前的输入归一化的结果
        gates = []  # 存放由RMSNorm得到的门控值
        for i, x in enumerate(xs):  # xs实际上对应pi0.py里的prefix_tokens和suffix_tokens两个专家的输入列表
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901，对每个专家的输入x做归一化，adarms_cond[i]是当前专家的时间嵌入条件信息
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(
            pre_attn,
            positions,
            attn_mask,
            kv_cache,
            freeze_as=freeze_as_vec,
            freeze_bs=freeze_bs_vec,
            use_task_banks=use_task_banks_vec,
            block_scales=block_scales_vec,
            task_id=task_id,
        )  # 经过Attention块，得到每个专家对应的输出列表和更新后的kv cache
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)  # 对每个专家的注意力输出做dropout
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]  # 分别对每个专家的输入x和注意力输出y按门控值gate做残差连接
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []  # 重新初始化门控值列表
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:

                curr_freeze = freeze_as_vec[i] if freeze_as_vec is not None else False
                curr_freeze_b = freeze_bs_vec[i] if freeze_bs_vec is not None else False
                curr_use_task_bank = use_task_banks_vec[i] if use_task_banks_vec is not None else False
                curr_scale = block_scales_vec[i] if block_scales_vec is not None else None

                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901 同理FFN前也要做归一化
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(
                    x,
                    freeze_a=curr_freeze,
                    block_scale=curr_scale,
                    task_id=task_id,
                    use_task_bank=curr_use_task_bank,
                    freeze_b=curr_freeze_b,
                )  # x经过含lora的前馈网络模块
            out.append(x)
            gates.append(gate if x is not None else None)  # 存放由RMSNorm得到的门控值

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]  # 用新门控值做残差连接
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache

# KV cache的形状是一个元组，包含key和value两个浮点张量，分别对应形状为[l, b, _t, _k, _h]和[l, b, _t, _v, _h]，代表第l层第b个batch的第_t个token的第_k个key头和第_v个value头的第_h个head维
KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):  # 此处是gemma的transformer整体实现（只有第一个专家有Embedder，而block的话是两个专家同时并行实例化的）
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert（专家配置列表）
    embed_dtype: str  # 嵌入的精度

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False  # 是否开启AdaRMSNorm

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)  # 确保所有专家的深度相同

        # 给第一个专家实例化Embedder模块
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )

        # 1. 使用局部变量 list (它绝对是列表，不会被 Flax 变成元组)
        block_weights_list = []

        for i, config in enumerate(self.configs):
            if hasattr(config, "use_block_weight") and config.use_block_weight and config.specific_pos:
                n_specific = len(config.specific_pos)
                weight_shape = (
                    (config.task_bank_size, n_specific)
                    if config.use_task_block_weight_bank
                    else (n_specific,)
                )
                bw = self.param(
                    f"cl_block_weights_{i}",
                    nn.initializers.zeros,
                    weight_shape,
                )
                block_weights_list.append(bw)  # 局部变量 append 绝对安全
            else:
                block_weights_list.append(None)

        # 2. 循环结束后，一次性赋值给 self
        # 此时 Flax 把它转成元组也没关系了，因为我们已经不需要再 append 了
        self.cl_block_weights = block_weights_list

        # 调用nn.remat把原先的Block块优化成block_cls，从而节省显存(不保存中间激活值，反传时重算)
        block_cls = nn.remat(
            Block,
            prevent_cse=False,  # 不沿用公共子表达式，要重算
            static_argnums=(5,),  # 此处5因为从0开始数，所以是Block.__call里的第6个参数，即deterministic这个布尔值不参加反传
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        # 用nn.scan把多个Block块串联起来形成深层transformer （本来block的参数字典是分别放着里面如注意力模块FFN等权重，而scan的用法是在这些参数的第0维拓展成深度D，形成多层的权重字典，如[18，"ATTN","FFN"...]）
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},  # 指定参数字典里的第0维是层数索引，每层有一个block的独立权重
            split_rngs={"params": True, "dropout": True},  # 每层block的参数和dropout都要独立采样随机数
            in_axes=(  # in_axes决定了输入给Block.__call的那些参数在循环过程中怎么处理（其中第一个输入xs会被当作carry项自动处理，即上一层的输出＝下一层的输入）
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,  # freeze_as_stack [Depth, Num_Experts]
                0,  # freeze_bs_stack [Depth, Num_Experts]
                0,  # use_task_bank_stack [Depth, Num_Experts]
                0,  # block_scales_stack [Depth, Num_Experts]
                nn.broadcast,  # task_id
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic （0代表按第0维即层维切分，每层的kv cache是独立的）（broadcast代表所有层共享）
            length=self.configs[0].depth,  # 循环的层数是专家的深度
        )(
            configs=self.configs,  # 在此处传入了专家列表
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )  # 具体实例化时传入每个block的configs和dropout参数
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]  # 为每个专家创建独立的最终的RMSNorm模块


    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]: # 对[B,T]形状的文本token id做嵌入编码，返回形状为[B,T,D]的嵌入表示
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],  # 各专家对应的嵌入表示列表，形状均为[B,T,D]，D为对应专家的width，比如[B,T1,D1]、[B,T2,D2]
        positions: at.Int[at.Array, "b t"],  # 位置索引，形状为[B,T1+T2]，专家共享（因为专家0接受视觉语言信息的嵌入，专家1接受含噪动作、状态、时间步的嵌入，虽然二者embedded不同，但要在Attention块内信息融合，所以position共用才能对齐序列顺序坐标系）
        mask: at.Bool[at.Array, "b t s"],  # 注意力掩码阵，形状为[B,T1+T2,T1+T2]，专家共享
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,  # 流匹配时间嵌入条件列表，形状均为[B,D]
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,  # 是否在推理模式
        task_id=None,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:  # 输出各专家对应的形状为[B,T,D]的结果列表，以及更新后的kv cache
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)  # 把各专家的嵌入表示统一成指定的embed_dtype精度
        mask = jnp.asarray(mask)[:, None, :, :]  # [B,T,D] -> [B,1,T,S]，在第1维插入一个维度以匹配Attention块里对掩码的要求
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        depth = self.configs[0].depth
        num_experts = len(self.configs)
        freeze_a_cols = []
        freeze_b_cols = []
        use_task_bank_cols = []
        scale_cols = []
        task_id = jnp.asarray(0 if task_id is None else task_id, dtype=jnp.int32)
        for i, config in enumerate(self.configs):
            # 准备 Freeze 列 (Shared 层为 True)
            # 默认全 False
            freeze_a_col = jnp.zeros((depth,), dtype=bool)
            if config.freeze_shared_lora and hasattr(config, "shared_pos") and config.shared_pos:
                # 将 shared_pos 对应的层设为 True
                # 注意 jax 数组更新语法
                idx = jnp.array(config.shared_pos, dtype=jnp.int32)
                freeze_a_col = freeze_a_col.at[idx].set(True)
            freeze_a_cols.append(freeze_a_col)

            freeze_b_col = jnp.zeros((depth,), dtype=bool)
            if config.freeze_lora_b:
                freeze_b_col = jnp.ones((depth,), dtype=bool)
            elif config.freeze_shared_lora and hasattr(config, "shared_pos") and config.shared_pos:
                idx = jnp.array(config.shared_pos, dtype=jnp.int32)
                freeze_b_col = freeze_b_col.at[idx].set(True)
            freeze_b_cols.append(freeze_b_col)

            use_task_bank_col = jnp.zeros((depth,), dtype=bool)
            if config.use_task_lora_b_bank and hasattr(config, "specific_pos") and config.specific_pos:
                idx = jnp.array(config.specific_pos, dtype=jnp.int32)
                use_task_bank_col = use_task_bank_col.at[idx].set(True)
            use_task_bank_cols.append(use_task_bank_col)

            # 准备 Scale 列 (Specific 层为 mu，其他为 1.0)
            # 默认全 1.0
            scale_col = jnp.ones((depth,), dtype=jnp.float32)

            raw_weights = self.cl_block_weights[i]
            if raw_weights is not None and hasattr(config, "specific_pos") and config.specific_pos:
                idx = jnp.array(config.specific_pos, dtype=jnp.int32)
                if config.use_task_block_weight_bank:
                    raw_weights = jnp.take(raw_weights, task_id, axis=0)

                # Bounded, positive, identity-at-init scaling:
                # raw=0 -> effective_scale=1.0; range ~= (0.5, 1.5)
                effective_weights = 1.0 + 0.5 * jnp.tanh(raw_weights.astype(jnp.float32))

                scale_col = scale_col.at[idx].set(effective_weights)


            scale_cols.append(scale_col)
        # 此时freeze_cols和scale_cols形状均为[专家0数组(18,), 专家1数组(18,)]
        # 堆叠成 [Depth, Num_Experts]
        freeze_a_stack = jnp.stack(freeze_a_cols, axis=1)  # [18,2]
        freeze_b_stack = jnp.stack(freeze_b_cols, axis=1)  # [18,2]
        use_task_bank_stack = jnp.stack(use_task_bank_cols, axis=1)  # [18,2]
        scale_stack = jnp.stack(scale_cols, axis=1)  # [18,2]

        embedded, kv_cache = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            deterministic,
            freeze_a_stack,
            freeze_b_stack,
            use_task_bank_stack,
            scale_stack,
            task_id,
        )  # 调用layers实例，输入是block定义要求的，输出是block里定义的return的xs和kv_cache

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)  # 确保所有专家的输出精度一致

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache  # 对每个专家的输出e，根据各自时间嵌入条件a在各自的最终RMSNorm层f做归一化，返回归一化后的结果列表和kv cache （只取[0]是因为RMSNorm返回的gate已没用）

    # 触发初始化方法（因为linen要求）
    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))  # 传一个[1,1]初始化嵌入环节
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# rope位置编码函数
def _apply_rope(x, *, positions, max_wavelength=10_000):  # positions里是每个token的绝对位置
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""  # B为batchsize, L为token长度, H为头数, D为头维
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)  # 频率指数，把D维两两分成一组，得到D/2个频率指数
    timescale = max_wavelength**freq_exponents  # 形状为(D/2,)
    radians = positions[..., None] / timescale[None, None, :]  # [B,L,1]/[1,1,D/2]通过广播机制得到形状为[B,L,D/2]的radians,即B里第L个token的D/2组复数对应角度值
    radians = radians[..., None, :]  # [B,L,1,D/2]，补回头维度
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)  # 计算sin和cos，形状均为[B,L,1,D/2]
    x1, x2 = jnp.split(x, 2, axis=-1)  # 把x在最后一维D上拆分成两个形状均为[B,L,H,D/2]的张量，分别是实部和虚部
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)  # 做旋转后拼回来本身的复数，形状为[B,L,H,D]
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


# 专家的参数的命名函数
def _name(name, i):
                     # 如果i=0即第一个专家，则直接返回name，而后续的专家全部返回name_i
                     # 第一个专家是负责视觉语言处理的PaliGemma，因此命名上要跟PaliGemma checkpoint保持一致，即不带索引i，方便直接加载已有训练点；而第二个专家则是新加的动作专家，带索引i，权重从头开始训练
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


# 残差连接函数
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)  # x和y要么同时为空，要么同时不为空
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate  # 把输入信号x和经过模块处理后的信号y按门控值gate加权后相加
