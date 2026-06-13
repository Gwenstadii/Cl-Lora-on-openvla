import math
import re
import jax
import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp
import numpy as np
import openpi.shared.array_typing as at

# 实际上Einsum（负责注意力计算）和FeedForward（前馈网络计算）都是Gemma模型里的组件，这里单独拿出来是为了插入LoRA功能

@struct.dataclass
class LoRAConfig:  # Lora adapter的配置(att和ffn里的都是这个配置类）)
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732  （秩稳定）
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes. （-2和-1指最后两个维度轴，即对这两个轴做低秩分解，A×B变成A×r和r×B）
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.  （用于在einsum字符串里标记lora的特殊标签，因此原始无lora时的字符串不能含有该标签））
    label: str = "L"
    
    ###add1: new parms for CL-LoRA config
    "shared放在浅层,则只训A矩阵,B矩阵正交初始化且冻结; specific放在深层,A、B都训"
    use_orthogonal_init: bool = False  # 静态阶段通过setup影响初始化方法
    ###end1

    @property  #该装饰器将方法转换为属性调用，使得调用时无需加括号，比如LoRAConfig.scaling_value而不是LoRAConfig.scaling_value()
    def scaling_value(self) -> float:  # 计算LoRA的缩放值,用于衡量LoRA的影响力
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank  # 标准缩放（分母rank）或秩稳定缩放(分母sqrt(rank)）

###add2: orthogonal_init_function
# 正交初始化函数(用于初始化shared adapter的B矩阵，保持其正交性以稳定训练)
def orthogonal_init(key, shape, dtype=jnp.float32):
    """Generates a random orthogonal matrix using Flax's robust built-in initializer."""
    # 直接调用 Flax 官方的正交初始化器，它能完美处理任意形状的多维张量和 QR 分解逻辑
    return nn.initializers.orthogonal()(key, shape, dtype)
###end2

# 用于Attention块内的lora分解
class Einsum(nn.Module):  # 初始化权重矩阵并决定张量计算方式以及是否插入LoRA
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight. （主权重w，形状以[8，16]为例）
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape) # 初始化主权重W，param是nn.Module的一个方法，用于定义可训练参数,用法是self.param(name, init_fn, shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)  #shape_a和shape_b分别是LoRA的W_a和W_b矩阵的形状,此时先写为[8,16]
            shape_a[config.axes[1]] = config.rank  
            shape_b[config.axes[0]] = config.rank  # 根据config里面的axes和rank对A和B做低秩分解，shape_a变为[8,r]，shape_b变为[r,16]
            
            ###add3: init two types of CL-LoRA adapters differently
            if config.use_orthogonal_init:
                # A正交初始化，B全零初始化（CL-Lora）
                self.w_a = self.param("lora_a", orthogonal_init, shape_a)
                self.w_b = self.param("lora_b", nn.initializers.zeros, shape_b)
            else:
                # 正常初始化
                self.w_a = self.param("lora_a", config.init_fn, shape_a)
                self.w_b = self.param("lora_b", config.init_fn, shape_b)
            ###end3
            
    @nn.compact
    def __call__(self, eqn: str, x, freeze_a: bool = False, block_scale: float | None=None, ):  # x为输入张量
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))  # 主权重W的einsum计算，其中eqn是Einsum公式字符串，比如"ab,bc->ac"，用来指定怎么乘

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)  # 用原始的eqn生成LoRA的两个einsum字符串
            
            ###add4: if shared，freeze A matrix
            w_a = self.w_a.astype(dtype)
            w_b = self.w_b.astype(dtype)
            # 使用 jnp.where 替代 if，完美兼容 JAX Tracer 并且能正确阻断梯度
            w_a = jnp.where(freeze_a, jax.lax.stop_gradient(w_a), w_a)
            ###end4
            
            lora = jnp.einsum(eqn_a, x, w_a) # x和W_a的einsum计算
            lora = jnp.einsum(eqn_b, lora, w_b) # W_b和上一步结果的einsum计算
            scale = config.scaling_value  # 获取LoRA的缩放值
            
            ###add5: if specific, apply block-wise scaling(实际上用block_sacle来调当前层的specific lora的影响力，视任务而定)
            if block_scale is not None:
                scale = scale * block_scale
            # 强制将 scale 转换为输入精度，斩断 JAX 向 float32 的自动类型提升
            scale = jnp.asarray(scale, dtype=dtype)
            ###end5
            result = result + lora * scale  # 将LoRA的结果按缩放值加到主权重结果上得到最终结果

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:  # 检查eqn字符串中是否含有LoRA的标签L，有则报错
            raise ValueError(f"L already in eqn: {eqn}")

        if not (m := re.match("(.*),(.*)->(.*)", eqn)):  # 检查eqn字符串是否符合einsum的格式，不符合则报错（re.match是判断eqn是不是"(.*),(.*)->(.*)"格式，若是，则通过:=赋值给m，使m变成一个实例，存储匹配的各项信息；若不是，则m是None）
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()  # 把实例m的各分组返回，即把eqn字符串拆分成左侧、中间和输出三部分，比如“b s h,h o -> b s o”拆分成“b s h”，“h o”，“b s o”

        assert self.lora_config is not None  # assert是断言语句，检查条件是否为真，不为真则报错
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)  # 根据config里的axes获取rhs字符串中对应位置的标签，比如axes=(-2,-1)，rhs="h o"，则a_label='h',b_label='o'
        label = self.lora_config.label  # 获取config里的LoRA标签，比如'L'

        # str.relace(old, new)方法用于字符串替换，把old替换成new，返回一个新字符串        
        a_rhs = rhs.replace(b_label, label)  # rhs从"ho"变成"hL"
        a_out = out.replace(b_label, label)  # out从"bso"变成"bsL"
        eqn_a = f"{lhs},{a_rhs}->{a_out}"  # 生成LoRA里W_a矩阵的einsum字符串，比如"b s h,h L -> b s L"

        b_rhs = rhs.replace(a_label, label) # rhs从"ho"变成"L o"
        eqn_b = f"{a_out},{b_rhs}->{out}"  # 生成LoRA里W_b矩阵的einsum字符串，比如"b s L,L o -> b s o"

        return eqn_a, eqn_b

# 用于FFN块内的lora分解
class FeedForward(nn.Module):  # 前馈网络的LoRA实现
    """Feed forward module."""
    
    features: int  # FFN的输入或输出维度
    hidden_dim: int  # FFN中间隐藏层的维度
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        # lecun_normal是一种权重初始化方法,in_axis和out_axis指定输入和输出维度的轴位置，batch_axis指定batch轴的位置以不参与归一化）
        # 第一类权重
        self.w_gating = self.param(  # 初始化门控权重W_gating，形状为(2, features, hidden_dim)，第一维度的2代表两个分支。其中w_gating[0]是gate分支的权重，用来计算门控值；w_gating[1]是ff1分支的权重，用来计算ff1值
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),  # 此处第一维就是batch轴
            (2, self.features, self.hidden_dim),
        )
        # 第二类权重
        self.w_linear = self.param(  # 初始化线性权重W_linear，形状为(hidden_dim, features)，是用来最后降维回features维度的
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        # Lora的参数初始化为None(禁用)
        self.w_gating_lora = None
        self.w_linear_lora = None
        
        if config := self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.            
            ###add6: get_init_fn
            def get_init_fn(is_down_projection):
                if config.use_orthogonal_init:
                    # 如果是下投影(Down)，用正交初始化；如果是上投影(Up)，用零初始化
                    return orthogonal_init if is_down_projection else nn.initializers.zeros
                return config.init_fn
            ###end6
            
            # 初始化gate分支的lora参数，把原来（2，features, hidden_dim）分解为(2, features, rank)和(2, rank, hidden_dim)
            
            ###add7: use get_init_fn instead of config.init_fn directly
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", get_init_fn(True), (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", get_init_fn(False), (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            # 初始化linear分支的lora参数，把原来（hidden_dim, features）分解为(hidden_dim, rank)和(rank, features)
            self.w_linear_lora = (
                self.param("linear_lora_a", get_init_fn(True), (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", get_init_fn(False), (self.lora_config.rank, self.features)),
            )
            ###end7

    @nn.compact
    def __call__(self, x, freeze_a: bool = False, block_scale: float | None = None):  # x的形状是（batch_size， seq_len, features）
        dtype = x.dtype  # original dtype, could be half-precision
        
        # gate分支的前向计算
        #（无lora时：gate_value=nn.gelu(x × (w_gating[0])。有lora时：gate_value=nn.gelu(x × (w_gating[0] + w_gating_lora_a[0] × w_gating_lora_b[0])）
        ###add9: pass block_scale and freeze_a to _dot for gate branch
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]), freeze_a, block_scale # A if 条件 else B 用法：如果条件为真则取A，否则取B
        )  
        gate_value = nn.gelu(ff_gate)  # 形状变为（batch_size， seq_len, hidden_dim）
        
        # ff1分支的前向计算
        #（无lora时：ff1=x × (w_gating[1])。有lora时：ff1=x × (w_gating[1] + w_gating_lora_a[1] × w_gating_lora_b[1])）
        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]), freeze_a, block_scale
        )  # 形状变为（batch_size， seq_len, hidden_dim）
        activations = gate_value * ff1  # 门控值和ff1值按元素对应相乘得到最终激活值，代表“经过门控筛选后的信息”，形状仍为（batch_size， seq_len, hidden_dim）

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora, freeze_a, block_scale)
        ###end9
        assert outputs.dtype == dtype
        return outputs

    # 含lora的点积计算函数
    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None, freeze_a: bool = False, block_scale: float | None = None) -> at.Array:  # 定义了参数的类型以及返回值的类型

        base = jnp.dot(x, w.astype(x.dtype))  # 主权重base由x和w点积计算得出（这里的dot会自动对最后两个轴做矩阵乘法）
        
        if lora_weights is None:
            return base
        w_a, w_b = lora_weights[0].astype(x.dtype), lora_weights[1].astype(x.dtype)
        
        ###add8: if shared, freeze A matrix by stop_gradient
        # 同样使用 jnp.where 替代 Python if 控制流
        w_a = jnp.where(freeze_a, jax.lax.stop_gradient(w_a), w_a)
        lora = jnp.dot(jnp.dot(x, w_a), w_b)

        scale = self.lora_config.scaling_value
        if block_scale is not None:
            scale = scale * block_scale
        # 强制将 scale 转换为输入精度，斩断 JAX 向 float32 的自动类型提升
        scale = jnp.asarray(scale, dtype=x.dtype)
        return base + lora * scale
        ###end8
