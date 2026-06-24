# CL-LoRA 核心技术实现文档

> 项目：基于 CL-LoRA 的 OpenVLA 在 LIBERO 任务上的持续学习评估  
> 本文档汇总 CL-LoRA 算法的核心代码实现，聚焦于关键技术而非辅助代码。

---

## 目录

1. [架构总览](#1-架构总览)
2. [CLLoRALinear：核心低秩适配层](#2-clloralinear核心低秩适配层)
3. [模型注入与层级划分](#3-模型注入与层级划分)
4. [PI Task Bank：跨任务参数管理](#4-pi-task-bank跨任务参数管理)
5. [正交初始化与 A 矩阵冻结](#5-正交初始化与-a-矩阵冻结)
6. [Block-Scale 门控机制](#6-block-scale-门控机制)
7. [多任务训练与损失组合](#7-多任务训练与损失组合)
8. [知识蒸馏：教师模型快照](#8-知识蒸馏教师模型快照)
9. [原型回放机制](#9-原型回放机制)
10. [推理评估中的 CL-LoRA 加载](#10-推理评估中的-cl-lora-加载)

---

## 1. 架构总览

CL-LoRA 在 OpenVLA 中的核心设计原则（五条冻结原则）：

| 原则 | 说明 | 对应论文 |
|------|------|----------|
| **冻结视觉编码器** | SigLIP 完全不注入 LoRA，跨任务保持特征稳定 | Sec 3.2 |
| **冻结 LLM 主干权重** | 仅 LoRA adapter 参数可训，base weight 全部冻结 | Sec 3.1 |
| **Shared 层** | 浅层 LoRA-A 正交初始化 + 冻结，保护旧知识 | Sec 3.3 |
| **Specific 层** | 深层 LoRA-A/B 可训 + block-scale 门控 | Sec 3.4 |
| **轻量 Action Head** | 仅 L1Regression 头可训，不引入额外复杂度 | Sec 4.1 |

核心文件分布：

| 文件 | 作用 |
|------|------|
| `vla-scripts/cl_lora.py` | CLLoRALinear 模块 + 注入函数 + Task Bank |
| `vla-scripts/train_cl_lora.py` | 多阶段训练循环（KD + replay） |
| `pi0.5代码/lora.py` | JAX 参考实现（Einsum / FeedForward） |
| `vla-scripts/build_replay_buffer_openvla.py` | 原型回放缓冲区构建 |
| `vla-scripts/replay_dataset.py` | 回放数据加载器 |
| `experiments/robot/openvla_utils.py` | 评估时 CL-LoRA 权重加载 |

---

## 2. CLLoRALinear：核心低秩适配层

**文件位置：** `openvla-oft/vla-scripts/cl_lora.py` 第 30-109 行  
**论文对应：** Section 3.2 — CL-LoRA Low-Rank Adaptation Formulation

### 2.1 模块定义

`CLLoRALinear` 继承自 `nn.Module`，替代标准 `nn.Linear` 作为 Attention 和 FFN 的线性层。其前向公式为：

```
output = W·x + (α/r) · g · B · A · x
```

其中：
- `W`：冻结的原始权重
- `A ∈ R^{r×d_in}`：低秩投影矩阵（降维）
- `B ∈ R^{d_out×r}`：低秩投影矩阵（升维）
- `α/r`：缩放因子（alpha/rank）
- `g`：block-scale 门控值（仅 specific 层生效）

```python
class CLLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,     # 原始 nn.Linear 层
        rank: int = 32,             # 低秩分解秩
        alpha: float = 32.0,        # LoRA 缩放因子
        dropout: float = 0.0,       # Dropout 率
        is_shared: bool = True,     # 是否属于 shared 层（浅层）
        orthogonal_init: bool = True, # 是否对 A 使用正交初始化
        freeze_a: bool = True,      # 是否冻结 A 矩阵
        use_block_scale: bool = True, # 是否启用 block-scale 门控
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank   # 缩放因子 = alpha / rank
        self.is_shared = is_shared

        # 冻结原始权重（原则 2）
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        self.bias = base_layer.bias if base_layer.bias is not None else None
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA 低秩矩阵
        #注意：和论文定义相反，论文中是 A ∈ R^{r×d_in}，这里我们是 A ∈ R^{r×d_out}，因为 LoRA 是从输出层注入的
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

        # Block-scale 门控（仅 specific 层，原则 4）
        if not self.is_shared and use_block_scale:
            self.block_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('block_scale', None)

        self._orthogonal_init = orthogonal_init
        self._freeze_a = freeze_a and is_shared  # 只有 shared 层冻结 A
        self.reset_parameters()
```

### 2.2 参数初始化

```python
def reset_parameters(self):
    # Shared 层：A 正交初始化，B 零初始化（原则 3）
    if self.is_shared and self._orthogonal_init:
        nn.init.orthogonal_(self.lora_a)      # 正交初始化确保初始方向稳定
    else:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # Specific 层用 Kaiming
    nn.init.zeros_(self.lora_b)               # B 始终零初始化，减少初始扰动

    if self._freeze_a:
        self.lora_a.requires_grad = False     # 切断 A 的梯度
```

**关键参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rank` | int | 32 | 低秩分解秩 r，控制 adapter 容量 |
| `alpha` | float | 32.0 | 缩放因子分子，实际缩放 = alpha/rank |
| `is_shared` | bool | True | 决定该层属于 shared 还是 specific |
| `orthogonal_init` | bool | True | 是否对 A 做正交初始化（论文 Sec 3.3 关键设计） |
| `freeze_a` | bool | True | 冻结 A 矩阵防止灾难性遗忘 |
| `use_block_scale` | bool | True | 控制 specific 层是否学习门控系数 |

### 2.3 前向传播

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. 原始权重计算（冻结，无梯度）
    result = F.linear(x, self.weight, self.bias)

    # 2. LoRA 低秩计算
    lora_out = F.linear(self.dropout(x), self.lora_a)   # x → A (d_in → r)
    lora_out = F.linear(lora_out, self.lora_b)           # → B (r → d_out)

    # 3. Block-scale 门控（仅 specific 层生效）
    scale = self.scaling
    if self.block_scale is not None:
        effective_scale = 1.0 + 0.5 * torch.tanh(self.block_scale)
        # tanh 将输出压缩到 (-1, 1)，经线性变换后 effective_scale ∈ (0.5, 1.5)
        scale = scale * effective_scale

    # 4. 残差连接
    return result + lora_out * scale
```

实现逻辑注释：
- **第 1 步**：原始权重始终冻结，不对其计算梯度，保护预训练知识
- **第 2 步**：LoRA 低秩分解，A 将 d_in 投影到 r 维，B 再将 r 维投影回 d_out
- **第 3 步**：`1.0 + 0.5*tanh(w)` 是平滑门控函数，使得 effective_scale 在 (0.5, 1.5) 区间，避免极端缩放
- **第 4 步**：LoRA 输出以残差方式加到原始输出上

---

## 3. 模型注入与层级划分

**文件位置：** `openvla-oft/vla-scripts/cl_lora.py` 第 112-173 行  
**论文对应：** Section 3.3 — Shared vs. Specific Layer Partitioning

### 3.1 注入函数

`inject_cl_lora_into_model()` 遍历模型的所有 `LlamaDecoderLayer`，将 Attention 和 FFN 的线性层替换为 `CLLoRALinear`。

```python
def inject_cl_lora_into_model(
    model,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    shared_split_ratio: float = 0.5,   # shared 层占总层数的比例
    orthogonal_init: bool = True,
    freeze_a: bool = True,
    use_block_scale: bool = True,
):
    # Step 1: 发现所有 LlamaDecoderLayer，确定深度排列
    llama_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaDecoderLayer":
            llama_layers.append((name, module))

    total_depth = len(llama_layers)                    # OpenVLA-7B 共 32 层
    shared_depth_count = max(1, int(total_depth * shared_split_ratio))
    # shared_depth=16 时，层 0-15 为 shared，层 16-31 为 specific

    # Step 2: 定义注入目标（与 PI 论文保持一致）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",   # Attention 四投影
                      "gate_proj", "up_proj", "down_proj"]       # FFN 三投影

    # Step 3: 逐层替换
    for layer_idx, (layer_name, layer_module) in enumerate(llama_layers):
        is_shared = layer_idx < shared_depth_count  # 前 N 层为 shared

        for name, module in layer_module.named_modules():
            if any(name.endswith(t) for t in target_modules) and isinstance(module, nn.Linear):
                # 创建 CLLoRALinear 替代原始 Linear
                cl_lora_layer = CLLoRALinear(
                    base_layer=module,
                    rank=rank, alpha=alpha, dropout=dropout,
                    is_shared=is_shared,
                    orthogonal_init=orthogonal_init,
                    freeze_a=freeze_a,
                    use_block_scale=use_block_scale,
                ).to(module.weight.device).to(module.weight.dtype)

                # 原位替换
                setattr(parent_module, child_name, cl_lora_layer)
```

### 3.2 层级划分示意

```
LlamaDecoderLayer 深度 (OpenVLA-7B: 32 层)
├── Shared 层 (0 ~ shared_depth-1): LoRA-A 正交初始化 + 冻结
│   ├── 保护预训练知识的"稳定区"
│   └── 论文 Section 3.3：防止旧任务特征被覆盖
│
└── Specific 层 (shared_depth ~ 31): LoRA-A/B 均可训 + block_scale
    ├── 学习新任务的"适配区"
    └── 论文 Section 3.4：block_scale 门控控制每层对新任务的贡献程度
```

**关键参数 `shared_split_ratio`：** 控制 shared 层占总层数的比例。默认 0.5 表示前 16 层 shared、后 16 层 specific。训练时通过 `--shared_depth 16` 指定。

---

## 4. PI Task Bank：跨任务参数管理

**文件位置：** `openvla-oft/vla-scripts/cl_lora.py` 第 176-252 行  
**论文对应：** Section 3.5 — Task-Specific Parameter Isolation

### 4.1 Stage 1 后冻结（保护共享知识）

```python
def freeze_stage1_params(model) -> None:
    """
    训练完 Stage 1（第一个任务）后：
    - shared LoRA-B 冻结：保护共享输出知识
    - specific LoRA-A 冻结：保护共享输入知识
    - specific LoRA-B + block_scale 保持可训：为后续任务保留适配能力
    """
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear):
            if module.is_shared:
                module.lora_b.requires_grad = False   # shared B 冻结
            else:
                module.lora_a.requires_grad = False   # specific A 冻结
```

冻结策略图示：
```
Stage 1 训练后:
├── Shared  层: LoRA-A [已冻结] + LoRA-B [新冻结]
└── Specific层: LoRA-A [新冻结] + LoRA-B [可训] + block_scale [可训]
```

### 4.2 新任务 Bank 重初始化

```python
def reinit_bank_for_new_task(model) -> None:
    """
    训练 Stage 2+ 之前：
    - 将 specific 层的 LoRA-B 和 block_scale 重置为零
    - 新任务学到自己的输出映射，不受旧任务干扰
    """
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            nn.init.zeros_(module.lora_b)              # B 重置
            if module.block_scale is not None:
                nn.init.zeros_(module.block_scale)     # block_scale 重置（初始 effective_scale=1.0）
```

### 4.3 任务 Bank 保存与恢复

```python
def save_task_bank(model, action_head, bank_dir, stage):
    """保存 per-task bank：specific LoRA-B + block_scale + action_head"""
    bank = {}
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            bank[f"{layer_key}.lora_b"] = module.lora_b.data.cpu().clone()
            if module.block_scale is not None:
                bank[f"{layer_key}.block_scale"] = module.block_scale.data.cpu().clone()
    if action_head is not None:
        for k, v in action_head.state_dict().items():
            bank[f"action_head.{k}"] = v.cpu().clone()
    torch.save(bank, f"{bank_dir}/task_{stage}_bank.pt")

def load_task_bank(model, action_head, bank_path):
    """加载 per-task bank：恢复 specific LoRA-B + block_scale + action_head"""
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    for name, module in model.named_modules():
        if isinstance(module, CLLoRALinear) and not module.is_shared:
            layer_key = name.replace('.', '_')
            for suffix in ['lora_b', 'block_scale']:
                key = f"{layer_key}.{suffix}"
                if key in bank:
                    getattr(module, suffix).data.copy_(bank[key].to(target.device))
    # 同理恢复 action_head
```

**跨任务知识迁移原理：**
- Shared 层的 A/B 矩阵在 Stage 1 后冻结，为所有后续任务提供统一的特征基础
- Specific 层的 A 矩阵冻结但 B 矩阵按任务独立保存，形成 per-task bank
- 评估时加载对应任务的 B + block_scale，切换任务无需重新训练
- 这实现了论文中描述的"隔离参数集"策略——共享层保证迁移，独立层保证无干扰

---

## 5. 正交初始化与 A 矩阵冻结

**文件位置：** `openvla-oft/pi0.5代码/lora.py` 第 39-47 行（参考实现）  
**论文对应：** Section 3.3 — Orthogonal Initialization for Stability

### 5.1 JAX 参考实现

```python
# pi0.5代码/lora.py
def orthogonal_init(key, shape, dtype=jnp.float32):
    """使用 Flax 官方正交初始化器生成正交矩阵"""
    return nn.initializers.orthogonal()(key, shape, dtype)
```

### 5.2 JAX 版 Einsum 中的 CL-LoRA 初始化流程

```python
# pi0.5代码/lora.py — Einsum.setup()
shape_a[config.axes[1]] = config.rank   # shape_a: [8, r]
shape_b[config.axes[0]] = config.rank   # shape_b: [r, 16]

if config.use_orthogonal_init:
    # Shared adapter: A 正交初始化 + B 零初始化
    self.w_a = self.param("lora_a", orthogonal_init, shape_a)
    self.w_b = self.param("lora_b", nn.initializers.zeros, shape_b)
else:
    # Specific adapter: 标准随机初始化
    self.w_a = self.param("lora_a", config.init_fn, shape_a)
    self.w_b = self.param("lora_b", config.init_fn, shape_b)
```

### 5.3 A 矩阵冻结（stop_gradient）

```python
# pi0.5代码/lora.py — Einsum.__call__()
w_a = jnp.where(freeze_a, jax.lax.stop_gradient(w_a), w_a)
# 使用 jnp.where 而非 Python if，兼容 JAX Tracer 且正确阻断梯度
```

```python
# vla-scripts/cl_lora.py — PyTorch 等效实现
if self._freeze_a:
    self.lora_a.requires_grad = False
```

**设计原理：**
- 正交初始化保证 A 矩阵的列向量线性无关，子空间更稳定
- B 零初始化使初始时刻 LoRA 输出为零，不扰动原始模型
- 冻结 A 迫使模型通过调整 B 来适应新任务，B 的修改幅度受 A 子空间约束
- 这构成对旧知识的结构化保护——LoRA 子空间方向不变，仅改变投影权重

---

## 6. Block-Scale 门控机制

**文件位置：** `openvla-oft/vla-scripts/cl_lora.py` 第 91-94 行  
**论文对应：** Section 3.4 — Adaptive Block-Scale Gating

### 6.1 门控参数定义

```python
# CLLoRALinear.__init__()
if not self.is_shared and use_block_scale:
    self.block_scale = nn.Parameter(torch.tensor(0.0))
    # 初始化为 0，对应 effective_scale = 1.0 + 0.5*tanh(0) = 1.0
else:
    self.register_parameter('block_scale', None)
```

### 6.2 门控函数与激活映射

```python
# CLLoRALinear.forward()
if self.block_scale is not None:
    effective_scale = 1.0 + 0.5 * torch.tanh(self.block_scale)
    scale = scale * effective_scale
```

门控函数 `g(w) = 1.0 + 0.5·tanh(w)` 的取值区间：
| w | tanh(w) | effective_scale | 效果 |
|---|---------|-----------------|------|
| -∞ | -1.0 | 0.5 | LoRA 贡献减半 |
| 0 | 0.0 | 1.0 | 标准 LoRA 贡献 |
| +∞ | +1.0 | 1.5 | LoRA 贡献增强 50% |

**设计意图：**
- 每层可学习不同的门控值，自适应调节该层对新任务的贡献程度
- `1.0 + 0.5*tanh(w)` 确保门控值不会太极端（始终在 0.5~1.5 之间）
- 初始化 w=0 使初始门控为 1.0，训练过程自动学习各层权重
- 与普通 LoRA 使用固定 `α/r` 缩放不同，block-scale 为每层提供自由度

### 6.3 JAX 参考实现

```python
# pi0.5代码/lora.py — Einsum.__call__()
if block_scale is not None:
    scale = scale * block_scale
scale = jnp.asarray(scale, dtype=dtype)  # 强制匹配输入精度
```

JAX 版本中 block_scale 作为外部传入值（在 Gemma 层定义时创建），使用方式与 PyTorch 版本一致。

---

## 7. 多任务训练与损失组合

**文件位置：** `openvla-oft/vla-scripts/train_cl_lora.py` 第 400-500 行  
**论文对应：** Section 4.2 — Multi-Objective Training

### 7.1 三支路训练框架

```python
# train_cl_lora.py — 训练配置
@dataclass
class TrainCLConfig:
    # ---- 训练分支控制 ----
    use_kd: bool = True              # 知识蒸馏开关
    use_replay: bool = False         # 回放开开关
    use_cl_lora: bool = True         # CL-LoRA 注入开关

    # 三个实验分支：
    # Branch 2: --no_kd --no_replay   → 纯 CL-LoRA
    # Branch 3: --use_kd --use_replay → CL-LoRA + KD + 原型回放
    # Branch 4: --use_kd --use_replay → CL-LoRA + KD + 均匀回放
```

### 7.2 总损失组合

```python
# 训练循环核心 (约第 780-810 行)
for batch_idx, batch in enumerate(dataloader):
    # 1. 当前任务损失 (Task Loss)
    loss_task, metrics, student_pred, _ = run_forward_pass_extended(
        vla=vla, action_head=action_head, ...,
        return_predictions=True,
    )

    # 2. 知识蒸馏损失 (KD Loss)
    loss_kd = torch.tensor(0.0)
    if cfg.use_kd and teacher_snapshot is not None:
        # 交换到教师参数 → 前向 → 恢复学生参数
        student_copy = swap_to_teacher(vla, action_head, teacher_snapshot)
        _, _, teacher_pred, _ = run_forward_pass_extended(..., return_predictions=True)
        restore_student(vla, action_head, student_copy)
        loss_kd = F.mse_loss(student_pred, teacher_pred.detach())

    # 3. 回放损失 (Replay Loss)
    loss_replay = torch.tensor(0.0)
    if cfg.use_replay and replay_iters is not None:
        if log_step % cfg.replay_every_n_steps == 0:
            replay_batch = _next_replay_batch(replay_iters, ...)
            loss_replay, _, _, _ = run_forward_pass_extended(
                batch=replay_batch, ...)

    # 4. 加权总损失
    loss_total = (loss_task
                  + cfg.lambda_kd * loss_kd
                  + cfg.replay_loss_weight * loss_replay)
```

**损失权重说明：**

| 损失项 | 权重参数 | 默认值 | 作用 |
|--------|----------|--------|------|
| `loss_task` | 1.0 | 固定 | 当前任务 L1 回归损失 |
| `loss_kd` | `lambda_kd` | 1.0 | 知识蒸馏 MSE 损失，防止新任务偏离旧模型 |
| `loss_replay` | `replay_loss_weight` | 1.0 | 回放 L1 损失，直接复习旧任务数据 |

### 7.3 CL-LoRA 参数冻结策略

```python
# 冻结所有参数，仅放开 LoRA
for name, param in vla.named_parameters():
    param.requires_grad = False

# 解冻 LoRA 参数（lora_a, lora_b, block_scale）
for name, param in vla.named_parameters():
    if any(x in name for x in ['lora_a', 'lora_b', 'block_scale']):
        param.requires_grad = True

# 重新应用 freeze_a：shared 层的 lora_a 必须保持冻结
for name, module in vla.named_modules():
    if isinstance(module, CLLoRALinear) and module.is_shared and module._freeze_a:
        module.lora_a.requires_grad = False
```

最终可训练参数统计：
- **Shared 层**：仅 LoRA-B 可训（LoRA-A 冻结）
- **Specific 层**：LoRA-A、LoRA-B、block_scale 均可训
- **Visual backbone / LLM 主干 / LM head**：全部冻结

---

## 8. 知识蒸馏：教师模型快照

**文件位置：** `openvla-oft/vla-scripts/train_cl_lora.py` 第 191-235 行  
**论文对应：** Section 4.3 — Knowledge Distillation for Anti-Forgetting

### 8.1 教师快照保存

```python
def save_teacher_snapshot(model, action_head, save_dir, step):
    """仅保存可训参数（CL-LoRA + action_head），最小化内存占用"""
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[f"model.{name}"] = param.data.cpu().clone()

    if action_head is not None:
        for name, param in action_head.named_parameters():
            if param.requires_grad:
                snapshot[f"action_head.{name}"] = param.data.cpu().clone()

    torch.save(snapshot, save_path)
```

### 8.2 师生参数交换

```python
def swap_to_teacher(model, action_head, teacher, device):
    """将教师参数交换进模型，返回学生参数副本"""
    student_copy = {}
    for name, param in model.named_parameters():
        key = f"model.{name}"
        if key in teacher:
            student_copy[name] = param.data.clone()
            param.data.copy_(teacher[key].to(device))
    # 同理处理 action_head
    return student_copy

def restore_student(model, action_head, student_copy):
    """KD 前向完成后恢复学生参数"""
    for name, param in model.named_parameters():
        if name in student_copy:
            param.data.copy_(student_copy[name])
```

KD 执行流程：
```
1. save_teacher_snapshot() → 保存 Stage 1 训练完成的参数
2. Stage 2 训练时：
   a. 学生前向 → 得到 student_pred
   b. swap_to_teacher() → 临时加载教师参数
   c. no_grad 教师前向 → 得到 teacher_pred
   d. restore_student() → 恢复学生参数
   e. loss_kd = MSE(student_pred, teacher_pred.detach())
```

**设计要点：**
- 仅保存可训参数（LoRA + action_head），教师快照极小（约几十 MB）
- 使用原地复制（`data.copy_`）而非重建模型，GPU 内存高效
- MSE 蒸馏使新任务输出不偏离旧知识太远

---

## 9. 原型回放机制

**文件位置：** `openvla-oft/vla-scripts/build_replay_buffer_openvla.py` + `replay_dataset.py`  
**论文对应：** Section 4.4 — Prototype Replay for Catastrophic Forgetting Mitigation

### 9.1 整体流程

```
原始 RLDS 数据集
    │
    ▼
运动量分割 (_compute_chunk_motions + _find_boundaries)
    │  按夹爪开合 + 运动主方向变化切分轨迹
    ▼
物理语义段 (segments)
    │
    ▼
视觉特征提取 (_extract_visual_feature)
    │  仅用 Vision Backbone (SigLIP + DINOv2)
    ▼
语义原型计算 (prototype = mean of normalized features)
    │
    ▼
Top-K 关键帧选择 (cosine similarity to prototype)
    │
    ▼
原型回放缓冲区 (.npz manifest)
```

### 9.2 物理运动量分割

```python
def _compute_chunk_motions(states, cfg):
    """计算连续帧之间的物理运动量并分类主运动方向"""
    for i in range(0, len(states) - chunk_frames):
        s0, s1 = states[i], states[i + chunk_frames]

        # 平移增量
        d_xyz = xyz1 - xyz0
        # 旋转增量（wrap to pi）
        d_rpy = _wrap_to_pi(rpy1 - rpy0)
        # 夹爪增量
        d_g = g1 - g0

        # 归一化运动量（除以阈值）
        norm_mag = raw_mag / [translation_threshold] * 3
                           + [rotation_threshold] * 3
                           + [gripper_threshold]

        # 确定主导运动方向
        dominant = ["tx","ty","tz","rx","ry","rz","grip"][argmax(norm_mag)]
```

**分割阈值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `translation_threshold_m` | 0.03 | 平移 ≥ 3cm 视为有效运动 |
| `rotation_threshold_rad` | 0.05 | 旋转 ≥ 0.05rad 视为有效运动 |
| `gripper_threshold` | 0.1 | 夹爪开合 ≥ 0.1 视为有效活动 |
| `chunk_frames` | 5 | 每 5 帧比较一次运动增量 |

边界检测规则：
1. 夹爪开合超过阈值 → 立即切分
2. 运动主方向变化（连续 3 帧确认）→ 切分

### 9.3 视觉特征提取与原型计算

```python
@torch.no_grad()
def _extract_visual_feature(model, processor, image, device):
    """仅用 Vision Backbone 提取特征，避开 7B LLM"""
    pil_img = Image.fromarray(image).convert("RGB")
    inputs = processor(text="dummy", images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

    # 仅前向视觉骨干网络
    patch_features = model.vision_backbone(pixel_values)  # [1, num_patches, embed_dim]

    # 平均池化得到全局特征
    pooled_feature = patch_features.mean(dim=1).squeeze(0).cpu().to(torch.float32).numpy()
    return pooled_feature
```

```python
# 按 segment 内所有帧的特征计算原型
feat_norm = feat_mat / np.linalg.norm(feat_mat, axis=-1, keepdims=True)
proto = np.mean(feat_norm, axis=0)           # 归一化特征的均值
proto = proto / np.linalg.norm(proto)        # 再归一化

# 选取与原型余弦相似度最高的 Top-K 帧
cosine = feat_norm @ proto
top_idx = np.argsort(-cosine)[:top_k_per_segment]
```

**设计原理：**
- 物理分割保证每段具有语义一致性（同一操作子技能）
- 视觉特征归一化后计算原型，选取最接近原型的帧作为回放样本
- Top-K 策略（默认 K=2）控制回放缓冲区大小，平衡记忆效果与存储开销

### 9.4 回放数据加载

```python
# replay_dataset.py
class PrototypeReplayDataset(Dataset):
    def __init__(self, replay_dir, batch_transform):
        # 读取 manifest.jsonl（路径指向 .npz 文件）
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __getitem__(self, idx):
        record = self.samples[idx]
        data = np.load(record["sample_path"])
        img_array = data["image"]
        action = data["action"]
        task_desc = str(data["task"])

        # 动作维度对齐：扩展为 Action Chunk [8, 7]
        if len(action.shape) == 1:
            action = np.tile(action, (8, 1))

        # 构造 dummy 轨迹，复用 OpenVLA 的 batch_transform
        dummy_step = {
            "observation": {"image_primary": np.expand_dims(img_array, 0)},
            "task": {"language_instruction": task_desc.encode('utf-8')},
            "action": action,
            "dataset_name": "libero_spatial_no_noops",
        }
        return self.batch_transform(dummy_step)
```

### 9.5 训练中的回放注入

```python
# train_cl_lora.py — 回放批次采样
def _next_replay_batch(iters, loaders, strategy):
    if strategy == "round_robin":
        # 循环遍历所有回放缓冲区
        idx = _next_replay_batch._rr_counter % len(iters)
        _next_replay_batch._rr_counter += 1
    elif strategy == "random":
        idx = random.randint(0, len(iters) - 1)

    try:
        return next(iters[idx])
    except StopIteration:
        iters[idx] = iter(loaders[idx])
        return next(iters[idx])
```

回放频率由 `replay_every_n_steps` 控制（默认每步都回放），损失权重由 `replay_loss_weight` 控制（默认 1.0）。

---

## 10. 推理评估中的 CL-LoRA 加载

**文件位置：** `openvla-oft/experiments/robot/openvla_utils.py` 第 253-340 行  
**论文对应：** Section 5.1 — Evaluation Protocol

### 10.1 CL-LoRA 权重加载逻辑

```python
def get_vla(cfg):
    # 1. 检测是否为 CL-LoRA checkpoint
    cl_lora_path = os.path.join(cfg.pretrained_checkpoint, "cl_lora_adapter.pt")
    is_cl_lora = os.path.exists(cl_lora_path)

    # 2. CL-LoRA 场景：从原始 OpenVLA-7B 加载骨架
    base_model_path = "/root/autodl-tmp/models/openvla-7b" if is_cl_lora \
                      else cfg.pretrained_checkpoint

    # 3. 加载基础模型
    vla = AutoModelForVision2Seq.from_pretrained(base_model_path, ...)

    # 4. 注入 CL-LoRA 架构
    if is_cl_lora:
        # 读取 cl_lora_config.json 获取配置
        cl_config_path = os.path.join(cfg.pretrained_checkpoint, "cl_lora_config.json")
        if os.path.exists(cl_config_path):
            cl_cfg = json.load(open(cl_config_path))
            cl_rank = cl_cfg.get("lora_rank", 32)
            cl_shared_ratio = cl_cfg.get("shared_split_ratio", 0.5)
        else:
            cl_rank, cl_shared_ratio = 32, 0.5

        vla = inject_cl_lora_into_model(
            vla, rank=cl_rank, alpha=cl_rank,
            shared_split_ratio=cl_shared_ratio,
        )

        # 5. 加载训练好的 CL-LoRA 权重
        cl_state_dict = torch.load(cl_lora_path, map_location="cpu")
        vla.load_state_dict(cl_state_dict, strict=False)

        # 6. (可选) 加载 per-task bank
        eval_task = getattr(cfg, 'eval_task_id', 0) or 0
        if eval_task > 0:
            bank_path = os.path.join(cfg.pretrained_checkpoint,
                                     f"task_{eval_task}_bank.pt")
            if os.path.exists(bank_path):
                load_task_bank(vla, None, bank_path)
```

### 10.2 Checkpoint 文件结构

```
LOGS/cl_lora/checkpoint-4000/
├── cl_lora_config.json          # CL-LoRA 配置（rank、shared_ratio 等）
├── cl_lora_adapter.pt           # LoRA 参数（lora_a, lora_b, block_scale）
├── action_head--4000_checkpoint.pt  # 动作头参数
├── teacher_snapshot--4000.pt    # 教师快照（用于下一阶段 KD）
├── task_1_bank.pt               # 任务 1 的 specific B + block_scale
├── task_2_bank.pt               # 任务 2 的 specific B + block_scale
├── dataset_statistics.json      # 数据统计（用于动作反标准化）
└── config.json                  # HuggingFace 模型配置
```

---

## 附录 A：关键超参数汇总

| 参数 | 默认值 | 说明 | 论文位置 |
|------|--------|------|----------|
| `lora_rank` | 32 | 低秩分解秩 r | Sec 3.2 |
| `alpha` | = rank | 缩放因子分子（α/r 为实际缩放） | Sec 3.2 |
| `shared_depth` | 16 | shared 层数（前 16 层） | Sec 3.3 |
| `orthogonal_init` | True | A 矩阵正交初始化 | Sec 3.3 |
| `freeze_a` | True | Shared 层冻结 A 矩阵 | Sec 3.3 |
| `use_block_scale` | True | Specific 层启用 block_scale 门控 | Sec 3.4 |
| `lambda_kd` | 1.0 | 知识蒸馏损失权重 | Sec 4.3 |
| `replay_loss_weight` | 1.0 | 回放损失权重 | Sec 4.4 |
| `replay_every_n_steps` | 1 | 回放频率 | Sec 4.4 |

## 附录 B：代码文件索引

| 文件 | 关键行号 | 核心内容 |
|------|----------|----------|
| `vla-scripts/cl_lora.py` | 22-109 | CLLoRALinear 类定义（参数初始化 + 前向传播） |
| `vla-scripts/cl_lora.py` | 112-173 | `inject_cl_lora_into_model()` 注入函数 |
| `vla-scripts/cl_lora.py` | 176-252 | Task Bank 管理（freeze/reinit/save/load） |
| `pi0.5代码/lora.py` | 11-230 | JAX 参考实现（LoRAConfig / Einsum / FeedForward） |
| `vla-scripts/train_cl_lora.py` | 62-158 | TrainCLConfig 配置类 + KD/replay 参数 |
| `vla-scripts/train_cl_lora.py` | 191-235 | 教师快照保存/交换/恢复 |
| `vla-scripts/train_cl_lora.py` | 400-500 | 三损失组合训练循环 |
| `vla-scripts/build_replay_buffer_openvla.py` | 64-130 | 运动分割 + 边界检测 |
| `vla-scripts/build_replay_buffer_openvla.py` | 132-168 | 视觉特征提取 + 原型计算 + Top-K 选择 |
| `vla-scripts/replay_dataset.py` | 12-73 | 原型回放数据加载 |
| `experiments/robot/openvla_utils.py` | 253-340 | 评估时 CL-LoRA 加载逻辑 |
