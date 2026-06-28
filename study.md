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
11. [CL-LoRA 无回放训练：Checkpoint 文件详解](#11-cl-lora-无回放训练checkpoint-文件详解)
12. [CL-LoRA 无回放训练全过程详解](#12-cl-lora-无回放训练全过程详解)

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

CL-LoRA 无回放训练结束后，会在 `run_root_dir` 下生成如下文件（以 step=4000 为例）：

```
LOGS/cl_lora/checkpoint-4000/
├── cl_lora_config.json              # CL-LoRA 超参数配置（JSON）
├── cl_lora_adapter.pt               # LoRA 参数存档（PyTorch）
├── action_head--4000_checkpoint.pt  # 动作头参数存档（PyTorch）
├── teacher_snapshot--4000.pt        # 教师快照（PyTorch）
├── task_1_bank.pt                   # 当前阶段 task bank（PyTorch）
├── dataset_statistics.json          # 数据集统计量（JSON）
└── (由 processor.save_pretrained 生成)
    ├── config.json                  # HuggingFace 模型配置
    ├── processor_config.json        # 图像处理器配置
    ├── preprocessor_config.json     # 预处理器配置
    ├── special_tokens_map.json      # 特殊 token 映射
    ├── tokenizer_config.json        # 分词器配置
    └── tokenizer.model              # SentencePiece 分词器模型
```

---

## 11. CL-LoRA 无回放训练：Checkpoint 文件详解

**命令示例（无回放分支）：**
```bash
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ../LOGS/cl_lora \
  --use_cl_lora --shared_depth 16 --orthogonal_init --freeze_a \
  --use_block_scale --lora_rank 32 --num_steps 4000
```

该命令等价于 `train_cl_lora.py --use_cl_lora --no_kd --no_replay`（Branch 2）。训练过程中每 `save_freq` 步（默认 10000）保存一次 checkpoint，训练结束后保存最终 checkpoint。

### 11.1 生成文件逐一说明

#### JSON 文件

| 文件 | 来源 | 内容 |
|------|------|------|
| `cl_lora_config.json` | `train_cl_lora.py` 手动写入 | 记录 CL-LoRA 注入时的超参数：`lora_rank`、`alpha`、`shared_depth`、`shared_split_ratio`、`orthogonal_init`、`freeze_a`、`use_block_scale`。**该文件为评估脚本提供注入参数**，确保评估时架构与训练一致 |
| `dataset_statistics.json` | `save_dataset_statistics()` | 数据集的动作统计量（mean/std 或 min/max），用于评估时**动作反标准化**（将模型输出的归一化动作还原为物理动作）。内容示例：`{"action": {"mean": [...], "std": [...]}, "proprio": {...}}` |
| `config.json` | `processor.save_pretrained()` | OpenVLA 模型的 HuggingFace 配置，包含 `model_type`、`hidden_size`、`num_hidden_layers`(32)、`max_position_embeddings`、`auto_map`（指向本地 `configuration_prismatic.py` / `modeling_prismatic.py`）等 |
| `preprocessor_config.json` | `processor.save_pretrained()` | PrismaticProcessor 的预处理配置，包含 `do_resize`、`size`(224)、`do_center_crop`、`image_mean`/`image_std` 归一化参数 |
| `special_tokens_map.json` | `processor.save_pretrained()` | 特殊 token 映射表，如 `{"bos_token": "<s>", "eos_token": "</s>", "pad_token": "</s>"}` |
| `tokenizer_config.json` | `processor.save_pretrained()` | 分词器完整配置，包括 `model_max_length`(2048)、`add_bos_token`、`add_eos_token`、`legacy` 等参数 |

#### PyTorch .pt 文件

| 文件 | 来源 | 内容 |
|------|------|------|
| `cl_lora_adapter.pt` | `torch.save(cl_lora_state)` | 仅保存**可训练的 LoRA 参数**（全部 32 层的 `lora_a`、`lora_b`、`block_scale`），不含冻结的 base weight。体积约几十 MB（对比完整 OpenVLA 约 14GB）。**评估时加载此文件恢复 LoRA 权重** |
| `action_head--4000_checkpoint.pt` | `torch.save(action_head_dict)` | `L1RegressionActionHead` 的完整 state_dict，包含 `predict_action` 线性层的 weight 和 bias。用于评估时构建动作头 |
| `teacher_snapshot--4000.pt` | `save_teacher_snapshot()` | 当前阶段所有可训参数的快照副本（LoRA + action_head）。在 Stage 2+ 训练时作为**知识蒸馏教师模型**使用 |
| `task_1_bank.pt` | `save_task_bank()` | Specific 层的 `lora_b` 和 `block_scale` + action_head 的存档。用于后续阶段**恢复旧任务参数**进行评估 |

#### 其他文件

| 文件 | 来源 | 内容 |
|------|------|------|
| `tokenizer.model` | `processor.save_pretrained()` | **SentencePiece 分词器模型文件**（LLaMA 使用）。负责将文本 tokenize 为 token ID 序列。OpenVLA 底层 LLM 基于 LLaMa-2，使用 `tokenizer.model` 将语言指令和各 action token 映射到词表索引。无此文件评估时会报错 |

### 11.2 文件依赖关系

```
训练阶段 (train_cl_lora.py)                      评估阶段 (run_libero_eval.py)
┌──────────────────────────────┐               ┌──────────────────────────────┐
│ processor.save_pretrained()  │               │ AutoProcessor.from_pretrained│
│  ├── config.json             │──────────────>│   → 读取模型结构配置          │
│  ├── tokenizer.model         │──────────────>│   → 加载分词器               │
│  ├── tokenizer_config.json   │               │                              │
│  └── special_tokens_map.json │               │                              │
├──────────────────────────────┤               ├──────────────────────────────┤
│ cl_lora_config.json          │──────────────>│ inject_cl_lora_into_model()  │
│                              │               │   → 根据 rank/shared_depth   │
│                              │               │     重建 CL-LoRA 架构        │
├──────────────────────────────┤               ├──────────────────────────────┤
│ cl_lora_adapter.pt           │──────────────>│ vla.load_state_dict()        │
│                              │               │   → 恢复每层 LoRA 权重       │
├──────────────────────────────┤               ├──────────────────────────────┤
│ action_head--4000.pt         │──────────────>│ L1RegressionActionHead()     │
│                              │               │   .load_state_dict()         │
├──────────────────────────────┤               ├──────────────────────────────┤
│ dataset_statistics.json      │──────────────>│ vla.norm_stats               │
│                              │               │   → 动作反标准化              │
└──────────────────────────────┘               └──────────────────────────────┘
```

---

## 12. CL-LoRA 无回放训练全过程详解

**命令（无回放，等价于 Branch 2）：**
```bash
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --use_cl_lora --shared_depth 16 --orthogonal_init --freeze_a \
  --use_block_scale --lora_rank 32 --learning_rate 5e-4 \
  --batch_size 8 --num_steps 4000
```

### 12.1 训练流程概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 0: 模型加载与 CL-LoRA 注入                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 1. 挂载 OpenVLA-7B 预训练权重 (float16 → bfloat16)                  │ │
│  │ 2. inject_cl_lora_into_model(): 替换 Attention + FFN 的 nn.Linear    │ │
│  │    → 前 16 层 (0-15): Shared CLLoRALinear (A 正交 + 冻结)           │ │
│  │    → 后 16 层 (16-31): Specific CLLoRALinear (A/B 可训 + gate)      │ │
│  │ 3. 全局冻结 → 选择性解冻 LoRA 参数                                    │ │
│  │    → 可训: shared lora_b + specific lora_a/b + block_scale           │ │
│  │    → 冻结: 所有 base weight + visual backbone + LM head              │ │
│  │ 4. 构建 L1RegressionActionHead (L1 回归动作头)                       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  阶段 1: 数据加载                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 5. RLDSDataset: 从 TFRecord 读取 libero_spatial_no_noops            │ │
│  │    → 原始数据: image(224×224) + state(7维) + action(7维) + language  │ │
│  │    → 动作归一化: z-score (mean=0, std=1) ← 来自 dataset_statistics   │ │
│  │ 6. RLDSBatchTransform: 逐样本转换为训练格式                          │ │
│  │    → Action Tokenization + Prompt 构建 + Image Transform             │ │
│  │ 7. PaddedCollatorForActionPrediction: 批内 padding + batch 组装     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  阶段 2: 训练循环 (step 0 → 4000)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ for each batch:                                                      │ │
│  │   8. 当前任务前向 → 计算 L1 loss (loss_task)                         │ │
│  │   9. (无 KD) loss_kd = 0                                             │ │
│  │  10. (无 Replay) loss_replay = 0                                     │ │
│  │  11. loss_total = loss_task                                          │ │
│  │  12. backward → optimizer.step → scheduler.step                      │ │
│  │   每 save_freq 步 → 保存 checkpoint                                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 12.2 数据管道详解

#### 12.2.1 RLDS 原始数据格式

LIBERO 数据集以 TFRecord 格式存储，每条 trajectory 包含：

```
trajectory {
  steps: [
    {
      observation: {
        image:        uint8[H, W, 3]    # 224×224 RGB 图像 (第三人称视角)
        wrist_image:  uint8[H, W, 3]    # 腕部相机图像
        state:        float32[7]        # [x, y, z, axisangle_x, axisangle_y, axisangle_z, gripper]
      }
      action:          float32[7]       # [dx, dy, dz, d_ax, d_ay, d_az, gripper_open]
      language_instruction: string     # 如 "pick up the black bowl and place it on the plate"
      is_terminal:     bool
      is_first:        bool
    },
    ...
  ]
}
```

#### 12.2.2 Action Tokenization（动作分词）

**代码位置：** `prismatic/vla/action_tokenizer.py`

OpenVLA 将连续动作空间离散化为 256 个 bin，每维单独 tokenize：

```
连续动作 x ∈ [-1, 1] → 离散 bin ∈ [0, 255] → action_token = bin + 29896
                                                              ↑ action_token_begin_idx
```

- 7 个动作维 × 256 个离散值 = 每步动作对应 **7 个 action token**
- Action Chunk 为 8 步（当前 + 未来 7 步），共 **7 × 8 = 56 个 action token**
- 加上一个 stop token（`</s>` 对应的 token ID），最终输出序列为 **57 个 token**

**关键参数：**

| 常量 | 值 | 说明 |
|------|-----|------|
| `NUM_ACTIONS_CHUNK` | 8 | 每次预测的动作步数（当前+未来7步=open-loop 8步执行） |
| `ACTION_DIM` | 7 | 每步动作维度 [x, y, z, rx, ry, rz, grip] |
| `ACTION_TOKEN_BEGIN_IDX` | 29896 | 动作 token 起始索引（词表中映射为 `<0>` ~ `<255>`） |
| `STOP_INDEX` | 29892 | 停止 token 索引（映射为 `</s>`） |
| `IGNORE_INDEX` | -100 | Padding/label 忽略索引 |

#### 12.2.3 RLDSBatchTransform：单样本转换

**代码位置：** `prismatic/vla/datasets/datasets.py` 第 28-78 行

每个样本从 RLDS 格式转换为模型输入的过程：

```python
# 1. 提取原始数据
img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])  # 224×224 RGB
lang = rlds_batch["task"]["language_instruction"].decode().lower()
actions = rlds_batch["action"]  # shape: [N, 7]

# 2. Action Tokenization
current_action = actions[0]                                     # 第一步动作
future_actions = actions[1:NUM_ACTIONS_CHUNK]                   # 未来 7 步
current_action_string = action_tokenizer(current_action)        # → 7 个 token
future_actions_string = action_tokenizer(future_actions)        # → 49 个 token
action_chunk_string = current_action_string + future_actions_string  # 共 56 token

# 3. 构建对话 Prompt
prompt = """<s>A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: What action should the robot take to {lang}?
ASSISTANT: {action_chunk_string}</s>"""

# 4. Tokenize
input_ids  = [1, 319, 13563, ...]  # prompt token IDs（含 language instruction part）
labels      = [-100, -100, ..., -100, a1, a2, ..., a56, </s>]
#               ↑ prompt 部分的 label 全设为 -100（不计算 loss）
#               ↑ 仅对 action token 部分计算 loss

# 5. Image Transform
pixel_values = image_transform(img)  # 转换为 tensor + normalize
```

**labels 结构示意：**
```
input_ids:  [<s>, A, chat, ..., USER:, What, action, ..., to, {lang}?, ASSISTANT:, a1, a2, ..., a56, </s>]
labels:     [-100, -100, -100, ..., -100, -100, -100, ..., -100, -100, a1, a2, ..., a56, </s>]
              ↑ prompt 部分全部 mask（IGNORE_INDEX=-100）          ↑ 仅 action token 保留真实值用于 loss 计算
```

#### 12.2.4 Action Chunk 的结构

```
┌──────────┬──────────────────────────────────────────────────────────┬──────┐
│  Step 0  │  Step 1  │  Step 2  │  ...  │  Step 7  │                │      │
│ (当前)    │ (+~0.5s) │ (+~1.0s) │       │ (+~3.5s) │                │      │
├──────────┼──────────┼──────────┼───────┼──────────┼────────────────┼──────┤
│7 tokens  │7 tokens  │7 tokens  │  ...  │7 tokens  │                │</s>  │
│(a1..a7)  │(a8..a14) │(a15..a21)│       │(a50..a56)│                │      │
└──────────┴──────────┴──────────┴───────┴──────────┴────────────────┴──────┘
                            56 action tokens                             1 stop
                        总共 57 个 token 参与 loss 计算
```

### 12.3 前向传播与 Loss 计算

**代码位置：** `train_cl_lora.py` 第 260-320 行（`run_forward_pass_extended`）

#### 12.3.1 模型前向路径

```
Input:
  pixel_values:  [B, 3, 224, 224]  (来自 camera image)
  input_ids:     [B, L]            (tokenized prompt + action sequence)
  attention_mask:[B, L]            (padding mask)
  labels:        [B, L]            (仅 action token 处非 IGNORE_INDEX)

Pipeline:
  1. Vision Backbone (SigLIP + DINOv2, frozen):
     pixel_values → vision_features [B, num_patches, embed_dim=4096]

  2. MLP Projector (frozen):
     vision_features → projected_vision [B, num_patches, llm_dim=4096]

  3. LLM Decoder (Llama-2 7B, base weight frozen, CL-LoRA injected):
     input_ids + projected_vision → hidden_states [B, L, 4096]

  4. Action Head (L1Regression, trainable):
     action_hidden_states = text_hidden_states[action_mask].reshape(B, 56, 4096)
     predicted_actions = fc(action_hidden_states)  # [B, 8, 7]

  5. Loss 计算:
     loss = L1Loss(predicted_actions, ground_truth_actions)
            ↑ [B, 8, 7]         ↑ [B, 8, 7]
```

#### 12.3.2 详细前向函数

```python
def run_forward_pass_extended(vla, action_head, ..., batch, ...):
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)  # [B, 8, 7]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # 模型前向（仅 L1 回归模式）
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,      # 需要最后一层 hidden states
        )

    # 从最后一层 hidden states 中提取动作 token 对应的位置
    last_hidden_states = output.hidden_states[-1]                       # [B, L, 4096]
    text_hidden_states = last_hidden_states[:, num_patches:-1]          # 去掉 vision patch token
    batch_size = batch["input_ids"].shape[0]

    # 获取 action token 位置的 mask
    ground_truth_token_ids = batch["labels"][:, 1:]                     # labels shifted right
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # 将 56 个 action token 的 hidden states 重排为 [B, 56, 4096]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    # L1 Action Head 预测
    predicted_actions = action_head.predict_action(actions_hidden_states)  # [B, 8, 7]

    # === L1 Loss 计算 ===
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
    #        ↑ 直接对归一化后 (mean=0, std=1) 的动作值计算 L1

    # Performance metrics
    curr_action_l1_loss = L1Loss(gt[:, 0], pred[:, 0])     # 当前步 L1
    next_actions_l1_loss = L1Loss(gt[:, 1:], pred[:, 1:])  # 未来 7 步 L1

    return loss, metrics, predicted_actions, actions_hidden_states
```

#### 12.3.3 Action Mask 机制

`get_current_action_mask` 和 `get_next_actions_mask` 用于从 `labels` 序列中定位 action token 的位置：

```python
# labels: [-100, -100, ..., a1, a2, ..., a56, </s>]
# mask 选出 action token 位置：
current_action_mask  = [False, ..., False, True, False, ..., False, True, ..., False]
#                                      ↑a1                    ↑a8 (step 1 的第一个 token)
next_actions_mask    = [False, ..., False, False, True, ..., True, True, ..., True]
#                                      除当前步外所有未来 7 步的 token
```

**用法：** `current_action_mask | next_actions_mask` 定位到所有 56 个 action token，再从 `text_hidden_states` 中取出对应位置的 hidden states 送入 action head。

#### 12.3.4 CL-LoRA 无回放时的总 Loss

```python
# 无 KD，无 Replay 时（Branch 2）：
loss_task   = L1Loss(predicted_actions, ground_truth_actions)   # 主要训练目标
loss_kd     = torch.tensor(0.0)                                  # 跳过（use_kd=False）
loss_replay = torch.tensor(0.0)                                  # 跳过（use_replay=False）

loss_total  = loss_task   # 等价于标准 L1 回归训练
```

**L1 Loss 公式：**

$$
\mathcal{L}_{\text{task}} = \frac{1}{B \cdot 8 \cdot 7} \sum_{b=1}^{B} \sum_{s=0}^{7} \sum_{d=0}^{6} \left| \hat{a}_{b,s,d} - a_{b,s,d} \right|
$$

其中：
- $B$ = batch_size（默认 8）
- 8 = NUM_ACTIONS_CHUNK
- 7 = ACTION_DIM
- $\hat{a}$ = 模型预测的归一化动作值（经 z-score 归一化后）
- $a$ = ground-truth 归一化动作值

### 12.4 训练超参数与优化器

| 参数 | CLI 标志 | 默认值 | 说明 |
|------|----------|--------|------|
| `max_steps` | `--num_steps` | 4000 | 总训练步数 |
| `batch_size` | `--batch_size` | 8 | 每步 batch size |
| `learning_rate` | `--learning_rate` | 5e-4 | 初始学习率 |
| `lr_warmup_steps` | — | 0 | LR warmup 步数（从 10% 线性升至 100%） |
| `num_steps_before_decay` | — | 100000 | LR 衰减步数（衰减因子 0.1，4000 步内不触发衰减） |
| `grad_accumulation_steps` | — | 1 | 梯度累积步数 |
| `optimizer` | — | AdamW | PyTorch AdamW 优化器 |
| `scheduler` | — | MultiStepLR | 阶梯衰减 LR 调度器 |
| `image_aug` | — | True | 是否启用图像增强 |

**图像增强细节（image_aug=True 时）：**
```python
frame_transform_kwargs["image_augment_kwargs"] = {
    "random_resized_crop": dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),  # 轻微缩放
    "random_brightness": [0.2],        # 亮度 ±20%
    "random_contrast":   [0.8, 1.2],   # 对比度 80%~120%
    "random_saturation": [0.8, 1.2],   # 饱和度 80%~120%
    "random_hue":        [0.05],       # 色相 ±5%
}
```

### 12.5 训练循环伪代码

```
for step in 1..max_steps:
    batch = dataloader.next()
    ├─ pixel_values: [B, 3, 224, 224]
    ├─ input_ids:    [B, L]
    ├─ attention_mask: [B, L]
    ├─ labels:       [B, L]
    └─ actions:      [B, 8, 7]

    # Step 1: 模型前向
    output = vla.forward(pixel_values, input_ids, attention_mask, labels)
    hidden = output.hidden_states[-1]
    action_hidden = hidden[:, action_positions].reshape(B, 56, 4096)

    # Step 2: Action Head 预测
    pred_actions = action_head.predict_action(action_hidden)   # [B, 8, 7]

    # Step 3: L1 Loss
    loss_task = L1Loss(pred_actions, batch.actions)            # scalar

    # Step 4: 总 Loss（无回放分支）
    loss_total = loss_task

    # Step 5: 反向传播
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # Step 6: 日志记录（每 wandb_log_freq=10 步）
    wandb.log({"CL Train/Loss": loss_total.item(),
               "CL Train/Curr Action L1 Loss": curr_l1,
               "CL Train/Next Actions L1 Loss": next_l1,
               "CL Train/Learning Rate": scheduler.get_last_lr()[0]})

    # Step 7: Checkpoint 保存（每 save_freq=10000 步）
    if step % save_freq == 0:
        processor.save_pretrained(checkpoint_dir)         # config.json + tokenizer 等
        torch.save(cl_lora_state, "cl_lora_adapter.pt")   # LoRA 权重
        torch.save(action_head_state, "action_head--X.pt")# 动作头权重
        json.dump(cl_lora_config, "cl_lora_config.json")  # CL-LoRA 配置
        json.dump(dataset_stats, "dataset_statistics.json")  # 数据集统计
```

### 12.6 关键设计决策说明

1. **为什么使用 L1 Loss 而非 CrossEntropy？** OpenVLA 有两种模式：标准模式用 CrossEntropy 对离散 action token 分类；连续模式用 L1 Regression 直接回归 7 维连续动作向量。CL-LoRA 训练使用 **L1 回归模式**（`use_l1_regression=True`），因为回归模式直接优化动作精度且不会受 token 离散化的量化误差影响。

2. **为什么仅对 action token 计算 loss？** 输入序列同时包含语言指令 prompt 和 action token 序列。CL-LoRA 的目标是预测动作，因此 prompt 部分全部 mask（labels=-100），仅对 57 个 action/stop token 计算 loss。这确保模型专注于学习"语言指令 → 动作"的映射。

3. **为什么需要 dataset_statistics.json？** LIBERO 的 action 值在训练前经过 z-score 归一化（每维减均值除标准差），模型输出的是归一化值。评估时需要根据 `dataset_statistics.json` 中的 `mean` 和 `std` 将预测值反归一化为物理动作值（米、弧度）。

4. **为什么 base weight 全部冻结？** CL-LoRA 遵循持续学习原则：冻结预训练权重保护原始知识，仅通过低秩 adapter 学习任务特定偏移。训练后无需保存完整 14GB 模型，仅保存几十 MB 的 `cl_lora_adapter.pt`。

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
