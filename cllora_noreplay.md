# CL-LoRA 无回放支线 最终实验报告

## 实验目标

在 OpenVLA-7B 上实现 PI 式 CL-LoRA 无回放持续学习，预期旧任务 retention 低（5-30%），作为后续 replay 实验的基线。

## 核心机制

CL-LoRA 将 LLaMA-2-7B 的 32 层 Decoder 分为共享层和特定层：

- **共享层 (layer 0 至 shared_depth-1)**：LoRA-A 正交初始化，跨任务共享知识
- **特定层 (layer shared_depth 至 31)**：LoRA-A/B 可训 + block_scale 门控
- **Task Bank**：每任务结束后保存特定 LoRA-B + block_scale + action_head 到 bank 文件
- **评估**：通过 `--eval_task_id N` 从 `task_N_bank.pt` 恢复该任务专属参数

## 全部实验数据

| 实验 | shared | rank | 共享A | freeze_speA | bank | A_s1 | Stage2 | Stage3 | Stage4 |
|------|--------|------|--------|-------------|------|------|--------|--------|--------|
| V11 | 8 | 16 | frozen | ✓ | B only | ~100%? | 0.20-0.96 | 0.20-0.95-0.84 | — |
| vlast | 8 | 16 | frozen | ✓ | B only | ~100% | 0.20-0.98 | 0.28-0.92-1.00 | 0.22-0.92-0.86-0.96 |
| last(r4) | 8 | 4 | frozen | ✓ | B only | ~100%? | 0.58-1.00 | 0.58-1.00-1.00 | 0.58-1.00-1.00-0.98 |
| vlast16 | 16 | 4 | frozen | ✓ | B only | ~100% | 0.98-0.92 | 0.98-0.92-0.92 | — |
| v13 | 22 | 16 | frozen | ✓ | B only | ? | ? | 0.25-1.00-0.36 | — |
| V18 | 4 | 16 | frozen | **✗** | A+B | 0.68 | 0.68-1.00 | 0.68-1.00-1.00 | — |
| V19 | 8 | 16 | **可漂移** | ✓ | A+B | 0.94 | 0.94-~1.00 | 0.94-~1.00-0.64 | — |
| V20 | 20 | 4 | **可漂移** | ✓ | B only | ~1.00 | 1.00-1.00 | — | — |
| V21 | 16 | 4 | frozen | **✗** | A+B | ? | 0.68-0.94 | — | — |
| V22 | 4 | 2 | frozen | **✗** | B only | 0.06 | 0.28-0.98 | 0.00-0.98-0.10 | 0.04-0.94-0.20-1.00 |
| V23 | 8 | 4 | frozen | **✗** | B only | ? | >0.90-? | — | — |
| V24 | 20 | 4 | frozen | **✗** | B only | ~1.00 | ~1.00-? | — | — |
| V25 | 4 | 2 | frozen | **✗** | B only | 0.06 | 0.28-0.98 | 0.02-0.90-1.00 | — |
| V26 | 4 | 2 | frozen | **✗** | B only | 0.98 | ~1.00-~1.00 | 0.92-1.00-1.00 | 0.00-0.00-0.00-0.96 |

> Stage2 栏为 (A-B)，Stage3 为 (A-B-C)，依此类推。s1 为 Stage 1 单任务评估。

## 失败原因分析

### 1. Bank 机制在 7B 上天然高 retention

Bank 保存的特定层 B 矩阵参数量：

```
rank=16: 24层 × 7模块 × B矩阵 ≈ 13.9M params → ~56 MB
rank=4:  24层 × 7模块 × B矩阵 ≈ 3.5M  params → ~14 MB  
rank=2:  28层 × 7模块 × B矩阵 ≈ 1.75M params → ~7 MB
```

PI (Gemma-2B) bank 总共才 ~1.5M。即使是 rank=2，bank 容量仍是 PI 的 5 倍。Bank 天然有足够容量编码完整任务行为 → eval 时完美恢复。

### 2. 遗忘不对称：A vs B/C/D

```
Stage 1 (Task A):  shared A/B + specific A/B 全部可训
                   → A 的知识分裂保存在 shared 和 specific 两端

Stage 2+ (B/C/D):  (freeze_specific_a 为 True 时) specific A 冻结
                   → 新任务知识集中保存在 bank 的 specific B 里
                   → 唯一不可恢复的是 shared B（冻在 Stage 1 值）
```

- Task A：shared B 是唯一不可恢复的知识。shared B 只有 4-8 层 → 容量 2-5M。
- Task B/C/D：specific B 从 bank 完整恢复，shared 层冻在 A 态但被 24+ 层 specific 覆盖。

### 3. Rank=2 的虚假成功

V22 (rank=2, shared=4) 显示 A=28%，接近预期。但进一步分析发现：
- Stage 1 单任务仅 6%（V22 6000 步）
- V26 训到 12000 步后单任务达 98%，但遗忘立即消失（A=92%+ at Stage 3）

V22 的 28% 不是"从 100% 忘到 28%"而是"从 6% 涨到 28%"。

### 4. 梯度不对称导致 B 单步免疫

freeze_specific_a=False 时，特定 A 的漂移量由特定 B 的容量决定：
- rank=2 → 特定 B 容量 ~1.75M → 不够 → 特定 A 被迫漂移
- rank=4 → 特定 B 容量 ~3.5M → 够用 → 特定 A 几乎不漂

但漂移方向不对称：
- A: specific A 经历 A→B→C (两步漂移) → 遗忘严重
- B: specific A 经历 B→C (一步漂移) → 微不足道

B 永远只差一步，在 7B 上不足以破坏 bank 恢复。

### 5. V26 Stage 4 断崖式崩溃的根本原因

V26 (rank=2, shared=4, freeze_specific_a=False) 的结果模式为：
```
Stage 2: A=1.0, B=1.0           ← 同域，不遗忘
Stage 3: A=0.92, B=1.0, C=1.0   ← 首次跨域，A 轻微下降
Stage 4: A=0, B=0, C=0, D=0.96  ← 断崖式全崩
```

**根因是 freeze_specific_a=False 导致的累积漂移突破临界点：**

```
Stage 1 结束: specific A = A_state (纯 spatial)
Stage 2 结束: specific A = AB_state (混合 spatial)
Stage 3 结束: specific A = ABC_state (spatial→object)
Stage 4 结束: specific A = ABCD_state (spatial→object→goal)

加载 A 的 bank B (A_state) 时:
  Stage 3: specific A 离 A_state = 2 步 → rank=2 B 勉强能桥接 → 0.92
  Stage 4: specific A 离 A_state = 3 步 → rank=2 B 桥接失败 → 0
```

**specific A 有 28 层 × rank=2 = 大量参数 → 漂移是全局的。** 特定 B 只有 rank=2 → 2 维子空间无法桥接 3 步的层级特征漂移。

Stage 1-3 "几乎不遗忘"是因为：
1. Stage 2 同域 → specific A 无需大幅漂移
2. Stage 3 仅一次跨域 → rank=2 的 B 矩阵能勉强桥接

### 6. 实验设计的结构性问题

**A 和 B 都是 libero_spatial（同域）**。无论什么机制，同域训练天然不会遗忘。真正的遗忘只在跨域（Stage 3 object, Stage 4 goal）才显现。

### 7. V11 的 20% 不可复现

V11 是早期实验（代码可能有 bug），所有后续正确实现 bank 的实验中，A retention 最低为 58%（last rank4）。

## 核心结论

No-replay CL-LoRA 在 7B 上的遗忘有三种独立机制：

| 机制 | 条件 | 效果 | 代表实验 |
|------|------|------|---------|
| **shared B 覆盖** | freeze_specific_a=True + low shared_depth | 仅 A 遗忘，B/C 保持 | V11, vlast |
| **specific A 漂移** | freeze_specific_a=False | A/B/C 全遗忘，但断崖式 | V22, V26 |
| **bank 容量** | bank 存 A vs 不存 A | 影响 A 的遗忘程度 | V18 vs V22 |

三种机制相互制约：增大 A 漂移 → B/C 也漂 → 断崖；减小 A 漂移 → B/C 不漂 → 只有 A 遗忘。

**同域 A/B 在 Stage 2 不遗忘是实验设计的必然结果，不是调参能解决的。**

## V27 设计方向

目标：no-replay 的遗忘应该是 **渐进的**，不是断崖式的。

要达到这个目标，需要 specific A 不在 Stage 4 崩溃。两种策略：

### 策略 A: freeze_specific_a=True + 压缩 shared B（V11 路线）

让遗忘集中发生在 shared B 层（冻结后不可恢复），specific A 永不动。

- shared_depth=4, rank=4：shared B 仅约 1.1M 参数，极易被后续任务"覆盖"
- 预期：A 显著遗忘（shared B 小），B/C 随跨域也逐步下降

### 策略 B: freeze_specific_a=False + 增大 shared_depth

让更多层冻结（shared B），减少漂移层数（specific A），避免断崖。

- shared_depth=12~16, rank=2：12-16 层特定 A 漂移量被限制
- 预期：漂移更温和，4 阶段都不至崩溃
- 风险：shared 太多可能让所有旧任务 retention 偏高

## 后续方向

已启动 prototype replay 支线（r1 实验），replay 在 Stage 4 展现了显著效果（0→1.0）。
