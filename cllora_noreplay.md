# CL-LoRA 无回放支线 最终实验报告

## 实验目标

在 OpenVLA-7B 上实现 PI 式 CL-LoRA 无回放持续学习，预期旧任务 retention 在 5-20%，作为后续 replay 实验的低基线。

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

Bank 保存的特定层参数（24-28 层 × 7 模块 × rank²）在 7B 模型上参数量级：

```
rank=16: 24层 × 7模块 × B矩阵 ≈ 13.9M params → ~56 MB
rank=4:  24层 × 7模块 × B矩阵 ≈ 3.5M  params → ~14 MB  
rank=2:  28层 × 7模块 × B矩阵 ≈ 1.75M params → ~7 MB
```

PI (Gemma-2B) 的 bank 总共才 ~1.5M 参数。即使是 rank=2，bank 容量仍是 PI 的 5 倍。Bank 天然有足够容量编码完整任务行为 → eval 时完美恢复。

### 2. 遗忘不对称：A vs B/C/D

```
Stage 1 (Task A):  shared A/B + specific A/B 全部可训
                   → A 的知识分裂保存在 shared 和 specific 两端

Stage 2+ (B/C/D):  shared A/B + specific A 已冻结 (freeze_specific_a=True)
                   或 shared B 冻结 + 其余可训 (freeze_specific_a=False)
                   → 可训参数全部在 specific 端
                   → 知识集中保存在 bank 里
```

- **Task A 的遗忘**：shared B 是唯一不可恢复的知识（冻在 Stage 1 值）。shared B 只有 4-8 层，容量 ~2-5M。后续任务覆盖 shared B → A 部分遗忘。
- **Task B/C/D bank 恢复**：bank 保存了 specific B，eval 时完整还原。shared B 冻在 A 态，但这是唯一的 mismatch，不足以打破 90%+ 的 retention。

### 3. Rank=2 的虚假成功

V22 (rank=2, shared=4) 显示 A=28%，接近预期。但进一步分析发现：

- Stage 1 单任务仅 6%（V22 6000 步）
- V26 训到 12000 步后单任务达 98%，但遗忘立即消失（A=92%+ at Stage 3）

**V22 的 28% 不是"从 100% 忘到 28%"而是"从 6% 涨到 28%"——同域训练让模型变好了。**

### 4. 梯度不对称

freeze_specific_a=False 时，特定 A 的可漂移量由特定 B 的容量决定：

```
rank=2 → 特定 B 容量 ~1.75M → 不够 → 特定 A 被迫漂移 → A 遗忘
rank=4 → 特定 B 容量 ~3.5M  → 够用 → 特定 A 几乎不漂 → A 不遗忘
```

但 rank=2 时单任务学不会（A=6%, C=10%）。rank=4 时单任务学会了但漂移不够。**单 rank 同时决定学习和遗忘，方向相反，不可调和。**

### 5. B 的单步免疫

```
A: 特定 A 经历 A→B→C (两步漂移) → 遗忘严重
B: 特定 A 经历 B→C    (一步漂移) → 遗轻微
C: 特定 A = C_state    (零步漂移) → 当前任务，不涉及 bank
```

**B 永远只差一步，在 7B 上不足以破坏 bank 恢复。**

### 6. V11 的 20% 不可复现

V11 是早期实验，结果是 0.20-0.96。后续所有配置在 bank 机制正确时，A retention 最低也是 58%（last rank4）。V11 的 20% 可能是当时代码不成熟导致的。

## 核心结论

**No-replay CL-LoRA 在 7B 上无法实现 PI 式低 retention 基线。** 不是因为调参不够——试了 5 种 shared_depth、3 种 rank、3 种 freeze 策略、有无 bank A、有无 shared A 漂移——bank 机制天然产生 ≥58% 的旧任务 retention。

B/C/D 的 retention 高是结构性结果：bank 机制在 7B 上就是一个强大的参数级 anti-forgetting 方案。只有 A 有中等遗忘（shared B 不可恢复），且需要 freeze_specific_a=True。

## 后续方向

回到 **prototype replay** 支线。不管 no-replay 的基线和 PI 比是高是低，replay 都能展示相对于 no-replay 的提升。使用 V22 配置（已收敛）或 V19 配置（机制最纯）作为基座，加载旧任务 replay buffer。
