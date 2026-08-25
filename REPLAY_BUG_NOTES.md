# 原型回放实验问题记录（REPLAY BUG NOTES）

> 目的：记录 CL-LoRA + 原型回放（RoboTwin）实验中发现的问题与结论，防止遗忘。
> 更新时间：2026-08（实验进行中，随时补充）

## 1. v39r 的实验定性问题（重要！）

**现象**：v39r（回放版 BCD）训练完成后，评估 A=0.9、B=0（C/D 当时未完成）。

**关键发现**：v39r 启动时服务器代码里包含 **cd003ff**（"Stage>1 无条件冻结 FiLM"），
直到 **f68d28b** 才把冻结改成默认关闭（`--freeze_film_stage2 False`）。
日志确认（train_v39_replay_BCD.log）：

```
[FiLM] Stage>1: vision_backbone (含 FiLM) 已冻结, 视觉特征恒定
# trainable params in vla.vision_backbone (frozen): 0
```

**结论**：v39r 实际是 **冻结 FiLM + 原型回放**，而不是原计划（漂移 FiLM + 回放）。
- A=0.9 符合"冻结保视觉（v39f 无回放 A=0.7）+ 回放强化"的组合预期
- **"漂移 FiLM + 回放"实验尚未真正跑过**——如需该配置，必须显式 `--freeze_film_stage2 False`

## 2. B=0（回放污染特定层槽位）——假设待诊断确认

**现象**：v39r 用 D ckpt 评估 B（eval_task_id=2）= 0；A=0.9。

**机制假设（回放梯度污染当前任务特定层）**：
- Stage 2 (B) 训练时，可训练参数只有 B 的特定层 lora_b + block_scale + action_head lora_b
- task loss（B 数据）→ 学 B；replay loss（A buffer 样本）→ 复习 A
- **两者梯度流向同一批参数** → B 特定层被 A 的回放样本污染，变成 A/B 混合体
- 评估 B 恢复的是被污染权重 → 单任务崩（B=0）
- 对比 A=0.9：A 是 stage 1（无回放污染），FiLM 冻结 + 回放强化 → 高保留

**待确认诊断**（若再遇到类似情况先跑）：
1. B 用自己 ckpt 自评（eval_task_id=2, 10 episode）——自评低 = 污染坐实；自评高 = bank/评估链问题
2. `md5sum` 对比 B ckpt 与 D ckpt 的 task_2_bank.pt——不一致 = 复制 bug

**修复方向**（按成本排序，未实施）：
- A. 回放专用/独立 LoRA 槽：每任务特定层参数独立，回放梯度进旧任务槽，不污染新任务（~30 行 + 一轮训练）
- B. 回放 loss 在旧任务 bank 参数上前向（swap bank 后算 replay loss，梯度不进当前特定层）——参考 PI0.5 实现（~40 行）
- C. 接受缺陷，改叙事（不推荐）

## 3. 评估加速注意事项

- 改 step_lim / num_open_loop_steps 会破坏与历史结果（0.98/0.04/0.42…）的可比性，**不要改**
- 零代价加速：`gpus="6,6,7,7"`（4 worker，每卡 2 进程，显存 ~30GB/卡足够）
- 训练进程占 CPU 时评估显著变慢——先确认训练结束再评估

## 4. 当前实验矩阵（2026-08 状态）

| 支线 | 配置 | 结果（D ckpt 全任务） |
|---|---|---|
| v39 漂移版（无回放） | FiLM 训练 + block_scale 漂移 | A=0, B=0.42, C=0.02, D=0.82 |
| v39b2 漂移版（无回放） | FiLM lr×0.2 + block_scale 冻结 | A=0, B=0.24, C=0, D=0.8 |
| v39f v1（冻结版） | FiLM 冻结 + block_scale 漂移 | A=0.7, B=0.72, C(Dckpt)=0, C(Cckpt)=高 |
| v39f2（冻结版 v2） | FiLM + block_scale 全冻结 | 待评估 |
| v39r（回放版） | **实际=冻结 FiLM + 回放** | A=0.9, B=0, C/D 待 |
| 锚定正则（保底，未跑） | film_anchor_reg λ | — |
| FiLM 进 bank（已完成） | bank 存 vision_backbone + film_gamma | γ=1: A/C≈0.9；γ∈[0.1,0.8]: C=0（全或无） |

**FiLM 恢复的"全或无"实证**：γ=0.1~0.8 下 C 全 0，γ=1 才活——参数插值给不了 0.2-0.5 中间残留。

## 5. 待办

- [ ] 诊断 v39r B=0（自评 + md5）
- [ ] 决定回放修复方向（独立槽 / swap bank 前向）
- [ ] 决定"漂移+回放"是否补跑（--freeze_film_stage2 False）
- [ ] v39f2 全任务评估（验证 block_scale 冻结后 C 是否恢复）
- [ ] 锚定正则 λ 实验（无回放"可控残留"最后手段）
