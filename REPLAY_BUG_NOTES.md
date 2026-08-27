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

## 2. B=0 / B/C/D 全崩（回放+KD 超载压垮新任务）——已确认

**最终结论（2026-08 评估确认）**：v39r 完整结果：

| checkpoint | A | B | C | D |
|---|---|---|---|---|
| v39r-C | 0.81 | 0.00 | 0.00 | — |
| v39r-D | 0.90 | — | 0.00 | 0.02 |

**决定性证据**：C 用自己 checkpoint 评估也是 0、D 自评 0.02 —— **新任务根本没学好**。
根因不是 bank 复制（md5 完好），是**训练强度三倍超载**：

- v39r 配置: replay_every=1（每步回放）+ replay_weight=1.0 + lambda_kd=1.0（每步 KD）
- B 阶段每步梯度构成: task(1份) : replay(1份) : KD(1份, 把新任务拉向旧teacher行为)
- ⇒ B 特定层 2/3 梯度在学 A 的行为 → rank16×8层容量被灌爆 → B/C/D 全崩
- A 是 stage1 学的 + 回放强化 → 0.9（幸存）

LIBERO 同样 1:1:1 没崩（7D 单臂简单）；RoboTwin 14D 双臂直接压垮。
**这是强度问题不是机制错误。**

**v2 修复（run_v39_replay_BCD.sh 已改，run_id rt_v39r2）**：
- `--freeze_film_stage2 False`：显式回到"漂移 FiLM + 回放"原叙事（v39r 意外冻结）
- replay_every 1→4、replay_weight 1.0→0.5
- lambda_kd 1.0→0.2

**诊断方法备忘**：新任务用自己 ckpt 自评（eval_task_id=N, 10episode）——低 = 学习被压垮（强度问题）；高 = bank/评估链问题。

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
