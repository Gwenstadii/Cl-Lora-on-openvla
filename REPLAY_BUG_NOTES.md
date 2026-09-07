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
| v39r2c（回放 v2 + 冻结FiLM 重训D） | 冻结 FiLM + buffers A×2+B×1+C×3 | A=0.50, B=0.64, C=0, D=0.79（含 proprio bug） |
| v39r2d（回放 + proprio 修复重训D） | 冻结 FiLM + proprio 从 prev 加载 | A=0.62, B=0.88, C=0.88, D=0.80 ← proprio bug 修复后 C 恢复（定稿） |
| v39b3（无回放, proprio 修复, A冻结） | 漂移FiLM lr0.2 + A冻结 + proprio修复 | A=0.26, B=0.88, C=0.72, D=0.82 |
| v39b4（无回放, proprio 修复, A漂移） | FiLM冻结 + freeze_specific_a=False | A=0, B=0.57, C=0, D=0.85 |
| 锚定正则（保底，未跑） | film_anchor_reg λ | — |
| FiLM 进 bank（已完成） | bank 存 vision_backbone + film_gamma | γ=1: A/C≈0.9；γ∈[0.1,0.8]: C=0（全或无） |

**实验配对关系（无回放 ↔ 回放，同配置才算干净对照）**：

| 无回放 | 回放 | 配对状态 |
|---|---|---|
| v39（漂移FiLM lr1.0, bug）0-0.42-0.02-0.82 | **v39r2**（lr1.0+回放降载）0.69-0.80-0-0.75 | ✅ 同 FiLM lr（均含 proprio bug，同环境） |
| v39b2（lr0.2, bug）0-0.24-0-0.86 | **无**（lr0.2+回放从未跑） | ❌ 空洞——b2 无同配置回放配对 |
| v39b3（lr0.2+修复）0.26-0.88-0.72-0.82 | 无（修复版漂移+回放未跑 = v39r2e 待跑） | ❌ 空洞 |
| v39b4（A漂移+修复）0-0.57-0-0.85 | 无（False+回放未跑 = v39r2f 待跑） | ❌ 空洞 |
| **无（冻结FiLM+True+修复+无回放未跑 = v39b5 待跑）** | **v39r2d**（冻结FiLM+True+修复+回放）0.62-0.88-0.88-0.80 | ❌ 空洞——**v39r2d 无严格无回放配对** |

**v39r2d 补充说明**：只重训了 D（从 v39r2 链的 C 起），B/C bank 来自 v39r2（含 proprio bug 训练，但 v39r2d 的 proprio=从 C 加载 → 评估组合自洽 → C=0.88 成立）；冻结 FiLM 下 D 训练 FiLM 恒等 C-FiLM。其严格对照（同配置无回放）需补跑：冻结 FiLM + freeze_specific_a=True + proprio 修复 + 无回放 BCD。

**FiLM 恢复的"全或无"实证**：γ=0.1~0.8 下 C 全 0，γ=1 才活——参数插值给不了 0.2-0.5 中间残留。
**⚠️ v39/v39b2/v39f/v39r/v39r2/v39r2b/v39r2c 数字均含 proprio 随机投影 bug（见 §5）；v39b3/v39b4/v39r2d 为修复后干净数字。**

## 5. proprio_projector 随机初始化 bug（C 类任务归零的隐藏根因，已修复 33202d9）

**现象**：v39r2c（冻结 FiLM，task_3_bank / vision_backbone 与 C 自评 **md5 完全一致**）评 C 仍 = 0，而 C 自评 0.95——文件级对比暴露唯一残留差异：**proprio_projector**。

**根因**：train_cl_lora.py 每次训练**全新随机初始化** ProprioProjector，从不从上一阶段加载（对比 FiLM 是从 prev 加载的）。各 stage 的 proprio 投影 = 不同随机矩阵：
- Stage N 训练：模型在随机投影 P_N 下学习（P_N 从不训练、仅被"适应"）
- 评估旧任务 K（用 stage M>K 的 ckpt）：加载 P_M ≠ P_K → proprio 输入错位
- 依赖 proprio 的视觉敏感任务（C 精确堆叠）→ 归零；A/B 容错大只降幅

**历史影响（重要）**：bug 存在于所有 stage 2+ 训练，"任何 D ckpt 评 C=0、C 自己 ckpt 评 C 高"
的现象可能**部分归因于 proprio 而非 FiLM/block_scale/回放**：
- v39f 的 C=0（当时归因 block_scale 漂移）需重审
- "漂移 FiLM + 回放救不回 C"（v39r2b 结论）需重审——proprio 修复后可能本来就该恢复
- 无回放基线（v39b2 的 A=0/C=0）可能被 proprio 因素夸大

**修复（33202d9）**：stage 2+ 时 proprio_projector 从 previous_checkpoint_dir 加载
（全任务共享 stage1 投影，与 FiLM 同机制）。日志标志：`[Proprio] Loaded proprio_projector from ...`。

**v39r2d（修复版重训 D）**：冻结 FiLM + proprio 修复 → C 恢复（成功率很高，数字待补）。
**待验证**：漂移 FiLM + proprio 修复（freeze_film_stage2=False）下 C 是否也恢复 → 决定"冻结 FiLM 是否必要"。

## 6. 待办

- [x] 诊断 v39r B=0（= 回放+KD 超载压垮新任务，见 §2）
- [ ] 漂移 FiLM + proprio 修复对照（freeze_film_stage2=False，只重训 D 或全 BCD）→ 决定冻结是否必要
- [ ] 决策：无回放基线是否带 proprio 修复重跑（历史基线被污染）
- [ ] v39r2d 完整数字入册（REPLAY_BUG_NOTES + 方法论.md）
- [ ] 方法论.md 同步（proprio bug、历史结论标注"待重审"）
- [ ] 锚定正则 λ 实验（无回放"可控残留"最后手段）
