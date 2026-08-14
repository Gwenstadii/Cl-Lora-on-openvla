# RoboTwin CL-LoRA 交接文档

> 用途：无缝接管「OpenVLA-7B + CL-LoRA + 原型回放」在 RoboTwin 上的持续学习实验。
> 生成日期：2026-08-14。请先通读本文，再根据「待办事项」继续。

---

## 0. 一句话现状

LIBERO 阶段已完成并收敛到 V38 配置；RoboTwin 阶段迁移到**新服务器（2×A800 80GB）**，Task A（handover_mic）用 V38 配置单任务训练达到 **88%（44/50）**。当前在做无回放顺序学习：Task A → Task B → Task C → Task D，目标是验证旧任务保留率，后续再上原型回放（Prototype Replay v2）。

**此刻卡在**：v39 配置（新的共享/特定层比例）需要从头重训 Task A；评估侧 `OPENVLA_BASE_PATH` 环境变量必须设置否则报错。

---

## 1. 项目背景

- **模型**：OpenVLA-7B（DinoSigLIP 视觉 + LLaMa-2 7B LLM + L1 回归 action head），带 FiLM 语言调制 + proprio 投影。
- **CL-LoRA**：把注意力 q/k/v/o 和 FFN gate/up/down 的 Linear 换成 `CLLoRALinear`。共享层（A 正交初始化+冻结，B 训练后冻结）+ 特定层（每任务一个 bank，存 specific B + block_scale + action_head LoRA）。前向：`Wx + scaling*block_scale*B@A@x`。
- **核心概念**：
  - `first_lora_layer`：从第几层开始注入（PI 式 "action expert"，只给最后 N 层装 LoRA）。
  - `shared_depth`：注入层中前几层是共享层（永久冻结）。
  - `freeze_specific_a`：Stage 1 结束后是否冻结特定层 A（必须 True，否则回放救不回来）。
  - **task bank**：每任务存 specific B + block_scale + action_head LoRA，评估用 `--eval_task_id N` 恢复。
- **两阶段规划**：LIBERO（已完成）→ RoboTwin（当前）。

## 2. 服务器环境（新服务器，重点）

| 项 | 值 |
|---|---|
| 机器 | 2× A800 80GB（可用卡 4、5 号） |
| 仓库根目录 | `/mnt/data/pengshengdi`（git 仓库，`RoboTwin-main/` 和 `openvla-oft/` 是其子目录） |
| conda 环境 | `/mnt/data/pengshengdi/openvla`（Python 3.10） |
| 基座模型 | 需确认位置，通常 `/mnt/data/pengshengdi/models/openvla-7b` |
| 数据 | `RoboTwin-main/data/{task}/processed_openvla/` |

**必须 export 的环境变量**（训练和评估前）：

```bash
export VLA_PATH=<基座模型路径>              # 训练用 --vla_path
export OPENVLA_BASE_PATH=<基座模型路径>      # 评估用（不设会回退到旧路径报错！）
export LOGS_ROOT=<checkpoint 输出目录>       # 训练用 --run_root_dir
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
```

**git 网络（GFW）**：GitHub HTTPS 常被掐断，已配置 `http.version HTTP/1.1` + `postBuffer 524288000` + `lowSpeedLimit 0` + `lowSpeedTime 999999`。pull 失败就重试或换代理/镜像（ghfast.top）。

**VS Code 终端**：若报 `__vsc_prompt_cmd_original: command not found`，`unset PROMPT_COMMAND`（无害）。

## 3. 仓库状态

- 分支：`master`，当前 HEAD `c39f98a`，工作区干净。
- **每次代码改动流程**：Windows 本地改 → commit + push → 服务器 `git pull`。**禁止从聊天窗口复制多行代码到服务器**（`\n`、`__file__`、反斜杠会被吃掉导致语法错误）。
- `prismatic` 是 editable 安装，pull 后**不用重装**。
- `task_config/` 目录在 .gitignore 里，**yml 改动不走 git**（服务器上的改动是永久的本地文件）。

## 4. 关键代码修复清单（按时间，全部已推送）

| commit | 文件 | 内容 |
|---|---|---|
| `8f05f71` | `train_cl_lora.py` | `_merge_previous_stats()`：新阶段 checkpoint 合并前一阶段的 `dataset_statistics.json`，使旧任务评估能查 `norm_stats[unnorm_key]`（修复 KeyError） |
| `8f05f71` 等 | `deploy_policy.py` | eval 里固定各任务的 prompt（`_FIXED_PROMPTS`），与训练数据一致 |
| `69d9aeb` | `deploy_policy.py` + `openvla_utils.py` | vla-scripts 路径改相对定位；基座模型路径改用 `OPENVLA_BASE_PATH` 环境变量 |
| `de08afb` | `eval.sh` | overrides 加 `--eval_video_log False`，评估默认不录视频 |
| `7a53919` | `eval.sh` | overrides 加 `--clear_cache_freq 50`，避免每 5 个 episode 清 Sapien 渲染缓存导致的几分钟卡顿 |
| `fda554f` | `train_cl_lora.py` | 修复 `cl_lora_config.json` 的 `shared_split_ratio` 口径：除以注入层数（`32-first_lora_layer`）而非固定 32 |
| `250a132` | `openvla-oft/prismatic/vla/datasets/datasets.py` | `language_instruction` 为 numpy 数组时 `.decode()` 报错 → 支持 ndarray/list/tuple（random.choice）/bytes |
| `c39f98a` | `openvla-oft/prismatic/vla/datasets/rlds/dataset.py` | `make_interleaved_dataset` 末尾加 `dataset.prefetch(tf.data.AUTOTUNE)`，数据准备与 GPU 计算重叠 |

> 注意：仓库有**两份** `prismatic`（`openvla-oft/prismatic` 和 `RoboTwin-main/policy/openvla-oft/prismatic`）。训练用的是前者。`datasets.py` 的 language_instruction 修复已同步到两份，后续改 prismatic 记得两边都看。

## 5. 已完成的实验与结果

### 5.1 LIBERO 阶段（已完成）
- 系统测了 ~30 个配置（v5–v48），收敛到 **V38**：`first_lora_layer=24, shared_depth=4, rank=16, freeze_specific_a=True`。
- 结论：无回放时 7B 结构性保留高；V38 + 回放可把所有任务恢复到 ≥0.88。V47（freeze_specific_a=False）回放救不回旧任务。
- 注意：LIBERO 评估 `run_libero_eval.py` 用 `_load_task_bank(model, None, bank_path)`，**action_head=None，没从 bank 恢复 action head**——这是当时无回放 A 只有 0.44 的主要 eval 端因素。RoboTwin 的 `deploy_policy.py` 是传了 `action_head` 的，已修正。

### 5.2 RoboTwin 阶段
- 任务：A=handover_mic，B=grab_roller，C=stack_bowls_two，D=open_laptop。双 14D 关节动作，3 相机（head/left/right wrist），proprio，FiLM。
- **Task A（V38 配置）成功率 88%（44/50）**，checkpoint `rt_v38_taskA--30000_chkpt`。
  - 之前卡在 0.3 是因为（1）eval 端 proprio_projector 随机初始化没从 checkpoint 加载；（2）eval prompt 与训练不一致。都已修复。

## 6. 当前实验配置

### 6.1 V38（已验证，Task A 88%）
```
注入 L24-31（8 层）：共享 L24-27（4 层）: 特定 L28-31（4 层）= 1:1
rank=16, freeze_specific_a=True, orthogonal_init, freeze_a, use_block_scale
use_proprio=True, use_film=True, num_images_in_input=3
batch_size=1, grad_accumulation_steps=8, lr=5e-4
```

### 6.2 V39（新配置，待重训 Task A）
```
注入 L16-31（16 层）：共享 L16-23（8 层）: 特定 L24-31（8 层）= 1:1
rank=16, freeze_specific_a=True, 其余与 V38 相同
```
- 理由：80GB 显存可承载更多注入层；特定层从 4 层扩到 8 层（B 学习容量更大），共享层从 4 扩到 8（A 知识冻结量翻倍），比例为 1:1。**必须从头重训 Task A**（新配置不能从 V38 checkpoint 续）。

## 7. 训练命令

### 7.1 V39 Task A（Stage 1，单卡或双卡）

单卡：
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --run_root_dir $LOGS_ROOT --run_id_override "rt_v39_taskA" \
  --max_steps 30000 --save_freq 6000 \
  --vla_path $VLA_PATH \
  --dataset_name aloha_handover_mic_clean \
  --batch_size 1 --grad_accumulation_steps 8 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --first_lora_layer 16 \
  --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 1 --image_aug True \
  --use_proprio True --use_film True --num_images_in_input 3
```

双卡（4、5 号卡，DDP）：
```bash
CUDA_VISIBLE_DEVICES=4,5 PYTORCH_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=offline torchrun --standalone --nproc_per_node 2 vla-scripts/train_cl_lora.py \
  --run_root_dir $LOGS_ROOT --run_id_override "rt_v39_taskA" \
  --max_steps 30000 --save_freq 6000 \
  --vla_path $VLA_PATH \
  --dataset_name aloha_handover_mic_clean \
  --batch_size 1 --grad_accumulation_steps 4 --learning_rate 5e-4 \
  ...（其余参数同上）
```
- 双卡时 `grad_accumulation_steps 8→4` 保持有效 batch=8 不变（1×4×2）。
- **判断是否真的加速**：看单位时间总步数，不是看日志里的秒/步（DDP 下每卡单步时间不变）。
- 若 GPU-Util 低（数据瓶颈）：检查 `nproc` 核数，必要时把 `dataset.py:153` 的 `num_parallel_calls=16` 减半。

### 7.2 V38 Task B（Stage 2，无回放，40000 步，参考）
```bash
... --run_id_override "rt_v38_taskB" --max_steps 40000 --save_freq 6000 \
  --dataset_name aloha_grab_roller_clean \
  --stage 2 --first_lora_layer 24 --shared_depth 4 \
  --previous_checkpoint_dir $LOGS_ROOT/rt_v38_taskA--30000_chkpt \
  --previous_checkpoint_step 30000
```
V39 的 Stage 2 同理：`--run_id_override rt_v39_taskB`、`--dataset_name aloha_grab_roller_clean`、`--stage 2`、`--first_lora_layer 16 --shared_depth 8`、`--previous_checkpoint_dir $LOGS_ROOT/rt_v39_taskA--30000_chkpt --previous_checkpoint_step 30000`。

## 8. 评估命令

```bash
cd /mnt/data/pengshengdi/RoboTwin-main && bash policy/openvla-oft/eval.sh \
  handover_mic demo_clean $CKPT 0 0 aloha_handover_mic_clean 1
```
- 参数：`task_name task_config checkpoint_path seed gpu_id unnorm_key eval_task_id`
- `eval_task_id`：0=当前权重，1=A bank，2=B bank，3=C，4=D。
- 每个任务有固定 prompt（在 `deploy_policy.py` 的 `_FIXED_PROMPTS`）。
- **评估旧任务前必须**：`export OPENVLA_BASE_PATH=...`；且确认 checkpoint 的 `dataset_statistics.json` 包含该任务的 key（后续用修复后的脚本训练会自动累积；老 checkpoint 需手动合并）。

## 9. 关键分析结论（下一阶段最相关）

### 9.1 为什么 Stage 2 后旧任务掉点（核心待办）
- bank 只覆盖 specific B + block_scale + action_head LoRA（LLM 侧）。
- **FiLM 泄漏**：`film_vit_wrapper.py` 给 SigLIP（~27 block）和 DINOv2（~24 block）的每个 ViT block 都挂 `scale/shift = Linear(4096→vision_dim)`，**约 450M 可训参数**，是 LLM 侧 bank（~3M）的 150 倍。它在 Stage 2 训练时自由漂向新任务，**不在 bank 里** → 恢复旧任务 bank 也救不回它。
- **proprio_projector 从未被训练**：`train_cl_lora.py` 的 `trainable_params` 只含 vla + action_head，proprio 不在 optimizer 里（A 能到 88% 说明模型基本忽略它，影响小）。
- **修复方案**：
  - 方案 A（推荐先试）：Stage>1 冻结 FiLM——
    ```python
    if cfg.use_film and cfg.stage > 1:
        for p in vla.vision_backbone.parameters():
            p.requires_grad = False
    ```
  - 方案 B（B 掉点时的备选）：把 FiLM 参数存进每个 task bank，评估按任务恢复。
- 验证方法：看 Stage 2 训练日志 `# total trainable params`，目前应是 ~4.5 亿量级（有 FiLM 泄漏）；冻结后应 ~300 万。

### 9.2 评估慢
- 已修：关闭视频录制（`--eval_video_log False`）、`clear_cache_freq=50`。
- 新服务器 flash-attn 可能未装：装好能加速 LLM 注意力，但**基座模型 config.json 里没有 `attn_implementation`/`use_flash_attention_2` 字段**（已确认输出 None）→ 加载走 eager。若要启用需在 `get_vla` 显式传 `attn_implementation="flash_attention_2"`，且注意 transformers fork 是双向注意力并行解码，需做数值 A/B 验证。
- 评估慢通常不是 GPU 算力问题，瓶颈在 CPU（仿真渲染/图像预处理）和磁盘。用 `nvidia-smi -l 2` 看 GPU-Util 判断。

### 9.3 任务数据管线
- `unify_prompts.py`：统一每个任务的语言指令。
- `preprocess_aloha.py`：转 ALOHA 格式（注意 episode 索引从 hdf5 文件名取）。
- RLDS builder：`aloha_{task}_clean`，已注册进 `mixtures.py` / `configs.py` / `transforms.py`。

## 10. 待办事项（按优先级）

1. **[阻塞] 设置评估环境**：`export OPENVLA_BASE_PATH=...`，确认基座模型在服务器上（`ls .../openvla-7b/config.json`）。
2. **v39 Task A 训练**（Stage 1，30000 步，双卡 4/5 号），完成后评估 Task A 单任务成功率（预期 ≥88%）。
3. **v39 Task B 训练**（Stage 2，40000 步，从 v39 Task A checkpoint），评估 Task A（eval_task_id=1）和 Task B（eval_task_id=2）。
4. **如果旧任务掉点**：实施方案 A（冻结 FiLM）重训 Stage 2 验证；若 B 也掉，转方案 B（FiLM 进 bank）。
5. Task C、D 同流程。
6. **原型回放阶段**（最终目标）：`build_replay_buffer_openvla.py` 建 buffer → 回放训练（`--use_kd --use_replay --replay_buffer_dirs`）。
7. **flash-attn 安装**：先查 `python -c "import torch; print(torch.__version__, torch.version.cuda)"` 再选版本（torch 2.4/2.5→2.5.5，2.6→2.7.2.post1，2.7+→2.8.0）；`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。
8. **验证 DDP 提速**：跑 5 分钟对比双卡 vs 单卡总步数。
9. 更新报告文档（1st.docx）。

## 11. 已知坑 / 注意事项

- **不要从聊天窗口复制多行代码到服务器**——用 git 同步，或复制后仔细检查 `\n`、`__file__`、反斜杠是否被吃掉。
- **task_config/ 不走 git**：改 yml 直接改服务器文件，但不会被 git 覆盖也不会被 git 同步。
- **两份 prismatic**：改代码先确认训练/评估实际 import 的是哪份（训练用 `openvla-oft/prismatic`，editable 安装）。
- **checkpoint 的 dataset_statistics.json**：多任务顺序训练后，每个 checkpoint 必须含所有历史任务的 stats（新代码会自动合并；旧 checkpoint 手动合并）。
- **多卡 DDP**：`CUDA_VISIBLE_DEVICES` 必须与 `nproc_per_node` 匹配；脚本用 accelerate `PartialState` 自动检测，不用改代码。
- **OIDN_DEVICE=cpu**：svulkan2 渲染噪声，无害。
- **评估旧任务**：`eval_task_id` 必须对（1=A，2=B…），用 0 评估旧任务会得到错误的低分（那是当前任务的 bank）。
