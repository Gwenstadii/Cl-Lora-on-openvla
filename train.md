# CL-LoRA 持续学习实验训练指南

## 四分支对照

| 分支 | 名称                         | 训练入口                           | 关键参数                                                       |
| -- | -------------------------- | ------------------------------ | ---------------------------------------------------------- |
| ①  | 普通 LoRA + no replay        | `vla-scripts/finetune.py`      | `--use_lora`（不加 CL-LoRA 参数）                                |
| ②  | CL-LoRA + no replay        | `vla-scripts/train_cl_lora.py` | `--use_kd False --use_replay False`                        |
| ③  | CL-LoRA + Prototype Replay | `vla-scripts/train_cl_lora.py` | `--use_kd --use_replay --replay_buffer_dirs <proto_dir>`   |
| ④  | CL-LoRA + Uniform Replay   | `vla-scripts/train_cl_lora.py` | `--use_kd --use_replay --replay_buffer_dirs <uniform_dir>` |

**分支① 使用** **`finetune.py`（标准 LoRA），分支②③④ 使用** **`train_cl_lora.py`（CL-LoRA）。**

***

## 环境前置

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft/LIBERO:/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft/:$PYTHONPATH"
export VLA_PATH="/root/autodl-tmp/models/openvla-7b"
export DATA_ROOT="/root/autodl-tmp/modified_libero_rlds"
export LOGS_ROOT="/root/autodl-tmp/LOGS-2"
/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft
```

***

## 任务规划（LIBERO 四阶段）

| 阶段     | 数据集                       | task\_suite\_name | 示例 max\_steps |
| ------ | ------------------------- | ----------------- | ------------- |
| Task A | `libero_spatial_no_noops` | `libero_spatial`  | 2000          |
| Task B | `libero_spatial_no_noops` | `libero_spatial`  | 4000          |
| Task C | `libero_object_no_noops`  | `libero_object`   | 32000         |
| Task D | `libero_goal_no_noops`    | `libero_goal`     | 4000          |

> Task A/B 是 `libero_spatial` 中的两个不同任务，通过修改 `run_libero_eval.py` 中的 `TARGET_TASK_ID` 来区分（见评估章节）。

***

## 阶段一：训练 Task A

### 分支① 普通 LoRA

```bash
cd /root/openvla-oft

torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "overfit_test_2000_steps_bs1" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_lora True \
  --lora_rank 16 \
  --max_steps 2000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

Checkpoint 输出路径: `$LOGS_ROOT/overfit_test_2000_steps_bs1--2000_chkpt/`

### 分支②  CL-LoRA + no replay

```bash
cd /root/openvla-oft/Cl-Lora-on-openvla/openvla-oft

WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_taskA_2k" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd False \
  --use_replay False \
  --stage 1 \
  --max_steps 2000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

> 分支③④ 的阶段一与此完全相同（都是用 `train_cl_lora.py --use_kd False --use_replay False`），共用同一个 checkpoint。

***

## 阶段一之后：构建回放缓冲

仅分支③④需要。先用阶段一的 checkpoint 构建 Task A 的回放缓冲。

### 原型回放缓冲（分支③）

编辑 `vla-scripts/build_replay_buffer_openvla.py` 中的 `BuildReplayBufferConfig` 参数后运行：

```bash
cd /root/openvla-oft
python vla-scripts/build_replay_buffer_openvla.py
```

关键参数（在 `BuildReplayBufferConfig` 类中修改）：

- `vla_path = "/root/autodl-tmp/models/openvla-7b"`
- `cl_lora_path = "/root/autodl-tmp/LOGS/cl_lora_taskA_2k--2000_chkpt/cl_lora_adapter.pt"`
- `data_root_dir = "/root/autodl-tmp/modified_libero_rlds"`
- `dataset_name = "libero_spatial_no_noops"`
- `target_task_name = "pick up the black bowl next to the cookie box and place it on the plate"`（Task A 的语言指令）
- `output_dir = "/root/autodl-tmp/replay_buffers/taskA_prototype"`
- `top_k_per_segment = 2`

### 均匀回放缓冲（分支④）

```bash
cd /root/openvla-oft

python vla-scripts/build_uniform_replay_buffer.py \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --target_task_name "pick up the black bowl next to the cookie box and place it on the plate" \
  --output_dir /root/autodl-tmp/replay_buffers/taskA_uniform \
  --match_budget_buffer_dir /root/autodl-tmp/replay_buffers/taskA_prototype \
  --overwrite True
```

> `--match_budget_buffer_dir` 指向原型回放缓冲目录，自动匹配相同样本数预算，保证公平对比。

***

## 阶段二：训练 Task B

**Task A checkpoint 路径可复用：**

```bash
TASK_A_CKPT="$LOGS_ROOT/overfit_test_2000_steps_bs1--2000_chkpt"        # 分支①
TASK_A_CL_CKPT="$LOGS_ROOT/cl_lora_taskA_2k--2000_chkpt"                # 分支②③④
```

### 分支① 普通 LoRA 顺序微调

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path $TASK_A_CKPT \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "task_b_cl_10k_from_90" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_lora True \
  --lora_rank 16 \
  --max_steps 4000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

> 关键：`--vla_path` 指向前一阶段的 checkpoint 目录（含 `config.json` 和 PEFT adapter）。PEFT 的 `from_pretrained()` 会自动加载 adapter 权重。**不需要** **`--resume`**（`--resume` 是用于恢复中断的训练，不是顺序微调）。

### 分支② CL-LoRA + no replay

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_taskB_no_replay" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd False \
  --use_replay False \
  --stage 2 \
  --previous_checkpoint_dir $TASK_A_CL_CKPT \
  --previous_checkpoint_step 2000 \
  --max_steps 4000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### 分支③ CL-LoRA + Prototype Replay

```bash
PROTO_BUFFER="/root/autodl-tmp/replay_buffers/taskA_prototype"

WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_taskB_proto_replay" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd True \
  --lambda_kd 1.0 \
  --teacher_checkpoint_dir $TASK_A_CL_CKPT \
  --teacher_checkpoint_step 2000 \
  --use_replay True \
  --replay_buffer_dirs $PROTO_BUFFER \
  --replay_loss_weight 1.0 \
  --replay_every_n_steps 1 \
  --stage 2 \
  --previous_checkpoint_dir $TASK_A_CL_CKPT \
  --previous_checkpoint_step 2000 \
  --max_steps 4000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### 分支④ CL-LoRA + Uniform Replay

```bash
UNIFORM_BUFFER="/root/autodl-tmp/replay_buffers/taskA_uniform"

# 参数同分支③，仅 --replay_buffer_dirs 换为均匀回放缓冲
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_taskB_uniform_replay" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd True \
  --lambda_kd 1.0 \
  --teacher_checkpoint_dir $TASK_A_CL_CKPT \
  --teacher_checkpoint_step 2000 \
  --use_replay True \
  --replay_buffer_dirs $UNIFORM_BUFFER \
  --replay_loss_weight 1.0 \
  --replay_every_n_steps 1 \
  --stage 2 \
  --previous_checkpoint_dir $TASK_A_CL_CKPT \
  --previous_checkpoint_step 2000 \
  --max_steps 4000 \
  --save_freq 2000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

***

## 阶段三：训练 Task C

Task C 切换数据集到 `libero_object_no_noops`，训练步数增大（32k）。

### 分支① 普通 LoRA

```bash
TASK_B_CKPT="$LOGS_ROOT/task_b_cl_10k_from_90--2000_chkpt"

WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path $TASK_B_CKPT \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_object_no_noops" \
  --run_id_override "normal_lora_taskC_object" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_lora True \
  --lora_rank 16 \
  --max_steps 32000 \
  --save_freq 16000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### 分支② CL-LoRA + no replay

```bash
TASK_B_CL_CKPT="$LOGS_ROOT/cl_lora_taskB_no_replay--4000_chkpt"

WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_object_no_noops" \
  --run_id_override "cl_lora_taskC_no_replay" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd False \
  --use_replay False \
  --stage 3 \
  --previous_checkpoint_dir $TASK_B_CL_CKPT \
  --previous_checkpoint_step 4000 \
  --max_steps 32000 \
  --save_freq 16000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### 分支③ CL-LoRA + Prototype Replay

阶段二完成后需要先构建 Task B 的回放缓冲（与阶段一之后同理），然后阶段三同时使用 Task A + Task B 两个 replay buffer。

```bash
# 先构建 Task B 原型回放缓冲（同阶段一之后步骤，改 target_task_name 为 Task B 指令）

TASK_B_CL_CKPT="$LOGS_ROOT/cl_lora_taskB_proto_replay--4000_chkpt"
PROTO_BUFFER_A="/root/autodl-tmp/replay_buffers/taskA_prototype"
PROTO_BUFFER_B="/root/autodl-tmp/replay_buffers/taskB_prototype"

WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name "libero_object_no_noops" \
  --run_id_override "cl_lora_taskC_proto_replay" \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_cl_lora True \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init True \
  --freeze_a True \
  --use_block_scale True \
  --use_kd True \
  --lambda_kd 1.0 \
  --teacher_checkpoint_dir $TASK_B_CL_CKPT \
  --teacher_checkpoint_step 4000 \
  --use_replay True \
  --replay_buffer_dirs $PROTO_BUFFER_A $PROTO_BUFFER_B \
  --replay_loss_weight 1.0 \
  --replay_every_n_steps 1 \
  --stage 3 \
  --previous_checkpoint_dir $TASK_B_CL_CKPT \
  --previous_checkpoint_step 4000 \
  --max_steps 32000 \
  --save_freq 16000 \
  --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### 分支④ CL-LoRA + Uniform Replay

同分支③，`--replay_buffer_dirs` 替换为对应的 uniform buffer。

***

## 阶段四：训练 Task D

Task D 切换到 `libero_goal_no_noops`。模式同阶段三：

- 分支①：`--vla_path` 指向前一阶段 checkpoint
- 分支②③④：`--stage 4 --previous_checkpoint_dir ... --teacher_checkpoint_dir ... --replay_buffer_dirs <A B C 三个 buffer>`

***

## 评估

### 前置操作：修改 TARGET\_TASK\_ID

评估前需修改 `experiments/robot/libero/run_libero_eval.py` 中的 `TARGET_TASK_ID` 变量：

| 评估目标   | task\_suite\_name | TARGET\_TASK\_ID |
| ------ | ----------------- | ---------------- |
| Task A | `libero_spatial`  | 6（或对应的 task ID）  |
| Task B | `libero_spatial`  | 0（或对应的 task ID）  |
| Task C | `libero_object`   | 按实际 task 设定      |
| Task D | `libero_goal`     | 按实际 task 设定      |

> 不同 `task_suite_name` 之间的 task ID 含义不同。`TARGET_TASK_ID` 对应 suite 内的任务编号。

### 跨数据集评估：合并 dataset\_statistics.json

当在数据集 A 上训练、在数据集 B 上评估时（如 Task C 训练于 `libero_object` 但要评估 Task A/B 的 `libero_spatial`），需要合并 `dataset_statistics.json`，否则动作反归一化会出错。

```bash
python -c "
import json

# 将 base 数据集的 statistics 合并进 target checkpoint
path_base = '/root/autodl-tmp/LOGS/task_b_cl_10k_from_90--2000_chkpt/dataset_statistics.json'
target_ckpt = '/root/autodl-tmp/LOGS/normal_lora_taskC_object--32000_chkpt'

with open(path_base, 'r') as f:
    base_stats = json.load(f)

with open(target_ckpt + '/dataset_statistics.json', 'r') as f:
    target_stats = json.load(f)

target_stats.update(base_stats)

with open(target_ckpt + '/dataset_statistics.json', 'w') as f:
    json.dump(target_stats, f)

print('Merged dataset_statistics.json for cross-dataset evaluation.')
"
```

> 原理：评估时动作反归一化依赖 `dataset_statistics.json` 中的 `action_mean` / `action_std`。当 checkpoint 是在 `libero_object` 上训练的，其 statistics 不包含 `libero_spatial` 的动作分布信息，需要从其他 checkpoint 中补充。

### 评估命令

```bash
cd /root/openvla-oft

# 评估单一任务
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS/overfit_test_2000_steps_bs1--2000_chkpt \
  --task_suite_name libero_spatial

# 跨数据集评估（确保已合并 dataset_statistics.json）
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS/normal_lora_taskC_object--32000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False

# CL-LoRA checkpoint 评估
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS/cl_lora_taskB_proto_replay--4000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False
```

> 对于 CL-LoRA 训练的 checkpoint，`run_libero_eval.py` 中需要加载 `cl_lora_adapter.pt`。如果评估脚本尚未支持 CL-LoRA 权重加载，需要在 `experiments/robot/openvla_utils.py` 的模型加载逻辑中添加对应路径。

***

## 训练运维

### RTX 5090 单次训练耗时估算

OpenVLA-7B（DinoSigLIP + LLaMA-2 7B），单卡 RTX 5090，`batch_size=1 --grad_accumulation_steps=8`，L1 regression 动作头：

| 阶段     | max\_steps | 数据集                       | 估算耗时           |
| ------ | ---------- | ------------------------- | -------------- |
| Task A | 2,000      | `libero_spatial_no_noops` | **\~2–3 小时**   |
| Task B | 4,000      | `libero_spatial_no_noops` | **\~4–6 小时**   |
| Task C | 32,000     | `libero_object_no_noops`  | **\~30–40 小时** |
| Task D | 4,000      | `libero_goal_no_noops`    | **\~4–6 小时**   |

> 瓶颈在每步都需要完整走 7B LLaMA decoder forward pass。若使用 `--use_diffusion`（DDIM 多步采样）会更慢。以上为 L1 regression 估算。

### 实时输出

训练过程中的实时输出来源：

| 输出渠道                     | 内容                                                                                                      | 频率                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------- |
| **控制台（stdout）**          | CL-LoRA 注入层数/范围、可训参数量、Teacher 快照加载确认、Replay loader 初始化、Checkpoint 保存路径                                  | 训练开始 + 每次 save checkpoint   |
| **tqdm 进度条**             | `max_steps` 总进度，当前 step 数，已耗时                                                                           | 每个 gradient step 更新         |
| **W\&B**                 | `CL Train/Loss`、`CL Train/Loss Task`、`CL Train/Loss Kd`、`CL Train/Loss Replay`、`CL Train/Learning Rate` | 每 `wandb_log_freq` 步（默认 10） |
| **`WANDB_MODE=offline`** | 日志写入本地 `wandb/` 目录，事后 `wandb sync` 上传                                                                   | —                           |

训练开始时的典型控制台输出：

```
--- Injecting CL-LoRA ---
Total Depth: 32
Shared Layers (Frozen A):    0 to 15
Specific Layers (Learnable):  16 to 31
Replaced 224 Linear layers with CLLoRALinear.

# trainable params in action_head: 589824
# total trainable params: 74563840

[Teacher] Loading snapshot from /root/.../teacher_snapshot--2000.pt (336 tensors)
[Replay] Initialized 1 replay loaders from: ['/root/.../taskA_prototype']

Checkpoint saved at step 2000 → /root/.../my_run--2000_chkpt
```

> 如果训练中途 W\&B 上传失败（网络问题），训练本身不会中断。使用 `WANDB_MODE=offline` 可完全避免此问题。

### 断点续训

#### 场景一：同一次训练中断后恢复（崩溃/OOM/超时）

训练意外中断时，从最近的 checkpoint 继续：

```bash
# finetune.py（分支①）
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/LOGS/my_run--2000_chkpt \
  --resume True \
  --resume_step 2000 \
  --max_steps 4000 \
  --run_id_override "my_run" \
  ...（其余参数与中断前保持一致）

# train_cl_lora.py（分支②③④）
torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --previous_checkpoint_dir /root/autodl-tmp/LOGS/my_cl_run--2000_chkpt \
  --previous_checkpoint_step 2000 \
  --resume True \
  --resume_step 2000 \
  --max_steps 4000 \
  --run_id_override "my_cl_run" \
  ...（其余参数与中断前保持一致）
```

**关键注意事项：**

- `--vla_path` / `--previous_checkpoint_dir` 必须指向最后一次 save 的 checkpoint 目录（如 `xxx--2000_chkpt`）
- `--resume_step` 必须与 checkpoint 的步数匹配
- `--max_steps` 设为最终目标步数（不是剩余步数）
- `--run_id_override` 保持一致，避免创建新目录

**当前局限：**`train_cl_lora.py` 的 checkpoint 保存了 CL-LoRA adapter + action\_head + teacher\_snapshot，但**未保存 optimizer state 和 scheduler state**。从断点恢复时 optimizer 会重新初始化，动量等状态会丢失，可能轻微影响后续训练动态。`finetune.py` 同样不保存 optimizer state。对于 LIBERO 单任务这种短期训练（2k–4k 步），影响通常可忽略。

#### 场景二：顺序微调下一阶段（正常流程，非断点）

```bash
# finetune.py：只改 --vla_path，不加 --resume
--vla_path /root/autodl-tmp/LOGS/taskA--2000_chkpt

# train_cl_lora.py：用 --previous_checkpoint_dir 加载上一阶段 CL-LoRA 权重
--previous_checkpoint_dir /root/autodl-tmp/LOGS/taskA--2000_chkpt
--previous_checkpoint_step 2000
--stage 2
```

> 场景二是正常的多阶段流程，optimizer 和 scheduler 从零开始是预期行为。**不要**在场景二使用 `--resume`。

***

## 性能矩阵记录

每阶段训练结束后，对**所有已学任务**进行评估：

```
                Task A    Task B    Task C    Task D
Stage 1 (A)        0.90       -         -         -
Stage 2 (A→B)      0.XX     0.XX        -         -
Stage 3 (A→B→C)    0.XX     0.XX      0.XX        -
Stage 4 (A→B→C→D)  0.XX     0.XX      0.XX      0.XX
```

BWT（Backward Transfer）：

```
BWT = mean(每个旧任务的最终成功率 - 其初始成功率)
```

***

## 分支对照逻辑

| 对照         | 目的                                        |
| ---------- | ----------------------------------------- |
| 分支① vs 分支② | 验证 CL-LoRA 结构本身能否缓解纯顺序微调的灾难性遗忘            |
| 分支② vs 分支③ | 验证旧任务少样本 prototype replay 是否是稳定多阶段性能的关键   |
| 分支③ vs 分支④ | 在相同 replay 样本预算下，prototype 筛选机制是否优于普通均匀抽样 |

***

## 关键超参数

### LoRA / CL-LoRA

| 参数                                       | 分支①   | 分支②③④ | 说明                            |
| ---------------------------------------- | ----- | ----- | ----------------------------- |
| `lora_rank`                              | 16    | 32    | CL-LoRA 需要更大秩容纳多任务知识          |
| `shared_depth`                           | —     | 16    | LLaMA-2 32 层的一半；越大抗遗忘越强但可塑性降低 |
| `orthogonal_init`                        | —     | True  | 使不同任务 LoRA 子空间尽量正交            |
| `freeze_a`                               | —     | True  | 冻结共享层 A 矩阵是抗遗忘核心              |
| `use_block_scale`                        | —     | True  | 每层自动调节新旧任务平衡                  |
| `batch_size` / `grad_accumulation_steps` | 1 / 8 | 1 / 8 | 有效 batch = 8                  |

### KD 与 Replay

| 参数                       | 推荐值           | 说明                       |
| ------------------------ | ------------- | ------------------------ |
| `lambda_kd`              | 1.0           | KD 损失权重，与 task loss 等权   |
| `replay_loss_weight`     | 1.0           | 回放损失权重，与 task loss 等权    |
| `replay_every_n_steps`   | 1             | 每步回放                     |
| `replay_sample_strategy` | `round_robin` | 多 buffer 时轮询；可选 `random` |

### 回放缓冲构建

| 参数                        | 推荐值  | 说明         |
| ------------------------- | ---- | ---------- |
| `top_k_per_segment`       | 2    | 每段保留帧数     |
| `translation_threshold_m` | 0.03 | 平动切分阈值（米）  |
| `rotation_threshold_rad`  | 0.05 | 转动切分阈值（弧度） |
| `gripper_threshold`       | 0.1  | 夹爪变化阈值     |

***

## 文件变更汇总

### 新建

| 文件                                           | 说明                          |
| -------------------------------------------- | --------------------------- |
| `vla-scripts/train_cl_lora.py`               | CL-LoRA 持续学习训练脚本（分支②③④统一入口） |
| `vla-scripts/build_uniform_replay_buffer.py` | 均匀回放缓冲构建器                   |

### 修改

| 文件                        | 改动                                                                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `vla-scripts/cl_lora.py`  | `CLLoRALinear` / `inject_cl_lora_into_model` 支持可配置的 `orthogonal_init`、`freeze_a`、`use_block_scale`                            |
| `vla-scripts/finetune.py` | `FinetuneConfig` 新增 `use_cl_lora`、`shared_depth`、`orthogonal_init`、`freeze_a`、`use_block_scale`、`clip_weight`；支持 CL-LoRA 注入路径 |

