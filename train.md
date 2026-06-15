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
export PYTHONPATH="/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft/LIBERO:/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft:$PYTHONPATH"
export VLA_PATH="/root/autodl-tmp/models/openvla-7b"
export DATA_ROOT="/root/autodl-tmp/modified_libero_rlds"
export LOGS_ROOT="/root/autodl-tmp/LOGS-2"

# 每次运行前先进入项目目录
cd /root/openvla-oft
```

***

## 实验总流程

提供两种执行策略，可根据实验习惯选择。

### 策略 A：分支优先（推荐）

一条分支从头跑到底（Task A→B→C→D），评估完毕后再启动下一条分支。优点是思维负担轻，中途出错只需回滚当前分支。

```
═══ 分支①：普通 LoRA + no replay ═════════════════════════════════════

  A1  训练 Task A                           finetune.py
  A2  评估 Task A                           修改 TARGET_TASK_ID
  A3  训练 Task B（--vla_path 指向 A1 的 checkpoint）
  A4  评估 Task A + Task B
  A5  训练 Task C（--vla_path 指向 A3 的 checkpoint，切换数据集 libero_object）
  A6  合并 dataset_statistics.json → 评估 Task A+B+C
  A7  训练 Task D（--vla_path 指向 A5 的 checkpoint，切换数据集 libero_goal）
  A8  合并 dataset_statistics.json → 评估 Task A+B+C+D  ← 分支①全部完成

═══ 分支②：CL-LoRA + no replay ═══════════════════════════════════════

  B1  训练 Task A                           train_cl_lora.py（--use_kd False --use_replay False）
  B2  评估 Task A
  B3  训练 Task B（--previous_checkpoint_dir 指向 B1，--use_kd False --use_replay False）
  B4  评估 Task A + B
  B5  训练 Task C（--previous_checkpoint_dir 指向 B3，切换数据集）
  B6  合并 dataset_statistics.json → 评估 Task A+B+C
  B7  训练 Task D（--previous_checkpoint_dir 指向 B5，切换数据集）
  B8  合并 dataset_statistics.json → 评估 Task A+B+C+D  ← 分支②全部完成

═══ 分支③：CL-LoRA + Prototype Replay ════════════════════════════════

  C1  训练 Task A                           train_cl_lora.py（同分支② Stage 1）
  C2  评估 Task A
  C3  构建 Task A 原型回放缓冲                build_replay_buffer_openvla.py
  C4  训练 Task B（--use_kd --use_replay --replay_buffer_dirs <taskA_proto>）
  C5  评估 Task A + B
  C6  构建 Task B 原型回放缓冲
  C7  训练 Task C（--use_kd --use_replay --replay_buffer_dirs <taskA_proto> <taskB_proto>）
  C8  合并 dataset_statistics.json → 评估 Task A+B+C
  C9  构建 Task C 原型回放缓冲
  C10 训练 Task D（--replay_buffer_dirs 三个 buffer）
  C11 合并 dataset_statistics.json → 评估 Task A+B+C+D  ← 分支③全部完成

═══ 分支④：CL-LoRA + Uniform Replay ══════════════════════════════════

  D1~D11  同分支③，全部「原型回放」替换为「均匀回放」
          构建缓冲用 build_uniform_replay_buffer.py
          训练时 --replay_buffer_dirs 指向对应的 uniform buffer
```

> **核心规则（两条策略通用）：**
>
> - 分支① 的 `--vla_path` 始终指向前一阶段分支①自己的 checkpoint
> - 分支②③④ 的 `--vla_path` 始终指向 base model（`$VLA_PATH`），用 `--previous_checkpoint_dir` 加载前一阶段 CL-LoRA 权重
> - 分支③④ 从阶段二开始 `--teacher_checkpoint_dir` 指向前一阶段的 checkpoint（供 KD loss）

### 策略 B：阶段优先（原流程）

四个分支在同一阶段交替推进，每个阶段结束后统一评估。适合需要横向对比各分支在同阶段表现的场景。

```
═══ 阶段一 ═══
  分支① Task A  →  分支② Task A  →  评估 Task A
  构建 taskA replay buffers

═══ 阶段二 ═══
  分支① Task B  →  分支② Task B  →  分支③ Task B  →  分支④ Task B
  评估全部（四个分支 × 两个任务）
  构建 taskB replay buffers

═══ 阶段三 ═══  (切换数据集 libero_object)
  四个分支 Task C  →  评估全部（四个分支 × 三个任务）
  构建 taskC replay buffers

═══ 阶段四 ═══  (切换数据集 libero_goal)
  四个分支 Task D  →  评估全部（四个分支 × 四个任务）
```

> 策略 B 的命令细节见下方各阶段章节。

### 对比

| <br />                   | 策略 A（分支优先）      | 策略 B（阶段优先） |
| ------------------------ | --------------- | ---------- |
| 已跑完的分支可立即写论文             | 需等全部完成才有完整矩阵    | <br />     |
| 出错影响范围小（仅当前分支）           | 出错影响该阶段全部四个分支   | <br />     |
| 分支③④需要中途构建 replay buffer | 也是中途构建，频率相同     | <br />     |
| GPU 空闲零碎，一次跑一条线          | GPU 可在阶段内连续跑多条线 | <br />     |
| **目前推荐**                 | 适合最终确认实验时交叉验证   | <br />     |

***

## 产生的文件树

```
LOGS-2/
├── [分支①]  overfit_test_2000_steps_bs1--2000_chkpt/
├── [分支①]  task_b_cl_10k_from_90--4000_chkpt/
├── [分支①]  normal_lora_taskC_object--32000_chkpt/
├── [分支①]  normal_lora_taskD_goal--4000_chkpt/
├── [②③④共用] cl_lora_taskA_2k--2000_chkpt/
├── [分支②]   cl_lora_taskB_no_replay--4000_chkpt/
├── [分支③]   cl_lora_taskB_proto_replay--4000_chkpt/
├── [分支④]   cl_lora_taskB_uniform_replay--4000_chkpt/
├── [分支②]   cl_lora_taskC_no_replay--32000_chkpt/
├── [分支③]   cl_lora_taskC_proto_replay--32000_chkpt/
├── [分支④]   cl_lora_taskC_uniform_replay--32000_chkpt/
├── [分支②]   cl_lora_taskD_no_replay--4000_chkpt/
├── [分支③]   cl_lora_taskD_proto_replay--4000_chkpt/
└── [分支④]   cl_lora_taskD_uniform_replay--4000_chkpt/

replay_buffers/
├── taskA_prototype/   taskA_uniform/
├── taskB_prototype/   taskB_uniform/
├── taskC_prototype/   taskC_uniform/
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

### 评估原理

`run_libero_eval.py` 的核心逻辑：

1. 加载指定 checkpoint 的模型权重
2. 遍历 `--task_suite_name` 对应 suite 下的所有任务
3. 对每个任务，检测其任务名称是否包含 `TARGET_TASK_NAME`（硬编码在脚本第 484 行）
4. 匹配成功后，执行 `--num_trials_per_task` 次 rollout，统计成功率

**每次评估前必须改的参数：**

- `run_libero_eval.py` 第 484 行的 `TARGET_TASK_NAME`（指定评估哪个具体任务）
- CLI 的 `--pretrained_checkpoint`（指定用哪个 checkpoint 评估）
- CLI 的 `--task_suite_name`（指定在哪个 benchmark suite 里找任务）

### 前置准备：获取任务名称

评估的第一步是知道你要评测的任务在 LIBERO 里的确切名称。可以用这个命令列出 suite 的所有任务：

```bash
cd /root/openvla-oft

python -c "
from libero.libero import benchmark
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
for i in range(task_suite.n_tasks):
    print(f'ID {i}: {task_suite.get_task(i).name}')
"


[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ID 0: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
ID 1: pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
ID 2: pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
ID 3: pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
ID 4: pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
ID 5: pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
ID 6: pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
ID 7: pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
ID 8: pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
ID 9: pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
```

将 `libero_spatial` 替换为 `libero_object`、`libero_goal` 查看其他 suite 的任务列表。记下你要评测的任务名称（完整字符串）。

***

### 策略 A 评估：分支① 普通 LoRA

分支① 的所有 checkpoint 都是标准 PEFT LoRA 格式，评估脚本可直接加载。

#### A2 — 评估 Task A（训练完 Task A 之后）

```bash
cd /root/openvla-oft

# ⚠️ 前置：修改 run_libero_eval.py 第 484 行
#     TARGET_TASK_NAME = "你的 Task A 任务名称"
#     例如: "pick up the black bowl next to the cookie box and place it on the plate"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/overfit_test_2000_steps_bs1--2000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False \
  --num_images_in_input 1 \
  --lora_rank 16
```

| 参数                        | 含义                       | 为什么设这个值                       |
| ------------------------- | ------------------------ | ----------------------------- |
| `--pretrained_checkpoint` | 要评估的 checkpoint 目录       | 分支① Task A 训练的输出目录            |
| `--task_suite_name`       | 在哪个 benchmark suite 里找任务 | Task A 来自 `libero_spatial`    |
| `--use_proprio False`     | 不使用 proprio 输入           | 训练时没开 proprio，评估也必须关          |
| `--num_images_in_input 1` | 只用一张第三视角图像               | 训练时没开 wrist camera，保持一致       |
| `--lora_rank 16`          | LoRA 秩                   | 必须等于训练时的 `lora_rank`（分支①用 16） |

#### A4 — 评估 Task A + Task B（训练完 Task B 之后）

Task B 的 checkpoint 包含 Task A 的 LoRA adapter（因为是从 Task A 的 checkpoint 继续训练的），评估时只需评估这个最新 checkpoint。

```bash
# ⚠️ 前置：修改 TARGET_TASK_NAME = "你的 Task A 任务名称"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/task_b_cl_10k_from_90--4000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False \
  --num_images_in_input 1 \
  --lora_rank 16

# ⚠️ 前置：修改 TARGET_TASK_NAME = "你的 Task B 任务名称"
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/task_b_cl_10k_from_90--4000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False \
  --num_images_in_input 1 \
  --lora_rank 16
```

> **同一条命令跑两次，唯一区别是改** **`TARGET_TASK_NAME`**——第一次评 Task A，第二次评 Task B。

#### A6 — 评估 Task A + B + C（训练完 Task C 之后，跨数据集）

**这一步需要额外操作**，因为 Task C 是在 `libero_object` 上训练的，而 Task A/B 在 `libero_spatial` 上。checkpoint 里的 `dataset_statistics.json` 只包含 `libero_object` 的动作分布。

**Step 1：合并 dataset\_statistics.json**

```bash
python -c "
import json, shutil

ckpt = '/root/autodl-tmp/LOGS-2/normal_lora_taskC_object--32000_chkpt'
stats_spatial = '/root/autodl-tmp/LOGS-2/task_b_cl_10k_from_90--4000_chkpt/dataset_statistics.json'

# 备份原始文件
shutil.copy(ckpt + '/dataset_statistics.json', ckpt + '/dataset_statistics.json.bak')

with open(stats_spatial, 'r') as f:
    spatial = json.load(f)
with open(ckpt + '/dataset_statistics.json', 'r') as f:
    obj = json.load(f)

obj.update(spatial)  # 把 libero_spatial 的统计信息合并进去

with open(ckpt + '/dataset_statistics.json', 'w') as f:
    json.dump(obj, f)

print('Done. libero_object + libero_spatial statistics merged.')
"
```

| 变量              | 含义                                                                 |
| --------------- | ------------------------------------------------------------------ |
| `ckpt`          | Task C 的 checkpoint 路径（需要补充 libero\_spatial 的统计信息）                 |
| `stats_spatial` | 一个在 `libero_spatial` 上训练过的 checkpoint 中的 `dataset_statistics.json` |

**原理：** 评估时动作反归一化靠 `dataset_statistics.json` 里的 `action_mean` / `action_std`。每个数据集（spatial/object/goal）的统计量不同。当用某数据集训练出来的 checkpoint 评估另一数据集的任务时，必须把目标数据集的统计信息合并进去。

**Step 2：执行评估**

```bash
# ⚠️ 每次改 TARGET_TASK_NAME

# 评估 Task A（libero_spatial）
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/normal_lora_taskC_object--32000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False --num_images_in_input 1 --lora_rank 16

# 评估 Task B（libero_spatial）
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/normal_lora_taskC_object--32000_chkpt \
  --task_suite_name libero_spatial \
  --use_proprio False --num_images_in_input 1 --lora_rank 16

# 评估 Task C 自身（libero_object，不需要合并 statistics）
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/LOGS-2/normal_lora_taskC_object--32000_chkpt \
  --task_suite_name libero_object \
  --use_proprio False --num_images_in_input 1 --lora_rank 16
```

#### A8 — 评估 Task A + B + C + D（训练完 Task D 之后，跨数据集）

与 A6 相同模式：Task D 是在 `libero_goal` 上训练的，评估 Task A/B 和 Task C 都需要合并 statistics。

```bash
# Step 1: 合并（需要 spatial + object 的 statistics）
python -c "
import json, shutil

ckpt = '/root/autodl-tmp/LOGS-2/normal_lora_taskD_goal--4000_chkpt'
stats_spatial = '/root/autodl-tmp/LOGS-2/task_b_cl_10k_from_90--4000_chkpt/dataset_statistics.json'
stats_object  = '/root/autodl-tmp/LOGS-2/normal_lora_taskC_object--32000_chkpt/dataset_statistics.json.bak'

shutil.copy(ckpt + '/dataset_statistics.json', ckpt + '/dataset_statistics.json.bak')

with open(stats_spatial) as f: s = json.load(f)
with open(stats_object) as f: o = json.load(f)
with open(ckpt + '/dataset_statistics.json') as f: d = json.load(f)

d.update(s); d.update(o)

with open(ckpt + '/dataset_statistics.json', 'w') as f:
    json.dump(d, f)
print('Done.')
"

# Step 2: 四次评估（每次改 TARGET_TASK_NAME）
# Task A: --task_suite_name libero_spatial
# Task B: --task_suite_name libero_spatial
# Task C: --task_suite_name libero_object
# Task D: --task_suite_name libero_goal（自身数据集，不需要合并但已合并也无害）
```

***

### 策略 A 评估：分支② CL-LoRA + no replay

**关键差异：** CL-LoRA checkpoint 不含 PEFT adapter，而是用 `cl_lora_adapter.pt` 存储权重。评估脚本中 `get_model()` 函数需要能够加载 CL-LoRA 格式的权重。

当前 `experiments/robot/openvla_utils.py` 的 `get_model()` 使用 PEFT 的 `from_pretrained()` 加载 adapter。对于 CL-LoRA checkpoint，需要在评估前**临时修补**模型加载逻辑：

确认 `experiments/robot/openvla_utils.py` 中的 `get_model()` 是否已支持 CL-LoRA。如果未支持（报错找不到 adapter\_model.bin），临时方案是用 `finetune.py` 以 `--use_lora` 模式加载同一个 checkpoint（但会产生一份新的 LoRA adapter），或者直接在 eval 前手动注入 CL-LoRA 并加载 cl\_lora\_adapter.pt。

**评估命令与分支①完全相同，只换 checkpoint 路径和** **`--lora_rank 32`：**

```bash
# B2 — 评估 Task A
#     checkpoint: /root/autodl-tmp/LOGS-2/cl_lora_taskA_2k--2000_chkpt
#     --task_suite_name libero_spatial  --lora_rank 32

# B4 — 评估 Task A + B
#     checkpoint: /root/autodl-tmp/LOGS-2/cl_lora_taskB_no_replay--4000_chkpt
#     --task_suite_name libero_spatial  --lora_rank 32
#     (跑两次，分别改 TARGET_TASK_NAME)

# B6 — 评估 Task A + B + C（需要合并 statistics，同分支① A6 模式）
#     checkpoint: /root/autodl-tmp/LOGS-2/cl_lora_taskC_no_replay--32000_chkpt
#     --lora_rank 32

# B8 — 评估 Task A + B + C + D（需要合并 statistics，同分支① A8 模式）
#     checkpoint: /root/autodl-tmp/LOGS-2/cl_lora_taskD_no_replay--4000_chkpt
#     --lora_rank 32
```

***

### 策略 A 评估：分支③④ CL-LoRA + Replay

**与分支②完全相同**，只换对应的 checkpoint 路径：

| 步骤              | 分支③ checkpoint                            | 分支④ checkpoint                              |
| --------------- | ----------------------------------------- | ------------------------------------------- |
| 评估 Task A       | `cl_lora_taskA_2k--2000_chkpt`            | （共用分支②的 Task A checkpoint）                  |
| 评估 Task A+B     | `cl_lora_taskB_proto_replay--4000_chkpt`  | `cl_lora_taskB_uniform_replay--4000_chkpt`  |
| 评估 Task A+B+C   | `cl_lora_taskC_proto_replay--32000_chkpt` | `cl_lora_taskC_uniform_replay--32000_chkpt` |
| 评估 Task A+B+C+D | `cl_lora_taskD_proto_replay--4000_chkpt`  | `cl_lora_taskD_uniform_replay--4000_chkpt`  |

参数设置完全一致：`--lora_rank 32 --use_proprio False --num_images_in_input 1`，跨数据集时合并 statistics。

***

### 评估参数速查

| 参数                      | 分支①       | 分支②③④    | 说明                      |
| ----------------------- | --------- | -------- | ----------------------- |
| `--lora_rank`           | 16        | 32       | 必须等于训练时使用的秩             |
| `--use_proprio`         | False     | False    | 当前训练都没开 proprio         |
| `--num_images_in_input` | 1         | 1        | 单相机（无 wrist camera）     |
| `--use_l1_regression`   | True（默认）  | True（默认） | 用的是 L1 回归动作头            |
| `--num_trials_per_task` | 50（默认）    | 50（默认）   | 每个任务测 50 次，算成功率         |
| `--task_suite_name`     | 取决于评估哪个任务 | 同左       | spatial / object / goal |

### 成功率计算

每次评估的输出示例：

```
[INFO] 正在执行精准评估任务: pick_up_the_black_bowl... (ID: 6)
[INFO] Trials: 50 | Successes: 45 | Success Rate: 90.00%
```

**记录格式（性能矩阵）：**

```
                Task A    Task B    Task C    Task D
分支① Stage 1    0.90       -         -         -
分支① Stage 2    0.XX     0.XX        -         -
分支① Stage 3    0.XX     0.XX      0.XX        -
分支① Stage 4    0.XX     0.XX      0.XX      0.XX
```

每一行就是上面 A2/A4/A6/A8 评估步骤得到的四个成功率。

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

