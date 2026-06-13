# CL-LoRA 持续学习实验训练指南

## 总览

本项目的四分支对照实验设计：

| 分支 | 名称 | 训练脚本 | 关键特征 |
|---|---|---|---|
| ① | 普通 LoRA + no replay | `finetune.py` | 标准 LoRA 顺序微调（灾难性遗忘基线） |
| ② | CL-LoRA + no replay | `train_cl_lora.py` | CL-LoRA 结构 + 关闭 KD 和 replay |
| ③ | CL-LoRA + Prototype Replay | `train_cl_lora.py` | CL-LoRA + 教师 KD + 原型回放（主方法） |
| ④ | CL-LoRA + Uniform Replay | `train_cl_lora.py` | CL-LoRA + 教师 KD + 均匀回放（公平对照） |

---

## 通用环境变量

```bash
export VLA_PATH="openvla/openvla-7b"
export DATA_ROOT="../modified_libero_rlds"
export LOGS_ROOT="../LOGS"
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="cl-lora-openvla"
```

---

## 实验工作流（以两阶段为例：Task A → Task B）

LIBERO-Spatial 中选取两个不同任务。例如：
- Task A: `"pick up the black bowl next to the cookie box and place it on the plate"`
- Task B: `"put the black bowl on the plate"`

### 阶段一：训练 Task A（所有分支共享此步骤）

#### 分支①：普通 LoRA 训练 Task A

```bash
cd openvla-oft
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $LOGS_ROOT/branch1_stage1 \
  --use_lora \
  --lora_rank 16 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --max_steps 4000 \
  --save_freq 1000 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note "branch1_taskA"
```

#### 分支②③④：CL-LoRA 训练 Task A

```bash
torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $LOGS_ROOT/branch234_stage1 \
  --use_cl_lora \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init \
  --freeze_a \
  --use_block_scale \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --max_steps 4000 \
  --save_freq 1000 \
  --use_kd=False \
  --use_replay=False \
  --stage 1 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note "branch234_taskA"
```

> 阶段一训练完成后，记录 Task A 评估成功率。checkpoint 保存在 `$LOGS_ROOT/branch234_stage1/<run_id>--4000_chkpt/`。

---

### 阶段一之后：构建回放缓冲（分支③④需要）

#### 构建原型回放缓冲（分支③）

```bash
# 修改 build_replay_buffer_openvla.py 中的 BuildReplayBufferConfig 参数后运行：
python vla-scripts/build_replay_buffer_openvla.py
```

关键参数说明：
- `vla_path`: OpenVLA-7B 路径
- `cl_lora_path`: 阶段一训练好的 `cl_lora_adapter.pt` 路径
- `data_root_dir`: RLDS 数据集目录
- `dataset_name`: `libero_spatial_no_noops`
- `target_task_name`: Task A 的语言指令
- `output_dir`: 回放缓冲输出目录（如 `$LOGS_ROOT/replay_buffers/taskA_prototype`）
- `top_k_per_segment`: 每段保留的帧数（默认 2）
- `translation_threshold_m`: 平动阈值（默认 0.03）
- `rotation_threshold_rad`: 转动阈值（默认 0.05）
- `gripper_threshold`: 夹爪阈值（默认 0.1）

#### 构建均匀回放缓冲（分支④）

```bash
python vla-scripts/build_uniform_replay_buffer.py \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --target_task_name "pick up the black bowl next to the cookie box and place it on the plate" \
  --output_dir $LOGS_ROOT/replay_buffers/taskA_uniform \
  --match_budget_buffer_dir $LOGS_ROOT/replay_buffers/taskA_prototype \
  --overwrite True
```

> `--match_budget_buffer_dir` 指向原型回放缓冲目录，自动匹配相同的样本数预算，保证公平对比。

---

### 阶段二：训练 Task B

#### 分支①：普通 LoRA 顺序微调 Task B

```bash
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path $LOGS_ROOT/branch1_stage1/<run_id>--4000_chkpt \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $LOGS_ROOT/branch1_stage2 \
  --use_lora \
  --lora_rank 16 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --max_steps 4000 \
  --save_freq 1000 \
  --resume \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note "branch1_taskB"
```

> 注意：`--vla_path` 需指向前一阶段的 checkpoint 目录（含 `config.json`）。`--resume` 会自动加载该目录下的 LoRA adapter 权重。

#### 分支②：CL-LoRA + no replay 训练 Task B

```bash
TASK_A_CKPT="$LOGS_ROOT/branch234_stage1/<run_id>--4000_chkpt"

torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $LOGS_ROOT/branch2_stage2 \
  --use_cl_lora \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init \
  --freeze_a \
  --use_block_scale \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --max_steps 4000 \
  --save_freq 1000 \
  --use_kd=False \
  --use_replay=False \
  --stage 2 \
  --previous_checkpoint_dir $TASK_A_CKPT \
  --previous_checkpoint_step 4000 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note "branch2_taskB"
```

#### 分支③：CL-LoRA + Prototype Replay 训练 Task B

```bash
TASK_A_CKPT="$LOGS_ROOT/branch234_stage1/<run_id>--4000_chkpt"
PROTO_BUFFER="$LOGS_ROOT/replay_buffers/taskA_prototype"

torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH \
  --data_root_dir $DATA_ROOT \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $LOGS_ROOT/branch3_stage2 \
  --use_cl_lora \
  --lora_rank 32 \
  --shared_depth 16 \
  --orthogonal_init \
  --freeze_a \
  --use_block_scale \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --use_l1_regression \
  --max_steps 4000 \
  --save_freq 1000 \
  --use_kd \
  --lambda_kd 1.0 \
  --teacher_checkpoint_dir $TASK_A_CKPT \
  --teacher_checkpoint_step 4000 \
  --use_replay \
  --replay_buffer_dirs $PROTO_BUFFER \
  --replay_loss_weight 1.0 \
  --replay_every_n_steps 1 \
  --stage 2 \
  --previous_checkpoint_dir $TASK_A_CKPT \
  --previous_checkpoint_step 4000 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note "branch3_taskB"
```

#### 分支④：CL-LoRA + Uniform Replay 训练 Task B

```bash
TASK_A_CKPT="$LOGS_ROOT/branch234_stage1/<run_id>--4000_chkpt"
UNIFORM_BUFFER="$LOGS_ROOT/replay_buffers/taskA_uniform"

# 参数同分支③，仅 replay_buffer_dirs 指向 uniform buffer
torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  ... \
  --use_replay \
  --replay_buffer_dirs $UNIFORM_BUFFER \
  ...
```

---

## 多阶段扩展（三阶段及以上）

对于三阶段（Task A → Task B → Task C）：

1. **阶段二训练完成后**：对 Task B 也构建回放缓冲
2. **阶段三训练时**：`--replay_buffer_dirs` 传入多个目录（Task A buffer + Task B buffer）
3. **Teacher checkpoint**：始终指向前一阶段的 checkpoint（即阶段二的 checkpoint）

```bash
# 阶段三示例
torchrun ... train_cl_lora.py \
  --stage 3 \
  --previous_checkpoint_dir $TASK_B_CKPT \
  --teacher_checkpoint_dir $TASK_B_CKPT \
  --replay_buffer_dirs $TASK_A_BUFFER $TASK_B_BUFFER \
  ...
```

多个 replay buffer 的采样策略通过 `--replay_sample_strategy` 控制：
- `round_robin`（默认）：轮流从每个 buffer 采样
- `random`：随机选择 buffer

---

## 评估

### LIBERO 仿真评估

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $LOGS_ROOT/branch3_stage2/<run_id>--4000_chkpt \
  --task_suite_name libero_spatial \
  --use_cl_lora
```

关键参数：
- `--pretrained_checkpoint`: 训练保存的 checkpoint 目录（含 `config.json` 和 `cl_lora_adapter.pt`）
- `--task_suite_name`: `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90`
- `--use_cl_lora`: 若使用 CL-LoRA 训练则必须加此参数
- `--num_trials_per_task`: 每个任务的评估次数（默认 20）

### 性能矩阵记录

每阶段训练结束后，对**所有已学任务**进行评估，记录成功率矩阵：

```
               Task A    Task B    Task C
Stage 1 (train A)   0.90      -         -
Stage 2 (train B)   0.XX     0.52       -
Stage 3 (train C)   0.XX     0.XX     0.XX
```

基于此矩阵计算 BWT（Backward Transfer）：
```
BWT = mean(旧任务最终成功率 - 旧任务初始成功率)
```

---

## 关键超参数选择指南

### LoRA / CL-LoRA

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `lora_rank` | 标准 LoRA: 16, CL-LoRA: 32 | CL-LoRA 需要更大秩以容纳多任务知识 |
| `shared_depth` | 16 (LLaMA-2 32 层的一半) | 共享层越多，旧知识保护越强，但新任务可塑性降低 |
| `orthogonal_init` | True | 正交初始化使不同任务 LoRA 子空间尽量正交 |
| `freeze_a` | True | 冻结共享层 A 矩阵是抗遗忘的核心 |
| `use_block_scale` | True | 让每层自动调节新旧任务平衡 |

### 训练

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `max_steps` | 单任务 2000-4000 | LIBERO 单任务收敛较快 |
| `batch_size` | 8 (单卡 RTX 4090/5090) | 可随显存增大 |
| `learning_rate` | 5e-4 | CL-LoRA 同样适用此学习率 |
| `image_aug` | True | 强烈建议开启 |

### KD 与 Replay

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `lambda_kd` | 1.0 | KD 损失权重，与 task loss 等权 |
| `replay_loss_weight` | 1.0 | 回放损失权重，与 task loss 等权 |
| `replay_every_n_steps` | 1 | 每步都进行回放（间隔 > 1 可加速但降低回放效果） |
| `replay_batch_size` | 同 `batch_size` 或更小 | 回放 batch 与当前任务 batch 独立控制 |

### 回放缓冲构建

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `top_k_per_segment` | 2 | 每段保留的帧数，控制压缩率 |
| `translation_threshold_m` | 0.03 | 平动切分阈值（米） |
| `rotation_threshold_rad` | 0.05 | 转动切分阈值（弧度） |
| `gripper_threshold` | 0.1 | 夹爪变化阈值 |
| `max_episodes` | None（使用全部 episode） | 限制 episode 数以控制 few-shot 程度 |

---

## 文件清单

### 新建文件

| 文件 | 说明 |
|---|---|
| `vla-scripts/train_cl_lora.py` | CL-LoRA 持续学习训练脚本（分支②③④统一入口） |
| `vla-scripts/build_uniform_replay_buffer.py` | 均匀回放缓冲构建器（分支④） |

### 修改文件

| 文件 | 改动 |
|---|---|
| `vla-scripts/cl_lora.py` | `CLLoRALinear` 和 `inject_cl_lora_into_model` 支持可配置的 `orthogonal_init`、`freeze_a`、`use_block_scale` 参数 |
| `vla-scripts/finetune.py` | `FinetuneConfig` 新增 `use_cl_lora`、`shared_depth`、`orthogonal_init`、`freeze_a`、`use_block_scale`、`clip_weight` 参数；模型初始化时支持 CL-LoRA 注入路径 |

### 未修改文件

- `vla-scripts/build_replay_buffer_openvla.py` — 原型回放缓冲构建器（无需改动）
- `vla-scripts/replay_dataset.py` — 回放数据加载器（兼容两种 buffer 格式）
- `vla-scripts/deploy.py` — ALOHA 部署服务器
- `vla-scripts/merge_lora_weights_and_save.py` — LoRA 权重合并
- `experiments/robot/libero/run_libero_eval.py` — LIBERO 评估
