# CL-LoRA 无回放支线 最终实验报告 (vlast)

## 实验方法

### 核心机制

CL-LoRA 将 LLaMA-2-7B 的 32 层 Decoder 分为共享层和特定层：

- **共享层 (layer 0-7)**：LoRA-A 正交初始化 → Stage 1 学习 → 之后**永久冻结**。提供跨任务稳定的低秩特征空间。
- **特定层 (layer 8-31)**：LoRA-A/B 可训（Stage 1 后 A 冻结）。每任务独立保存 **特定 B + block_scale + 动作头** 到 task bank，评估时按 task ID 恢复。

```
Stage 1 (Task A):  全部可训
                   保存 task_1_bank.pt (特定B + block + action_head)
                   冻结 → shared A, shared B, specific A

Stage 2 (Task B):  加载 Stage 1 → 冻结 → reinit 特定B/block → 仅训 bank 参数
                   保存 task_2_bank.pt

Stage 3 (Task C):  同理 (切换数据集 libero_object)
Stage 4 (Task D):  同理 (切换数据集 libero_goal)

评估: --eval_task_id N → 从 task_N_bank.pt 恢复该任务专属参数
```

### 参数配置

| 参数 | 值 |
|---|---|
| LoRA rank | 16 |
| alpha | 16 (scaling = 1.0) |
| 共享层数 | 8 (layer 0-7) |
| 特定层数 | 24 (layer 8-31) |
| 学习率 | 5e-4 |
| warmup | 200 steps |
| 每阶段步数 | 6000 |
| batch size | 1 × grad_accum 8 = 有效 8 |
| 冻结策略 | shared A/B + specific A (Stage 1 后) |
| bank 内容 | specific B + block_scale + action_head |

### 注入范围

仅 LlamaDecoderLayer 的 attn (q/k/v/o_proj) + ffn (gate/up/down_proj)，共 224 层。视觉骨干 (SigLIP)、lm_head、投影器全部冻结，注入率为 0。

## 与 PI 系列对比

| | PI 系列 (Gemma-2B) | vlast (Llama-7B) | 是否一致 |
|---|---|---|---|
| **总层数** | 18 | 32 | — |
| **共享层** | ~12 (67%) | 8 (25%) | △ |
| **特定层** | ~6 (33%) | 24 (75%) | △ |
| **LoRA rank** | 16/32 | 16 | — |
| **Stage 1 后冻结 shared A/B** | ✓ | ✓ | ✓ |
| **Stage 1 后冻结 specific A** | ✓ | ✓ | ✓ |
| **bank 内容** | B + block + head | B + block + head | ✓ |
| **bank B 矩阵数** | ~42 | **168** | ✗ (4×) |
| **bank 参数量** | ~1.5M | **~14M** | ✗ (9×) |
| **视觉 LoRA** | PaliGemma 冻结 | SigLIP 原本冻结 | ✓ |
| **动作头** | 可训 | 可训 | ✓ |
| **评估方式** | task bank 插拔 | task bank 插拔 | ✓ |

**结论：机制完全对齐，参数量未对齐。** 每个 B 矩阵在 Llama-7B 上的参数量是 Gemma-2B 的 ~3 倍（hidden dim 更大），且特定层数量是 PI 的 4 倍（24 vs 6），银行总参数是 PI 的 9 倍。

## 实验结果

| | Task A | Task B | Task C | Task D |
|---|---|---|---|---|
| **Stage 2 后** | 0.20 | 0.98 | — | — |
| **Stage 3 后** | 0.28 | 0.92 | 1.00 | — |
| **Stage 4 后** | 0.22 | 0.92 | 0.86 | 0.96 |

**关键发现：**
- Task A：retention 达到预期 (20-28%)，Stage 1 无冻结保护 → 后续跨域后自然遗忘
- Task B/C/D：retention 远超预期 (86-92%)，受益于 Stage 1 后的冻结保护，bank 参数足够独立恢复
- 新任务 (C/D)：学习能力正常 (86-96%)

## 未达到预期的原因分析

PI 系列预期旧任务 retention 在 5% 左右，vlast 中 B/C/D 达到 86-96%。核心原因：

1. **bank 参数容量过大**：每个 B 矩阵在 Llama-7B 上约 100K 参数，168 个矩阵总计 ~14M。PI 仅 ~42 个矩阵、每个 ~30K、总计 ~1.5M。9 倍的 bank 容量意味着每个任务的专属知识被保存得过于完整。

2. **时序不对称**：Task A 在 Stage 1 无冻结保护，与 specific A 强耦合 → 跨域后无法恢复 → retention 低 (22%)。Task B/C/D 在冻结的 specific A 下训练，B 矩阵学会了独立工作 → bank 恢复后几乎无损 (86-96%)。

3. **层数不可调**：调整 shared_depth 无法同时解决"旧任务 retention 过高"和"新任务学习充分"。共享层多 → 可训容量不足，C/D 学不好；共享层少 → bank B 矩阵更多，retention 更高。

4. **模型规模差异**：PI 使用 Gemma-2B (18层)，vlast 使用 Llama-2-7B (32层)。层数 × 维度差异使得相同机制的 bank 在实际信息容量上有数量级差距。

## 训练命令

### 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="/root/autodl-tmp/openvla-oft/LIBERO:/root/autodl-tmp/openvla-oft:$PYTHONPATH"
export VLA_PATH="/root/autodl-tmp/models/openvla-7b"
export DATA_ROOT="/root/autodl-tmp/modified_libero_rlds"
export LOGS_ROOT="/root/autodl-tmp/LOGS-2"
cd /root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft
```

### Stage 1 (Task A)

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_vlast_taskA" \
  --batch_size 1 --grad_accumulation_steps 8 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 1 \
  --max_steps 6000 --save_freq 2000 --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### Stage 2 (Task B)

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH --data_root_dir $DATA_ROOT \
  --dataset_name "libero_spatial_no_noops" \
  --run_id_override "cl_lora_vlast_taskB" \
  --batch_size 1 --grad_accumulation_steps 8 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 2 \
  --previous_checkpoint_dir $LOGS_ROOT/cl_lora_vlast_taskA--6000_chkpt \
  --previous_checkpoint_step 6000 \
  --max_steps 6000 --save_freq 2000 --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### Stage 3 (Task C)

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH --data_root_dir $DATA_ROOT \
  --dataset_name "libero_object_no_noops" \
  --run_id_override "cl_lora_vlast_taskC" \
  --batch_size 1 --grad_accumulation_steps 8 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 3 \
  --previous_checkpoint_dir $LOGS_ROOT/cl_lora_vlast_taskB--6000_chkpt \
  --previous_checkpoint_step 6000 \
  --max_steps 6000 --save_freq 2000 --image_aug True \
  --run_root_dir $LOGS_ROOT
```

### Stage 4 (Task D)

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node 1 vla-scripts/train_cl_lora.py \
  --vla_path $VLA_PATH --data_root_dir $DATA_ROOT \
  --dataset_name "libero_goal_no_noops" \
  --run_id_override "cl_lora_vlast_taskD" \
  --batch_size 1 --grad_accumulation_steps 8 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 4 \
  --previous_checkpoint_dir $LOGS_ROOT/cl_lora_vlast_taskC--6000_chkpt \
  --previous_checkpoint_step 6000 \
  --max_steps 6000 --save_freq 2000 --image_aug True \
  --run_root_dir $LOGS_ROOT
```

## 关键代码文件

| 文件 | 作用 |
|---|---|
| `vla-scripts/cl_lora.py` | CLLoRALinear (前向计算) + inject + freeze_stage1_params + save/load_task_bank |
| `vla-scripts/train_cl_lora.py` | 训练主循环，Stage 1 后冻结 + reinit bank + bank 保存/复制 |
| `experiments/robot/openvla_utils.py` | 评估时 CL-LoRA 注入 + adapter 加载 + bank 恢复 |
| `experiments/robot/libero/run_libero_eval.py` | 评估入口，--eval_task_id 选择 bank |
