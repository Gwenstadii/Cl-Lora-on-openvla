#!/bin/bash
# =============================================================================
# run_v39_taskB.sh — V39 Task B (grab_roller) Stage 2 训练启动脚本
#
# 用法（服务器上, tmux 里前台跑, 实时看输出）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   bash run_v39_taskB.sh 2>&1 | tee train_v39_taskB.log
#
# 前置: VLA_PATH / LOGS_ROOT 已由 server_env.sh export（脚本会自检）
# 配置: 与 Task A 完全一致 (first_lora_layer=16, shared_depth=8, rank=16)
#       40000 步, 每 10000 步保存; 从 rt_v39_taskA--30000_chkpt 续训
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"          # 训练代码目录（prismatic editable 安装的这份）
CKPT_DIR="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"        # Task A 最终 checkpoint
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"    # 任务数据根目录（只做提示）

echo "================ 前置检查 ================"
[ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
[ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
[ -f "$VLA_PATH/config.json" ]  || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
[ -d "$CKPT_DIR" ]              || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_DIR"; exit 1; }
ls "$CKPT_DIR"/cl_lora_adapter.pt "$CKPT_DIR"/cl_lora_config.json >/dev/null 2>&1 \
    || { echo "[FAIL] Task A checkpoint 缺少 cl_lora_adapter.pt / cl_lora_config.json"; exit 1; }
ls -d "$DATA_DIR"/*grab_roller* >/dev/null 2>&1 \
    || echo "[WARN] 没在 $DATA_DIR 下找到 grab_roller 数据目录, 请确认 RLDS 数据实际位置"
echo "[OK] VLA_PATH      = $VLA_PATH"
echo "[OK] LOGS_ROOT     = $LOGS_ROOT"
echo "[OK] previous_ckpt = $CKPT_DIR"
echo "================ 启动训练 ================"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }
echo "[INFO] cwd = $(pwd)"

# 参数与 Task A 完全一致, 仅换 dataset / 步数 / 保存频率 / Stage 2 续训
exec env CUDA_VISIBLE_DEVICES=4,5 PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
torchrun --standalone --nproc_per_node 2 vla-scripts/train_cl_lora.py \
  --run_root_dir "$LOGS_ROOT" --run_id_override "rt_v39_taskB" \
  --max_steps 40000 --save_freq 10000 \
  --vla_path "$VLA_PATH" \
  --dataset_name aloha_grab_roller_clean \
  --batch_size 1 --grad_accumulation_steps 4 --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 \
  --shared_depth 8 --first_lora_layer 16 \
  --orthogonal_init True --freeze_a True --use_block_scale True \
  --freeze_specific_a True \
  --use_kd False --use_replay False --stage 2 --image_aug True \
  --use_proprio True --use_film True --num_images_in_input 3 \
  --previous_checkpoint_dir "$CKPT_DIR" \
  --previous_checkpoint_step 30000
