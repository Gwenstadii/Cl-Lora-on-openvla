#!/bin/bash
# =============================================================================
# run_v39r2b_stageD.sh — v39r2 修复版: 只重训 Stage 4 (D) + C buffer 加权
#
# 背景: v39r2 全任务评估 A=0.69 / B≈0.8 / C=0 / D=0.75。
#   C 自评(γ=0)高 → C 学好了, 崩在 D 阶段: 3 buffer round-robin 下 C 每 12 步
#   才复习 1 次, FiLM 漂移约束剂量不足 → C 的 bank 与 D 的 FiLM 错位归零。
#   修复: 只重训 D, replay buffers 加权为 A×2 + B×1 + C×3 (6 条目轮询,
#   C 复习频率 1/12 → 1/8), replay 总次数不变 → D 学习干扰不变。
#
# 提速: batch_size=2 × grad_accum=1 × 4卡 = 有效 batch 8 (口径不变), 吞吐+30~60%
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainD2
#   bash run_v39r2b_stageD.sh 2>&1 | tee train_v39r2b_D.log
#
# 产物: $LOGS_ROOT/rt_v39r2b_taskD--40000_chkpt
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_C="$LOGS_ROOT/rt_v39r2_taskC--40000_chkpt"       # 回放版 C (D 的起点 + teacher)
BUFFER_ROOT="$LOGS_ROOT/replay_buffers"

GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"                          # 每卡 batch (显存富余, 默认 2)
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))               # 有效 batch = 8 不变

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_C" ] || { echo "[FAIL] 回放版 C checkpoint 不存在: $CKPT_C"; exit 1; }
for d in taskA taskB taskC; do
    [ -f "$BUFFER_ROOT/$d/manifest.jsonl" ] || { echo "[FAIL] buffer 缺失: $BUFFER_ROOT/$d"; exit 1; }
done
echo "[OK] Stage3 ckpt = $CKPT_C (起点 + teacher)"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"
echo "[OK] buffers 加权: A×2 + B×1 + C×3 (6 条目轮询, C 复习频率 1/12→1/8)"
echo "============ 开始 Stage 4 (D) 加权回放重训 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# C buffer 写 3 份、A 写 2 份: 旧任务里 C 最需要 FiLM 约束
BUFFERS_CSV="$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskB,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC"

env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
  --run_root_dir "$LOGS_ROOT" --run_id_override rt_v39r2b_taskD \
  --max_steps 40000 --save_freq 10000 \
  --vla_path "$VLA_PATH" \
  --dataset_name aloha_open_laptop_clean \
  --stage 4 \
  --previous_checkpoint_dir "$CKPT_C" \
  --previous_checkpoint_step 40000 \
  --teacher_checkpoint_dir "$CKPT_C" --teacher_checkpoint_step 40000 \
  --replay_buffer_dirs "$BUFFERS_CSV" \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
  --lr_warmup_steps 200 --num_steps_before_decay 100000 \
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True \
  --use_kd True --use_replay True --freeze_film_stage2 False \
  --replay_every_n_steps 4 --replay_loss_weight 0.5 --lambda_kd 0.2 \
  --image_aug True --use_proprio True --use_film True --num_images_in_input 3

rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] Stage 4 训练失败 (exit=$rc)"
    exit $rc
fi
echo "[OK] Stage 4 完成 -> $LOGS_ROOT/rt_v39r2b_taskD--40000_chkpt"
echo "    之后评估: FILM_GAMMA=0 bash RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
echo "      \$LOGS_ROOT/rt_v39r2b_taskD--40000_chkpt 4,4,5,5,6,6,7,7 50 v39r2bD A B C D"
