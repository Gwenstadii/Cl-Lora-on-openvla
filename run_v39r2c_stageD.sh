#!/bin/bash
# =============================================================================
# run_v39r2c_stageD.sh — 方案 B: 只重训 Stage 4 (D) + D 阶段冻结 FiLM
#
# 背景: v39r2b (加权重训D) 结果 A=0.54 / B=0.88 / C=0 / D=0.80 —— C 依然 0。
#   机制结论: C 对 FiLM 漂移零容忍 (γ=0.95 都崩), 而回放对 FiLM 只是弱约束
#   (D 阶段 replay 梯度占比 ~6%), 跨不过 C 的阈值 —— "漂移FiLM+回放"框架内
#   C 救不回来。方案 B: D 阶段冻结 FiLM (freeze_film_stage2=True),
#   D 的 FiLM 恒等于 C-FiLM → C 评估组合完全匹配 → C 应恢复 0.6-0.9。
#
# 对照干净性: 与 run_v39r2b_stageD.sh 唯一差异 = --freeze_film_stage2 True
#   (buffers 加权 A×2+B×1+C×3 保持, 起点/teacher/步数/权重全部相同)
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainD3
#   bash run_v39r2c_stageD.sh 2>&1 | tee train_v39r2c_D.log
#
# 产物: $LOGS_ROOT/rt_v39r2c_taskD--40000_chkpt
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
echo "[OK] 方案B: D 阶段冻结 FiLM (freeze_film_stage2=True) —— C 的 FiLM 与 D 训练时一致"
echo "[OK] buffers 加权: A×2 + B×1 + C×3 (与 v39r2b 相同, 唯一变量=冻结)"
echo "============ 开始 Stage 4 (D) 冻结FiLM 重训 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

BUFFERS_CSV="$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskB,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC"

env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
  --run_root_dir "$LOGS_ROOT" --run_id_override rt_v39r2c_taskD \
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
  --use_kd True --use_replay True --freeze_film_stage2 True \
  --replay_every_n_steps 4 --replay_loss_weight 0.5 --lambda_kd 0.2 \
  --image_aug True --use_proprio True --use_film True --num_images_in_input 3

rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] Stage 4 训练失败 (exit=$rc)"
    exit $rc
fi
echo "[OK] Stage 4 完成 -> $LOGS_ROOT/rt_v39r2c_taskD--40000_chkpt"
echo "    之后评估: FILM_GAMMA=0 bash RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
echo "      \$LOGS_ROOT/rt_v39r2c_taskD--40000_chkpt 4,4,5,5,6,6,7,7 50 v39r2cD A B C D"
