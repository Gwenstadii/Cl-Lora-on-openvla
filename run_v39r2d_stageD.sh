#!/bin/bash
# =============================================================================
# run_v39r2d_stageD.sh — v39r2d: 只重训 Stage 4 (D) + 冻结 FiLM + proprio 加载修复
#
# 修复 (train_cl_lora.py): proprio_projector stage 2+ 从上一阶段加载,
#   不再每次随机初始化 —— 此前每个 stage 的 proprio 投影不同, 旧任务评估时
#   proprio 输入错位 → 依赖 proprio 的任务 (C) 归零 (C bank/FiLM 全一致仍 0
#   的根因)。修复后全任务共享 stage1 的 proprio 投影。
#
# 其余配置与 v39r2c 相同: 冻结 FiLM (freeze_film_stage2=True)、
#   buffers 加权 A×2+B×1+C×3、batch=2×4卡、训完自动评估 (γ=0, 8 worker)。
#
# 用法（tmux 里前台跑）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainD4
#   bash run_v39r2d_stageD.sh 2>&1 | tee train_v39r2d_D.log
#
# 产物: $LOGS_ROOT/rt_v39r2d_taskD--40000_chkpt
# 结果: eval_result/v39r2dD_summary.txt (自动评估汇总)
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
ls "$CKPT_C"/proprio_projector--40000_checkpoint.pt >/dev/null 2>&1 \
    || echo "[WARN] C ckpt 缺 proprio_projector--40000_checkpoint.pt (修复将跳过 proprio 加载)"
for d in taskA taskB taskC; do
    [ -f "$BUFFER_ROOT/$d/manifest.jsonl" ] || { echo "[FAIL] buffer 缺失: $BUFFER_ROOT/$d"; exit 1; }
done
echo "[OK] Stage3 ckpt = $CKPT_C (起点 + teacher + proprio 来源)"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"
echo "[OK] 修复: proprio 从 C ckpt 加载 (不再随机初始化) | FiLM 冻结 | buffers A×2+B×1+C×3"
echo "============ 开始 Stage 4 (D) 修复版重训 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

BUFFERS_CSV="$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskA,$BUFFER_ROOT/taskB,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC,$BUFFER_ROOT/taskC"

env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
  --run_root_dir "$LOGS_ROOT" --run_id_override rt_v39r2d_taskD \
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
echo "[OK] Stage 4 完成 -> $LOGS_ROOT/rt_v39r2d_taskD--40000_chkpt"

# ---------- 训练完成, 自动全任务评估 ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    CKPT_FINAL="$LOGS_ROOT/rt_v39r2d_taskD--40000_chkpt"
    EVAL_LOG="$(dirname "$TRAIN_DIR")/train_v39r2d_eval.log"
    echo ""
    echo "==== 训练完成, 自动开始全任务评估 (γ=0, ${EVAL_GPUS:-4,4,5,5,6,6,7,7}) ===="
    FILM_GAMMA=0 bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh \
        "$CKPT_FINAL" "${EVAL_GPUS:-4,4,5,5,6,6,7,7}" "${EVAL_EPISODES:-50}" v39r2dD A B C D \
        2>&1 | tee "$EVAL_LOG" | grep -v "svulkan2.*error"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo ""
        echo "[WARN] 自动评估异常 (exit=$rc), 训练产物完好, 可手动重跑评估"
    else
        echo ""
        echo "==== 自动评估完成, 汇总: eval_result/v39r2dD_summary.txt ===="
        grep "Merged success rate" "$EVAL_LOG"
    fi
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi
