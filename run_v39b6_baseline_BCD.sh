#!/bin/bash
# =============================================================================
# run_v39b6_baseline_BCD.sh — 无回放基线 v6 (b3 配置 + 加强 FiLM 漂移, 找中间态)
#
# 目标: b3 (FiLM lr×0.2 + A冻结 + proprio修复) = A 0.26 / B 0.88 / C 0.72 / D 0.82
#       b4 (FiLM 冻结 + A漂移)                = A 0.00 / B 0.57 / C 0.00 / D 0.85
#       v6 = b3 的唯一改动: 提高 FILM_LR_SCALE (默认 0.5, 可调), 让 FiLM 漂移更强,
#       把 retention 从 b3 水平连续压低, 目标落在 b3 与 b4 之间的"均匀中间态"
#       (如 A~0.1 / C~0.3-0.5), 供"低残留无回放基线"叙事使用。
#
# 前置诊断(建议先跑): 量化 b3 的 A-bank FiLM vs D-FiLM 相对漂移, 依据幅度选 FILM_LR_SCALE
#
# 用法（tmux 里前台跑, 训完自动评估）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB6
#   bash run_v39b6_baseline_BCD.sh 2>&1 | tee train_v39b6_baseline.log
#   # 调漂移强度: FILM_LR_SCALE=0.8 bash run_v39b6_baseline_BCD.sh
#
# 产物: $LOGS_ROOT/rt_v39b6_taskB/C/D--40000_chkpt
# 结果: train_v39b6_eval.log 末尾汇总
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage 1 最终 checkpoint (Stage 2 的起点)

GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

# 关键旋钮: FiLM 漂移强度 (b3=0.2 → A0.26/C0.72; 越大漂移越强 → retention 越低)
FILM_LR_SCALE="${FILM_LR_SCALE:-0.5}"

check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_A"; exit 1; }
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"
echo "[OK] v6 配置 = b3 配置(漂移FiLM+A冻结+proprio修复+block_scale冻结) 且 FILM_LR_SCALE=$FILM_LR_SCALE"
echo "     (b3=0.2 参考: A0.26/B0.88/C0.72/D0.82; b4=冻结+A漂移: A0/B0.57/C0/D0.85)"
echo "============ 开始 无回放基线v6 Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

COMMON_ARGS=(--batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True
  --use_kd False --use_replay False --image_aug True
  --use_proprio True --use_film True --num_images_in_input 3
  --freeze_film_stage2 False --film_lr_scale "$FILM_LR_SCALE")

run_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_checkpoint_dir  $5=prev_step
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5
    echo ""
    echo "############ Stage $stage : $ds (from $prev_dir) ############"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" \
        --previous_checkpoint_step "$prev_step" \
        "${COMMON_ARGS[@]}"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

run_stage 2 aloha_grab_roller_clean rt_v39b6_taskB "$CKPT_A" 30000
run_stage 3 aloha_stack_bowls_two_clean rt_v39b6_taskC "$LOGS_ROOT/rt_v39b6_taskB--40000_chkpt" 40000
run_stage 4 aloha_open_laptop_clean rt_v39b6_taskD "$LOGS_ROOT/rt_v39b6_taskC--40000_chkpt" 40000

echo ""
echo "==== 无回放基线v6 全部完成 (FILM_LR_SCALE=$FILM_LR_SCALE) ===="

# ---------- 自动全任务评估 (γ=0) ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    CKPT_FINAL="$LOGS_ROOT/rt_v39b6_taskD--40000_chkpt"
    EVAL_LOG="$(dirname "$TRAIN_DIR")/train_v39b6_eval.log"
    echo ""
    echo "==== 自动开始全任务评估 (γ=0, ${EVAL_GPUS:-4,4,5,5,6,6,7,7}) ===="
    FILM_GAMMA=0 bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh \
        "$CKPT_FINAL" "${EVAL_GPUS:-4,4,5,5,6,6,7,7}" "${EVAL_EPISODES:-50}" v39b6D A B C D \
        2>&1 | tee "$EVAL_LOG" | grep -v "svulkan2.*error"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "[WARN] 自动评估异常 (exit=$rc), 训练产物完好, 可手动重跑"
    else
        echo ""
        echo "==== 自动评估完成 ===="
        grep "Merged success rate" "$EVAL_LOG"
        echo ""
        echo "对照: b3(0.2)=0.26/0.88/0.72/0.82 | b4(冻结+A漂移)=0/0.57/0/0.85"
    fi
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi
