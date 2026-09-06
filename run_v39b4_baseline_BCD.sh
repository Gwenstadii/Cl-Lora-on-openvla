#!/bin/bash
# =============================================================================
# run_v39b4_baseline_BCD.sh — 无回放基线 v4 (freeze_specific_a=False 制造遗忘源)
#
# 背景: v39b3 (proprio 修复 + FiLM 冻结 + freeze_specific_a=True 无回放)
#   = A=0.26, B=0.88, C=0.72, D=0.82 —— B/C retention 过高。
#   v4: FiLM 保持冻结 + proprio 修复, 唯一改动 = freeze_specific_a=False
#   (特定层 A 随新任务漂移 = 单一可控遗忘源, LIBERO V39 实证过该机制)。
#   预期: A(三轮漂移)~0-0.2, B(两轮)~0.3-0.6, C(一轮)~0.4-0.7, D~0.8
#   —— 形成"低残留无回放基线", 给回放支线留叙事空间。
#
# 注意对比口径: 若基线用 False, 回放支线也应 False 同口径 (LIBERO V39 做法)。
#
# 用法（tmux 里前台跑, 训完自动评估）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB4
#   bash run_v39b4_baseline_BCD.sh 2>&1 | tee train_v39b4_baseline.log
#
# 产物: $LOGS_ROOT/rt_v39b4_taskB/C/D--40000_chkpt
# 结果: eval_result/v39b4D_summary.txt (自动评估汇总)
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage 1 最终 checkpoint (Stage 2 的起点)

GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"                     # 每卡 batch
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))          # 有效 batch = 8 不变

# ---------- 前置检查 ----------
check_env() {
    [ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 请先: source server_env.sh"; exit 1; }
    [ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
}

echo "================ 前置检查 ================"
check_env
[ -d "$CKPT_A" ] || { echo "[FAIL] Task A checkpoint 不存在: $CKPT_A"; exit 1; }
echo "[OK] VLA_PATH  = $VLA_PATH"
echo "[OK] LOGS_ROOT = $LOGS_ROOT"
echo "[OK] Stage1 ckpt = $CKPT_A"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"
echo "[OK] 遗忘源配置: FiLM 冻结 + freeze_specific_a=False (特定层A漂移) + proprio 修复"
echo "============ 开始 无回放基线v4 Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

COMMON_ARGS=(--batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a False
  --use_kd False --use_replay False --image_aug True
  --use_proprio True --use_film True --num_images_in_input 3
  --freeze_film_stage2 True)

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

# Stage 2: Task B (特定 A 开始漂移, proprio 从 A 加载)
run_stage 2 aloha_grab_roller_clean rt_v39b4_taskB "$CKPT_A" 30000

# Stage 3: Task C
run_stage 3 aloha_stack_bowls_two_clean rt_v39b4_taskC "$LOGS_ROOT/rt_v39b4_taskB--40000_chkpt" 40000

# Stage 4: Task D
run_stage 4 aloha_open_laptop_clean rt_v39b4_taskD "$LOGS_ROOT/rt_v39b4_taskC--40000_chkpt" 40000

echo ""
echo "==== 无回放基线v4(freeze_specific_a=False) 全部完成 ===="
echo "    B: $LOGS_ROOT/rt_v39b4_taskB--40000_chkpt"
echo "    C: $LOGS_ROOT/rt_v39b4_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39b4_taskD--40000_chkpt"

# ---------- 训练完成, 自动全任务评估 (γ=0 纯口径) ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    CKPT_FINAL="$LOGS_ROOT/rt_v39b4_taskD--40000_chkpt"
    EVAL_LOG="$(dirname "$TRAIN_DIR")/train_v39b4_eval.log"
    echo ""
    echo "==== 训练完成, 自动开始全任务评估 (γ=0, ${EVAL_GPUS:-4,4,5,5,6,6,7,7}) ===="
    FILM_GAMMA=0 bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh \
        "$CKPT_FINAL" "${EVAL_GPUS:-4,4,5,5,6,6,7,7}" "${EVAL_EPISODES:-50}" v39b4D A B C D \
        2>&1 | tee "$EVAL_LOG" | grep -v "svulkan2.*error"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo ""
        echo "[WARN] 自动评估异常 (exit=$rc), 训练产物完好, 可手动重跑评估"
    else
        echo ""
        echo "==== 自动评估完成, 汇总: eval_result/v39b4D_summary.txt ===="
        grep "Merged success rate" "$EVAL_LOG"
    fi
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi
