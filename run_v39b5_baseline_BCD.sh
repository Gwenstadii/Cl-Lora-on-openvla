#!/bin/bash
# =============================================================================
# run_v39b5_baseline_BCD.sh — 无回放基线 v5（v39r2d / v39v 的"同配置去掉回放"臂）
#
# 定位: **全程冻结 FiLM + specific-A 冻结 + proprio 修复 + 无回放**，从 rt_v39_taskA
#   起完整重训 B/C/D —— 即"原型回放那条线（0.62-0.88-0.88-0.80）的配置，去掉回放"。
#
# ⚠️ 与 v39r2d 的差异（必须心里有数，别当成严格单变量）:
#   ① v39r2d 是"**只重训 D 阶段**"（B/C bank 继承自 v39r2 链：漂移 FiLM 1.0× + proprio bug 时代）
#      ⇒ 它的 B/C 不是"冻结 FiLM + A 冻结"训练出来的；本脚本全程按干净配置重训，**更干净**；
#   ② v39r2d 开了 KD λ0.2，本脚本默认 `USE_KD=False`（方法定义=纯回放/无蒸馏口径）。
#      想与 v39r2d 逐项对齐（含 KD）：`USE_KD=True bash run_v39b5_baseline_BCD.sh`
#   ⇒ 真正的干净单变量对是 **b5(本脚本) ↔ v39v**（`run_v39v_prototype_replay.sh`）：
#      两者唯一差异 = 回放开关（冻结 FiLM / A 冻结 / 无 KD / 步数 / 起点全同）。
#
# 预期与判读:
#   · 预警: 冻结 FiLM + 全冻结共享 + bank 精确恢复 ⇒ **无回放也可能很高**
#     （v39f 实证 A/B≈0.7；b3 在漂移 FiLM 下 B/C 已 0.88/0.72）⇒ b5 可能 ≥ v39r2d；
#   · 若 b5 ≈ v39v ⇒ 回放边际价值 ≈ 0，叙事改为"结构隔离为主 + 回放增强 X"；
#   · 若 b5 明显 < v39v ⇒ 回放确有价值，且是干净单变量证据。
#
# 用法（tmux 里前台跑, 训完自动评估）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB5
#   bash run_v39b5_baseline_BCD.sh 2>&1 | tee train_v39b5.log
#
# 产物: $LOGS_ROOT/rt_v39b5_taskB/C/D--40000_chkpt
# 结果: eval_result/v39b5D_summary.txt (自动评估汇总, γ=0 口径)
# =============================================================================

set -u

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
DATA_DIR="/mnt/data/pengshengdi/RoboTwin-main/data"
CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage 1 最终 checkpoint (Stage 2 的起点)

GPUS="${GPUS:-4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"                     # 每卡 batch
USE_KD="${USE_KD:-False}"                         # 与 v39r2d 逐项对齐时设 True（λ=0.2）
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
echo "[OK] 配置: 全程冻结 FiLM + freeze_specific_a=True + block_scale 冻结 + proprio 修复 + 无回放 | USE_KD=$USE_KD"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch_size=$BATCH_SIZE | grad_accum=$GRAD_ACCUM (有效batch=8)"
echo "============ 开始 无回放基线v5 Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

COMMON_ARGS=(--batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4
  --lr_warmup_steps 200 --num_steps_before_decay 100000
  --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16
  --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True
  --use_kd "$USE_KD" --use_replay False --image_aug True
  --use_proprio True --use_film True --num_images_in_input 3
  --freeze_film_stage2 True)

run_stage() {  # $1=stage  $2=dataset  $3=run_id  $4=prev_checkpoint_dir  $5=prev_step
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5
    if [ -d "$LOGS_ROOT/$rid--40000_chkpt" ]; then
        echo "[SKIP] $rid--40000_chkpt 已存在"
        return 0
    fi
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
        --teacher_checkpoint_dir "$prev_dir" \
        --teacher_checkpoint_step "$prev_step" \
        "${COMMON_ARGS[@]}"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] Stage $stage ($ds) 训练失败 (exit=$rc), 终止后续 Stage"
        exit $rc
    fi
    echo "[OK] Stage $stage ($ds) 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

# Stage 2: Task B (冻结 FiLM, proprio 从 A 加载)
run_stage 2 aloha_grab_roller_clean rt_v39b5_taskB "$CKPT_A" 30000

# Stage 3: Task C
run_stage 3 aloha_stack_bowls_two_clean rt_v39b5_taskC "$LOGS_ROOT/rt_v39b5_taskB--40000_chkpt" 40000

# Stage 4: Task D
run_stage 4 aloha_open_laptop_clean rt_v39b5_taskD "$LOGS_ROOT/rt_v39b5_taskC--40000_chkpt" 40000

echo ""
echo "==== 无回放基线v5(v39r2d镜像) 全部完成 ===="
echo "    B: $LOGS_ROOT/rt_v39b5_taskB--40000_chkpt"
echo "    C: $LOGS_ROOT/rt_v39b5_taskC--40000_chkpt"
echo "    D: $LOGS_ROOT/rt_v39b5_taskD--40000_chkpt"

# ---------- 训练完成, 自动全任务评估 (γ=0 纯口径) ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    CKPT_FINAL="$LOGS_ROOT/rt_v39b5_taskD--40000_chkpt"
    EVAL_LOG="$(dirname "$TRAIN_DIR")/train_v39b5_eval.log"
    echo ""
    echo "==== 训练完成, 自动开始全任务评估 (γ=0, ${EVAL_GPUS:-4,4,5,5,6,6,7,7}) ===="
    FILM_GAMMA=0 bash /mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh \
        "$CKPT_FINAL" "${EVAL_GPUS:-4,4,5,5,6,6,7,7}" "${EVAL_EPISODES:-50}" v39b5D A B C D \
        2>&1 | tee "$EVAL_LOG" | grep -v "svulkan2.*error"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo ""
        echo "[WARN] 自动评估异常 (exit=$rc), 训练产物完好, 可手动重跑评估"
    else
        echo ""
        echo "==== 自动评估完成, 汇总: eval_result/v39b5D_summary.txt ===="
        grep "Merged success rate" "$EVAL_LOG"
    fi
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0, 跳过自动评估"
fi
