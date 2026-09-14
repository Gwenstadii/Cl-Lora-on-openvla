#!/usr/bin/env bash
# =============================================================================
# run_v39v_prototype_replay.sh
#   原型回放(prototype) + 全程冻结 FiLM + proprio 修复 + 无 KD（纯回放）
#
# 唯一旋钮: FREEZE_SPECIFIC_A
#   True  -> 支线 v39v  (specific-A 冻结) = 方法本体定稿配置（此前从未跑过）
#   False -> 支线 v39r3 (specific-A 漂移) = "回放能否替代 A 冻结"（此前从未跑过）
#
# 为什么补这两条（缺口核对结论）:
#   · 所有回放支线(r/r2/r2b/r2c/r2d/u)都是 freeze_specific_a=True
#   · 唯一 freeze_specific_a=False 的支线是 b4，而 b4 无回放
#   · v39r2d 的 --freeze_film_stage2 True 只作用于它重训的 stage D，
#     B/C bank 继承自 v39r2 链（漂移 FiLM + proprio bug 期训练）→ 标签是"仅 D 冻 FiLM"
#   ⇒ "冻结 FiLM + A 漂移 + 回放" 与 "冻结 FiLM + A 冻结 + 原型回放" 都是空白格
#
# 干净单变量对照:
#   v39r3 vs v39b4  : 唯一差异 = 加原型回放（A 漂移 + 冻结 FiLM + 无 KD 两侧一致）
#   v39r3 vs v39v   : 唯一差异 = freeze_specific_a
#   v39v  vs v39u   : 唯一差异 = 回放选帧规则（prototype vs uniform；均 A 冻结 + 无 KD）
#
# 机制提示（cl_lora.py）: stage2+ 只 reinit specific-B + block_scale，**specific-A 不重置**；
#   而 task bank 只存 specific-B + block_scale（不存 A）→ A 漂移后，旧任务恢复的 B_K
#   与 A_final 配对错位。回放的梯度会流进 A，因此本实验就是在问:
#   "回放能否把 A 拉回到旧任务兼容的值" → 能则 A 冻结非必要，不能则 A 冻结是硬前提。
#
# 流程: A 复用 → Stage2(B) → B 自评门禁 → Stage3(C) → Stage4(D) → 全任务评估(A/B/C/D, 50ep, γ=0)
#
# 用法（tmux 前台）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainV
#   bash run_v39v_prototype_replay.sh 2>&1 | tee train_v39v.log                 # 方法本体
#   FREEZE_SPECIFIC_A=False bash run_v39v_prototype_replay.sh 2>&1 | tee train_v39r3.log
#
# 产物: $LOGS_ROOT/rt_<v39v|v39r3>_task{B,C,D}--40000_chkpt
#       RoboTwin-main/eval_result/<v39v|v39r3>D_summary.txt
# =============================================================================

set -u

# 未 source server_env.sh 时给出人话报错（而不是 unbound variable）
: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
EVAL_SEQ="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
EVAL_ONE="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"

CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # Stage1 产物（两条线共用，无需重训）
BUF_P="$LOGS_ROOT/replay_buffers"                 # 原型 buffer（与 v39r2d 同源，无需重建）

GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.80}"          # B 自评门禁
FORCE="${FORCE:-0}"                               # 1 = B 未达标也继续
FILM_GAMMA_EVAL="${FILM_GAMMA_EVAL:-0}"           # 与 v39u/b4 口径一致（冻结 FiLM 下 γ 无实质影响）

# 方法定义: 纯回放, 无蒸馏
USE_KD="${USE_KD:-False}"
FREEZE_SPECIFIC_A="${FREEZE_SPECIFIC_A:-True}"

if [ "$FREEZE_SPECIFIC_A" = "True" ]; then
    PREFIX="v39v";  ARM_DESC="specific-A 冻结（= 方法本体定稿配置）"
else
    PREFIX="v39r3"; ARM_DESC="specific-A 漂移（= 回放能否替代 A 冻结）"
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

# ---------- 前置检查 ----------
echo "================ 前置检查 ================"
[ -n "${VLA_PATH:-}" ]  || { echo "[FAIL] VLA_PATH 未设置 —— 先 source server_env.sh"; exit 1; }
[ -n "${LOGS_ROOT:-}" ] || { echo "[FAIL] LOGS_ROOT 未设置 —— 先 source server_env.sh"; exit 1; }
[ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
[ -d "$CKPT_A" ] || { echo "[FAIL] A ckpt 不存在: $CKPT_A"; exit 1; }
for t in taskA taskB taskC; do
    [ -f "$BUF_P/$t/manifest.jsonl" ] || { echo "[FAIL] 原型 buffer 缺失: $BUF_P/$t"; exit 1; }
done

echo "[OK] 支线 = $PREFIX | $ARM_DESC"
echo "[OK] 配置: 冻结 FiLM(stage≥2, 全程) + freeze_specific_a=$FREEZE_SPECIFIC_A + proprio 修复"
echo "[OK]       + 原型回放 every4/w0.5 + USE_KD=$USE_KD + block_scale 冻结 + shared A/B 冻结"
echo "[OK] A ckpt = $CKPT_A"
echo "[OK] buffers = $BUF_P/task{A,B,C} ($(wc -l < "$BUF_P/taskA/manifest.jsonl") / $(wc -l < "$BUF_P/taskB/manifest.jsonl") / $(wc -l < "$BUF_P/taskC/manifest.jsonl") samples)"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch=$BATCH_SIZE | accum=$GRAD_ACCUM (有效 batch=8)"
echo "============ $PREFIX: Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

# ---------- Stage 训练 ----------
run_stage() {  # $1=stage $2=dataset $3=run_id $4=prev_dir $5=prev_step $6=teacher_dir $7=teacher_step $8..=buffers
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5 tdir=$6 tstep=$7
    shift 7
    local buffers_csv; buffers_csv=$(IFS=,; echo "$*")
    if [ -d "$LOGS_ROOT/$rid--40000_chkpt" ]; then
        echo "[SKIP] $rid--40000_chkpt 已存在"
        return 0
    fi
    echo ""
    echo "############ Stage $stage : $ds ############"
    echo "    prev/teacher: $prev_dir @$prev_step"
    echo "    buffers: $buffers_csv"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" --previous_checkpoint_step "$prev_step" \
        --teacher_checkpoint_dir "$tdir" --teacher_checkpoint_step "$tstep" \
        --replay_buffer_dirs "$buffers_csv" \
        --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a "$FREEZE_SPECIFIC_A" \
        --use_kd "$USE_KD" --use_replay True --freeze_film_stage2 True \
        --replay_every_n_steps 4 --replay_loss_weight 0.5 --lambda_kd 0.2 \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    local rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] Stage $stage 训练失败 (rc=$rc)"; exit 1; }
    echo "[OK] Stage $stage 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

CKPT_B="$LOGS_ROOT/rt_${PREFIX}_taskB--40000_chkpt"
CKPT_C="$LOGS_ROOT/rt_${PREFIX}_taskC--40000_chkpt"
CKPT_D="$LOGS_ROOT/rt_${PREFIX}_taskD--40000_chkpt"

# Stage 2 (B): buffers A
run_stage 2 aloha_grab_roller_clean "rt_${PREFIX}_taskB" \
    "$CKPT_A" 30000 "$CKPT_A" 30000 "$BUF_P/taskA"

# ---------- B 自评门禁 ----------
echo ""
echo "==== B 自评 (20ep, 8 worker) —— 门禁 ≥ $PASS_THRESHOLD ===="
bash "$EVAL_ONE" grab_roller demo_clean "$CKPT_B" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 "${PREFIX}_B_self" 0 \
    2>&1 | tee "/tmp/${PREFIX}_B.log" | grep -v "svulkan2.*error"
B_RATE=$(grep "Merged success rate" "/tmp/${PREFIX}_B.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
echo "B 自评: ${B_RATE:-异常}  (对照 v39r2d B=0.88)"
if ! python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "[WARN] B 自评 < $PASS_THRESHOLD"
    if [ "$FORCE" != "1" ]; then
        echo "[STOP] 已停在 Stage 2 之后（想继续: FORCE=1 重跑同一命令，会跳过已完成的 B 训练）"
        echo "       排查: /tmp/${PREFIX}_B.log 与 $LOGS_ROOT/rt_${PREFIX}_taskB--40000_chkpt 训练日志"
        exit 2
    fi
    echo "[FORCE=1] 继续训练 C/D"
fi

# Stage 3 (C): buffers A+B
run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" \
    "$CKPT_B" 40000 "$CKPT_B" 40000 "$BUF_P/taskA" "$BUF_P/taskB"

# Stage 4 (D): buffers 加权 A×2 + B×1 + C×3（与 v39r2d/v39u 一致）
run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" \
    "$CKPT_C" 40000 "$CKPT_C" 40000 \
    "$BUF_P/taskA" "$BUF_P/taskA" "$BUF_P/taskB" "$BUF_P/taskC" "$BUF_P/taskC" "$BUF_P/taskC"

echo ""
echo "==== $PREFIX 三个阶段训练完成 ===="

# ---------- 全任务评估 ----------
if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    echo "==== 全任务评估 (A/B/C/D, 50ep, γ=$FILM_GAMMA_EVAL) ===="
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" \
        "$CKPT_D" "$EVAL_GPUS" 50 "${PREFIX}D" A B C D 2>&1 | grep -v "svulkan2.*error"
    SUM="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PREFIX}D_summary.txt"
    echo ""
    echo "================ $PREFIX (freeze_specific_a=$FREEZE_SPECIFIC_A) 结果 ================"
    [ -f "$SUM" ] && cat "$SUM" || echo "[WARN] 未找到汇总: $SUM"
    echo ""
    echo "对照: b4 (A漂移+无回放)=0-0.57-0-0.85 | v39r2d (脏版原型回放+KD)=0.62-0.88-0.88-0.80"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi
