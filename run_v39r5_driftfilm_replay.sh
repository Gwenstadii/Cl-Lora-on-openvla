#!/usr/bin/env bash
# =============================================================================
# run_v39r5_driftfilm_replay.sh — b3 配置的"回放孪生臂"（单变量：加原型回放）
#
# 目的: b3（漂移 FiLM lr0.2 + specific-A 冻结 + proprio 修复 + **无回放**）
#       实测 0.26-0.88-0.72-0.82 —— B/C 保留本来就高。
#       ⇒ 要证明"回放有价值"，必须与**同 FiLM 设置**的无回放臂单变量对比，
#         而不是拿 b4（A 漂移）当基线（差 3 个变量，评审一眼看穿）。
#   本脚本 = b3 的配置，**唯一改动 = 打开原型回放**（无 KD，保持方法定义）。
#
# 干净单变量对照:
#   v39r5 vs v39b3  : 唯一差异 = 有无原型回放（FiLM/A/KD/proprio/步数全同）
#   v39r5 vs v39r2  : v39r2 是漂移 FiLM lr1.0 + KD + proprio bug 时代的旧版；v39r5 是
#                     "方法定义口径（无 KD）+ proprio 修复 + FiLM 0.2×" 的干净重跑
#   v39r5 vs v39v   : FiLM 处理不同（漂移 0.2× vs 冻结）→ 用于回答"冻结 FiLM 是否必要"
#
# 判读:
#   · 若 v39r5 明显优于 b3（尤其 A、C）⇒ 回放在"该基线配置"下确有价值 → 可把 b3 当正式基线
#   · 若 v39r5 ≈ b3 ⇒ **CL-LoRA 结构隔离才是主机制，回放边际价值小** → 论文叙事必须调整
#     （别等到投稿前才发现；这正是本次实验要提前排掉的风险）
#
# 流程: 复用 rt_v39_taskA → B → B 自评门禁 → C → D → 全任务评估(γ=0, 与 b3 同口径)
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainR5
#   bash run_v39r5_driftfilm_replay.sh 2>&1 | tee train_v39r5.log
#
# 产物: $LOGS_ROOT/rt_v39r5_task{B,C,D}--40000_chkpt
#       RoboTwin-main/eval_result/v39r5D_summary.txt
# 对照: b3(无回放) = 0.26-0.88-0.72-0.82 | v39r2d(旧版: 冻结FiLM+KD) = 0.62-0.88-0.88-0.80
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
EVAL_SEQ="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
EVAL_ONE="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"

CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"     # 与 b3 同一起点，复用
BUF_P="$LOGS_ROOT/replay_buffers"                 # 原型 buffer（与 v39r2d 同源，复用）

PREFIX="${PREFIX:-v39r5}"                          # 漂移 1.0× 版本: FILM_LR_SCALE=1.0 PREFIX=v39r6
GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.80}"
FORCE="${FORCE:-0}"
FILM_LR_SCALE="${FILM_LR_SCALE:-0.2}"             # 与 b3 相同：漂移力度 0.2×
USE_KD="${USE_KD:-False}"                         # 方法定义 = 纯回放，无蒸馏
FILM_GAMMA_EVAL="${FILM_GAMMA_EVAL:-0}"           # 与 b3 的评估口径一致

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

echo "================ 前置检查 ================"
[ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
[ -d "$CKPT_A" ] || { echo "[FAIL] A ckpt 不存在: $CKPT_A"; exit 1; }
for t in taskA taskB taskC; do
    [ -f "$BUF_P/$t/manifest.jsonl" ] || { echo "[FAIL] 原型 buffer 缺失: $BUF_P/$t"; exit 1; }
done
echo "[OK] 支线 = $PREFIX（= b3 配置 + 原型回放，唯一差异就是这一项）"
echo "[OK] 配置: 漂移 FiLM freeze_film_stage2=False film_lr_scale=$FILM_LR_SCALE"
echo "[OK]       + specific-A 冻结(True) + block_scale 冻结 + proprio 修复"
echo "[OK]       + 原型回放 every4/w0.5 + USE_KD=$USE_KD + 有效 batch=8"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch=$BATCH_SIZE | accum=$GRAD_ACCUM"
echo "============ $PREFIX: Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

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
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True \
        --use_kd "$USE_KD" --use_replay True \
        --freeze_film_stage2 False --film_lr_scale "$FILM_LR_SCALE" \
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
echo "==== B 自评 (20ep) —— 门禁 ≥ $PASS_THRESHOLD（对照 b3 的 B=0.88）===="
bash "$EVAL_ONE" grab_roller demo_clean "$CKPT_B" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 "${PREFIX}_B_self" 0 \
    2>&1 | tee "/tmp/${PREFIX}_B.log" | grep -v "svulkan2.*error"
B_RATE=$(grep "Merged success rate" "/tmp/${PREFIX}_B.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
echo "B 自评: ${B_RATE:-异常}"
if ! python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "[WARN] B 自评 < $PASS_THRESHOLD"
    if [ "$FORCE" != "1" ]; then
        echo "[STOP] 停在 Stage 2 之后（继续: FORCE=1 重跑，已完成的 B 训练会 SKIP）"
        exit 2
    fi
    echo "[FORCE=1] 继续"
fi

# Stage 3 (C): buffers A+B
run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" \
    "$CKPT_B" 40000 "$CKPT_B" 40000 "$BUF_P/taskA" "$BUF_P/taskB"

# Stage 4 (D): buffers 加权 A×2 + B×1 + C×3（与原型回放支线一致）
run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" \
    "$CKPT_C" 40000 "$CKPT_C" 40000 \
    "$BUF_P/taskA" "$BUF_P/taskA" "$BUF_P/taskB" "$BUF_P/taskC" "$BUF_P/taskC" "$BUF_P/taskC"

echo ""
echo "==== $PREFIX 三个阶段训练完成 ===="

if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    echo "==== 全任务评估 (A/B/C/D, 50ep, γ=$FILM_GAMMA_EVAL, 与 b3 同口径) ===="
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" \
        "$CKPT_D" "$EVAL_GPUS" 50 "${PREFIX}D" A B C D 2>&1 | grep -v "svulkan2.*error"
    SUM="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PREFIX}D_summary.txt"
    echo ""
    echo "================ $PREFIX（b3 配置 + 原型回放）结果 ================"
    [ -f "$SUM" ] && cat "$SUM" || echo "[WARN] 未找到汇总: $SUM"
    echo ""
    echo "判读: 与 b3（无回放 0.26-0.88-0.72-0.82）逐任务比 —— 提升幅度 = 回放的真实边际价值"
    echo "      若 A/C 无提升 ⇒ CL-LoRA 结构隔离才是主机制，论文叙事需调整"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi
