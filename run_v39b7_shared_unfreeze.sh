#!/usr/bin/env bash
# =============================================================================
# run_v39b7_shared_unfreeze.sh — 解冻 shared A/B 的"共享通路漂移"臂
#
# 目的（两个用途，都属**标注清楚的机制消融**，不是"挑弱基线"）:
#   ① 遗忘曲线: `SHARED_LR_SCALE` 是**连续可调**的漂移剂量旋钮。
#      与 FiLM/specific-A 漂移不同，shared 层**不进 bank**（只在 stage1 学一次），
#      因此没有任何恢复路径 ⇒ 损伤同时作用于**所有任务**，且随剂量**平滑变化**
#      —— 这正是此前"0.2~0.5 中间残留不可得"缺的那条连续通道。
#   ② 回放对照组: `USE_REPLAY=True` 得到同配置 + 原型回放的孪生臂（唯一差异=回放），
#      用来测"回放能否抑制共享通路漂移"（回放梯度会流进 shared A/B）。
#
# 与 b5 的关系: b5 = 全冻结(shared A/B 冻结, specific-A 冻结, FiLM 冻结) + 无回放 → A/B >0.9
#   本脚本 = b5 **只解冻 shared A/B**（其余全同）⇒ 保留率应当下降，且下降幅度由 SHARED_LR_SCALE 控制。
#
# ⚠️ 定位提醒: 解冻 shared 等于**拆掉 CL-LoRA 设计的核心保护**。它可以作为
#   "逐项消融保护机制"的一行（− shared freeze），但**不能**被当作"CL-LoRA 无回放基线"
#   来讲故事——评审会问"你为什么把共享层解冻"。主基线仍应用 b3 / b5。
#
# 流程: 复用 rt_v39_taskA → B → B 自评门禁 → C → D → 全任务评估(γ=0)
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB7
#   bash run_v39b7_shared_unfreeze.sh 2>&1 | tee train_v39b7.log                 # 剂量 0.1
#   SHARED_LR_SCALE=0.3 bash run_v39b7_shared_unfreeze.sh 2>&1 | tee train_v39b7_s30.log
#   USE_REPLAY=True bash run_v39b7_shared_unfreeze.sh 2>&1 | tee train_v39r7.log # 回放孪生臂
#
# 产物: $LOGS_ROOT/rt_<v39b7|v39r7>_task{B,C,D}--40000_chkpt（不同剂量用 PREFIX 区分，见下）
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
EVAL_SEQ="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
EVAL_ONE="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"

CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"
BUF_P="$LOGS_ROOT/replay_buffers"

SHARED_LR_SCALE="${SHARED_LR_SCALE:-0.1}"     # 共享通路漂移剂量旋钮（1.0=主干 lr）
USE_REPLAY="${USE_REPLAY:-False}"            # True → 同配置 + 原型回放（回放孪生臂）
USE_KD="${USE_KD:-False}"
FREEZE_FILM_STAGE2="${FREEZE_FILM_STAGE2:-True}"   # 默认与 b5 一致（FiLM 冻死）

# tag 规则: 回放臂 → v39r7；无回放臂 → v39b7_s<剂量百分比>
if [ "$USE_REPLAY" = "True" ]; then
    PREFIX="v39r7"; ARM_DESC="shared A/B 解冻 + 原型回放"
else
    case "$SHARED_LR_SCALE" in
        1.0|1) PREFIX="v39b7" ;;
        *)     PREFIX="v39b7_s$(echo "$SHARED_LR_SCALE" | tr -d '.')" ;;
    esac
    ARM_DESC="shared A/B 解冻（无回放）"
fi

GPUS="${GPUS:-4,5,6,7}"
EVAL_GPUS="${EVAL_GPUS:-4,4,5,5,6,6,7,7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.80}"
FORCE="${FORCE:-0}"
FILM_GAMMA_EVAL="${FILM_GAMMA_EVAL:-0}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NPROC=${#GPU_ARR[@]}
GRAD_ACCUM=$((8 / (BATCH_SIZE * NPROC)))

echo "================ 前置检查 ================"
[ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在: $VLA_PATH/config.json"; exit 1; }
[ -d "$CKPT_A" ] || { echo "[FAIL] A ckpt 不存在: $CKPT_A"; exit 1; }
if [ "$USE_REPLAY" = "True" ]; then
    for t in taskA taskB taskC; do
        [ -f "$BUF_P/$t/manifest.jsonl" ] || { echo "[FAIL] 原型 buffer 缺失: $BUF_P/$t"; exit 1; }
    done
fi
echo "[OK] 支线 = $PREFIX | $ARM_DESC"
echo "[OK] 配置: freeze_shared=False（shared A/B 可训练, lr×$SHARED_LR_SCALE）"
echo "[OK]       + specific-A 冻结(True) + block_scale 冻结 + proprio 修复"
echo "[OK]       + FiLM freeze_film_stage2=$FREEZE_FILM_STAGE2 | 回放=$USE_REPLAY | USE_KD=$USE_KD"
echo "[OK] ⚠️ shared 层不进 bank ⇒ 漂移无恢复路径, 预期所有旧任务同时下降"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch=$BATCH_SIZE | accum=$GRAD_ACCUM (有效 batch=8)"
echo "============ $PREFIX: Stage 2 -> 3 -> 4 连续训练 ============"

cd "$TRAIN_DIR" || { echo "[FAIL] 目录不存在: $TRAIN_DIR"; exit 1; }

run_stage() {  # $1=stage $2=dataset $3=run_id $4=prev_dir $5=prev_step $6..=buffers(可空)
    local stage=$1 ds=$2 rid=$3 prev_dir=$4 prev_step=$5
    shift 5
    local buffers_csv; buffers_csv=$(IFS=,; echo "$*")
    if [ -d "$LOGS_ROOT/$rid--40000_chkpt" ]; then
        echo "[SKIP] $rid--40000_chkpt 已存在"
        return 0
    fi
    local replay_args=()
    if [ "$USE_REPLAY" = "True" ]; then
        replay_args=(--use_replay True --replay_buffer_dirs "$buffers_csv"
                     --replay_every_n_steps 4 --replay_loss_weight 0.5)
    else
        replay_args=(--use_replay False)
    fi
    echo ""
    echo "############ Stage $stage : $ds ############"
    env CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
    torchrun --standalone --nproc_per_node $NPROC vla-scripts/train_cl_lora.py \
        --run_root_dir "$LOGS_ROOT" --run_id_override "$rid" \
        --max_steps 40000 --save_freq 10000 \
        --vla_path "$VLA_PATH" \
        --dataset_name "$ds" \
        --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" --previous_checkpoint_step "$prev_step" \
        --teacher_checkpoint_dir "$prev_dir" --teacher_checkpoint_step "$prev_step" \
        --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a True \
        --freeze_shared False --shared_lr_scale "$SHARED_LR_SCALE" \
        --use_kd "$USE_KD" --freeze_film_stage2 "$FREEZE_FILM_STAGE2" \
        --lambda_kd 0.2 \
        "${replay_args[@]}" \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    local rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] Stage $stage 训练失败 (rc=$rc)"; exit 1; }
    echo "[OK] Stage $stage 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

CKPT_B="$LOGS_ROOT/rt_${PREFIX}_taskB--40000_chkpt"
CKPT_C="$LOGS_ROOT/rt_${PREFIX}_taskC--40000_chkpt"
CKPT_D="$LOGS_ROOT/rt_${PREFIX}_taskD--40000_chkpt"

if [ "$USE_REPLAY" = "True" ]; then
    run_stage 2 aloha_grab_roller_clean "rt_${PREFIX}_taskB" "$CKPT_A" 30000 "$BUF_P/taskA"
else
    run_stage 2 aloha_grab_roller_clean "rt_${PREFIX}_taskB" "$CKPT_A" 30000
fi

echo ""
echo "==== B 自评 (20ep) —— 门禁 ≥ $PASS_THRESHOLD（对照 b5 的 B>0.9）===="
bash "$EVAL_ONE" grab_roller demo_clean "$CKPT_B" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 "${PREFIX}_B_self" 0 \
    2>&1 | tee "/tmp/${PREFIX}_B.log" | grep -v "svulkan2.*error"
B_RATE=$(grep "Merged success rate" "/tmp/${PREFIX}_B.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
echo "B 自评: ${B_RATE:-异常}"
if ! python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "[WARN] B 自评 < $PASS_THRESHOLD"
    if [ "$FORCE" != "1" ]; then
        echo "[STOP] 停在 Stage 2 之后（继续: FORCE=1 重跑同一命令, 已完成的会 SKIP）"
        exit 2
    fi
    echo "[FORCE=1] 继续"
fi

if [ "$USE_REPLAY" = "True" ]; then
    run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" "$CKPT_B" 40000 \
        "$BUF_P/taskA" "$BUF_P/taskB"
    run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" "$CKPT_C" 40000 \
        "$BUF_P/taskA" "$BUF_P/taskA" "$BUF_P/taskB" "$BUF_P/taskC" "$BUF_P/taskC" "$BUF_P/taskC"
else
    run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" "$CKPT_B" 40000
    run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" "$CKPT_C" 40000
fi

echo ""
echo "==== $PREFIX 三个阶段训练完成 ===="

if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    echo "==== 全任务评估 (A/B/C/D, 50ep, γ=$FILM_GAMMA_EVAL) ===="
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" \
        "$CKPT_D" "$EVAL_GPUS" 50 "${PREFIX}D" A B C D 2>&1 | grep -v "svulkan2.*error"
    SUM="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PREFIX}D_summary.txt"
    echo ""
    echo "================ $PREFIX 结果 ================"
    [ -f "$SUM" ] && cat "$SUM" || echo "[WARN] 未找到汇总: $SUM"
    echo ""
    echo "对照: b5（全冻结, 无回放）= A/B >0.9 | b3（FiLM 漂 0.2×, 无回放）= 0.26-0.88-0.72-0.82"
    echo "判读: 若各任务同步下降且随 SHARED_LR_SCALE 单调 ⇒ 拿到连续可调的遗忘曲线"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi
