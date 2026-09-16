#!/usr/bin/env bash
# =============================================================================
# run_v39b8_speca_knob.sh — specific-A "冻结量"旋钮：两条路线
#
# 背景（痛点）: "冻结 FiLM + 不冻 specific-A"（b4 无回放 0-0.57-0-0.85；r3 有回放 0-0.80-0.20）
#   旧任务 A 恒为 0。原因: bank **只存 B**，A 自由漂移后 B_K 与 A_final 配不上。
#
# 两条路线:
#   ① 减少解冻量（本脚本默认）: `SPEC_A_LR_SCALE` 给 specific-A 单独设低 lr（"部分解冻"）
#      → 期望: A 漂移变慢、B_K 失配变小、旧任务保留回升。**但注意 Adam 的漂移量 ≈ lr×步数**，
#        漂移对 lr 的响应偏"阈值型"（参照 FiLM: 0.2× 与 1.0× 结果几乎一样）⇒ 0.1~0.5 这种
#        "温和"缩放**很可能仍然跨阈值**。真正能落在中间区的缩放可能很小(≤0.05)，那时 A≈冻死。
#   ② A 入 bank（`A_IN_BANK=True`）: A 照常全速训练（保留全部可塑性），但每个任务的 bank
#      **额外存一份 A_K 快照** → 评估时 A_K + B_K 同时恢复 ⇒ 配对精确复原 ⇒ 保留率≈自评。
#      ⇒ 这才是"既解冻 A 又不丢保留率"的正解；代价每任务 +约 11MB。
#
# 便宜探针: `STOP_AFTER_STAGE=2` 只训 B 就停 → 用 1/3 成本测"1 个 stage 的 A 漂移够不够毁掉 A"。
#
# 用法:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB8
#   # ① 探针（只训 B，1 段；剂量 0.05）
#   STOP_AFTER_STAGE=2 SPEC_A_LR_SCALE=0.05 bash run_v39b8_speca_knob.sh 2>&1 | tee train_v39b8_p05.log
#   # ② 中间剂量全链
#   SPEC_A_LR_SCALE=0.1 bash run_v39b8_speca_knob.sh 2>&1 | tee train_v39b8_s10.log
#   # ③ A 入 bank（推荐先跑这条：能直接达成目标）
#   A_IN_BANK=True bash run_v39b8_speca_knob.sh 2>&1 | tee train_v39r4.log
#   # ④ A 入 bank + 原型回放（测"回放还需要吗"）
#   A_IN_BANK=True USE_REPLAY=True bash run_v39b8_speca_knob.sh 2>&1 | tee train_v39r4r.log
#
# 产物: $LOGS_ROOT/rt_<tag>_task{B,C,D}--40000_chkpt
#   tag 规则: A_IN_BANK → v39r4(_r)；否则 v39b8_s<剂量>(_r)
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
EVAL_SEQ="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
EVAL_ONE="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"

CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"
BUF_P="$LOGS_ROOT/replay_buffers"

SPEC_A_LR_SCALE="${SPEC_A_LR_SCALE:-0.05}"    # 路线①: specific-A 的 lr 缩放
A_IN_BANK="${A_IN_BANK:-False}"               # 路线②: bank 存 A 快照（A 全速训练）
USE_REPLAY="${USE_REPLAY:-False}"
USE_KD="${USE_KD:-False}"
FREEZE_FILM_STAGE2="${FREEZE_FILM_STAGE2:-True}"   # 与 b4/r3 一致（冻结 FiLM）
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-4}"     # 2 = 只训 B（便宜探针）

if [ "$A_IN_BANK" = "True" ]; then
    BASE_TAG="v39r4"
else
    BASE_TAG="v39b8_s$(echo "$SPEC_A_LR_SCALE" | tr -d '.')"
fi
if [ "$USE_REPLAY" = "True" ]; then PREFIX="${BASE_TAG}_r"; else PREFIX="$BASE_TAG"; fi

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
[ -f "$VLA_PATH/config.json" ] || { echo "[FAIL] 基座模型不存在"; exit 1; }
[ -d "$CKPT_A" ] || { echo "[FAIL] A ckpt 不存在: $CKPT_A"; exit 1; }
if [ "$USE_REPLAY" = "True" ]; then
    for t in taskA taskB taskC; do
        [ -f "$BUF_P/$t/manifest.jsonl" ] || { echo "[FAIL] 原型 buffer 缺失: $BUF_P/$t"; exit 1; }
    done
fi
echo "[OK] 支线 tag = $PREFIX | 路线$([ "$A_IN_BANK" = "True" ] && echo '② A 入 bank（A 全速训练）' || echo "① 部分解冻 spec-A lr×$SPEC_A_LR_SCALE")"
echo "[OK] 配置: freeze_specific_a=False + FiLM freeze=$FREEZE_FILM_STAGE2 + 回放=$USE_REPLAY + USE_KD=$USE_KD"
echo "[OK] STOP_AFTER_STAGE=$STOP_AFTER_STAGE（2=只训 B 的便宜探针）"
echo "[OK] GPUS=$GPUS | NPROC=$NPROC | batch=$BATCH_SIZE | accum=$GRAD_ACCUM (有效 batch=8)"
echo "============ $PREFIX: Stage 2 -> $STOP_AFTER_STAGE ============"

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
        --vla_path "$VLA_PATH" --dataset_name "$ds" --stage "$stage" \
        --previous_checkpoint_dir "$prev_dir" --previous_checkpoint_step "$prev_step" \
        --teacher_checkpoint_dir "$prev_dir" --teacher_checkpoint_step "$prev_step" \
        --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" --learning_rate 5e-4 \
        --lr_warmup_steps 200 --num_steps_before_decay 100000 \
        --use_cl_lora True --lora_rank 16 --shared_depth 8 --first_lora_layer 16 \
        --orthogonal_init True --freeze_a True --use_block_scale True --freeze_specific_a False \
        --specific_a_lr_scale "$SPEC_A_LR_SCALE" --bank_save_specific_a "$A_IN_BANK" \
        --bank_film_mode film \
        --use_kd "$USE_KD" --freeze_film_stage2 "$FREEZE_FILM_STAGE2" --lambda_kd 0.2 \
        "${replay_args[@]}" \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    local rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] Stage $stage 训练失败 (rc=$rc)"; exit 1; }
    echo "[OK] Stage $stage 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

CKPT_B="$LOGS_ROOT/rt_${PREFIX}_taskB--40000_chkpt"
CKPT_C="$LOGS_ROOT/rt_${PREFIX}_taskC--40000_chkpt"
CKPT_D="$LOGS_ROOT/rt_${PREFIX}_taskD--40000_chkpt"
BUF_ARGS_B=("$BUF_P/taskA")
BUF_ARGS_C=("$BUF_P/taskA" "$BUF_P/taskB")
BUF_ARGS_D=("$BUF_P/taskA" "$BUF_P/taskA" "$BUF_P/taskB" "$BUF_P/taskC" "$BUF_P/taskC" "$BUF_P/taskC")

run_stage 2 aloha_grab_roller_clean "rt_${PREFIX}_taskB" "$CKPT_A" 30000 "${BUF_ARGS_B[@]}"

echo ""
echo "==== B 自评 (20ep) —— 门禁 ≥ $PASS_THRESHOLD ===="
bash "$EVAL_ONE" grab_roller demo_clean "$CKPT_B" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 "${PREFIX}_B_self" 0 \
    2>&1 | tee "/tmp/${PREFIX}_B.log" | grep -v "svulkan2.*error"
B_RATE=$(grep "Merged success rate" "/tmp/${PREFIX}_B.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
echo "B 自评: ${B_RATE:-异常}"
if ! python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "[WARN] B 自评 < $PASS_THRESHOLD"
    [ "$FORCE" != "1" ] && { echo "[STOP] 停在 Stage 2 之后（继续: FORCE=1 重跑, 已完成会 SKIP）"; exit 2; }
    echo "[FORCE=1] 继续"
fi

if [ "$STOP_AFTER_STAGE" -ge 3 ] 2>/dev/null; then
    run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" "$CKPT_B" 40000 "${BUF_ARGS_C[@]}"
else
    echo "[PROBE] STOP_AFTER_STAGE=$STOP_AFTER_STAGE → 只训到 B, 跳过 C/D"
fi
if [ "$STOP_AFTER_STAGE" -ge 4 ] 2>/dev/null; then
    run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" "$CKPT_C" 40000 "${BUF_ARGS_D[@]}"
fi

echo ""
echo "==== $PREFIX 训练阶段完成 ===="

# 探针模式: 直接在 B ckpt 上评估 A（关键问题: 1 个 stage 的 A 漂移有没有毁掉 A）
if [ "$STOP_AFTER_STAGE" -lt 4 ] 2>/dev/null; then
    echo ""
    echo "==== [探针] 在 B ckpt (stage 2) 上评估 A + B ===="
    echo "     A 用 bank(eval_task_id=1) 恢复 B_A; 若 A 已≈0 ⇒ 说明该剂量下 1 个 stage 就跨阈值"
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" "$CKPT_B" "$EVAL_GPUS" 50 "${PREFIX}BonB" A B \
        2>&1 | grep -v "svulkan2.*error"
    echo ""
    echo "对照: b4 的 D ckpt 上 A=0（3 阶段全速漂移）| b5（A 冻结）A>0.9"
    exit 0
fi

if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    # A 入 bank 路线: 继承来的 task_1_bank.pt 是旧代码写的（不含 A 快照）⇒ 评估前补上 A_1
    if [ "$A_IN_BANK" = "True" ]; then
        echo ""
        echo "==== [A 入 bank] 给继承的 task_1_bank.pt 补 specific-A 快照 ===="
        echo "     否则任务 A 仍会用漂移后的 A_final（A 会继续崩）；来源: $CKPT_A"
        python vla-scripts/patch_bank_specific_a.py \
            --ckpt-dir "$CKPT_D" --src-ckpt "$CKPT_A" --vla-step 30000 --ah-step 30000 \
            || { echo "[FAIL] bank 补 A 失败 —— A 的保留率会失真, 请先排查"; exit 1; }
    fi
    echo "==== 全任务评估 (A/B/C/D, 50ep, γ=$FILM_GAMMA_EVAL) ===="
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" \
        "$CKPT_D" "$EVAL_GPUS" 50 "${PREFIX}D" A B C D 2>&1 | grep -v "svulkan2.*error"
    SUM="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PREFIX}D_summary.txt"
    echo ""
    echo "================ $PREFIX 结果 ================"
    [ -f "$SUM" ] && cat "$SUM" || echo "[WARN] 未找到汇总: $SUM"
    echo ""
    echo "对照: b4（不冻 A, 无回放）0-0.57-0-0.85 | r3（不冻 A + 回放）0-0.80-0.20 | b5（全冻结）A/B>0.9"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi
