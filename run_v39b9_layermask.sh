#!/usr/bin/env bash
# =============================================================================
# run_v39b9_layermask.sh — "按层保护量"旋钮：冻结部分 specific-A 层（L24-31 内）
#
# 与 b5 的关系: b5 = **8/8 层 A 全冻** ⇒ A/B >0.9（保留率天花板）
#               k 层解冻（= 8-k 层冻结）⇒ 每解冻一层，那层的 B_K·A 就配不上 ⇒ 保留率应逐级下降
#               k=8 层全解冻 = 与 b4 同构（但 b4 还额外让动作头 A 漂）
#
# 为什么这个旋钮**比 lr 缩放更可能给出连续曲线**（实测依据见 REPLAY_BUG_NOTES §10.8）:
#   冻结层的贡献 = s·g_K·B_K·A_1 **精确复原**；只有解冻层局部损坏
#   ⇒ 损伤按"坏掉几层"分级；而 lr 缩放让**所有层同时轻微漂移**，实测 1 个 stage @lr×0.2 就 A=0/56。
#
# 用法（先探针后全链）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s trainB9
#   # 探针（只训 B, 1 段）：解冻 2 层 / 4 层 / 6 层，各评一次 A 保留率
#   STOP_AFTER_STAGE=2 TRAIN_LAYERS="30-31" bash run_v39b9_layermask.sh   # 解冻 2 层
#   STOP_AFTER_STAGE=2 TRAIN_LAYERS="28-31" bash run_v39b9_layermask.sh   # 解冻 4 层
#   STOP_AFTER_STAGE=2 TRAIN_LAYERS="26-31" bash run_v39b9_layermask.sh   # 解冻 6 层
#   # 全链（含 C/D + 全任务评估）
#   TRAIN_LAYERS="28-31" bash run_v39b9_layermask.sh
#   # 加原型回放（测"回放能否替代 A 保护"）
#   TRAIN_LAYERS="28-31" USE_REPLAY=True bash run_v39b9_layermask.sh
#
# 产物 tag: v39b9_n<解冻层数>(_r)。例: TRAIN_LAYERS="28-31" → v39b9_n4
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

TRAIN_DIR="/mnt/data/pengshengdi/openvla-oft"
EVAL_SEQ="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_sequence.sh"
EVAL_ONE="/mnt/data/pengshengdi/RoboTwin-main/policy/openvla-oft/eval_multi_gpu.sh"

CKPT_A="$LOGS_ROOT/rt_v39_taskA--30000_chkpt"
BUF_P="$LOGS_ROOT/replay_buffers"

TRAIN_LAYERS="${TRAIN_LAYERS:-28-31}"          # 保持可训练的 specific-A 层（24-31 内）；"none"=全部冻结 LLM 侧 A
FREEZE_AH_A="${FREEZE_AH_A:-True}"             # 动作头 A 是否冻结（默认冻结 ⇒ 只动 LLM 层，单变量更干净）
AH_KEEP="${AH_KEEP:-}"                         # 动作头 A 精细控制: ""=按 FREEZE_AH_A | "all" | "none" | 子串(如 "fc2")
USE_REPLAY="${USE_REPLAY:-False}"
USE_KD="${USE_KD:-False}"
FREEZE_FILM_STAGE2="${FREEZE_FILM_STAGE2:-True}"   # 与 b5/b4 一致
FILM_LR_SCALE="${FILM_LR_SCALE:-1.0}"              # >0 且 freeze_film_stage2=False 时生效: FiLM 漂移力度（b3/b6 用 0.2 / 1.0）
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-4}"

case "$TRAIN_LAYERS" in
  none|no|-|0) N_TRAIN=0 ;;
  *) N_TRAIN=$(python - <<PY
s="$TRAIN_LAYERS"; n=0
for p in s.split(','):
    p=p.strip()
    if not p: continue
    if '-' in p:
        a,b=p.split('-'); n+=int(b)-int(a)+1
    else: n+=1
print(n)
PY
) ;;
esac
if [ "$N_TRAIN" = "0" ] && [ "$FREEZE_AH_A" != "True" ]; then
    PREFIX="v39b9_ahonly"          # 只让动作头 A 漂移（决定性对照）
else
    PREFIX="v39b9_n${N_TRAIN}"
fi
PREFIX="${TAG:-$PREFIX}"           # TAG=v40 等自定义名（覆盖自动命名）
[ "$USE_REPLAY" = "True" ] && PREFIX="${PREFIX}_r"

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
echo "[OK] tag = $PREFIX | specific-A 解冻层 = $TRAIN_LAYERS（共 $N_TRAIN 层，其余 L24-31 的 A 冻结）"
echo "[OK] 动作头 A: $([ "$FREEZE_AH_A" = "True" ] && echo 冻结 || echo 解冻) | FiLM freeze=$FREEZE_FILM_STAGE2 (lr×$FILM_LR_SCALE)"
echo "[OK] 回放=$USE_REPLAY | USE_KD=$USE_KD | STOP_AFTER_STAGE=$STOP_AFTER_STAGE"
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
        --specific_a_trainable_layers "$TRAIN_LAYERS" \
        --specific_a_freeze_action_head "$FREEZE_AH_A" \
        --specific_a_action_head_keep "$AH_KEEP" \
        --bank_film_mode film \
        --use_kd "$USE_KD" --freeze_film_stage2 "$FREEZE_FILM_STAGE2" --film_lr_scale "$FILM_LR_SCALE" --lambda_kd 0.2 \
        "${replay_args[@]}" \
        --image_aug True --use_proprio True --use_film True --num_images_in_input 3
    local rc=$?
    [ $rc -ne 0 ] && { echo "[FAIL] Stage $stage 训练失败 (rc=$rc)"; exit 1; }
    echo "[OK] Stage $stage 完成 -> $LOGS_ROOT/$rid--40000_chkpt"
}

CKPT_B="$LOGS_ROOT/rt_${PREFIX}_taskB--40000_chkpt"
CKPT_C="$LOGS_ROOT/rt_${PREFIX}_taskC--40000_chkpt"
CKPT_D="$LOGS_ROOT/rt_${PREFIX}_taskD--40000_chkpt"

run_stage 2 aloha_grab_roller_clean "rt_${PREFIX}_taskB" "$CKPT_A" 30000 "$BUF_P/taskA"

echo ""
echo "==== B 自评 (20ep) —— 门禁 ≥ $PASS_THRESHOLD（对照 b5 的 B>0.9）===="
bash "$EVAL_ONE" grab_roller demo_clean "$CKPT_B" 0 "$EVAL_GPUS" \
    aloha_grab_roller_clean 2 20 "${PREFIX}_B_self" 0 \
    2>&1 | tee "/tmp/${PREFIX}_B.log" | grep -v "svulkan2.*error"

# 稳健解析: 合并行缺失（某 worker 崩了会让合并被跳过）时，回退到各 worker 成功率均值
parse_rate() {  # $1=log → 打印 0-1 的成功率，或空
    local merged
    merged=$(grep "Merged success rate" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || true)
    if [ -n "$merged" ]; then echo "$merged"; return; fi
    python - "$1" <<'PY'
import re, sys
vals = []
try:
    for ln in open(sys.argv[1], errors="ignore"):
        m = re.search(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", ln)
        if m:
            vals.append(float(m.group(3)) / 100.0)
except FileNotFoundError:
    pass
print(f"{sum(vals)/len(vals):.4f}" if vals else "")
PY
}

B_RATE=$(parse_rate "/tmp/${PREFIX}_B.log")
if [ -z "$B_RATE" ]; then
    echo "[FAIL] B 自评日志里既无合并行也无 worker 成功率 → 看 /tmp/${PREFIX}_B.log 的报错"
    echo "       想跳过门禁直接评估: SKIP_B_GATE=1 重跑（训练会 SKIP, 直接进探针评估）"
    [ "$FORCE" != "1" ] && exit 2
fi
echo "B 自评: ${B_RATE:-异常}$(grep -q '有 worker 失败' "/tmp/${PREFIX}_B.log" 2>/dev/null && echo ' （有 worker 崩溃, 合并行缺失 → 用各 worker 均值）')"
if [ "${SKIP_B_GATE:-0}" = "1" ]; then
    echo "[SKIP_B_GATE=1] 跳过 B 门禁，继续"
elif ! python -c "import sys; sys.exit(0 if float('${B_RATE:-0}') >= $PASS_THRESHOLD else 1)"; then
    echo "[WARN] B 自评 < $PASS_THRESHOLD"
    [ "$FORCE" != "1" ] && { echo "[STOP] 停在 Stage 2 之后（继续: FORCE=1 或 SKIP_B_GATE=1 重跑, 已完成会 SKIP）"; exit 2; }
    echo "[FORCE=1] 继续"
fi

if [ "$STOP_AFTER_STAGE" -ge 3 ] 2>/dev/null; then
    run_stage 3 aloha_stack_bowls_two_clean "rt_${PREFIX}_taskC" "$CKPT_B" 40000 \
        "$BUF_P/taskA" "$BUF_P/taskB"
else
    echo "[PROBE] STOP_AFTER_STAGE=$STOP_AFTER_STAGE → 只训到 B, 跳过 C/D"
fi
if [ "$STOP_AFTER_STAGE" -ge 4 ] 2>/dev/null; then
    run_stage 4 aloha_open_laptop_clean "rt_${PREFIX}_taskD" "$CKPT_C" 40000 \
        "$BUF_P/taskA" "$BUF_P/taskA" "$BUF_P/taskB" "$BUF_P/taskC" "$BUF_P/taskC" "$BUF_P/taskC"
fi

echo ""
echo "==== $PREFIX 训练阶段完成 ===="

if [ "$STOP_AFTER_STAGE" -lt 4 ] 2>/dev/null; then
    # 探针: 在最后一个已训 stage 的 ckpt 上评估"到该阶段为止的全部任务"
    if [ "$STOP_AFTER_STAGE" -ge 3 ] 2>/dev/null; then
        PROBE_CKPT="$CKPT_C"; PROBE_TAG="${PREFIX}ConC"; PROBE_TASKS="A B C"
    else
        PROBE_CKPT="$CKPT_B"; PROBE_TAG="${PREFIX}BonB"; PROBE_TASKS="A B"
    fi
    PROBE_EPISODES="${PROBE_EPISODES:-50}"
    echo ""
    echo "==== [探针] 在 $(basename "$PROBE_CKPT") 上评估 $PROBE_TASKS（A 用 bank 恢复 B_A）===="
    echo "     解冻层 = $TRAIN_LAYERS（$N_TRAIN 层）；episodes=$PROBE_EPISODES"
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" "$PROBE_CKPT" "$EVAL_GPUS" "$PROBE_EPISODES" \
        "$PROBE_TAG" $PROBE_TASKS 2>&1 | grep -v "svulkan2.*error"
    echo ""
    echo "---- ${PROBE_TAG} 探针结果（合并行缺失时自动用各 worker 均值兜底）----"
    for t in $PROBE_TASKS; do
        lf="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PROBE_TAG}_task${t}.log"
        r=$(parse_rate "$lf")
        printf "  Task %s: %s\n" "$t" "${r:-无数据（看 $lf）}"
    done
    echo ""
    echo "继续下一段（B/C 已训好的会自动 SKIP，只补 D 并做全任务评估）:"
    echo "  STOP_AFTER_STAGE=4 TRAIN_LAYERS=\"$TRAIN_LAYERS\" AH_KEEP=\"$AH_KEEP\" FREEZE_FILM_STAGE2=$FREEZE_FILM_STAGE2 FILM_LR_SCALE=$FILM_LR_SCALE TAG=$PREFIX bash run_v39b9_layermask.sh"
    echo ""
    echo "b3 对照（FiLM 漂 0.2×, A 全冻, 无回放, D ckpt）= 0.26-0.88-0.72-0.82"
    echo "b4 对照（specific-A 全漂含动作头, FiLM 冻, 无回放）= 0-0.57-0-0.85"
    exit 0
fi

if [ "${EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    echo "==== 全任务评估 (A/B/C/D, 50ep, γ=$FILM_GAMMA_EVAL) ===="
    FILM_GAMMA="$FILM_GAMMA_EVAL" bash "$EVAL_SEQ" \
        "$CKPT_D" "$EVAL_GPUS" 50 "${PREFIX}D" A B C D 2>&1 | grep -v "svulkan2.*error"
    SUM="/mnt/data/pengshengdi/RoboTwin-main/eval_result/${PREFIX}D_summary.txt"
    echo ""
    echo "================ $PREFIX 结果（解冻 $N_TRAIN 层 A）================"
    [ -f "$SUM" ] && cat "$SUM" || echo "[WARN] 未找到汇总: $SUM"
    echo ""
    echo "对照: b5(全冻 A) A/B>0.9 | b4(全解冻) 0-0.57-0-0.85 | r3(全解冻+回放) 0-0.80-0.20"
else
    echo "[SKIP] EVAL_AFTER_TRAIN=0"
fi
