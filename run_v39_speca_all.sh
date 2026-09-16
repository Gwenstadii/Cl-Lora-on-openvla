#!/usr/bin/env bash
# =============================================================================
# run_v39_speca_all.sh — ④ 全套：A 冻结量路线（探针 + A 入 bank）一条命令跑完
#
# 依次执行:
#   ① 探针 0.05 : A 部分解冻 lr×0.05 → 只训 B(1 段) → 在 B ckpt 上评 A/B
#                  （A 是"只漂 1 个 stage"的成绩，作为"漂 3 个 stage"的上界）
#   ② 探针 0.20 : 同上，剂量 0.2
#   ③ v39r4    : A 入 bank（A 全速训练 + bank 每任务存 A_K）→ 全链 B/C/D + 全任务评估
#                  （评估前自动 patch 继承来的 task_1_bank.pt，补 A_1 快照）
#   ④ v39r4_r  : 同 ③ + 原型回放（回答"A 都不用冻了，回放还需要吗"）
#
# 判读:
#   · 探针两档 A 都 ≈0 ⇒ A 漂移无中间区（阈值型），"部分解冻"路线判死
#   · 某档 A 明显 >0  ⇒ 存在可用中间区，可做"可调残留"曲线
#   · v39r4 A/B/C 都 0.85+ ⇒ **不冻 A 也能高保留**：bank 存 A 快照 = 与"冻结 A"并列的范式
#   · v39r4 D 若 ≥ b5 的 D ⇒ A 可塑性带来的真实收益，值得单独记
#
# 用法（tmux 前台）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s train4
#   bash run_v39_speca_all.sh 2>&1 | tee train_v39_speca_all.log
#
#   跳过某部分: RUN_BANK=0 / RUN_BANK_REPLAY=0 / PROBE_SCALES=""
#   只跑一个探针档: PROBE_SCALES="0.2" RUN_BANK=0 RUN_BANK_REPLAY=0 bash run_v39_speca_all.sh
#   卡被别人占用时限制卡数: GPUS=6,7 EVAL_GPUS=6,6,7,7 bash run_v39_speca_all.sh
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-/mnt/data/pengshengdi}"
EVAL_DIR_OUT="/mnt/data/pengshengdi/RoboTwin-main/eval_result"

PROBE_SCALES="${PROBE_SCALES:-0.05 0.2}"      # 探针剂量列表（空格分隔；空串=跳过探针）
RUN_BANK="${RUN_BANK:-1}"                     # 1 = 跑 v39r4（A 入 bank）
RUN_BANK_REPLAY="${RUN_BANK_REPLAY:-1}"       # 1 = 再跑 v39r4 的回放孪生臂
FREE_PCT_MIN="${FREE_PCT_MIN:-30}"            # 每卡空闲显存占比下限（%），低于则提示卡被占用
FORCE="${FORCE:-0}"                           # 1 = 显存不足也硬跑

INNER="$SELF_DIR/run_v39b8_speca_knob.sh"
[ -f "$INNER" ] || { echo "[FAIL] 找不到 $INNER（先 git pull）"; exit 1; }

# ---------- 显存预检（b5 评估等占卡时避免 OOM 秒退） ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "================ GPU 显存预检（空闲占比下限 ${FREE_PCT_MIN}%）================"
    LOW=0
    while IFS=, read -r idx total free util; do
        idx=$(echo "$idx" | tr -d ' '); total=$(echo "$total" | tr -d ' ')
        free=$(echo "$free" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        pct=$(( free * 100 / (total > 0 ? total : 1) ))
        flag="OK"
        [ "$pct" -lt "$FREE_PCT_MIN" ] && { flag="LOW"; LOW=1; }
        echo "  GPU $idx: 空闲 ${free}/${total} MB (${pct}%)  util=${util}%  [$flag]"
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu \
             --format=csv,noheader,nounits)
    if [ "$LOW" = "1" ] && [ "$FORCE" != "1" ]; then
        echo ""
        echo "  当前占卡进程:"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo ""
        echo "[WARN] 有卡显存不足 —— 若 b5 评估仍在跑, 起训练会 OOM 秒退。处理:"
        echo "       ① 等评估收尾后重跑本脚本（已完成的 ckpt 会自动 SKIP）;"
        echo "       ② 或限制卡数: GPUS=6,7 EVAL_GPUS=6,6,7,7 bash run_v39_speca_all.sh"
        echo "       ③ 或确认无碍后: FORCE=1 bash run_v39_speca_all.sh"
        exit 3
    fi
fi

run_arm() {  # $1=描述 $2=日志名 $3..=环境变量赋值
    local desc=$1 logname=$2; shift 2
    echo ""
    echo "############################################################################"
    echo "#### $desc"
    echo "############################################################################"
    env "$@" bash "$INNER" 2>&1 | tee "$LOGDIR/$logname"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 2 ]; then
        echo "[STOP] $desc 停在 B 自评门禁 (rc=2)；排查后重跑本脚本（已完成 ckpt 会 SKIP）"
        exit 2
    elif [ "$rc" -ne 0 ]; then
        echo "[FAIL] $desc 失败 (rc=$rc)，见 $LOGDIR/$logname"
        exit "$rc"
    fi
    echo "[OK] $desc 完成"
}

echo "============ ④ A 冻结量路线全套 ============"
echo "探针剂量: [${PROBE_SCALES:-（跳过）}] | v39r4=$RUN_BANK | v39r4_r=$RUN_BANK_REPLAY"
echo "共享: A ckpt = $LOGS_ROOT/rt_v39_taskA--30000_chkpt | buffers = $LOGS_ROOT/replay_buffers"

# ---------- ①② 探针（各 1 段训练） ----------
for s in $PROBE_SCALES; do
    TAG="v39b8_s$(echo "$s" | tr -d '.')"
    run_arm "探针 A 部分解冻 lr×$s（只训 B，1 段）→ $TAG" "train_${TAG}.log" \
        STOP_AFTER_STAGE=2 SPEC_A_LR_SCALE="$s" FORCE="${FORCE:-0}"
done

# ---------- ③ v39r4: A 入 bank ----------
if [ "$RUN_BANK" = "1" ]; then
    run_arm "v39r4: A 入 bank（A 全速训练 + 每任务 A 快照）" "train_v39r4.log" \
        A_IN_BANK=True FORCE="${FORCE:-0}"
fi

# ---------- ④ v39r4_r: A 入 bank + 原型回放 ----------
if [ "$RUN_BANK_REPLAY" = "1" ]; then
    run_arm "v39r4_r: A 入 bank + 原型回放" "train_v39r4r.log" \
        A_IN_BANK=True USE_REPLAY=True FORCE="${FORCE:-0}"
fi

# ---------- 汇总 ----------
echo ""
echo "======================== ④ 全套结果汇总 ========================"
print_sum() {  # $1=tag  $2=说明
    local tag=$1 note=$2 f="$EVAL_DIR_OUT/${tag}_summary.txt"
    echo "---- $tag  ($note) ----"
    if [ -f "$f" ]; then
        grep -E "Task [ABCD] " "$f" | sed 's/^/  /'
    else
        local any=0
        for t in A B; do
            local lf="$EVAL_DIR_OUT/${tag}_task${t}.log"
            if [ -f "$lf" ]; then
                any=1
                printf "  Task %s: %s\n" "$t" "$(grep 'Merged success rate' "$lf" | tail -1)"
            fi
        done
        [ "$any" = "0" ] && echo "  [WARN] 无结果（可跑: bash show_eval_rates.sh $tag）"
    fi
}
for s in $PROBE_SCALES; do
    print_sum "v39b8_s$(echo "$s" | tr -d '.')BonB" "探针 lr×$s, 只漂 1 个 stage 的 A 保留率"
done
[ "$RUN_BANK" = "1" ] && print_sum "v39r4D" "A 入 bank, 全链"
[ "$RUN_BANK_REPLAY" = "1" ] && print_sum "v39r4_rD" "A 入 bank + 原型回放, 全链"

echo ""
echo "---- 参照行 ----"
echo "  b4  （不冻 A, 无回放）       0    - 0.57 - 0    - 0.85"
echo "  r3  （不冻 A + 原型回放）    0    - 0.80 - 0.20 - 未评"
echo "  b5  （全冻结, 无回放）       >0.9 - >0.9 - C/D 待补"
echo ""
echo "判读: 探针 A≈0 ⇒ 部分解冻无中间区; v39r4 A/B/C 0.85+ ⇒ bank 存 A = 与冻结 A 并列的范式。"
echo "验证 A 快照是否补进 bank（应为 60）:"
echo "  python -c \"import torch,os;b=torch.load(os.environ['LOGS_ROOT']+'/rt_v39r4_taskD--40000_chkpt/task_1_bank.pt',map_location='cpu',weights_only=True);print(sum(1 for k in b if k.endswith('lora_a')))\""
