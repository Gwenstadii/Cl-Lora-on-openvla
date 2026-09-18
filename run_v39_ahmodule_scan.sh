#!/usr/bin/env bash
# =============================================================================
# run_v39_ahmodule_scan.sh — 动作头 A "单模块/多模块"扫描（哪个模块漂移才致命？）
#
# 背景: 已排除两个嫌疑人 ——
#   · LLM 侧 L24-31 的 A 漂 6/8 层（全速, 1 stage）→ Task A 仍 0.964（n6）
#   · 动作头 model.fc2 的 A 单独漂            → Task A 仍 0.982（v40_fc2）
#   而"LLM 8 层 + 动作头 4 个 全漂"（lr×0.2 探针）→ Task A = 0/56。
#   ⇒ 剩余嫌疑: 动作头 fc1 / 两个 resnet block / 多模块叠加效应。
#
# 默认扫描: fc1 → mlp_resnet_blocks → fc2(复核) → 四个全解冻(all)
#   每个臂 = 训 B 一个 stage + 探针评估 A(bank恢复)/B(当前权重)
#
# 用法（tmux 前台，跑完自动打对照表）:
#   cd /mnt/data/pengshengdi && git pull && source server_env.sh
#   tmux new -s scan
#   bash run_v39_ahmodule_scan.sh 2>&1 | tee train_v40_scan.log
#
#   只扫一个:  AH_LIST="fc1" bash run_v39_ahmodule_scan.sh
#   重跑已有:  FORCE_RERUN=1 bash run_v39_ahmodule_scan.sh
#   限卡:      GPUS=6,7 EVAL_GPUS=6,6,7,7 bash run_v39_ahmodule_scan.sh
# =============================================================================

set -u

: "${LOGS_ROOT:?请先 source server_env.sh（LOGS_ROOT 未设置）}"
: "${VLA_PATH:?请先 source server_env.sh（VLA_PATH 未设置）}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$SELF_DIR/run_v39b9_layermask.sh"
[ -f "$INNER" ] || { echo "[FAIL] 找不到 $INNER（先 git pull）"; exit 1; }

LLM_LAYERS="${LLM_LAYERS:-none}"                       # LLM 侧 specific-A 默认全冻
AH_LIST="${AH_LIST:-fc1 mlp_resnet_blocks fc2 all}"    # 要扫的动作头 A 配置（子串匹配模块名 / all）
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-2}"              # 2 = 只训 B（探针）
FORCE_RERUN="${FORCE_RERUN:-0}"
FREE_PCT_MIN="${FREE_PCT_MIN:-30}"
LOGDIR="${LOGDIR:-/mnt/data/pengshengdi}"
EVAL_OUT="/mnt/data/pengshengdi/RoboTwin-main/eval_result"

tag_of() {  # $1=AH_KEEP 规格 → tag
    case "$1" in
        fc1) echo "v40_fc1" ;;
        fc2) echo "v40_fc2" ;;
        mlp_resnet_blocks|blocks) echo "v40_blocks" ;;
        all) echo "v40_ahall" ;;
        *) echo "v40_$(echo "$1" | tr ',' '_' | tr -cd 'a-zA-Z0-9_')" ;;
    esac
}

parse_rate() {  # $1=log → 0-1 成功率（合并行缺失时用 worker 均值），或空
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

# ---------- 显存预检 ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "================ GPU 显存预检（空闲占比下限 ${FREE_PCT_MIN}%）================"
    LOW=0
    while IFS=, read -r idx total free util; do
        idx=$(echo "$idx" | tr -d ' '); total=$(echo "$total" | tr -d ' ')
        free=$(echo "$free" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        pct=$(( free * 100 / (total > 0 ? total : 1) ))
        printf "  GPU %s: 空闲 %s/%s MB (%s%%)  util=%s%%\n" "$idx" "$free" "$total" "$pct" "$util"
        [ "$pct" -lt "$FREE_PCT_MIN" ] && LOW=1
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu \
             --format=csv,noheader,nounits)
    if [ "$LOW" = "1" ]; then
        echo "  当前占卡进程:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "[WARN] 有卡显存偏紧（评估 worker 占卡时训练会 OOM）。建议 GPUS=6,7 EVAL_GPUS=6,6,7,7 重跑，或等评估结束。"
        sleep 5
    fi
fi

echo ""
echo "============ 动作头 A 模块扫描 ============"
echo "LLM 侧 specific-A: $LLM_LAYERS（none=全冻） | 扫描列表: [$AH_LIST] | 每臂训到 stage $STOP_AFTER_STAGE"

DONE_TAGS=()
FAILED_TAGS=()
for spec in $AH_LIST; do
    tag=$(tag_of "$spec")
    sumf="$EVAL_OUT/${tag}BonB_summary.txt"
    echo ""
    echo "############################################################################"
    echo "#### 臂 $tag（动作头 A = $spec）"
    echo "############################################################################"
    if [ -f "$sumf" ] && [ "$FORCE_RERUN" != "1" ]; then
        echo "[SKIP] 已有探针结果: $sumf（重跑用 FORCE_RERUN=1）"
        DONE_TAGS+=("$tag")
        continue
    fi
    STOP_AFTER_STAGE="$STOP_AFTER_STAGE" TRAIN_LAYERS="$LLM_LAYERS" AH_KEEP="$spec" TAG="$tag" \
        SKIP_B_GATE=1 bash "$INNER" 2>&1 | tee "$LOGDIR/train_${tag}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[WARN] $tag 结束码 rc=$rc（训练/评估有问题），继续下一臂；日志: $LOGDIR/train_${tag}.log"
        FAILED_TAGS+=("$tag")
    else
        DONE_TAGS+=("$tag")
    fi
done

echo ""
echo "======================== 扫描结果汇总 ========================"
printf "%-14s %-12s %-10s %-10s\n" "臂" "动作头A" "TaskA(保留)" "TaskB(新任务)"
for spec in $AH_LIST; do
    tag=$(tag_of "$spec")
    a=$(parse_rate "$EVAL_OUT/${tag}BonB_taskA.log")
    b=$(parse_rate "$EVAL_OUT/${tag}BonB_taskB.log")
    printf "%-14s %-12s %-10s %-10s\n" "$tag" "$spec" "${a:-无数据}" "${b:-无数据}"
done
echo ""
echo "---- 参照（1 个 stage 后 Task A 保留率）----"
echo "  b5       全冻 A（含动作头）                > 0.9"
echo "  n6       LLM 6/8 层 A 漂, 动作头 A 冻       0.9643"
echo "  v40_fc2  仅动作头 fc2 的 A 漂               0.9821"
echo "  lr_probe LLM 8 层 + 动作头 4 个全漂(lr/5)   0.0000"
echo "  b4       LLM 8 层 + 动作头 4 个全漂(3 stage) 0"
echo ""
echo "判读: 哪个模块单独漂就把 A 打崩 ⇒ 那就是单点故障；若单个都不崩、只有 all 崩 ⇒ 是多模块叠加效应。"
