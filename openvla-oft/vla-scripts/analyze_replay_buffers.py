#!/usr/bin/env python3
"""
analyze_replay_buffers.py — 回放 buffer 规模统计与"公平性"核对

回答三类问题：
  1. 每个任务的 buffer 有多少样本？为什么不同？（episodes / segments / top_k 拆解）
  2. prototype 与 uniform 是否严格同预算？（同任务内应逐任务相等）
  3. 样本数差异会不会造成梯度占比差异？（回答：不会，见 round-robin 说明）

用法:
  cd /mnt/data/pengshengdi/openvla-oft
  python vla-scripts/analyze_replay_buffers.py \
      --roots $LOGS_ROOT/replay_buffers $LOGS_ROOT/replay_buffers_uniform \
      --weights taskA=2,taskB=1,taskC=3          # 可选: D 阶段的目录重复加权

产物: 终端表格（stdout），可重定向存档做论文附录
"""

import argparse
import json
import os
from pathlib import Path


def read_json(p: Path):
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_lines(p: Path) -> int:
    if not p.exists():
        return -1
    n = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def collect(buf_dir: Path) -> dict:
    """从 meta.json 取权威统计, 缺失时回退到数文件行数。"""
    meta = read_json(buf_dir / "meta.json") or {}
    samples = meta.get("saved_replay_samples")
    if samples is None:
        samples = count_lines(buf_dir / "manifest.jsonl")
    segments = meta.get("num_segments")
    if segments is None:
        segments = count_lines(buf_dir / "segments.jsonl")
    source_frames = meta.get("source_frames")

    # diagnostics.jsonl: 每行一个"实际处理过的" episode（被跳过的 episode 不落盘）
    episodes_processed = count_lines(buf_dir / "diagnostics.jsonl")
    ep_frames = []
    seg_per_ep = []
    diag = buf_dir / "diagnostics.jsonl"
    if diag.exists():
        with diag.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                ep_frames.append(d.get("num_frames", 0))
                seg_per_ep.append(d.get("num_segments", 0))
    if source_frames is None:
        source_frames = sum(ep_frames)

    seg_lengths = []
    segs = buf_dir / "segments.jsonl"
    if segs.exists():
        with segs.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seg_lengths.append(json.loads(line).get("num_frames", 0))

    return {
        "name": buf_dir.name,
        "dataset": meta.get("dataset", "?"),
        "top_k": meta.get("top_k"),
        "num_episodes_cfg": meta.get("num_episodes"),
        "episodes_processed": episodes_processed if episodes_processed >= 0 else None,
        "segments": segments,
        "samples": samples,
        "source_frames": source_frames,
        "compression": meta.get("compression_ratio"),
        "seg_per_ep": (sum(seg_per_ep) / len(seg_per_ep)) if seg_per_ep else None,
        "avg_seg_frames": (sum(seg_lengths) / len(seg_lengths)) if seg_lengths else None,
        "avg_ep_frames": (sum(ep_frames) / len(ep_frames)) if ep_frames else None,
        "segmentation": meta.get("segmentation"),
        "exists": buf_dir.is_dir(),
    }


def fmt(v, nd=2):
    if v is None:
        return "?"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(rows, title):
    print(f"\n===== {title} =====")
    hdr = ["task", "dataset", "ep(实/配)", "segments", "samples", "源帧数",
           "压缩率", "段/ep", "样本/段", "段均帧", "集均帧"]
    print("  ".join(f"{h:<12}" for h in hdr))
    for r in rows:
        ep = f"{r['episodes_processed']}/{r['num_episodes_cfg']}"
        spr = (r["samples"] / r["segments"]) if (r["samples"] and r["segments"]) else None
        cells = [
            r["name"], r["dataset"], ep, fmt(r["segments"]), fmt(r["samples"]),
            fmt(r["source_frames"]), fmt(r["compression"], 4), fmt(r["seg_per_ep"]),
            fmt(spr), fmt(r["avg_seg_frames"], 1), fmt(r["avg_ep_frames"], 1),
        ]
        print("  ".join(f"{c:<12}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description="Replay buffer 规模统计 / 预算公平性核对")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="一个或多个 buffer 根目录（每个下面含 taskA/taskB/...）")
    ap.add_argument("--weights", default="",
                    help='可选目录重复加权, 如 "taskA=2,taskB=1,taskC=3"（D 阶段用）')
    args = ap.parse_args()

    all_rows = {}
    for root in args.roots:
        root_p = Path(os.path.expandvars(root)).expanduser()
        if not root_p.is_dir():
            print(f"[WARN] 目录不存在: {root_p}")
            continue
        subdirs = sorted([d for d in root_p.iterdir() if d.is_dir()
                          and (d / "manifest.jsonl").exists()])
        rows = [collect(d) for d in subdirs]
        all_rows[str(root_p)] = rows
        print_table(rows, f"{root_p}")

    # ---- 分段/配额参数一致性 ----
    print("\n===== 参数一致性核对（分段阈值 / episode 预算 / top_k 应跨任务一致）=====")
    for root, rows in all_rows.items():
        us = {(json.dumps(r["segmentation"], sort_keys=True), r["num_episodes_cfg"], r["top_k"])
              for r in rows}
        if len(us) == 1:
            seg, nep, tk = next(iter(us))
            print(f"[OK] {root}: 全部任务一致 → num_episodes={nep}, top_k={tk}, segmentation={seg}")
        else:
            print(f"[WARN] {root}: 参数不一致（不同任务用了不同设置）:")
            for r in rows:
                print(f"        {r['name']}: num_episodes={r['num_episodes_cfg']} "
                      f"top_k={r['top_k']} seg={r['segmentation']}")
    print("说明: 样本数 = Σ_episode Σ_segment min(top_k, 可选帧数)；"
          "num_episodes 与 top_k 是任务无关的固定配额，"
          "**随任务变化的是每集运动分段数**（见'段/ep'列）——这正是各任务样本数不同的唯一来源。")

    # ---- prototype vs uniform 逐任务预算对照 ----
    if len(all_rows) >= 2:
        keys = list(all_rows.keys())
        print(f"\n===== 逐任务预算对照: {keys[0]}  vs  {keys[1]} =====")
        base = {r["name"]: r for r in all_rows[keys[0]]}
        other = {r["name"]: r for r in all_rows[keys[1]]}
        for name in sorted(set(base) & set(other)):
            a, b = base[name]["samples"], other[name]["samples"]
            flag = "[OK]" if a == b else "[DIFF]"
            print(f"{flag} {name}: {a} vs {b}"
                  + ("" if a == b else f"  (差 {b - a})"))

    # ---- 加权目录的等效复习占比 ----
    if args.weights:
        w = {}
        for kv in args.weights.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                w[k.strip()] = int(v)
        tot = sum(w.values())
        print(f"\n===== D 阶段加权目录的等效复现占比 (--replay_buffer_dirs 重复: {args.weights}) =====")
        for k, v in w.items():
            n = None
            for rows in all_rows.values():
                for r in rows:
                    if r["name"] == k:
                        n = r["samples"]
            print(f"  {k}: 目录出现 {v} 次 → 每轮 replay 占比 {v}/{tot} = {v / tot:.1%}"
                  f"（该 buffer 样本数={n}）")
        print("说明: 训练端每个 buffer 目录 = 一个独立 DataLoader，"
              "轮询策略 round_robin 每次只向其中一个取 1 个 batch\n"
              "      ⇒ **梯度占比由目录出现次数决定，与各 buffer 样本数无关**；"
              "样本数差异只影响该任务被'不同样本'覆盖的多样性，不影响权重。")
    print()


if __name__ == "__main__":
    main()
