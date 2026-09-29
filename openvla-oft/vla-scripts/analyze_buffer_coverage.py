#!/usr/bin/env python3
"""
analyze_buffer_coverage.py — 对比两套 replay buffer "装了什么帧"（覆盖差异的定量证据）

动机：prototype 与 uniform 在**样本数完全相同**的情况下，保留率差了 0.27（0.70 vs 0.43）。
假设是"两者覆盖的行为相位不同"：机器人轨迹在时间上是**不均匀**的——大部分帧是慢速搬运/接近静止，
少数帧才是抓取/对位/释放等决定性时刻。均匀时间采样会大量落在"低信息帧"上，
而按运动分段挑原型帧则会**按相位均衡覆盖**、偏向"有信息"的帧。

本脚本用 buffer 里已存的 npz 直接量化这一点（**零训练、零 GPU**）：
  · 动作块前向速度 ‖a_{k+1}-a_k‖（运动强度代理）
  · 首步动作幅度 ‖a_0‖
  · 夹爪通道活动 |a[:,6]|、|a[:,13]|（夹爪=关键事件的代理）
  · 分位数分布 + "高活动帧占比"

用法：
  cd /mnt/data/pengshengdi/openvla-oft
  python vla-scripts/analyze_buffer_coverage.py \
      --roots $LOGS_ROOT/replay_buffers $LOGS_ROOT/replay_buffers_uniform \
      --labels prototype uniform --tasks taskA taskB taskC
"""

import argparse
import glob
import json
import os

import numpy as np

# RoboTwin 14D: 左臂 6 关节 + 左夹爪, 右臂 6 关节 + 右夹爪
GRIPPER_IDX = [6, 13]


def sample_stats(npz_path):
    try:
        d = np.load(npz_path, allow_pickle=True)
        a = np.asarray(d["action"], dtype=np.float32)     # [25, 14] 归一化真实 chunk
    except Exception:
        return None
    if a.ndim != 2 or a.shape[0] < 2:
        return None
    vel = np.linalg.norm(np.diff(a, axis=0), axis=-1)      # [24]
    return {
        "fwd_speed_mean": float(vel.mean()),
        "fwd_speed_max": float(vel.max()),
        "a0_norm": float(np.linalg.norm(a[0])),
        "grip_activity": float(np.abs(np.diff(a[:, GRIPPER_IDX], axis=0)).max()),
        "chunk_range": float(np.abs(a.max(axis=0) - a.min(axis=0)).mean()),
    }


def collect(root, task):
    files = sorted(glob.glob(os.path.join(root, task, "samples", "*.npz")))
    rows = [s for s in (sample_stats(f) for f in files) if s]
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=np.float32) for k in keys}, len(files)


def main():
    ap = argparse.ArgumentParser(description="对比两套 replay buffer 的帧覆盖（运动/夹爪活动分布）")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="+", default=["taskA", "taskB", "taskC"])
    ap.add_argument("--hi-percentile", type=float, default=75.0,
                    help="定义'高活动帧'的分位阈值（在 all 上取，默认 75 分位）")
    args = ap.parse_args()

    labels = args.labels if args.labels and len(args.labels) == len(args.roots) \
        else [os.path.basename(r.rstrip("/\\")) for r in args.roots]

    data = {}
    for lab, root in zip(labels, args.roots):
        for t in args.tasks:
            got = collect(root, t)
            if got:
                data[(lab, t)] = got

    if not data:
        print("[FAIL] 没读到任何 samples/*.npz —— 检查 --roots/--tasks")
        return

    # 统一的"高活动"阈值（基于全部样本的 fwd_speed 的 hi 分位）
    all_speed = np.concatenate([v[0]["fwd_speed_mean"] for v in data.values()])
    thr = float(np.percentile(all_speed, args.hi_percentile))

    print(f"\n===== 每套 buffer 的帧覆盖对比（高活动阈值 = fwd_speed 的 {args.hi_percentile:.0f} 分位 = {thr:.4f}）=====\n")
    hdr = ["buffer/task", "n", "speed均值", "speed中位", "speed P90",
           "首步幅度", "夹爪活动均", "高活动帧占比"]
    print("  ".join(f"{h:<13}" for h in hdr))
    for (lab, t), (st, n) in sorted(data.items()):
        hi = float((st["fwd_speed_mean"] > thr).mean())
        cells = [f"{lab}/{t}", str(n),
                 f"{st['fwd_speed_mean'].mean():.4f}",
                 f"{np.median(st['fwd_speed_mean']):.4f}",
                 f"{np.percentile(st['fwd_speed_mean'], 90):.4f}",
                 f"{st['a0_norm'].mean():.4f}",
                 f"{st['grip_activity'].mean():.4f}",
                 f"{hi:.1%}"]
        print("  ".join(f"{c:<13}" for c in cells))

    # 汇总（跨任务合并）
    print("\n===== 跨任务汇总 =====")
    for lab in labels:
        keys = [k for k in data if k[0] == lab]
        if not keys:
            continue
        sp = np.concatenate([data[k][0]["fwd_speed_mean"] for k in keys])
        gp = np.concatenate([data[k][0]["grip_activity"] for k in keys])
        n = sum(data[k][1] for k in keys)
        hi = float((sp > thr).mean())
        print(f"  {lab:<12} n={n:<6} speed均值={sp.mean():.4f}  P90={np.percentile(sp,90):.4f}  "
              f"夹爪活动均={gp.mean():.4f}  P90={np.percentile(gp,90):.4f}  高活动帧占比={hi:.1%}")

    print("\n判读：")
    print("  · 若 prototype 的「高活动帧占比 / 夹爪活动」明显高于 uniform ⇒ 证实'均匀采样偏向低信息帧'，")
    print("    与其'短期够用、长期失守'的实测表现一致（约束都落在容易满足的地方 ⇒ 梯度弱 ⇒ 抗漂移能力差）。")
    print("  · 若两者分布接近 ⇒ 覆盖假设不成立，需另找原因（如帧的时序位置/任务辨识度差异）。")


if __name__ == "__main__":
    main()
