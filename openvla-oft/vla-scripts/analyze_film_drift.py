#!/usr/bin/env python3
"""
analyze_film_drift.py — 量化 FiLM(scale/shift) 在 CL 各阶段之间的漂移量

回答的问题:
  · 每个 stage 训练让 FiLM 漂了多远？（相对 L2 变化 / cos 相似度）
  · 评估任务 K 时的"失配剂量" = 任务 K 训练结束时的 FiLM → D 阶段 FiLM 的距离
    （γ=0 口径下，评估用的就是 D 阶段 FiLM，而任务 K 是在它自己的 FiLM 下学的）
  · 漂移集中在 SigLIP 还是 DINOv2 主干？

数据来源（按优先级自动回退）:
  1) <ckpt>/vision_backbone--<step>_checkpoint.pt   （取 step 最大者）
  2) <ckpt>/task_<K>_bank.pt 里的 "vision_backbone" 条目（仅 film_mode != none 时存在）

用法:
  cd /mnt/data/pengshengdi/openvla-oft
  python vla-scripts/analyze_film_drift.py \
      --ckpts $LOGS_ROOT/rt_v39_taskA--30000_chkpt \
              $LOGS_ROOT/rt_v39b3_taskB--40000_chkpt \
              $LOGS_ROOT/rt_v39b3_taskC--40000_chkpt \
              $LOGS_ROOT/rt_v39b3_taskD--40000_chkpt \
      --names A B C D

  只想知道"评估任务 K 的失配剂量":
  python vla-scripts/analyze_film_drift.py --task-dirs A=$LOGS_ROOT/rt_v39_taskA--30000_chkpt \
      B=$LOGS_ROOT/rt_v39b3_taskB--40000_chkpt C=$LOGS_ROOT/rt_v39b3_taskC--40000_chkpt \
      --eval-dir $LOGS_ROOT/rt_v39b3_taskD--40000_chkpt
"""

import argparse
import glob
import os
import re

import torch


def is_film(k: str) -> bool:
    return "scale" in k or "shift" in k


def group_of(k: str) -> str:
    if "fused_featurizer" in k:
        return "dinov2(fused)"
    if "featurizer" in k:
        return "siglip"
    return "other"


def load_film(ckpt_dir: str):
    """返回 (film_dict, 来源描述)"""
    vb = sorted(glob.glob(os.path.join(ckpt_dir, "vision_backbone--*_checkpoint.pt")))
    if vb:
        def step_of(p):
            m = re.search(r"--(\d+)_checkpoint", p)
            return int(m.group(1)) if m else -1
        p = max(vb, key=step_of)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        film = {k: v.float() for k, v in sd.items() if is_film(k) and v.dtype.is_floating_point}
        if film:
            return film, f"vision_backbone--{step_of(p)}"
    banks = sorted(glob.glob(os.path.join(ckpt_dir, "task_*_bank.pt")))
    for p in banks:
        b = torch.load(p, map_location="cpu", weights_only=True)
        vb_sd = b.get("vision_backbone")
        if isinstance(vb_sd, dict):
            film = {k: v.float() for k, v in vb_sd.items() if is_film(k) and v.dtype.is_floating_point}
            if film:
                return film, os.path.basename(p)
    raise FileNotFoundError(f"{ckpt_dir} 里既无 vision_backbone--*.pt, 也无含 FiLM 的 task_*_bank.pt")


def rel_and_cos(a: dict, b: dict):
    """整体相对 L2 变化 + 逐组统计"""
    tot_a = torch.cat([a[k].flatten() for k in sorted(a)]).norm()
    tot_b = torch.cat([b[k].flatten() for k in sorted(b)]).norm()
    rel = float((tot_b - tot_a).norm() / max(float(tot_a), 1e-12))
    cos = float(torch.dot(tot_a, tot_b) / (tot_a.norm() * tot_b.norm() + 1e-12))
    per_group = {}
    for g in ("siglip", "dinov2(fused)", "other"):
        keys = [k for k in a if k in b and group_of(k) == g]
        if not keys:
            continue
        va = torch.cat([a[k].flatten() for k in keys])
        vb = torch.cat([b[k].flatten() for k in keys])
        per_group[g] = (len(keys), float((vb - va).norm() / max(float(va.norm()), 1e-12)))
    return rel, cos, per_group


def main():
    ap = argparse.ArgumentParser(description="FiLM(scale/shift) 跨阶段漂移量化")
    ap.add_argument("--ckpts", nargs="*", default=[], help="按时间顺序的 ckpt 目录（相邻两两比较 + 与第一个累计比较）")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--task-dirs", nargs="*", default=[],
                    help='形如 A=<dir> B=<dir> ...：算"任务K训练时 FiLM → eval-dir FiLM"的失配剂量')
    ap.add_argument("--eval-dir", default="", help="与 --task-dirs 配合（通常是 D 阶段 ckpt）")
    args = ap.parse_args()

    if args.ckpts:
        names = args.names if args.names and len(args.names) == len(args.ckpts) \
            else [os.path.basename(c.rstrip("/\\")) for c in args.ckpts]
        films = []
        for c, n in zip(args.ckpts, names):
            f, src = load_film(c)
            films.append(f)
            print(f"载入 {n}: {len(f)} 个 FiLM 张量 (来源 {src})")
        print()
        for i in range(len(films) - 1):
            rel, cos, pg = rel_and_cos(films[i], films[i + 1])
            print(f"== {names[i]} → {names[i+1]}: 相对L2变化 {rel:.5f} | cos {cos:.6f}")
            for g, (n, r) in pg.items():
                print(f"     {g:<14} n={n:<4} 相对L2变化 {r:.5f}")
        print("\n########## 相对首个 ckpt 的累计漂移 ##########")
        for i in range(1, len(films)):
            rel, cos, _ = rel_and_cos(films[0], films[i])
            print(f"  {names[0]} → {names[i]}: 相对L2变化 {rel:.5f} | cos {cos:.6f}")

    if args.task_dirs:
        if not args.eval_dir:
            print("[FAIL] --task-dirs 需配合 --eval-dir")
            return
        ev, ev_src = load_film(args.eval_dir)
        print(f"\n########## 评估端失配剂量（任务 K 训练结束时 FiLM → {os.path.basename(args.eval_dir)} 的 {ev_src}）##########")
        print("γ=0 口径下评估用的就是这个 eval FiLM；剂量越大，任务 K 的视觉输入偏移越严重。")
        for kv in args.task_dirs:
            if "=" not in kv:
                continue
            name, d = kv.split("=", 1)
            f, src = load_film(d)
            rel, cos, _ = rel_and_cos(f, ev)
            print(f"  任务 {name}: 相对L2变化 {rel:.5f} | cos {cos:.6f}   (训练时 {src})")
        print("\n读法: 剂量按任务训练顺序递减(A 最大 → D 为 0)。若 A 剂量 ≈ B 剂量但 A 成功率远低，")
        print("      说明差异来自任务容错而非 FiLM 剂量；若剂量逐级递减且成功率同向，则是 FiLM 剂量主导。")


if __name__ == "__main__":
    main()
