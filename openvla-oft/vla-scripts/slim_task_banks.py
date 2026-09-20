#!/usr/bin/env python3
"""
slim_task_banks.py — 给已有的 task bank "瘦身": 把整份 vision_backbone 压成只含 FiLM(scale/shift)

背景:
  `save_task_bank` 旧实现无条件把 **整份** `vision_backbone.state_dict()`
  （≈1.2B 冻结 ViT 参数, bf16 ≈2.4GB）塞进每个 `task_*_bank.pt`；
  而 `load_task_bank` 只用其中的 scale/shift（`_in_film_scope`）→ 99.99% 是死重量。
  一个训练目录通常有 4 个 bank（task_1..4_bank.pt）→ 白白占 ~10GB。

本脚本原地重写 bank，只保留 key 含 scale/shift 的 FiLM 张量:
  · 功能完全不变（loader 本来也只挑这些 key；γ 与 film_scope 能力保留）
  · bank 体积从 ~2.4GB 降到 ~0.2MB
  · 默认先备份为 <name>.full.bak，确认无误后可 --no-backup 或自行删除

用法:
  cd /mnt/data/pengshengdi/openvla-oft
  # 冒烟: 先看能省多少, 不写盘
  python vla-scripts/slim_task_banks.py --roots $LOGS_ROOT/rt_v39b3_taskD--40000_chkpt --dry-run
  # 批量: 传多个 ckpt 目录, 或传 LOGS_ROOT 用 --pattern 匹配
  python vla-scripts/slim_task_banks.py --roots $LOGS_ROOT/rt_v39r3_taskD--40000_ckpt $LOGS_ROOT/rt_v39u_taskD--40000_chkpt
  python vla-scripts/slim_task_banks.py --roots $LOGS_ROOT --pattern "rt_v39*_task*--40000_chkpt"

注意:
  · 只处理含 vision_backbone 且其张量数 > 200 的 bank（已经是 FiLM-only 的会被跳过）
  · 不动 vision_backbone--*_checkpoint.pt（那是 deploy 加载的文件, 缩小会打乱 missing 诊断）
"""

import argparse
import glob
import os
import shutil
import sys

import torch


def is_film(k: str) -> bool:
    return "scale" in k or "shift" in k


def mb(nbytes: float) -> str:
    return f"{nbytes / 1e6:.1f} MB"


def bank_bytes(bank: dict) -> int:
    total = 0
    for v in bank.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            total += sum(t.numel() * t.element_size() for t in v.values() if torch.is_tensor(t))
    return total


def slim_one(path: str, dry_run: bool, backup: bool, drop_film: bool = False) -> tuple:
    bank = torch.load(path, map_location="cpu", weights_only=True)
    before = bank_bytes(bank)
    vb = bank.get("vision_backbone")
    if not isinstance(vb, dict):
        return path, before, before, "skip(no vision_backbone)"
    if drop_film:
        # 整项删除：评估端 γ=0 时本来就不读它；删除后 γ 变 no-op（loader 会显式告警）
        del bank["vision_backbone"]
        after = bank_bytes(bank)
        if dry_run:
            return path, before, after, f"dry-run(drop film, 可省 {100 * (1 - after / before):.2f}%)"
        if backup:
            bak = path + ".full.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
        torch.save(bank, path)
        return path, before, after, f"OK(drop film, 省 {100 * (1 - after / before):.2f}%)"
    film = {k: v for k, v in vb.items() if is_film(k)}
    if len(film) == len(vb):
        return path, before, before, f"skip(已是 FiLM-only, {len(vb)} tensors)"
    if not film:
        return path, before, before, f"WARN(vision_backbone 里没有 scale/shift? {len(vb)} tensors)"

    bank["vision_backbone"] = film
    after = bank_bytes(bank)
    if dry_run:
        return path, before, after, f"dry-run(可省 {100 * (1 - after / before):.2f}%)"

    if backup:
        bak = path + ".full.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
    torch.save(bank, path)
    return path, before, after, f"OK(省 {100 * (1 - after / before):.2f}%)"


def main():
    ap = argparse.ArgumentParser(description="把 task bank 里的整份 vision_backbone 压成 FiLM-only")
    ap.add_argument("--roots", nargs="+", required=True, help="ckpt 目录（或含 ckpt 的父目录 + --pattern）")
    ap.add_argument("--pattern", default="task_*_bank.pt",
                    help='bank 文件名匹配（默认 "task_*_bank.pt"；传父目录时可用 "rt_v39*_task*--40000_chkpt"）')
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .full.bak 备份")
    ap.add_argument("--drop-film", action="store_true",
                    help="整项删除 vision_backbone（不是压成 FiLM-only）——γ=0 口径下本来就不读它")
    args = ap.parse_args()

    paths = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[WARN] 不是目录, 跳过: {root}")
            continue
        direct = sorted(glob.glob(os.path.join(root, "task_*_bank.pt")))
        if direct:
            paths.extend(direct)
        else:
            for d in sorted(glob.glob(os.path.join(root, args.pattern))):
                paths.extend(sorted(glob.glob(os.path.join(d, "task_*_bank.pt"))))

    if not paths:
        print("[FAIL] 没找到任何 task_*_bank.pt —— 检查 --roots/--pattern")
        sys.exit(1)

    print(f"共 {len(paths)} 个 bank 文件{'（dry-run）' if args.dry_run else ''}\n")
    tb = ta = 0
    for p in paths:
        path, before, after, msg = slim_one(p, args.dry_run, not args.no_backup, args.drop_film)
        tb += before
        ta += after
        print(f"  {os.path.relpath(path)}: {mb(before)} → {mb(after)}  [{msg}]")
    print(f"\n合计: {mb(tb)} → {mb(ta)}  (省 {mb(tb - ta)}, {100 * (1 - ta / tb) if tb else 0:.2f}%)")
    if not args.dry_run:
        print("备份文件为 <bank>.full.bak；确认评估结果不变后可删除。")


if __name__ == "__main__":
    main()
