#!/usr/bin/env python3
"""
patch_bank_specific_a.py — 给已有 task bank 补 specific-A 快照（lora_a）

为什么需要:
  `--bank_save_specific_a True` 只对**本次训练写出的** bank 生效。
  链式训练里 `task_1_bank.pt` 是从 stage-1 ckpt（如 rt_v39_taskA--30000_chkpt）**继承**来的，
  那份 bank 由旧代码写出 ⇒ **不含 lora_a** ⇒ 评估任务 A 时仍会用漂移后的 A_final（A 依旧崩）。
  本工具从源 ckpt 的 `cl_lora_adapter.pt`（LLM 侧）+ `action_head--*.pt`（动作头侧）取出
  specific 层的 A，补写进指定 bank。

用法:
  # 补单个 bank（通常只需补继承来的 task_1_bank.pt）
  python vla-scripts/patch_bank_specific_a.py \
      --bank $LOGS_ROOT/rt_v39r4_taskD--40000_chkpt/task_1_bank.pt \
      --src-ckpt $LOGS_ROOT/rt_v39_taskA--30000_chkpt \
      --vla-step 30000 --ah-step 30000

  # 一次补一个 ckpt 目录里的所有 bank（会跳过已含 lora_a 的）
  python vla-scripts/patch_bank_specific_a.py --ckpt-dir $LOGS_ROOT/rt_v39r4_taskD--40000_chkpt \
      --src-ckpt $LOGS_ROOT/rt_v39_taskA--30000_chkpt --vla-step 30000 --ah-step 30000

注意:
  · 只补 **non-shared（specific）** 层：shared 层 A 是永久冻结的共享参数，不属于任何任务的 bank；
  · specific 层范围从源 ckpt 的 `cl_lora_config.json` 推断（first_lora_layer + shared_depth）；
  · 默认先备份为 <bank>.noA.bak；`--no-backup` 可关。
"""

import argparse
import glob
import json
import os
import re
import shutil

import torch


def load_cl_cfg(ckpt_dir: str) -> dict:
    p = os.path.join(ckpt_dir, "cl_lora_config.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {"first_lora_layer": 16, "shared_depth": 8}


def specific_layer_range(cfg: dict):
    first = int(cfg.get("first_lora_layer", 16))
    shared_depth = int(cfg.get("shared_depth", 8))
    denom = max(1, 32 - first)
    ratio = cfg.get("shared_split_ratio", None)
    if ratio is not None:
        shared = max(1, round(float(ratio) * denom))
    else:
        shared = shared_depth
    return first + shared, 32  # [lo, hi)


def collect_from_vla(ckpt_dir: str, lo: int, hi: int) -> dict:
    """从 cl_lora_adapter.pt 取 specific 层的 lora_a（bank key 命名: 点换下划线）。"""
    p = os.path.join(ckpt_dir, "cl_lora_adapter.pt")
    if not os.path.isfile(p):
        print(f"[WARN] 缺少 {p}（LLM 侧 A 无法补）")
        return {}
    sd = torch.load(p, map_location="cpu", weights_only=False)
    out = {}
    for k, v in sd.items():
        if not k.endswith(".lora_a"):
            continue
        m = re.search(r"layers\.(\d+)\.", k)
        if not m:
            continue
        if not (lo <= int(m.group(1)) < hi):
            continue
        mod = k[: -len(".lora_a")]
        out[f"{mod.replace('.', '_')}.lora_a"] = v.cpu().clone()
    return out


def collect_from_action_head(ckpt_dir: str, ah_step: int = None) -> dict:
    pats = [os.path.join(ckpt_dir, f"action_head--{ah_step}_checkpoint.pt")] if ah_step \
        else sorted(glob.glob(os.path.join(ckpt_dir, "action_head--*_checkpoint.pt")))
    p = next((x for x in pats if os.path.isfile(x)), None)
    if p is None:
        print(f"[WARN] {ckpt_dir} 里没有 action_head--*.pt（动作头侧 A 无法补）")
        return {}
    sd = torch.load(p, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if not k.endswith(".lora_a"):
            continue
        mod = k[: -len(".lora_a")]
        out[f"action_head.{mod.replace('.', '_')}.lora_a"] = v.cpu().clone()
    print(f"[INFO] 动作头 A 来源: {os.path.basename(p)}（{len(out)} 个张量）")
    return out


def patch_one(bank_path: str, ckpt_dir: str, vla_step: int, ah_step: int, dry_run: bool,
              backup: bool, overwrite: bool):
    cfg = load_cl_cfg(ckpt_dir)
    lo, hi = specific_layer_range(cfg)
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    have = sum(1 for k in bank if k.endswith(".lora_a"))
    if have and not overwrite:
        print(f"[SKIP] {os.path.basename(bank_path)} 已含 {have} 个 lora_a（--overwrite 可覆盖）")
        return
    adds = {}
    adds.update(collect_from_vla(ckpt_dir, lo, hi))
    adds.update(collect_from_action_head(ckpt_dir, ah_step if ah_step else None))
    if not adds:
        print(f"[FAIL] {bank_path}: 没取到任何 A（检查 --src-ckpt / --vla-step）")
        return
    bank.update(adds)
    print(f"[OK] {os.path.basename(bank_path)}: 补入 {len(adds)} 个 lora_a "
          f"(specific L{lo}-{hi-1} + action_head), bank key 数 {len(bank)}")
    if dry_run:
        print("     (dry-run, 未写盘)")
        return
    if backup:
        bak = bank_path + ".noA.bak"
        if not os.path.exists(bak):
            shutil.copy2(bank_path, bak)
    torch.save(bank, bank_path)


def main():
    ap = argparse.ArgumentParser(description="给 task bank 补 specific-A 快照")
    ap.add_argument("--bank", default="", help="单个 bank 文件路径")
    ap.add_argument("--ckpt-dir", default="", help="ckpt 目录（对其下所有 task_*_bank.pt 生效）")
    ap.add_argument("--src-ckpt", required=True, help="A 快照来源 ckpt 目录（如 stage-1 的 rt_v39_taskA）")
    ap.add_argument("--vla-step", type=int, default=0, help="LLM 侧步数（仅用于日志；adapter 文件不带步数）")
    ap.add_argument("--ah-step", type=int, default=0, help="动作头 checkpoint 步数（0=自动取最新）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.bank:
        patch_one(args.bank, args.src_ckpt, args.vla_step, args.ah_step,
                  args.dry_run, not args.no_backup, args.overwrite)
    elif args.ckpt_dir:
        banks = sorted(glob.glob(os.path.join(args.ckpt_dir, "task_*_bank.pt")))
        if not banks:
            print(f"[FAIL] {args.ckpt_dir} 下没有 task_*_bank.pt")
            return
        for b in banks:
            patch_one(b, args.src_ckpt, args.vla_step, args.ah_step,
                      args.dry_run, not args.no_backup, args.overwrite)
    else:
        print("[FAIL] 需要 --bank 或 --ckpt-dir")


if __name__ == "__main__":
    main()
