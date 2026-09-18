#!/usr/bin/env python3
"""
analyze_cl_lora_drift.py — 量化 CL-LoRA 各阶段之间 lora_a / lora_b 的漂移量

用途（回答 r3/b4 这类"A 漂移支线"为什么旧任务塌）:
  · A 在各 stage 之间到底漂了多远？（cos 相似度 + 相对 L2 变化）
  · shared(L16-23) 与 specific(L24-31) 分开看（shared A/B 应完全不动）
  · specific-A 逐层列出，定位漂移最重的层

背景机制: task bank **只存 specific-B + block_scale**（`save_task_bank`，cl_lora.py:283），
不存 specific-A。评估旧任务 K 时，模型算的是 W + s·g_K·B_K·A_final，
而训练时是 W + s·g_K·B_K·A_K。⇒ A 漂移量 = 旧任务恢复误差的下界。

用法:
  cd /mnt/data/pengshengdi/openvla-oft
  python vla-scripts/analyze_cl_lora_drift.py \
      --ckpts $LOGS_ROOT/rt_v39_taskA--30000_chkpt \
              $LOGS_ROOT/rt_v39r3_taskB--40000_chkpt \
              $LOGS_ROOT/rt_v39r3_taskC--40000_chkpt \
              $LOGS_ROOT/rt_v39r3_taskD--40000_chkpt \
      --names A B C D
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import torch

_WITH_ACTION_HEAD = False   # 由 --with-action-head 打开


def load_adapter(ckpt_dir: str) -> dict:
    """载入该 ckpt 的 CL-LoRA 参数：
      · LLM 侧: cl_lora_adapter.pt
      · 动作头侧: action_head--<step>_checkpoint.pt（key 加 "action_head." 前缀，需 --with-action-head）
    动作头侧必须单独读 —— 它不在 cl_lora_adapter.pt 里（那份只含 vla.module 的 key）。
    """
    out = {}
    p = os.path.join(ckpt_dir, "cl_lora_adapter.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到 {p}（确认这是 CL-LoRA 的 ckpt 目录）")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    out.update({k: v.float() for k, v in sd.items() if v.dtype.is_floating_point})

    if _WITH_ACTION_HEAD:
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "action_head--*_checkpoint.pt")))
        if cands:
            def step_of(x):
                m = re.search(r"--(\d+)_checkpoint", x)
                return int(m.group(1)) if m else -1
            ah = torch.load(max(cands, key=step_of), map_location="cpu", weights_only=True)
            for k, v in ah.items():
                if any(t in k for t in ("lora_a", "lora_b", "block_scale")) and v.dtype.is_floating_point:
                    out[f"action_head.{k}"] = v.float()
    return out


def layer_of(key: str):
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def group_of(key: str, shared_lo=16, shared_hi=23) -> str:
    """返回 shared / specific / other / action_head"""
    if key.startswith("action_head"):
        return "action_head"
    L = layer_of(key)
    if L is None:
        return "other"
    if shared_lo <= L <= shared_hi:
        return "shared"
    if L > shared_hi:
        return "specific"
    return "other"


def param_of(key: str) -> str:
    for p in ("lora_a", "lora_b", "block_scale"):
        if key.endswith(p):
            return p
    return "?"


def proj_of(key: str) -> str:
    parts = key.split(".")
    return parts[-2] if len(parts) >= 2 else "?"


def drift(a: torch.Tensor, b: torch.Tensor):
    """返回 (cos, 相对L2变化, a范数, b范数)"""
    fa, fb = a.flatten(), b.flatten()
    na, nb = fa.norm(), fb.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan"), float("nan"), float(na), float(nb)
    cos = float(torch.dot(fa, fb) / (na * nb))
    rel = float((fb - fa).norm() / na)
    return cos, rel, float(na), float(nb)


def compare(sd_a: dict, sd_b: dict, name_a: str, name_b: str):
    common = sorted(set(sd_a) & set(sd_b))
    if not common:
        print(f"[WARN] {name_a} 与 {name_b} 没有公共 key")
        return
    agg = defaultdict(list)          # (group, param) -> [(cos, rel)]
    per_layer_a = defaultdict(list)  # layer -> [(cos, rel)]  (specific lora_a)
    for k in common:
        cos, rel, _, _ = drift(sd_a[k], sd_b[k])
        g, p = group_of(k), param_of(k)
        agg[(g, p)].append((cos, rel))
        if g == "specific" and p == "lora_a":
            per_layer_a[layer_of(k)].append((cos, rel))

    print(f"\n===== {name_a} → {name_b} 漂移（{len(common)} 个张量）=====")
    print(f"{'分组':<14}{'参数':<13}{'张量数':<8}{'cos 均值':<11}{'cos 最小':<11}{'相对L2变化 均值':<16}")
    for (g, p) in sorted(agg):
        vals = agg[(g, p)]
        cos = [c for c, _ in vals if c == c]
        rel = [r for _, r in vals if r == r]
        if not cos:
            continue
        print(f"{g:<14}{p:<13}{len(vals):<8}{sum(cos)/len(cos):<11.5f}{min(cos):<11.5f}"
              f"{sum(rel)/len(rel):<16.5f}")

    if per_layer_a:
        print(f"  -- specific lora_a 逐层 --")
        for L in sorted(per_layer_a):
            vals = per_layer_a[L]
            cos = [c for c, _ in vals if c == c]
            rel = [r for _, r in vals if r == r]
            if not cos:
                continue
            print(f"     layer {L}: cos={sum(cos)/len(cos):.5f}  relL2={sum(rel)/len(rel):.5f}  (n={len(vals)})")

    # 判定
    spec_a = agg.get(("specific", "lora_a"), [])
    shared_a = agg.get(("shared", "lora_a"), [])
    if spec_a:
        mc = sum(c for c, _ in spec_a if c == c) / max(1, len([c for c, _ in spec_a if c == c]))
        verdict = ("几乎未动" if mc > 0.999 else "轻微" if mc > 0.99
                   else "显著" if mc > 0.9 else "严重（bank 恢复必然失效）")
        print(f"  判定: specific-A 平均 cos={mc:.5f} → {verdict}")
        print(f"        ⇒ 旧任务 K 的恢复误差下界 ≈ {1-mc:.1%}（bank 只存 B，无法补偿 A 的漂移）")
    if shared_a:
        ms = sum(c for c, _ in shared_a if c == c) / max(1, len([c for c, _ in shared_a if c == c]))
        flag = "[OK] 未漂移，符合预期" if ms > 0.9999 else "[WARN] shared-A 竟然动了（应为永久冻结）"
        print(f"        shared-A 平均 cos={ms:.6f} {flag}")


def main():
    ap = argparse.ArgumentParser(description="CL-LoRA 跨阶段漂移量化（lora_a / lora_b / block_scale）")
    ap.add_argument("--ckpts", nargs="+", required=True, help="按时间顺序排列的 ckpt 目录（≥2）")
    ap.add_argument("--names", nargs="*", default=None, help="对应的阶段名（如 A B C D）")
    ap.add_argument("--pair-only", action="store_true",
                    help="只比较相邻两两，不与第一个 ckpt 累计比较")
    ap.add_argument("--with-action-head", action="store_true",
                    help="同时读动作头 action_head--*.pt 的 lora_a/lora_b/block_scale（动作头 A 是当前重点嫌疑）")
    args = ap.parse_args()

    global _WITH_ACTION_HEAD
    _WITH_ACTION_HEAD = args.with_action_head

    names = args.names if args.names and len(args.names) == len(args.ckpts) \
        else [os.path.basename(c.rstrip("/\\")) for c in args.ckpts]
    sds = [load_adapter(c) for c in args.ckpts]
    print(f"载入 {len(sds)} 个 ckpt: " + ", ".join(f"{n}({len(s)} tensors)" for n, s in zip(names, sds)))

    for i in range(len(sds) - 1):
        compare(sds[i], sds[i + 1], names[i], names[i + 1])

    if not args.pair_only and len(sds) > 2:
        print("\n########## 相对初始 ckpt 的累计漂移（bank 恢复误差的直接来源）##########")
        for i in range(1, len(sds)):
            compare(sds[0], sds[i], names[0], names[i])


if __name__ == "__main__":
    main()
