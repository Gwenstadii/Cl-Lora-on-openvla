"""
patch_task_banks.py — 给已有 checkpoint 的 task bank 补 FiLM (vision_backbone)

背景: 新版本 save_task_bank 会把每任务 FiLM 直接存进 bank (评估时按 film_gamma 恢复)。
      旧 checkpoint (v39/v39b2/v39f 等) 的 bank 没有 FiLM —— 但每任务的 FiLM 本来就
      存在各任务自己的 checkpoint 里 (vision_backbone--{step}_checkpoint.pt), 无需重训,
      本脚本把它们注入对应 task_N_bank.pt 即可立即启用 FiLM bank 恢复。

用法 (每个任务跑一次, --bank-id 与 --film-source 对应):
  python vla-scripts/patch_task_banks.py \
      --target-ckpt $LOGS_ROOT/rt_v39b2_taskD--40000_chkpt \
      --bank-id 1 --film-source $LOGS_ROOT/rt_v39_taskA--30000_chkpt --film-step 30000
  # bank-id 1→A, 2→B, 3→C, 4→D; film-step: A 用 30000, B/C/D 用 40000

一行跑完 4 个:
  for id in 1 2 3 4; do
    case $id in
      1) SRC=$LOGS_ROOT/rt_v39_taskA--30000_chkpt; STEP=30000;;
      2) SRC=$LOGS_ROOT/rt_v39b2_taskB--40000_chkpt; STEP=40000;;
      3) SRC=$LOGS_ROOT/rt_v39b2_taskC--40000_chkpt; STEP=40000;;
      4) SRC=$LOGS_ROOT/rt_v39b2_taskD--40000_chkpt; STEP=40000;;
    esac
    python vla-scripts/patch_task_banks.py --target-ckpt $LOGS_ROOT/rt_v39b2_taskD--40000_chkpt \
        --bank-id $id --film-source $SRC --film-step $STEP
  done
"""

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Inject FiLM (vision_backbone) into task bank")
    parser.add_argument("--target-ckpt", required=True, help="含 task_N_bank.pt 的 checkpoint 目录")
    parser.add_argument("--bank-id", type=int, required=True, help="要注入的 bank 编号 (1=A,2=B,3=C,4=D)")
    parser.add_argument("--film-source", required=True, help="该任务 FiLM 来源 checkpoint (含 vision_backbone--*.pt)")
    parser.add_argument("--film-step", type=int, default=40000, help="FiLM 来源 step (A 用 30000, 其余 40000)")
    parser.add_argument("--overwrite", action="store_true", help="bank 已有 FiLM 时也覆盖")
    args = parser.parse_args()

    bank_path = os.path.join(args.target_ckpt, f"task_{args.bank_id}_bank.pt")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"bank 不存在: {bank_path}")

    vb_path = os.path.join(args.film_source, f"vision_backbone--{args.film_step}_checkpoint.pt")
    if not os.path.exists(vb_path):
        raise FileNotFoundError(f"FiLM checkpoint 不存在: {vb_path}")

    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    if "vision_backbone" in bank and not args.overwrite:
        print(f"[SKIP] {bank_path} 已有 vision_backbone, 加 --overwrite 覆盖")
        return

    vb_sd = torch.load(vb_path, map_location="cpu", weights_only=True)
    bank["vision_backbone"] = vb_sd
    torch.save(bank, bank_path)
    print(f"[OK] task_{args.bank_id}_bank.pt ← FiLM from {vb_path}")
    print(f"     bank 现有 {len(bank)} 个 key (含 vision_backbone {len(vb_sd)} tensors)")


if __name__ == "__main__":
    main()
