"""
build_uniform_replay_buffer_robotwin.py — Uniform-temporal replay buffer for RoboTwin (CL-LoRA 消融)

与 prototype replay (build_replay_buffer_robotwin.py) 的严格对照:
  - 相同的 RLDS 加载 (tfds episode 级)、相同的任务/相机/proprio/action-chunk/归一化
  - 相同的输出格式 (samples/*.npz + manifest.jsonl + meta.json, replay_dataset 直接可用)
  - 相同的**样本预算** (--match-budget-buffer-dir 读原型 buffer 的 manifest 行数)
  - **唯一差异 = 选帧规则**: 均匀时间采样, 而非"物理分段 + 原型 Top-K"

用法 (服务器, openvla-oft 目录下):
  python vla-scripts/build_uniform_replay_buffer_robotwin.py \
    --data-root-dir datasets/rlds \
    --dataset-name aloha_handover_mic_clean \
    --output-dir $LOGS_ROOT/replay_buffers_uniform/taskA \
    --stats-path $LOGS_ROOT/rt_v39_taskA--30000_chkpt/dataset_statistics.json \
    --match-budget-buffer-dir $LOGS_ROOT/replay_buffers/taskA \
    --num-episodes 10 --overwrite
"""

import argparse
import json
import os
import pathlib
import shutil
from typing import List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class UniformReplayConfig:
    data_root_dir: str = "datasets/rlds"
    dataset_name: str = "aloha_handover_mic_clean"
    output_dir: str = ""
    stats_path: str = ""
    match_budget_buffer_dir: Optional[str] = None
    target_num_samples: Optional[int] = None
    num_episodes: int = 10
    overwrite: bool = False


def _count_manifest(buffer_dir: str) -> int:
    p = pathlib.Path(buffer_dir) / "manifest.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"参考 manifest 不存在: {p}")
    n = sum(1 for line in p.open("r", encoding="utf-8") if line.strip())
    if n <= 0:
        raise ValueError(f"参考 manifest 为空: {p}")
    return n


def _get_language_instruction(step) -> str:
    candidates = []
    if "language_instruction" in step:
        candidates.append(step["language_instruction"])
    if "task" in step and isinstance(step["task"], dict):
        candidates.append(step["task"].get("language_instruction"))
    obs = step.get("observation", {})
    if isinstance(obs, dict) and "natural_language_instruction" in obs:
        candidates.append(obs["natural_language_instruction"])
    for raw in candidates:
        if raw is None:
            continue
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        if isinstance(raw, np.ndarray):
            raw = raw.item() if raw.size == 1 else raw[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw:
            return str(raw)
    raise KeyError(f"找不到语言指令, step keys: {list(step.keys())}")


def _normalize_bounds(values: np.ndarray, low, high) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    out = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    out[..., low == high] = 0.0
    return out.astype(np.float32)


def _load_stats(stats_path: str, dataset_name: str):
    if not stats_path or not os.path.isfile(stats_path):
        print(f"[WARN] stats 缺失: {stats_path!r} —— 不归一化")
        return None, None
    with open(stats_path) as f:
        stats = json.load(f)
    if dataset_name not in stats:
        print(f"[WARN] stats 无 {dataset_name} (keys={sorted(stats.keys())})")
        return None, None
    s = stats[dataset_name]
    a = {k: np.asarray(v, dtype=np.float64) for k, v in s.get("action", {}).items()}
    p = {k: np.asarray(v, dtype=np.float64) for k, v in s.get("proprio", {}).items()}
    print(f"[OK] 归一化统计已加载 (action/proprio min/max)")
    return a, p


def build(cfg: UniformReplayConfig) -> None:
    out_dir = pathlib.Path(cfg.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not cfg.overwrite:
            raise FileExistsError(f"{out_dir} 非空, 加 --overwrite")
        shutil.rmtree(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # 样本预算: 与原型 buffer 对齐
    if cfg.match_budget_buffer_dir:
        budget = _count_manifest(cfg.match_budget_buffer_dir)
        print(f"[Budget] 匹配原型 buffer 预算: {budget} samples (来自 {cfg.match_budget_buffer_dir})")
    elif cfg.target_num_samples:
        budget = int(cfg.target_num_samples)
        print(f"[Budget] 指定预算: {budget} samples")
    else:
        raise ValueError("需要 --match-budget-buffer-dir 或 --target-num-samples")

    action_stats, proprio_stats = _load_stats(cfg.stats_path, cfg.dataset_name)

    print(f"[Data] 加载 RLDS: {cfg.dataset_name}")
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    ds = builder.as_dataset(split="all")

    # 第一遍: 收集 episodes (帧数 + 语言指令)
    episodes = []
    for ep_idx, episode in enumerate(ds):
        if ep_idx >= cfg.num_episodes:
            break
        steps = list(episode["steps"])
        if len(steps) < NUM_ACTIONS_CHUNK + 1:
            continue
        episodes.append((ep_idx, steps, _get_language_instruction(steps[0])))
    if not episodes:
        raise RuntimeError("没有可用 episode")
    total_frames = sum(len(s) for _, s, _ in episodes)
    print(f"[Data] {len(episodes)} episodes, 总帧数 {total_frames}, 预算 {budget}")

    # 按 episode 长度比例分配预算, 每 episode 内均匀取样
    counts = []
    remaining = budget
    for i, (_, steps, _) in enumerate(episodes):
        if i == len(episodes) - 1:
            n = remaining
        else:
            n = max(1, int(round(budget * len(steps) / total_frames)))
            n = min(n, remaining - (len(episodes) - 1 - i))
        counts.append(max(1, n))
        remaining -= counts[-1]
    print(f"[Plan] 每 episode 采样数: {counts} (合计 {sum(counts)})")

    segment_count = 0
    sample_count = 0
    seg_manifest = out_dir / "segments.jsonl"
    samp_manifest = out_dir / "manifest.jsonl"

    with seg_manifest.open("w", encoding="utf-8") as seg_f, \
         samp_manifest.open("w", encoding="utf-8") as samp_f:
        for (ep_idx, steps, lang), n_pick in zip(episodes, counts):
            T = len(steps)
            # 均匀时间采样 (含端点, 去重)
            frame_ids = sorted(set(int(round(t)) for t in np.linspace(0, T - 1, n_pick)))
            print(f"  episode {ep_idx}: T={T}, task={lang!r}, 采样 {len(frame_ids)} 帧")

            seg_f.write(json.dumps({
                "segment_id": segment_count, "episode_index": ep_idx, "task": lang,
                "num_frames": T, "selected_frame_indices": frame_ids, "selection": "uniform",
            }, ensure_ascii=False) + "\n")

            for rank, fidx in enumerate(frame_ids):
                s = steps[fidx]
                obs = s["observation"]
                state = np.asarray(obs["state"].numpy(), dtype=np.float32)          # [14]
                if state.shape[-1] != 14:
                    raise RuntimeError(f"state 维度异常: {state.shape}")
                proprio = (_normalize_bounds(state, proprio_stats["min"], proprio_stats["max"])
                           if proprio_stats and "min" in proprio_stats else state)
                act = np.asarray(s["action"].numpy(), dtype=np.float32)             # [14]
                act = (_normalize_bounds(act, action_stats["min"], action_stats["max"])
                       if action_stats else act)
                # 真实未来 chunk [25, 14]
                acts = np.asarray([np.asarray(steps[min(fidx + k, T - 1)]["action"].numpy(),
                                              dtype=np.float32) for k in range(NUM_ACTIONS_CHUNK)])
                acts = (_normalize_bounds(acts, action_stats["min"], action_stats["max"])
                        if action_stats else acts)

                sp = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                np.savez_compressed(
                    sp,
                    image_primary=obs["image"].numpy(),
                    left_wrist_image=obs["left_wrist_image"].numpy(),
                    right_wrist_image=obs["right_wrist_image"].numpy(),
                    proprio=proprio,
                    action=acts,
                    task=lang,
                    dataset_name=cfg.dataset_name,
                )
                samp_f.write(json.dumps({
                    "sample_id": sample_count, "sample_path": str(sp.relative_to(out_dir)),
                    "task": lang, "episode_index": ep_idx, "episode_frame_index": int(fidx),
                    "segment_id": segment_count, "coverage_gain": 0.0,
                }, ensure_ascii=False) + "\n")
                sample_count += 1
            segment_count += 1

    meta = {
        "format": "openvla_uniform_replay_v1_robotwin",
        "dataset": cfg.dataset_name,
        "saved_replay_samples": sample_count,
        "budget_source": cfg.match_budget_buffer_dir or f"target={cfg.target_num_samples}",
        "num_episodes": len(episodes),
        "source_frames": total_frames,
        "compression_ratio": float(sample_count) / max(1, total_frames),
        "selection": "uniform_temporal",
        "per_episode_counts": counts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[Done] {sample_count} samples → {out_dir} (uniform temporal sampling)")


def main():
    parser = argparse.ArgumentParser(description="Uniform replay buffer builder (RoboTwin)")
    parser.add_argument("--data-root-dir", default="datasets/rlds")
    parser.add_argument("--dataset-name", default="aloha_handover_mic_clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--match-budget-buffer-dir", default=None)
    parser.add_argument("--target-num-samples", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = UniformReplayConfig()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    build(cfg)


if __name__ == "__main__":
    main()
