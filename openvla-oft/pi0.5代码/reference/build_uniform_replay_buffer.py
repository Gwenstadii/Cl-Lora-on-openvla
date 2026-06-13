from __future__ import annotations

"""Build an offline replay buffer with uniform temporal sampling.

This script is designed as a same-budget ordinary replay baseline for the
prototype replay pipeline in ``scripts/build_replay_buffer.py``.

Fair-comparison principles:
1) Reuse the same task filtering and first-K episode selection logic.
2) Match the total number of stored replay samples to a reference prototype
   buffer (or to an explicitly provided target sample count).
3) Keep the saved sample structure compatible with the existing replay loader:
   ``manifest.jsonl`` + ``samples/*.npz`` + ``meta.json``.
4) Only change the frame selection rule: uniform temporal sampling instead of
   prototype-based Top-K selection.
"""

import dataclasses
import json
import logging
import pathlib
from typing import Any, Literal

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

import openpi.training.config as _config

import build_replay_buffer as _proto


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildUniformReplayBufferConfig:
    # Training config name used to resolve dataset defaults.
    train_config_name: str

    # Output root folder. A task-specific subfolder is created inside.
    output_dir: str = "./replay_buffers_uniform"
    overwrite: bool = False

    # Optional dataset/task overrides. If None, use config defaults.
    dataset_root: str | None = None
    target_task_name: str | None = None
    max_episodes: int | None = None

    # Replay sample budget. Exactly one of these should be set.
    target_num_samples: int | None = None
    match_budget_buffer_dir: str | None = None
    reference_manifest_filename: str = "manifest.jsonl"

    # Allocation policy inside the selected episodes.
    allocation_strategy: Literal["proportional_by_episode_length"] = "proportional_by_episode_length"
    ensure_one_per_episode_if_possible: bool = True


@dataclasses.dataclass(frozen=True)
class _EpisodeInfo:
    episode_index: int
    task: str
    start: int
    end: int

    @property
    def num_frames(self) -> int:
        return self.end - self.start


def _count_manifest_records(buffer_dir: str | pathlib.Path, manifest_filename: str) -> int:
    manifest_path = pathlib.Path(buffer_dir) / manifest_filename
    if not manifest_path.exists():
        raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")

    count = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    if count <= 0:
        raise ValueError(f"Reference manifest is empty: {manifest_path}")
    return count


def _resolve_target_num_samples(cfg: BuildUniformReplayBufferConfig) -> tuple[int, str]:
    has_explicit = cfg.target_num_samples is not None
    has_reference = cfg.match_budget_buffer_dir is not None
    if has_explicit == has_reference:
        raise ValueError(
            "Set exactly one of target_num_samples or match_budget_buffer_dir. "
            f"Got target_num_samples={cfg.target_num_samples!r}, "
            f"match_budget_buffer_dir={cfg.match_budget_buffer_dir!r}."
        )

    if has_explicit:
        target_num_samples = int(cfg.target_num_samples)
        if target_num_samples <= 0:
            raise ValueError(f"target_num_samples must be > 0, got {target_num_samples}")
        return target_num_samples, "explicit"

    assert cfg.match_budget_buffer_dir is not None
    target_num_samples = _count_manifest_records(cfg.match_budget_buffer_dir, cfg.reference_manifest_filename)
    return target_num_samples, "matched_from_reference_buffer"


def _distribute_with_caps(weights: list[float], capacities: list[int], total: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if len(weights) != len(capacities):
        raise ValueError("weights and capacities must have the same length")
    if not capacities:
        return []

    total_capacity = int(sum(capacities))
    if total > total_capacity:
        raise ValueError(f"Cannot distribute {total} samples with total capacity {total_capacity}.")
    if total == 0:
        return [0 for _ in capacities]

    safe_weights = [float(w) if cap > 0 else 0.0 for w, cap in zip(weights, capacities, strict=True)]
    if sum(safe_weights) <= 0.0:
        safe_weights = [1.0 if cap > 0 else 0.0 for cap in capacities]

    weight_sum = sum(safe_weights)
    raw = [total * w / weight_sum for w in safe_weights]
    alloc = [min(int(np.floor(x)), cap) for x, cap in zip(raw, capacities, strict=True)]
    remaining = total - sum(alloc)

    while remaining > 0:
        best_idx = None
        best_key = None
        for i, (a, cap, r, w) in enumerate(zip(alloc, capacities, raw, safe_weights, strict=True)):
            if a >= cap:
                continue
            frac = r - np.floor(r)
            key = (float(frac), float(w), int(cap - a), -i)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = i
        if best_idx is None:
            raise RuntimeError("Failed to allocate the full replay budget.")
        alloc[best_idx] += 1
        remaining -= 1

    return alloc


def _allocate_episode_sample_counts(
    lengths: list[int],
    total_samples: int,
    *,
    ensure_one_per_episode_if_possible: bool,
) -> list[int]:
    if total_samples <= 0:
        raise ValueError(f"total_samples must be > 0, got {total_samples}")
    if not lengths:
        return []
    if any(length < 0 for length in lengths):
        raise ValueError(f"Episode lengths must be non-negative, got {lengths}")

    total_frames = int(sum(lengths))
    if total_frames <= 0:
        raise ValueError("No valid frames available to build replay buffer.")

    if total_samples >= total_frames:
        return list(lengths)

    num_non_empty = sum(1 for x in lengths if x > 0)
    if ensure_one_per_episode_if_possible and total_samples >= num_non_empty:
        base = [1 if length > 0 else 0 for length in lengths]
        remaining = total_samples - sum(base)
        capacities = [max(length - b, 0) for length, b in zip(lengths, base, strict=True)]
        extras = _distribute_with_caps(lengths, capacities, remaining)
        alloc = [b + e for b, e in zip(base, extras, strict=True)]
    else:
        alloc = _distribute_with_caps(lengths, lengths, total_samples)

    expected = min(total_samples, total_frames)
    if sum(alloc) != expected:
        raise RuntimeError(f"Allocated {sum(alloc)} samples, expected {expected}.")
    if any(a > l for a, l in zip(alloc, lengths, strict=True)):
        raise RuntimeError("Allocated sample count exceeds episode length.")
    return alloc


def _uniform_frame_indices(num_frames: int, num_samples: int) -> list[int]:
    if num_frames <= 0 or num_samples <= 0:
        return []
    if num_samples >= num_frames:
        return list(range(num_frames))

    ideal_positions = ((np.arange(num_samples, dtype=np.float64) + 0.5) * num_frames / num_samples) - 0.5
    used: set[int] = set()
    result: list[int] = []

    for pos in ideal_positions:
        base = int(np.clip(round(float(pos)), 0, num_frames - 1))
        if base not in used:
            used.add(base)
            result.append(base)
            continue

        found = None
        for delta in range(1, num_frames):
            left = base - delta
            right = base + delta
            if left >= 0 and left not in used:
                found = left
                break
            if right < num_frames and right not in used:
                found = right
                break
        if found is None:
            raise RuntimeError("Failed to find a unique uniformly sampled frame index.")
        used.add(found)
        result.append(found)

    result.sort()
    if len(result) != num_samples:
        raise RuntimeError(f"Expected {num_samples} sampled frames, got {len(result)}")
    return result


def _gather_episode_infos(dataset: Any, episode_indices: list[int]) -> tuple[list[_EpisodeInfo], int]:
    infos: list[_EpisodeInfo] = []
    skipped_empty_episodes = 0
    for ep_idx in episode_indices:
        ep_from = int(dataset.episode_data_index["from"][ep_idx].item())
        ep_to = int(dataset.episode_data_index["to"][ep_idx].item())
        if ep_to <= ep_from:
            skipped_empty_episodes += 1
            continue
        infos.append(
            _EpisodeInfo(
                episode_index=ep_idx,
                task=_proto._get_episode_task(dataset, ep_idx),
                start=ep_from,
                end=ep_to,
            )
        )
    return infos, skipped_empty_episodes


def main(cfg: BuildUniformReplayBufferConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.set_printoptions(precision=4, suppress=True)

    train_cfg = _config.get_config(cfg.train_config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.repo_id is None:
        raise ValueError("Data config has no repo_id; cannot build replay buffer.")

    dataset_root = cfg.dataset_root if cfg.dataset_root is not None else data_cfg.root
    target_task_name = cfg.target_task_name if cfg.target_task_name is not None else data_cfg.target_task_name
    task_slug = _proto._slugify(target_task_name if target_task_name else "all_tasks")

    out_dir = pathlib.Path(cfg.output_dir).resolve() / f"{cfg.train_config_name}__{task_slug}"
    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}. Use --overwrite True to replace it.")
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    target_num_samples, budget_source = _resolve_target_num_samples(cfg)

    LOGGER.info("Loading dataset repo_id=%s root=%s", data_cfg.repo_id, dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(data_cfg.repo_id, root=dataset_root)
    episode_indices = _proto._select_episode_indices(dataset, target_task_name)
    if cfg.max_episodes is not None:
        episode_indices = episode_indices[: cfg.max_episodes]
    if not episode_indices:
        raise ValueError(f"No episodes found for target_task_name={target_task_name!r}.")

    episode_infos, skipped_empty_episodes = _gather_episode_infos(dataset, episode_indices)
    if not episode_infos:
        raise ValueError("No non-empty episodes available after filtering.")

    episode_lengths = [info.num_frames for info in episode_infos]
    num_candidate_frames = int(sum(episode_lengths))
    alloc_counts = _allocate_episode_sample_counts(
        episode_lengths,
        target_num_samples,
        ensure_one_per_episode_if_possible=cfg.ensure_one_per_episode_if_possible,
    )

    LOGGER.info("Selected episodes: %d / %d", len(episode_infos), int(dataset.num_episodes))
    LOGGER.info(
        "Uniform replay target budget: %d samples from %d candidate frames",
        int(sum(alloc_counts)),
        num_candidate_frames,
    )

    task_to_id: dict[str, int] = {}
    sample_manifest_path = out_dir / "manifest.jsonl"
    episode_manifest_path = out_dir / "segments.jsonl"

    sample_count = 0
    group_count = 0

    with sample_manifest_path.open("w", encoding="utf-8") as sample_f, episode_manifest_path.open(
        "w", encoding="utf-8"
    ) as group_f:
        iterator = zip(episode_infos, alloc_counts, strict=True)
        for info, num_select in tqdm.tqdm(iterator, total=len(episode_infos), desc="Episodes"):
            if num_select <= 0:
                continue

            task_id = task_to_id.setdefault(info.task, len(task_to_id))
            prompt = info.task
            frame_ids = _uniform_frame_indices(info.num_frames, num_select)

            frames = []
            for gidx in range(info.start, info.end):
                sample = dataset[gidx]
                frames.append(_proto._extract_frame_record(sample, fallback_task=info.task))
            actions = [frame.action for frame in frames]

            group_record = {
                "segment_id": group_count,
                "episode_index": info.episode_index,
                "task": info.task,
                "task_id": task_id,
                "segment_start": 0,
                "segment_end": info.num_frames,
                "num_frames": info.num_frames,
                "num_selected_frames": len(frame_ids),
                "selected_episode_frame_indices": [int(x) for x in frame_ids],
                "selection_strategy": "uniform_time",
            }
            group_f.write(json.dumps(group_record, ensure_ascii=False) + "\n")

            for rank, ep_frame_idx in enumerate(frame_ids):
                global_frame_idx = info.start + ep_frame_idx
                frame = frames[ep_frame_idx]
                action_chunk = _proto._build_action_chunk(actions, ep_frame_idx, horizon=train_cfg.model.action_horizon)

                sample_id = sample_count
                sample_path = out_dir / "samples" / f"sample_{sample_id:08d}.npz"
                np.savez_compressed(
                    sample_path,
                    image=frame.image,
                    wrist_image=frame.wrist_image,
                    state=frame.state.astype(np.float32),
                    action=frame.action.astype(np.float32),
                    action_chunk=action_chunk.astype(np.float32),
                    task=np.asarray(info.task),
                    prompt=np.asarray(prompt),
                    task_id=np.asarray(task_id, dtype=np.int32),
                    episode_index=np.asarray(info.episode_index, dtype=np.int32),
                    episode_frame_index=np.asarray(ep_frame_idx, dtype=np.int32),
                    global_frame_index=np.asarray(global_frame_idx, dtype=np.int32),
                    episode_num_frames=np.asarray(info.num_frames, dtype=np.int32),
                    episode_selection_rank=np.asarray(rank, dtype=np.int32),
                    selection_strategy=np.asarray("uniform_time"),
                )

                sample_record = {
                    "sample_id": sample_id,
                    "sample_path": str(sample_path.relative_to(out_dir)),
                    "task": info.task,
                    "task_id": task_id,
                    "prompt": prompt,
                    "episode_index": info.episode_index,
                    "episode_frame_index": ep_frame_idx,
                    "global_frame_index": global_frame_idx,
                    "segment_id": group_count,
                    "segment_start": 0,
                    "segment_end": info.num_frames,
                    "segment_rank": rank,
                    "selection_strategy": "uniform_time",
                }
                sample_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                sample_count += 1

            group_count += 1

    if sample_count != sum(alloc_counts):
        raise RuntimeError(f"Built {sample_count} samples, expected {sum(alloc_counts)}")

    meta = {
        "train_config_name": cfg.train_config_name,
        "repo_id": data_cfg.repo_id,
        "dataset_root": dataset_root,
        "target_task_name": target_task_name,
        "selection": {
            "strategy": "uniform_time",
            "allocation_strategy": cfg.allocation_strategy,
            "ensure_one_per_episode_if_possible": cfg.ensure_one_per_episode_if_possible,
            "target_num_samples": target_num_samples,
            "budget_source": budget_source,
            "match_budget_buffer_dir": cfg.match_budget_buffer_dir,
            "reference_manifest_filename": cfg.reference_manifest_filename,
        },
        "stats": {
            "num_selected_episodes": len(episode_infos),
            "num_skipped_empty_episodes": skipped_empty_episodes,
            "num_groups": group_count,
            "num_samples": sample_count,
            "num_candidate_frames": num_candidate_frames,
            "num_tasks": len(task_to_id),
            "task_to_id": task_to_id,
            "episode_lengths": episode_lengths,
            "samples_per_episode": alloc_counts,
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Uniform replay buffer build complete.")
    LOGGER.info("Output dir: %s", out_dir)
    LOGGER.info("Episodes: %d | Groups: %d | Samples: %d", len(episode_infos), group_count, sample_count)


if __name__ == "__main__":
    tyro.cli(main)
