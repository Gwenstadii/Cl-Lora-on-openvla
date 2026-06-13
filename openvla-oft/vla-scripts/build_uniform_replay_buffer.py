"""
Build an offline replay buffer with uniform temporal sampling.

Fair-comparison baseline for the prototype replay pipeline
(build_replay_buffer_openvla.py). Reuses the same:
  - Task filtering and episode selection
  - Total sample budget (matched from reference prototype buffer)
  - Output format (manifest.jsonl + samples/*.npz + meta.json)

Only the frame selection rule changes: uniform temporal sampling
instead of physics-based segmentation + prototype Top-K selection.

Reference: PI build_uniform_replay_buffer.py
"""

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

tf.config.set_visible_devices([], 'GPU')

LOGGER = logging.getLogger(__name__)


@dataclass
class BuildUniformReplayBufferConfig:
    data_root_dir: str = "/root/autodl-tmp/modified_libero_rlds"
    dataset_name: str = "libero_spatial_no_noops"
    target_task_name: str = "pick up the black bowl next to the cookie box and place it on the plate"

    output_dir: str = "/root/autodl-tmp/replay_buffers/taskA_uniform"
    overwrite: bool = False

    # Sample budget: exactly one of these should be set
    target_num_samples: Optional[int] = None
    match_budget_buffer_dir: Optional[str] = None
    reference_manifest_filename: str = "manifest.jsonl"

    max_episodes: Optional[int] = None

    # Action chunking
    action_horizon: int = 8

    # Allocation
    ensure_one_per_episode_if_possible: bool = True


# ==============================================================================
# Budget resolution
# ==============================================================================

def _count_manifest_records(buffer_dir: str, manifest_filename: str) -> int:
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


def _resolve_target_num_samples(cfg: BuildUniformReplayBufferConfig) -> tuple:
    has_explicit = cfg.target_num_samples is not None
    has_reference = cfg.match_budget_buffer_dir is not None
    if has_explicit == has_reference:
        raise ValueError(
            "Set exactly one of target_num_samples or match_budget_buffer_dir."
        )
    if has_explicit:
        target = int(cfg.target_num_samples)
        if target <= 0:
            raise ValueError(f"target_num_samples must be > 0, got {target}")
        return target, "explicit"
    target = _count_manifest_records(cfg.match_budget_buffer_dir, cfg.reference_manifest_filename)
    return target, "matched_from_reference_buffer"


# ==============================================================================
# Allocation helpers
# ==============================================================================

def _distribute_with_caps(weights, capacities, total):
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

    weights = [float(w) if cap > 0 else 0.0 for w, cap in zip(weights, capacities)]
    if sum(weights) <= 0.0:
        weights = [1.0 if cap > 0 else 0.0 for cap in capacities]
    weight_sum = sum(weights)
    raw = [total * w / weight_sum for w in weights]
    alloc = [min(int(np.floor(x)), cap) for x, cap in zip(raw, capacities)]
    remaining = total - sum(alloc)

    while remaining > 0:
        best_idx, best_key = None, None
        for i, (a, cap, r, w) in enumerate(zip(alloc, capacities, raw, weights)):
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


def _allocate_episode_sample_counts(lengths, total_samples, ensure_one_per_episode_if_possible):
    if total_samples <= 0:
        raise ValueError(f"total_samples must be > 0, got {total_samples}")
    if not lengths:
        return []
    if any(l < 0 for l in lengths):
        raise ValueError("Episode lengths must be non-negative")
    total_frames = int(sum(lengths))
    if total_frames <= 0:
        raise ValueError("No valid frames available.")
    if total_samples >= total_frames:
        return list(lengths)

    num_non_empty = sum(1 for x in lengths if x > 0)
    if ensure_one_per_episode_if_possible and total_samples >= num_non_empty:
        base = [1 if l > 0 else 0 for l in lengths]
        remaining = total_samples - sum(base)
        capacities = [max(l - b, 0) for l, b in zip(lengths, base)]
        extras = _distribute_with_caps(lengths, capacities, remaining)
        alloc = [b + e for b, e in zip(base, extras)]
    else:
        alloc = _distribute_with_caps(lengths, lengths, total_samples)

    expected = min(total_samples, total_frames)
    if sum(alloc) != expected:
        raise RuntimeError(f"Allocated {sum(alloc)} samples, expected {expected}.")
    if any(a > l for a, l in zip(alloc, lengths)):
        raise RuntimeError("Allocated sample count exceeds episode length.")
    return alloc


def _uniform_frame_indices(num_frames, num_samples):
    if num_frames <= 0 or num_samples <= 0:
        return []
    if num_samples >= num_frames:
        return list(range(num_frames))

    ideal_positions = ((np.arange(num_samples, dtype=np.float64) + 0.5) * num_frames / num_samples) - 0.5
    used = set()
    result = []

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
    return result


# ==============================================================================
# Main
# ==============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.set_printoptions(precision=4, suppress=True)

    cfg = BuildUniformReplayBufferConfig()

    out_dir = pathlib.Path(cfg.output_dir).resolve()
    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}. Use --overwrite True to replace it.")
        import shutil
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    target_num_samples, budget_source = _resolve_target_num_samples(cfg)
    LOGGER.info("Target sample budget: %d (source: %s)", target_num_samples, budget_source)

    # Load RLDS dataset
    LOGGER.info("Loading RLDS dataset: %s", cfg.dataset_name)
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    dataset = builder.as_dataset(split='all')

    # Filter episodes by task
    episodes = []
    for ep_idx, episode in enumerate(dataset):
        steps = list(episode['steps'])
        step_zero = steps[0]
        if 'language_instruction' in step_zero:
            raw_inst = step_zero['language_instruction']
        elif 'natural_language_instruction' in step_zero.get('observation', {}):
            raw_inst = step_zero['observation']['natural_language_instruction']
        else:
            continue
        if hasattr(raw_inst, 'numpy'):
            raw_inst = raw_inst.numpy()
        if isinstance(raw_inst, bytes):
            language_instruction = raw_inst.decode('utf-8')
        else:
            language_instruction = str(raw_inst)

        if cfg.target_task_name not in language_instruction:
            continue

        episodes.append({
            'episode_index': ep_idx,
            'task': language_instruction,
            'steps': steps,
            'num_frames': len(steps),
        })

    if cfg.max_episodes is not None:
        episodes = episodes[:cfg.max_episodes]

    if not episodes:
        raise ValueError(f"No episodes found for target_task_name={cfg.target_task_name!r}.")

    lengths = [ep['num_frames'] for ep in episodes]
    num_candidate_frames = int(sum(lengths))
    alloc_counts = _allocate_episode_sample_counts(
        lengths, target_num_samples,
        ensure_one_per_episode_if_possible=cfg.ensure_one_per_episode_if_possible,
    )

    LOGGER.info("Selected episodes: %d / %d", len(episodes), len(list(dataset)))
    LOGGER.info("Uniform replay target budget: %d samples from %d candidate frames",
                int(sum(alloc_counts)), num_candidate_frames)

    # Build buffer
    sample_manifest_path = out_dir / "manifest.jsonl"
    group_manifest_path = out_dir / "segments.jsonl"
    task_to_id = {}

    sample_count = 0
    group_count = 0

    with sample_manifest_path.open("w", encoding="utf-8") as samp_f, \
         group_manifest_path.open("w", encoding="utf-8") as group_f:

        for ep, num_select in tqdm.tqdm(zip(episodes, alloc_counts), total=len(episodes), desc="Episodes"):
            if num_select <= 0:
                continue

            task = ep['task']
            task_id = task_to_id.setdefault(task, len(task_to_id))
            steps = ep['steps']
            num_frames = ep['num_frames']

            frame_ids = _uniform_frame_indices(num_frames, int(num_select))

            # Write group record
            group_record = {
                "segment_id": group_count,
                "episode_index": ep['episode_index'],
                "task": task,
                "task_id": task_id,
                "segment_start": 0,
                "segment_end": num_frames,
                "num_frames": num_frames,
                "num_selected_frames": len(frame_ids),
                "selected_episode_frame_indices": [int(x) for x in frame_ids],
                "selection_strategy": "uniform_time",
            }
            group_f.write(json.dumps(group_record, ensure_ascii=False) + "\n")

            # Save sample frames
            for rank, ep_frame_idx in enumerate(frame_ids):
                step_data = steps[ep_frame_idx]
                action = step_data['action'].numpy()

                # Build action chunk
                action_chunk = []
                for offset in range(cfg.action_horizon):
                    idx = min(ep_frame_idx + offset, num_frames - 1)
                    action_chunk.append(steps[idx]['action'].numpy())
                action_chunk = np.stack(action_chunk, axis=0)

                sample_path = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                np.savez_compressed(
                    sample_path,
                    image=step_data['observation']['image'].numpy(),
                    state=step_data['observation']['state'].numpy(),
                    action=action,
                    action_chunk=action_chunk,
                    task=task,
                )

                sample_record = {
                    "sample_id": sample_count,
                    "sample_path": str(sample_path.relative_to(out_dir)),
                    "task": task,
                    "task_id": task_id,
                    "episode_index": ep['episode_index'],
                    "episode_frame_index": int(ep_frame_idx),
                    "global_frame_index": int(ep_frame_idx),
                    "segment_id": group_count,
                    "segment_start": 0,
                    "segment_end": num_frames,
                    "segment_rank": rank,
                    "selection_strategy": "uniform_time",
                }
                samp_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                sample_count += 1

            group_count += 1

    if sample_count != int(sum(alloc_counts)):
        raise RuntimeError(f"Built {sample_count} samples, expected {int(sum(alloc_counts))}")

    # Write meta
    meta = {
        "dataset_name": cfg.dataset_name,
        "data_root_dir": cfg.data_root_dir,
        "target_task_name": cfg.target_task_name,
        "selection": {
            "strategy": "uniform_time",
            "target_num_samples": target_num_samples,
            "budget_source": budget_source,
            "match_budget_buffer_dir": cfg.match_budget_buffer_dir,
            "reference_manifest_filename": cfg.reference_manifest_filename,
        },
        "stats": {
            "num_selected_episodes": len(episodes),
            "num_groups": group_count,
            "num_samples": sample_count,
            "num_candidate_frames": num_candidate_frames,
            "num_tasks": len(task_to_id),
            "task_to_id": task_to_id,
            "episode_lengths": lengths,
            "samples_per_episode": alloc_counts,
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Uniform replay buffer build complete.")
    LOGGER.info("Output dir: %s", out_dir)
    LOGGER.info("Episodes: %d | Groups: %d | Samples: %d", len(episodes), group_count, sample_count)


if __name__ == "__main__":
    main()
