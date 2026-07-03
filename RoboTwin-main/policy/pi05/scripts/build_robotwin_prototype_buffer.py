#!/usr/bin/env python3
"""Build a RoboTwin/pi05 prototype replay buffer from LeRobot training data."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from lerobot.common.datasets import lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_JOINT_IDXS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int32)
GRIPPER_IDXS = np.asarray([6, 13], dtype=np.int32)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_int(x: Any) -> int:
    if hasattr(x, "item"):
        x = x.item()
    return int(x)


def _get_value(sample: dict, dotted_key: str) -> Any:
    if dotted_key in sample:
        return sample[dotted_key]
    current = sample
    for part in dotted_key.split("."):
        current = current[part]
    return current


def _maybe_get_value(sample: dict, dotted_key: str) -> Any | None:
    try:
        return _get_value(sample, dotted_key)
    except (KeyError, TypeError):
        return None


def _ensure_chw_image(image: Any) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected rank-3 image, got {image.shape}")
    if image.shape[0] in (1, 3):
        return image
    if image.shape[-1] in (1, 3):
        return np.moveaxis(image, -1, 0)
    raise ValueError(f"Cannot infer image layout from shape {image.shape}")


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmin(image)) >= -1.0 and float(np.nanmax(image)) <= 1.0:
            if float(np.nanmin(image)) < 0.0:
                image = (image + 1.0) / 2.0
            image = image * 255.0
        image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def _ensure_action_chunk(action: Any, horizon: int) -> np.ndarray:
    action = _to_numpy(action).astype(np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        raise ValueError(f"Expected action chunk [horizon, dim], got {action.shape}")
    if action.shape[0] > horizon:
        action = action[:horizon]
    elif action.shape[0] < horizon:
        pad = np.repeat(action[-1:, :], horizon - action.shape[0], axis=0)
        action = np.concatenate([action, pad], axis=0)
    return action


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if x.size == 0 or window <= 1:
        return x
    window = min(window, x.size)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def _parse_episode_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _episode_ranges(dataset) -> list[tuple[int, int]]:
    index = getattr(dataset, "episode_data_index", None)
    if isinstance(index, dict) and "from" in index and "to" in index:
        starts = [_to_int(v) for v in _to_numpy(index["from"])]
        ends = [_to_int(v) for v in _to_numpy(index["to"])]
        return list(zip(starts, ends, strict=True))

    ranges: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        ep_idx = _maybe_get_value(sample, "episode_index")
        if ep_idx is None:
            raise ValueError("Dataset has no episode_data_index and samples have no episode_index.")
        ep_idx = _to_int(ep_idx)
        ranges.setdefault(ep_idx, [idx, idx + 1])
        ranges[ep_idx][1] = idx + 1
    return [(v[0], v[1]) for _, v in sorted(ranges.items())]


def _prompt_from_sample(sample: dict, tasks: dict[int, str], fixed_prompt: str | None) -> str:
    if fixed_prompt:
        return fixed_prompt
    prompt = _maybe_get_value(sample, "prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            prompt = prompt.item()
        return str(prompt)
    task = _maybe_get_value(sample, "task")
    if task is not None:
        if not isinstance(task, str):
            task = task.item()
        return str(task)
    task_index = _maybe_get_value(sample, "task_index")
    if task_index is not None:
        return str(tasks[_to_int(task_index)])
    raise ValueError("Cannot infer prompt from sample; pass --fixed-prompt.")


def _select_episodes(
    ranges: list[tuple[int, int]],
    *,
    num_episodes: int,
    episode_indices: list[int] | None,
    random_episodes: bool,
    seed: int,
) -> list[int]:
    if episode_indices is not None:
        selected = episode_indices
    elif random_episodes:
        rng = np.random.default_rng(seed)
        count = min(num_episodes, len(ranges))
        selected = sorted(rng.choice(len(ranges), size=count, replace=False).tolist())
    else:
        selected = list(range(min(num_episodes, len(ranges))))

    for ep_idx in selected:
        if ep_idx < 0 or ep_idx >= len(ranges):
            raise ValueError(f"Episode index {ep_idx} is out of range [0, {len(ranges)})")
    return selected


def _collect_physical_signals(dataset, start: int, end: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    states = []
    first_actions = []
    for global_idx in range(start, end):
        sample = dataset[global_idx]
        state = _to_numpy(_get_value(sample, "observation.state")).astype(np.float32)
        action = _ensure_action_chunk(_get_value(sample, "action"), horizon)
        states.append(state[:14])
        first_actions.append(action[0, :14])
    return np.asarray(states, dtype=np.float32), np.asarray(first_actions, dtype=np.float32)


def _merge_short_segments(segments: list[tuple[int, int]], min_len: int) -> list[tuple[int, int]]:
    if not segments:
        return segments

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
            continue
        if end - start < min_len:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_len:
        first_start, _ = merged[0]
        _, second_end = merged[1]
        merged = [(first_start, second_end), *merged[2:]]
    return merged


def _split_long_segments(segments: list[tuple[int, int]], max_len: int) -> list[tuple[int, int]]:
    if max_len <= 0:
        return segments
    out: list[tuple[int, int]] = []
    for start, end in segments:
        length = end - start
        if length <= max_len:
            out.append((start, end))
            continue
        chunks = int(math.ceil(length / max_len))
        edges = np.linspace(start, end, chunks + 1).round().astype(int)
        out.extend((int(edges[i]), int(edges[i + 1])) for i in range(chunks) if edges[i + 1] > edges[i])
    return out


def _build_skill_segments(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    min_segment_len: int,
    max_segment_len: int,
    min_boundary_gap: int,
) -> list[tuple[int, int]]:
    num_frames = states.shape[0]
    if num_frames <= max(1, min_segment_len):
        return [(0, num_frames)]

    state_delta = np.diff(states[:, :14], axis=0)
    action_delta = np.diff(actions[:, :14], axis=0)
    arm_energy = (
        np.linalg.norm(state_delta[:, ARM_JOINT_IDXS], axis=-1)
        + 0.5 * np.linalg.norm(action_delta[:, ARM_JOINT_IDXS], axis=-1)
    )
    gripper_energy = (
        np.abs(state_delta[:, GRIPPER_IDXS]).sum(axis=-1)
        + np.abs(action_delta[:, GRIPPER_IDXS]).sum(axis=-1)
    )
    left_energy = np.linalg.norm(state_delta[:, :7], axis=-1) + 0.5 * np.linalg.norm(action_delta[:, :7], axis=-1)
    right_energy = np.linalg.norm(state_delta[:, 7:14], axis=-1) + 0.5 * np.linalg.norm(action_delta[:, 7:14], axis=-1)

    arm_energy = _moving_average(arm_energy)
    gripper_energy = _moving_average(gripper_energy)

    motion_threshold = max(float(np.percentile(arm_energy, 65)), float(arm_energy.mean() + 0.2 * arm_energy.std()))
    moving = arm_energy > motion_threshold
    dominant = np.full_like(left_energy, fill_value=2, dtype=np.int32)
    dominant[left_energy > right_energy * 1.15] = 0
    dominant[right_energy > left_energy * 1.15] = 1
    gripper_threshold = max(float(np.percentile(gripper_energy, 80)), float(gripper_energy.mean() + gripper_energy.std()))

    boundaries = {0, num_frames}
    last_boundary = 0
    for i in range(1, num_frames - 1):
        is_boundary = False
        if moving[i] != moving[i - 1]:
            is_boundary = True
        if dominant[i] != dominant[i - 1]:
            is_boundary = True
        if gripper_energy[i] > gripper_threshold:
            is_boundary = True
        if is_boundary and (i + 1 - last_boundary) >= min_boundary_gap:
            boundaries.add(i + 1)
            last_boundary = i + 1

    ordered = sorted(boundaries)
    segments = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1) if ordered[i + 1] > ordered[i]]
    segments = _merge_short_segments(segments, min_segment_len)
    segments = _split_long_segments(segments, max_segment_len)
    segments = _merge_short_segments(segments, min_segment_len)
    return segments or [(0, num_frames)]


def _collate_transformed(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _load_model(config: _config.TrainConfig, checkpoint_dir: Path):
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_dir}")
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    return model


def _make_pi05_feature_fn(model):
    def extract(obs: _model.Observation) -> jnp.ndarray:
        obs = _model.preprocess_observation(None, obs, train=False)
        pooled = []
        for name in _model.IMAGE_KEYS:
            image_tokens, _ = model.PaliGemma.img(obs.images[name], train=False)
            feature = jnp.mean(image_tokens.astype(jnp.float32), axis=1)
            mask = obs.image_masks[name].astype(jnp.float32)[:, None]
            pooled.append(feature * mask)
        feature = jnp.concatenate(pooled, axis=-1)
        norm = jnp.linalg.norm(feature, axis=-1, keepdims=True)
        return feature / jnp.maximum(norm, 1e-8)

    return jax.jit(extract)


def _image_stats_feature(raw_sample: dict) -> np.ndarray:
    features = []
    for key in CAMERA_KEYS:
        image = _ensure_chw_image(_get_value(raw_sample, f"observation.images.{key}")).astype(np.float32) / 255.0
        features.extend(image.mean(axis=(1, 2)).tolist())
        features.extend(image.std(axis=(1, 2)).tolist())
        h_mid, w_mid = image.shape[1] // 2, image.shape[2] // 2
        center = image[:, max(0, h_mid - 16) : h_mid + 16, max(0, w_mid - 16) : w_mid + 16]
        features.extend(center.mean(axis=(1, 2)).tolist())
    feature = np.asarray(features, dtype=np.float32)
    return feature / max(float(np.linalg.norm(feature)), 1e-8)


def _extract_features(
    raw_dataset,
    transformed_dataset,
    frame_indices: list[int],
    *,
    feature_backend: str,
    model,
    batch_size: int,
) -> np.ndarray:
    if feature_backend == "image_stats":
        return np.asarray([_image_stats_feature(raw_dataset[idx]) for idx in frame_indices], dtype=np.float32)

    extract_fn = _make_pi05_feature_fn(model)
    outputs = []
    for offset in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[offset : offset + batch_size]
        batch_dict = _collate_transformed([transformed_dataset[idx] for idx in batch_indices])
        obs = _model.Observation.from_dict(batch_dict)
        outputs.append(np.asarray(jax.device_get(extract_fn(obs)), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _save_replay_sample(
    raw_sample: dict,
    sample_path: Path,
    *,
    horizon: int,
) -> None:
    images = {
        key: _to_uint8_image(_ensure_chw_image(_get_value(raw_sample, f"observation.images.{key}")))
        for key in CAMERA_KEYS
    }
    state = _to_numpy(_get_value(raw_sample, "observation.state")).astype(np.float32)[:14]
    action = _ensure_action_chunk(_get_value(raw_sample, "action"), horizon)[:, :14]
    np.savez_compressed(
        sample_path,
        cam_high=images["cam_high"],
        cam_left_wrist=images["cam_left_wrist"],
        cam_right_wrist=images["cam_right_wrist"],
        state=state,
        action=action,
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_checkpoint_dir(config: _config.TrainConfig, args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).expanduser().resolve()
    if not args.exp_name:
        raise ValueError("--exp-name is required unless --checkpoint-dir is provided.")
    return (
        Path(config.checkpoint_base_dir)
        / args.train_config_name
        / args.exp_name
        / str(args.checkpoint_id)
    ).expanduser().resolve()


def build_buffer(args: argparse.Namespace) -> None:
    config = _config.get_config(args.train_config_name)
    if args.exp_name:
        config = dataclasses.replace(config, exp_name=args.exp_name)

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != args.repo_id:
        print(f"[W] config repo_id={data_config.repo_id!r} differs from --repo-id={args.repo_id!r}; using --repo-id.")

    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    base_dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    raw_dataset = base_dataset
    if data_config.prompt_from_task:
        raw_dataset = _data_loader.TransformedDataset(raw_dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])

    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=args.skip_norm_stats)
    ranges = _episode_ranges(base_dataset)
    episode_indices = _select_episodes(
        ranges,
        num_episodes=args.num_episodes,
        episode_indices=_parse_episode_indices(args.episode_indices),
        random_episodes=args.random_episodes,
        seed=args.seed,
    )

    checkpoint_dir = _resolve_checkpoint_dir(config, args)
    model = None
    if args.feature_backend == "pi05_vision":
        model = _load_model(config, checkpoint_dir)

    output_dir = Path(args.output_dir).expanduser().resolve()
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    norm_assets_rel = Path("assets")
    if data_config.norm_stats is not None:
        _normalize.save(output_dir / norm_assets_rel / args.repo_id, data_config.norm_stats)

    manifest_records: list[dict] = []
    segment_records: list[dict] = []
    prototypes: dict[str, np.ndarray] = {}
    total_source_frames = 0
    sample_id = 0
    segment_id = 0

    for ep_idx in episode_indices:
        start, end = ranges[ep_idx]
        states, first_actions = _collect_physical_signals(raw_dataset, start, end, config.model.action_horizon)
        segments = _build_skill_segments(
            states,
            first_actions,
            min_segment_len=args.min_segment_len,
            max_segment_len=args.max_segment_len,
            min_boundary_gap=args.min_boundary_gap,
        )
        frame_indices = list(range(start, end))
        features = _extract_features(
            raw_dataset,
            transformed_dataset,
            frame_indices,
            feature_backend=args.feature_backend,
            model=model,
            batch_size=args.feature_batch_size,
        )
        features = _normalize_rows(features.astype(np.float32))
        total_source_frames += len(frame_indices)

        for local_start, local_end in segments:
            local_features = features[local_start:local_end]
            if local_features.size == 0:
                continue
            prototype = local_features.mean(axis=0)
            prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-8)
            sims = local_features @ prototype
            top_count = min(args.top_k, sims.shape[0])
            top_local = np.argsort(-sims)[:top_count]

            proto_key = f"prototype_{segment_id:06d}"
            prototypes[proto_key] = prototype.astype(np.float32)
            selected = []
            for rank, local_offset in enumerate(top_local.tolist()):
                global_idx = start + local_start + local_offset
                raw_sample = raw_dataset[global_idx]
                prompt = _prompt_from_sample(raw_sample, meta.tasks, args.fixed_prompt)
                sample_rel_path = Path("samples") / f"sample_{sample_id:06d}.npz"
                _save_replay_sample(raw_sample, output_dir / sample_rel_path, horizon=config.model.action_horizon)

                record = {
                    "sample_path": str(sample_rel_path).replace("\\", "/"),
                    "task_name": args.task_name,
                    "repo_id": args.repo_id,
                    "episode_index": ep_idx,
                    "frame_index": int(global_idx - start),
                    "global_index": int(global_idx),
                    "segment_id": f"ep{ep_idx:04d}_seg{segment_id:04d}",
                    "segment_start": int(local_start),
                    "segment_end": int(local_end),
                    "rank": rank,
                    "similarity": float(sims[local_offset]),
                    "prompt": prompt,
                }
                manifest_records.append(record)
                selected.append(record)
                sample_id += 1

            segment_records.append(
                {
                    "task_name": args.task_name,
                    "repo_id": args.repo_id,
                    "episode_index": ep_idx,
                    "segment_id": f"ep{ep_idx:04d}_seg{segment_id:04d}",
                    "start": int(local_start),
                    "end": int(local_end),
                    "length": int(local_end - local_start),
                    "prototype_key": proto_key,
                    "coverage_error": float(np.mean(1.0 - (local_features @ prototype))),
                    "selected": selected,
                }
            )
            segment_id += 1

    if not manifest_records:
        raise RuntimeError("No replay samples were selected; check segmentation/top-k settings.")

    saved_frame_count = len(manifest_records)
    compression_ratio = saved_frame_count / max(total_source_frames, 1)

    np.savez_compressed(output_dir / "prototypes.npz", **prototypes)
    _write_jsonl(output_dir / "manifest.jsonl", manifest_records)
    _write_jsonl(output_dir / "segments.jsonl", segment_records)

    meta_record = {
        "format": "robotwin_pi05_prototype_replay_v1",
        "task_name": args.task_name,
        "repo_id": args.repo_id,
        "train_config_name": args.train_config_name,
        "cl_task_id": int(getattr(config.model, "cl_task_id", 0)),
        "source_checkpoint": str(checkpoint_dir),
        "feature_backend": args.feature_backend,
        "selected_episode_indices": episode_indices,
        "num_source_episodes": len(episode_indices),
        "source_frame_count": total_source_frames,
        "saved_frame_count": saved_frame_count,
        "sample_count": saved_frame_count,
        "segment_count": len(segment_records),
        "top_k": args.top_k,
        "compression_ratio": compression_ratio,
        "source_to_saved_frame_compression_ratio": compression_ratio,
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": config.model.action_horizon,
        "fps": meta.fps,
        "norm_assets_dir": str(norm_assets_rel).replace("\\", "/"),
        "norm_asset_id": args.repo_id,
        "min_segment_len": args.min_segment_len,
        "max_segment_len": args.max_segment_len,
        "min_boundary_gap": args.min_boundary_gap,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2, ensure_ascii=False)

    print("Prototype replay buffer written")
    print(f"  output_dir: {output_dir}")
    print(f"  episodes: {len(episode_indices)}")
    print(f"  source frames: {total_source_frames}")
    print(f"  segments: {len(segment_records)}")
    print(f"  saved replay frames: {saved_frame_count}")
    print(f"  saved replay samples: {saved_frame_count}")
    print(f"  compression_ratio: {meta_record['compression_ratio']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--checkpoint-id", default="10000")
    parser.add_argument("--checkpoint-dir", default=None, help="Step checkpoint directory containing params/.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None, help="Comma-separated episode ids; overrides --num-episodes.")
    parser.add_argument("--random-episodes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--max-segment-len", type=int, default=80)
    parser.add_argument("--min-boundary-gap", type=int, default=6)
    parser.add_argument("--feature-backend", choices=("pi05_vision", "image_stats"), default="pi05_vision")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--fixed-prompt", default=None)
    parser.add_argument("--skip-norm-stats", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_buffer(parse_args())
