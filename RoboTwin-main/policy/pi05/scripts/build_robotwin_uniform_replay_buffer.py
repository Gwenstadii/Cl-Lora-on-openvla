#!/usr/bin/env python3
"""Build a budget-matched uniform replay buffer for RoboTwin/pi05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.common.datasets import lerobot_dataset
import numpy as np

import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

from build_robotwin_prototype_buffer import CAMERA_KEYS
from build_robotwin_prototype_buffer import _episode_ranges
from build_robotwin_prototype_buffer import _prompt_from_sample
from build_robotwin_prototype_buffer import _save_replay_sample
from build_robotwin_prototype_buffer import _write_jsonl


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _count_manifest_records(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Reference manifest not found: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if "sample_path" not in record:
                raise ValueError(f"Reference manifest record has no sample_path at {path}:{line_no}")
            count += 1
    if count == 0:
        raise ValueError(f"Reference manifest is empty: {path}")
    return count


def _reference_spec(reference_dir: Path, *, repo_id: str, task_name: str) -> tuple[dict, list[int], int]:
    meta = _read_json(reference_dir / "meta.json")
    manifest_count = _count_manifest_records(reference_dir / "manifest.jsonl")

    reference_repo_id = meta.get("repo_id")
    if reference_repo_id != repo_id:
        raise ValueError(f"Reference repo_id={reference_repo_id!r} does not match --repo-id={repo_id!r}.")
    reference_task_name = meta.get("task_name")
    if reference_task_name != task_name:
        raise ValueError(
            f"Reference task_name={reference_task_name!r} does not match --task-name={task_name!r}."
        )

    saved_frame_count = int(meta.get("saved_frame_count", manifest_count))
    if saved_frame_count != manifest_count:
        raise ValueError(
            "Reference prototype budget is inconsistent: "
            f"meta saved_frame_count={saved_frame_count}, manifest records={manifest_count}."
        )

    episode_indices = meta.get("selected_episode_indices")
    if not isinstance(episode_indices, list) or not episode_indices:
        raise ValueError("Reference meta.json must contain a non-empty selected_episode_indices list.")
    episode_indices = [int(index) for index in episode_indices]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Reference selected_episode_indices contains duplicates.")

    return meta, episode_indices, saved_frame_count


def _prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory or remove the old buffer explicitly."
        )
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def build_buffer(args: argparse.Namespace) -> None:
    reference_dir = Path(args.reference_prototype_buffer).expanduser().resolve()
    reference_meta, episode_indices, replay_budget = _reference_spec(
        reference_dir,
        repo_id=args.repo_id,
        task_name=args.task_name,
    )

    config = _config.get_config(args.train_config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != args.repo_id:
        raise ValueError(
            f"Config repo_id={data_config.repo_id!r} does not match --repo-id={args.repo_id!r}; "
            "use the source task's training config."
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    base_dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    raw_dataset = base_dataset
    if data_config.prompt_from_task:
        raw_dataset = _data_loader.TransformedDataset(
            raw_dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )

    ranges = _episode_ranges(base_dataset)
    candidates: list[tuple[int, int, int]] = []
    for episode_index in episode_indices:
        if episode_index < 0 or episode_index >= len(ranges):
            raise ValueError(f"Episode index {episode_index} is outside [0, {len(ranges)}).")
        start, end = ranges[episode_index]
        candidates.extend(
            (global_index, episode_index, global_index - start)
            for global_index in range(start, end)
        )

    source_frame_count = len(candidates)
    reference_source_frames = reference_meta.get("source_frame_count")
    if reference_source_frames is not None and int(reference_source_frames) != source_frame_count:
        raise ValueError(
            "Source data no longer matches the prototype buffer: "
            f"reference source_frame_count={reference_source_frames}, current={source_frame_count}."
        )
    if replay_budget > source_frame_count:
        raise ValueError(
            f"Replay budget {replay_budget} exceeds the {source_frame_count} frames in the reference episodes."
        )

    rng = np.random.default_rng(args.seed)
    selected_candidate_ids = rng.choice(source_frame_count, size=replay_budget, replace=False)
    selected = sorted((candidates[int(index)] for index in selected_candidate_ids), key=lambda item: item[0])

    output_dir = Path(args.output_dir).expanduser().resolve()
    samples_dir = _prepare_output_dir(output_dir)
    norm_assets_rel = Path("assets")
    if data_config.norm_stats is not None:
        _normalize.save(output_dir / norm_assets_rel / args.repo_id, data_config.norm_stats)

    manifest_records: list[dict] = []
    selected_per_episode = {episode_index: 0 for episode_index in episode_indices}
    for sample_id, (global_index, episode_index, frame_index) in enumerate(selected):
        raw_sample = raw_dataset[global_index]
        prompt = _prompt_from_sample(raw_sample, dataset_meta.tasks, args.fixed_prompt)
        sample_rel_path = Path("samples") / f"sample_{sample_id:06d}.npz"
        _save_replay_sample(
            raw_sample,
            samples_dir / sample_rel_path.name,
            horizon=config.model.action_horizon,
        )
        selected_per_episode[episode_index] += 1
        manifest_records.append(
            {
                "sample_path": str(sample_rel_path).replace("\\", "/"),
                "task_name": args.task_name,
                "repo_id": args.repo_id,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "global_index": global_index,
                "sampling_method": "global_uniform_without_replacement",
                "sampling_seed": args.seed,
                "prompt": prompt,
            }
        )

    _write_jsonl(output_dir / "manifest.jsonl", manifest_records)
    _write_jsonl(
        output_dir / "selection.jsonl",
        [
            {
                "task_name": args.task_name,
                "repo_id": args.repo_id,
                "episode_index": episode_index,
                "start": ranges[episode_index][0],
                "end": ranges[episode_index][1],
                "source_frame_count": ranges[episode_index][1] - ranges[episode_index][0],
                "selected_frame_count": selected_per_episode[episode_index],
            }
            for episode_index in episode_indices
        ],
    )

    compression_ratio = replay_budget / max(source_frame_count, 1)
    meta_record = {
        "format": "robotwin_pi05_uniform_replay_v1",
        "task_name": args.task_name,
        "repo_id": args.repo_id,
        "train_config_name": args.train_config_name,
        "cl_task_id": int(getattr(config.model, "cl_task_id", 0)),
        "sampling_method": "global_uniform_without_replacement",
        "sampling_seed": args.seed,
        "reference_prototype_buffer": str(reference_dir),
        "reference_prototype_format": reference_meta.get("format"),
        "selected_episode_indices": episode_indices,
        "num_source_episodes": len(episode_indices),
        "source_frame_count": source_frame_count,
        "full_dataset_frame_count": len(base_dataset),
        "saved_frame_count": replay_budget,
        "sample_count": replay_budget,
        "matched_prototype_budget": replay_budget,
        "compression_ratio": compression_ratio,
        "source_to_saved_frame_compression_ratio": compression_ratio,
        "full_dataset_compression_ratio": replay_budget / max(len(base_dataset), 1),
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": config.model.action_horizon,
        "fps": dataset_meta.fps,
        "norm_assets_dir": str(norm_assets_rel).replace("\\", "/"),
        "norm_asset_id": args.repo_id,
        "selected_frames_per_episode": {str(key): value for key, value in selected_per_episode.items()},
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2, ensure_ascii=False)

    print("Budget-matched uniform replay buffer written")
    print(f"  output_dir: {output_dir}")
    print(f"  reference prototype buffer: {reference_dir}")
    print(f"  episodes: {len(episode_indices)}")
    print(f"  source frames: {source_frame_count}")
    print(f"  matched replay budget: {replay_budget}")
    print(f"  saved replay frames: {len(manifest_records)}")
    print(f"  compression_ratio: {compression_ratio:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument(
        "--reference-prototype-buffer",
        required=True,
        help="Prototype buffer whose episode set and final sample count define the uniform baseline budget.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-prompt", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    build_buffer(parse_args())
