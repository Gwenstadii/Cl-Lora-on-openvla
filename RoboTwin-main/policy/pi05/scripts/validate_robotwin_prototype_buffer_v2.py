#!/usr/bin/env python3
"""Validate a completed RoboTwin Prototype Replay V2 buffer without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _validate_image(image: np.ndarray, key: str, sample_path: Path) -> None:
    if image.ndim != 3:
        raise ValueError(f"{sample_path}: {key} must be rank 3, got {image.shape}")
    if image.shape[0] not in (1, 3) and image.shape[-1] not in (1, 3):
        raise ValueError(f"{sample_path}: cannot infer channel dimension for {key} with shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{sample_path}: {key} must be uint8, got {image.dtype}")


def validate_buffer(buffer_dir: Path) -> dict[str, int | float]:
    buffer_dir = buffer_dir.expanduser().resolve()
    meta_path = buffer_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    if meta.get("format") != "robotwin_pi05_prototype_replay_v2":
        raise ValueError(f"Unexpected buffer format: {meta.get('format')!r}")

    manifest = _read_jsonl(buffer_dir / "manifest.jsonl")
    segments = _read_jsonl(buffer_dir / "segments.jsonl")
    expected_count = int(meta["sample_count"])
    if len(manifest) != expected_count or int(meta["saved_frame_count"]) != expected_count:
        raise ValueError(
            f"Sample count mismatch: manifest={len(manifest)}, sample_count={meta['sample_count']}, "
            f"saved_frame_count={meta['saved_frame_count']}"
        )
    if len(segments) != int(meta["segment_count"]):
        raise ValueError(f"Segment count mismatch: records={len(segments)}, meta={meta['segment_count']}")

    horizon = int(meta["action_horizon"])
    seen_paths: set[str] = set()
    seen_frames: set[tuple[int, int]] = set()
    manifest_by_segment: dict[str, list[dict[str, Any]]] = {}
    for record in manifest:
        relative_path = str(record["sample_path"])
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate sample path in manifest: {relative_path}")
        seen_paths.add(relative_path)
        frame_key = (int(record["episode_index"]), int(record["frame_index"]))
        if frame_key in seen_frames:
            raise ValueError(f"Duplicate replay frame in manifest: episode={frame_key[0]}, frame={frame_key[1]}")
        seen_frames.add(frame_key)
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"Replay record has an empty prompt: {relative_path}")
        sample_path = buffer_dir / relative_path
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        with np.load(sample_path, allow_pickle=False) as sample:
            for camera in CAMERA_KEYS:
                if camera not in sample:
                    raise KeyError(f"{sample_path}: missing camera {camera}")
                _validate_image(np.asarray(sample[camera]), camera, sample_path)
            state = np.asarray(sample["state"])
            action = np.asarray(sample["action"])
            if state.shape != (14,):
                raise ValueError(f"{sample_path}: expected state shape (14,), got {state.shape}")
            if action.shape != (horizon, 14):
                raise ValueError(f"{sample_path}: expected action shape ({horizon}, 14), got {action.shape}")
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                raise ValueError(f"{sample_path}: state/action contains non-finite values")
        manifest_by_segment.setdefault(str(record["segment_id"]), []).append(record)

    prototype_path = buffer_dir / "prototypes.npz"
    feature_path = buffer_dir / "behavior_features.npz"
    if not prototype_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Missing prototypes.npz or behavior_features.npz")
    with np.load(prototype_path, allow_pickle=False) as prototypes, np.load(
        feature_path, allow_pickle=False
    ) as features:
        for segment in segments:
            segment_id = str(segment["segment_id"])
            selected = manifest_by_segment.get(segment_id, [])
            if len(selected) != len(segment.get("selected", [])):
                raise ValueError(f"Selected-record mismatch for {segment_id}")
            if not selected:
                raise ValueError(f"Segment {segment_id} has no selected replay samples")
            prototype_key = str(segment["prototype_key"])
            if prototype_key not in prototypes:
                raise KeyError(f"Missing prototype key {prototype_key}")
            prefix = prototype_key.removesuffix("_fused")
            candidate_key = f"{prefix}_fused_candidates"
            selected_key = f"{prefix}_selected_offsets"
            if candidate_key not in features or selected_key not in features:
                raise KeyError(f"Missing behavior feature keys for {segment_id}")
            candidate_count = int(features[candidate_key].shape[0])
            if candidate_count != int(segment["length"]):
                raise ValueError(
                    f"Candidate count mismatch for {segment_id}: features={candidate_count}, length={segment['length']}"
                )
            selected_offsets = np.asarray(features[selected_key], dtype=np.int32)
            if selected_offsets.shape != (len(selected),):
                raise ValueError(f"Selected offset shape mismatch for {segment_id}: {selected_offsets.shape}")
            if np.any(selected_offsets < 0) or np.any(selected_offsets >= candidate_count):
                raise ValueError(f"Selected offset out of range for {segment_id}")
            coverage_error = float(segment["coverage_error"])
            if not 0.0 <= coverage_error <= 2.0 + 1e-5:
                raise ValueError(f"Invalid cosine coverage error for {segment_id}: {coverage_error}")

    norm_stats = buffer_dir / str(meta["norm_assets_dir"]) / str(meta["norm_asset_id"]) / "norm_stats.json"
    if not norm_stats.exists():
        raise FileNotFoundError(f"Missing replay normalization stats: {norm_stats}")
    source_frames = int(meta["source_frame_count"])
    expected_ratio = expected_count / max(source_frames, 1)
    if not np.isclose(float(meta["compression_ratio"]), expected_ratio, atol=1e-10):
        raise ValueError("compression_ratio is inconsistent with source/sample counts")
    return {
        "samples": expected_count,
        "segments": len(segments),
        "source_frames": source_frames,
        "compression_ratio": expected_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer_dir")
    args = parser.parse_args()
    summary = validate_buffer(Path(args.buffer_dir))
    print("Prototype Replay V2 buffer validation passed")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
