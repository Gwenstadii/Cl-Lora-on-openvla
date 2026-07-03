#!/usr/bin/env python3
"""Build a RoboTwin/pi05 Prototype Replay V2 buffer.

V2 combines three complementary mechanisms:
1. AtomicVLA-inspired dual-arm end-effector kinematic segmentation.
2. Task-conditioned context/action latents fused with physical motion features.
3. Robust, temporally separated k-center selection inside every segment.

The saved replay schema is intentionally identical to the V1 buffer schema, so
existing replay loaders and training configs can consume V2 buffers unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any

import h5py
import jax
from lerobot.common.datasets import lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

try:
    import build_robotwin_prototype_buffer as _v1
except ModuleNotFoundError:  # Imported as scripts.build_robotwin_prototype_buffer_v2 in tests.
    from scripts import build_robotwin_prototype_buffer as _v1

try:
    from validate_robotwin_prototype_buffer_v2 import validate_buffer as _validate_completed_buffer
except ModuleNotFoundError:
    from scripts.validate_robotwin_prototype_buffer_v2 import validate_buffer as _validate_completed_buffer


CAMERA_KEYS = _v1.CAMERA_KEYS
MODE_NAMES = (
    "idle",
    "x_positive",
    "x_negative",
    "y_positive",
    "y_negative",
    "z_positive",
    "z_negative",
    "rot_x_positive",
    "rot_x_negative",
    "rot_y_positive",
    "rot_y_negative",
    "rot_z_positive",
    "rot_z_negative",
    "gripper_opening",
    "gripper_closing",
)


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    joints: np.ndarray  # [raw_frames, 14]
    poses: np.ndarray  # [raw_frames, 2, 7], xyz + wxyz quaternion
    grippers: np.ndarray  # [raw_frames, 2]


@dataclass(frozen=True)
class MotionSignals:
    descriptors: np.ndarray  # [training_frames, physical_dim]
    modes: np.ndarray  # [training_frames, 2]
    translation: np.ndarray  # [training_frames, 2, 3]
    rotation: np.ndarray  # [training_frames, 2, 3]
    gripper_delta: np.ndarray  # [training_frames, 2]


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: list[int]
    prototype: np.ndarray
    prototype_error: float
    coverage_error: float
    inlier_coverage_error: float
    inlier_count: int
    outlier_count: int
    effective_temporal_gap: int
    coverage_gains: list[float]


@dataclass(frozen=True)
class _FixedPromptTransform:
    prompt: str

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "prompt": self.prompt}


def _require_dataset(h5: h5py.File, key: str) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Raw episode {h5.filename} is missing required dataset {key}")
    value = np.asarray(h5[key][()])
    if value.size == 0:
        raise ValueError(f"Raw episode {h5.filename} has an empty dataset {key}")
    return value


def _flatten_scalar_series(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    raise ValueError(f"Expected {name} to have shape [frames] or [frames, 1], got {value.shape}")


def _normalize_quaternions(quaternions: np.ndarray, source: Path) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"Raw episode {source} contains a near-zero end-effector quaternion")
    return quaternions / norms


def _load_raw_episode(path: Path, quaternion_order: str) -> RawEpisode:
    with h5py.File(path, "r") as h5:
        left_arm = _require_dataset(h5, "/joint_action/left_arm").astype(np.float32)
        right_arm = _require_dataset(h5, "/joint_action/right_arm").astype(np.float32)
        left_joint_gripper = _flatten_scalar_series(
            _require_dataset(h5, "/joint_action/left_gripper"), "left joint gripper"
        )
        right_joint_gripper = _flatten_scalar_series(
            _require_dataset(h5, "/joint_action/right_gripper"), "right joint gripper"
        )
        left_pose = _require_dataset(h5, "/endpose/left_endpose").astype(np.float32)
        right_pose = _require_dataset(h5, "/endpose/right_endpose").astype(np.float32)
        left_gripper = _flatten_scalar_series(_require_dataset(h5, "/endpose/left_gripper"), "left gripper")
        right_gripper = _flatten_scalar_series(_require_dataset(h5, "/endpose/right_gripper"), "right gripper")

    if left_arm.ndim != 2 or right_arm.ndim != 2 or left_arm.shape[1] != 6 or right_arm.shape[1] != 6:
        raise ValueError(f"Expected two 6D arm joint arrays in {path}, got {left_arm.shape} and {right_arm.shape}")
    if left_pose.ndim != 2 or right_pose.ndim != 2 or left_pose.shape[1] != 7 or right_pose.shape[1] != 7:
        raise ValueError(
            f"Expected two [frames, 7] end-pose arrays in {path}, got {left_pose.shape} and {right_pose.shape}"
        )

    lengths = {
        left_arm.shape[0],
        right_arm.shape[0],
        left_joint_gripper.shape[0],
        right_joint_gripper.shape[0],
        left_pose.shape[0],
        right_pose.shape[0],
        left_gripper.shape[0],
        right_gripper.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError(f"Raw episode arrays have inconsistent frame counts in {path}: {sorted(lengths)}")

    joints = np.concatenate(
        [
            left_arm,
            left_joint_gripper[:, None],
            right_arm,
            right_joint_gripper[:, None],
        ],
        axis=-1,
    )
    poses = np.stack([left_pose, right_pose], axis=1)
    if quaternion_order == "xyzw":
        poses = poses[..., [0, 1, 2, 6, 3, 4, 5]]
    poses[..., 3:7] = _normalize_quaternions(poses[..., 3:7], path)
    grippers = np.stack([left_gripper, right_gripper], axis=-1)

    for name, value in (("joints", joints), ("poses", poses), ("grippers", grippers)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Raw episode {path} contains non-finite {name} values")
    return RawEpisode(path=path, joints=joints, poses=poses, grippers=grippers)


def _resolve_raw_episode(raw_data_dir: Path, episode_index: int) -> Path:
    candidates = (
        raw_data_dir / f"episode{episode_index}.hdf5",
        raw_data_dir / f"episode_{episode_index}.hdf5",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find raw episode {episode_index} under {raw_data_dir}. "
        f"Tried: {', '.join(str(path.name) for path in candidates)}"
    )


def _list_raw_episode_paths(raw_data_dir: Path) -> dict[int, Path]:
    raw_paths: dict[int, Path] = {}
    for pattern in ("episode*.hdf5", "episode_*.hdf5"):
        for path in raw_data_dir.glob(pattern):
            stem = path.stem
            if stem.startswith("episode_"):
                suffix = stem[len("episode_") :]
            elif stem.startswith("episode"):
                suffix = stem[len("episode") :]
            else:
                continue
            if not suffix.isdigit():
                continue
            raw_paths[int(suffix)] = path.resolve()
    if not raw_paths:
        raise FileNotFoundError(f"Cannot find raw episode*.hdf5 files under {raw_data_dir}")
    return dict(sorted(raw_paths.items()))


def _validate_raw_alignment(raw: RawEpisode, lerobot_states: np.ndarray, *, atol: float, rtol: float) -> int:
    min_raw_frames = lerobot_states.shape[0] + 1
    if raw.joints.shape[0] < min_raw_frames:
        raise ValueError(
            f"Raw/LeRobot frame mismatch for {raw.path}: raw={raw.joints.shape[0]}, "
            f"LeRobot={lerobot_states.shape[0]}; expected at least raw=LeRobot+1 because process_data.py pairs "
            "state[t] with action[t+1]."
        )
    target_states = lerobot_states[:, :14]
    max_start = raw.joints.shape[0] - target_states.shape[0]
    best: tuple[float, int, tuple[int, int]] | None = None
    for start_offset in range(max_start + 1):
        raw_states = raw.joints[start_offset : start_offset + target_states.shape[0]]
        if np.allclose(raw_states, target_states, atol=atol, rtol=rtol):
            return start_offset
        error = np.abs(raw_states - target_states)
        max_index = np.unravel_index(int(np.argmax(error)), error.shape)
        max_error = float(error[max_index])
        if best is None or max_error < best[0]:
            best = (max_error, start_offset, (int(max_index[0]), int(max_index[1])))

    assert best is not None
    max_error, best_offset, max_index = best
    raise ValueError(
        f"Raw episode {raw.path} is not aligned with the LeRobot episode: best_offset={best_offset}, "
        f"max_abs_error={max_error:.6g} at frame={max_index[0]}, dim={max_index[1]}. "
        "Check episode ordering and source task/setting."
    )


def _match_raw_episode(
    *,
    raw_paths: dict[int, Path],
    raw_cache: dict[int, RawEpisode],
    used_raw_indices: set[int],
    preferred_raw_index: int,
    lerobot_states: np.ndarray,
    quaternion_order: str,
    atol: float,
    rtol: float,
) -> tuple[int, RawEpisode, int]:
    candidate_indices = []
    if preferred_raw_index in raw_paths:
        candidate_indices.append(preferred_raw_index)
    candidate_indices.extend(index for index in raw_paths if index != preferred_raw_index)

    errors: list[str] = []
    for raw_index in candidate_indices:
        if raw_index in used_raw_indices:
            continue
        raw_episode = raw_cache.get(raw_index)
        if raw_episode is None:
            raw_episode = _load_raw_episode(raw_paths[raw_index], quaternion_order)
            raw_cache[raw_index] = raw_episode
        try:
            raw_start_offset = _validate_raw_alignment(raw_episode, lerobot_states, atol=atol, rtol=rtol)
        except ValueError as exc:
            if len(errors) < 5:
                errors.append(f"raw episode {raw_index}: {exc}")
            continue
        return raw_index, raw_episode, raw_start_offset

    details = "\n  ".join(errors)
    raise ValueError(
        f"Could not align this LeRobot episode with any unused raw episode under "
        f"{next(iter(raw_paths.values())).parent}. This usually means the LeRobot repo was generated from a "
        "different raw directory, or the raw state schema/order changed during conversion."
        + (f"\nFirst failed candidates:\n  {details}" if details else "")
    )


def _quaternion_rotation_vector(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Return the shortest relative rotation vector for wxyz quaternions."""
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    w0, xyz0 = q0[0], q0[1:]
    w1, xyz1 = q1[0], q1[1:]
    rel_w = w1 * w0 + float(np.dot(xyz1, xyz0))
    rel_xyz = w0 * xyz1 - w1 * xyz0 - np.cross(xyz1, xyz0)
    rel_w = float(np.clip(rel_w, -1.0, 1.0))
    sin_half = float(np.linalg.norm(rel_xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, rel_w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return (rel_xyz / sin_half * angle).astype(np.float32)


def _moving_average_columns(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return np.stack([_v1._moving_average(values[:, dim], window) for dim in range(values.shape[1])], axis=-1)


def _compute_motion_signals(
    raw: RawEpisode,
    num_training_frames: int,
    *,
    start_offset: int,
    window: int,
    translation_threshold: float,
    rotation_threshold: float,
    gripper_threshold: float,
    descriptor_clip: float,
) -> MotionSignals:
    if min(translation_threshold, rotation_threshold, gripper_threshold) <= 0:
        raise ValueError("All kinematic thresholds must be positive")
    if window < 1:
        raise ValueError("--kinematic-window must be at least 1")

    translation = np.zeros((num_training_frames, 2, 3), dtype=np.float32)
    rotation = np.zeros((num_training_frames, 2, 3), dtype=np.float32)
    gripper_delta = np.zeros((num_training_frames, 2), dtype=np.float32)
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if start_offset + num_training_frames > raw.poses.shape[0]:
        raise ValueError(
            f"Raw episode has too few frames for start_offset={start_offset} and "
            f"num_training_frames={num_training_frames}"
        )

    raw_last = raw.poses.shape[0] - 1
    for frame in range(num_training_frames):
        raw_frame = start_offset + frame
        future = min(raw_frame + window, raw_last)
        for arm in range(2):
            translation[frame, arm] = raw.poses[future, arm, :3] - raw.poses[raw_frame, arm, :3]
            rotation[frame, arm] = _quaternion_rotation_vector(
                raw.poses[raw_frame, arm, 3:7], raw.poses[future, arm, 3:7]
            )
            gripper_delta[frame, arm] = raw.grippers[future, arm] - raw.grippers[raw_frame, arm]

    scaled = np.concatenate(
        [
            translation / translation_threshold,
            rotation / rotation_threshold,
            gripper_delta[..., None] / gripper_threshold,
        ],
        axis=-1,
    )
    smoothed = np.stack(
        [_moving_average_columns(scaled[:, arm], window) for arm in range(2)], axis=1
    )
    smoothed = np.clip(smoothed, -descriptor_clip, descriptor_clip)

    modes = np.zeros((num_training_frames, 2), dtype=np.int32)
    for frame in range(num_training_frames):
        for arm in range(2):
            component = int(np.argmax(np.abs(smoothed[frame, arm])))
            signed_value = float(smoothed[frame, arm, component])
            if abs(signed_value) >= 1.0:
                modes[frame, arm] = 1 + 2 * component + int(signed_value < 0.0)

    current_gripper = np.clip(raw.grippers[start_offset : start_offset + num_training_frames], -1.0, 1.0)
    arm_activity = np.linalg.norm(smoothed, axis=-1)
    descriptors = np.concatenate(
        [
            smoothed[:, 0],
            current_gripper[:, 0:1],
            smoothed[:, 1],
            current_gripper[:, 1:2],
            arm_activity,
        ],
        axis=-1,
    ).astype(np.float32)
    return MotionSignals(
        descriptors=descriptors,
        modes=modes,
        translation=translation,
        rotation=rotation,
        gripper_delta=gripper_delta,
    )


def _run_length_encode_modes(modes: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    if modes.shape[0] == 0:
        return []
    runs = []
    start = 0
    current = tuple(int(v) for v in modes[0])
    for index in range(1, modes.shape[0]):
        mode = tuple(int(v) for v in modes[index])
        if mode != current:
            runs.append((start, index, current))
            start = index
            current = mode
    runs.append((start, modes.shape[0], current))
    return runs


def _mean_descriptor(descriptors: np.ndarray, segment: tuple[int, int]) -> np.ndarray:
    start, end = segment
    feature = descriptors[start:end].mean(axis=0)
    norm = float(np.linalg.norm(feature))
    return feature / max(norm, 1e-8)


def _descriptor_distance(descriptors: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> float:
    return 1.0 - float(np.dot(_mean_descriptor(descriptors, first), _mean_descriptor(descriptors, second)))


def _merge_short_segments_by_similarity(
    segments: list[tuple[int, int]], descriptors: np.ndarray, min_len: int
) -> list[tuple[int, int]]:
    segments = list(segments)
    while len(segments) > 1:
        short_index = next((i for i, (start, end) in enumerate(segments) if end - start < min_len), None)
        if short_index is None:
            break
        start, end = segments[short_index]
        choices: list[tuple[float, int]] = []
        if short_index > 0:
            choices.append((_descriptor_distance(descriptors, segments[short_index], segments[short_index - 1]), -1))
        if short_index + 1 < len(segments):
            choices.append((_descriptor_distance(descriptors, segments[short_index], segments[short_index + 1]), 1))
        _, direction = min(choices, key=lambda item: item[0])
        if direction < 0:
            previous_start, _ = segments[short_index - 1]
            segments[short_index - 1 : short_index + 1] = [(previous_start, end)]
        else:
            _, next_end = segments[short_index + 1]
            segments[short_index : short_index + 2] = [(start, next_end)]
    return segments


def _build_kinematic_segments(
    signals: MotionSignals,
    *,
    persistence_frames: int,
    min_boundary_gap: int,
    min_segment_len: int,
    max_segment_len: int,
) -> list[tuple[int, int]]:
    num_frames = signals.modes.shape[0]
    if num_frames == 0:
        return []
    runs = _run_length_encode_modes(signals.modes)
    boundaries = [0]
    for start, end, _ in runs[1:]:
        if end - start < persistence_frames:
            continue
        if start - boundaries[-1] >= min_boundary_gap:
            boundaries.append(start)
    if boundaries[-1] != num_frames:
        boundaries.append(num_frames)
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    segments = _merge_short_segments_by_similarity(segments, signals.descriptors, min_segment_len)
    if max_segment_len > 0:
        segments = _v1._split_long_segments(segments, max_segment_len)
        segments = _merge_short_segments_by_similarity(segments, signals.descriptors, min_segment_len)
    return segments or [(0, num_frames)]


def _deterministic_noise(indices: list[int], shape: tuple[int, ...], seed: int, sample_index: int) -> np.ndarray:
    noise = []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, sample_index]))
        noise.append(rng.standard_normal(shape, dtype=np.float32))
    return np.asarray(noise, dtype=np.float32)


def _make_behavior_feature_fn(model, task_id: int):
    @jax.jit
    def extract(obs: _model.Observation, actions: jax.Array, noise: jax.Array, time: jax.Array):
        return model.extract_behavior_features(obs, actions, noise, time, task_id=task_id)

    return extract


def _extract_model_features(
    transformed_dataset,
    frame_indices: list[int],
    *,
    model,
    task_id: int,
    batch_size: int,
    flow_time: float,
    noise_seed: int,
    noise_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < flow_time < 1.0:
        raise ValueError("--flow-time must be strictly between 0 and 1")
    if noise_samples < 1:
        raise ValueError("--noise-samples must be at least 1")
    extract_fn = _make_behavior_feature_fn(model, task_id)
    context_outputs = []
    action_outputs = []
    for offset in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[offset : offset + batch_size]
        batch_dict = _v1._collate_transformed([transformed_dataset[index] for index in batch_indices])
        if "actions" not in batch_dict:
            raise KeyError("Transformed dataset item has no 'actions' field")
        actions = np.asarray(batch_dict["actions"], dtype=np.float32)
        obs = _model.Observation.from_dict(batch_dict)
        time = np.full((len(batch_indices),), flow_time, dtype=np.float32)
        context = None
        action_sum = None
        for noise_index in range(noise_samples):
            noise = _deterministic_noise(
                batch_indices, tuple(actions.shape[1:]), noise_seed, noise_index
            )
            batch_context, batch_action = extract_fn(obs, actions, noise, time)
            batch_context = np.asarray(jax.device_get(batch_context), dtype=np.float32)
            batch_action = np.asarray(jax.device_get(batch_action), dtype=np.float32)
            if context is None:
                context = batch_context
                action_sum = batch_action.copy()
            else:
                action_sum += batch_action
        context_outputs.append(context)
        action_outputs.append(action_sum / float(noise_samples))
    return np.concatenate(context_outputs, axis=0), np.concatenate(action_outputs, axis=0)


def _pool_local_action_features(
    action_tokens: np.ndarray,
    segment_start: int,
    segment_end: int,
    local_horizon: int,
) -> np.ndarray:
    pooled = []
    for episode_offset in range(segment_start, segment_end):
        steps_inside_segment = segment_end - episode_offset
        steps = max(1, min(local_horizon, steps_inside_segment, action_tokens.shape[1]))
        pooled.append(action_tokens[episode_offset, :steps].mean(axis=0))
    return np.asarray(pooled, dtype=np.float32)


def _fuse_behavior_features(
    context: np.ndarray,
    action: np.ndarray,
    physical: np.ndarray,
    *,
    context_weight: float,
    action_weight: float,
    physical_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    weights = np.asarray([context_weight, action_weight, physical_weight], dtype=np.float32)
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Feature weights must be non-negative and have a positive sum")
    weights /= weights.sum()
    normalized = {
        "context": _v1._normalize_rows(context.astype(np.float32)),
        "action": _v1._normalize_rows(action.astype(np.float32)),
        "physical": _v1._normalize_rows(physical.astype(np.float32)),
    }
    fused = np.concatenate(
        [
            math.sqrt(float(weights[0])) * normalized["context"],
            math.sqrt(float(weights[1])) * normalized["action"],
            math.sqrt(float(weights[2])) * normalized["physical"],
        ],
        axis=-1,
    )
    return _v1._normalize_rows(fused), normalized


def _coverage_error(features: np.ndarray, selected: list[int]) -> float:
    if not selected:
        return 1.0
    best_similarity = np.max(features @ features[selected].T, axis=1)
    return float(np.mean(1.0 - best_similarity))


def _select_coverage_representatives(
    features: np.ndarray,
    *,
    top_k: int,
    temporal_min_gap: int,
    outlier_mad_scale: float,
) -> SelectionResult:
    features = _v1._normalize_rows(np.asarray(features, dtype=np.float32))
    count = features.shape[0]
    if count == 0:
        raise ValueError("Cannot select representatives from an empty segment")
    target_count = min(max(1, top_k), count)
    similarity = features @ features.T
    medoid = int(np.argmax(similarity.mean(axis=1)))
    distance_to_medoid = 1.0 - similarity[:, medoid]

    inliers = np.arange(count, dtype=np.int32)
    if count > target_count + 2 and outlier_mad_scale > 0.0:
        median = float(np.median(distance_to_medoid))
        mad = float(np.median(np.abs(distance_to_medoid - median)))
        if mad > 1e-8:
            cutoff = median + outlier_mad_scale * 1.4826 * mad
            inliers = np.flatnonzero(distance_to_medoid <= cutoff).astype(np.int32)
            if inliers.size < target_count:
                inliers = np.argsort(distance_to_medoid)[:target_count].astype(np.int32)

    inlier_similarity = similarity[np.ix_(inliers, inliers)]
    medoid = int(inliers[int(np.argmax(inlier_similarity.mean(axis=1)))])
    if target_count > 1:
        span = int(inliers.max() - inliers.min()) if inliers.size else 0
        feasible_gap = span // (target_count - 1)
        effective_gap = min(max(0, temporal_min_gap), max(0, feasible_gap))
    else:
        effective_gap = 0

    selected = [medoid]
    gains = [_coverage_error(features, []) - _coverage_error(features, selected)]
    actual_gap = effective_gap
    while len(selected) < target_count:
        remaining = [int(index) for index in inliers if int(index) not in selected]
        gap = effective_gap
        valid = [index for index in remaining if all(abs(index - chosen) >= gap for chosen in selected)]
        while not valid and gap > 0:
            gap -= 1
            valid = [index for index in remaining if all(abs(index - chosen) >= gap for chosen in selected)]
        if not valid:
            break
        minimum_distances = {
            index: float(np.min(1.0 - similarity[index, selected])) for index in valid
        }
        chosen = max(valid, key=lambda index: (minimum_distances[index], -distance_to_medoid[index], -index))
        previous_error = _coverage_error(features, selected)
        selected.append(chosen)
        actual_gap = min(actual_gap, gap)
        gains.append(previous_error - _coverage_error(features, selected))

    prototype = features[inliers].mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    prototype_error = float(np.mean(1.0 - features @ prototype))
    selected_inlier_positions = [int(np.flatnonzero(inliers == index)[0]) for index in selected]
    return SelectionResult(
        selected_indices=selected,
        prototype=prototype.astype(np.float32),
        prototype_error=prototype_error,
        coverage_error=_coverage_error(features, selected),
        inlier_coverage_error=_coverage_error(features[inliers], selected_inlier_positions),
        inlier_count=int(inliers.size),
        outlier_count=int(count - inliers.size),
        effective_temporal_gap=actual_gap,
        coverage_gains=gains,
    )


def _modal_mode_name(modes: np.ndarray, arm: int) -> str:
    values, counts = np.unique(modes[:, arm], return_counts=True)
    return MODE_NAMES[int(values[int(np.argmax(counts))])]


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)


def _build_dataset_context(args: argparse.Namespace):
    config = _config.get_config(args.train_config_name)
    if args.exp_name:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != args.repo_id:
        print(f"[W] config repo_id={data_config.repo_id!r} differs from --repo-id={args.repo_id!r}; using --repo-id")
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    base_dataset = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={
            key: [step / meta.fps for step in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    raw_dataset = base_dataset
    if data_config.prompt_from_task:
        raw_dataset = _data_loader.TransformedDataset(
            raw_dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)]
        )
    if args.fixed_prompt:
        raw_dataset = _data_loader.TransformedDataset(raw_dataset, [_FixedPromptTransform(args.fixed_prompt)])
    ranges = _v1._episode_ranges(base_dataset)
    episode_indices = _v1._select_episodes(
        ranges,
        num_episodes=args.num_episodes,
        episode_indices=_v1._parse_episode_indices(args.episode_indices),
        random_episodes=args.random_episodes,
        seed=args.seed,
    )
    return config, data_config, meta, raw_dataset, ranges, episode_indices


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_args = {
        "num_episodes": args.num_episodes,
        "kinematic_window": args.kinematic_window,
        "persistence_frames": args.persistence_frames,
        "min_boundary_gap": args.min_boundary_gap,
        "min_segment_len": args.min_segment_len,
        "action_local_horizon": args.action_local_horizon,
        "noise_samples": args.noise_samples,
        "feature_batch_size": args.feature_batch_size,
        "top_k": args.top_k,
    }
    invalid = [name for name, value in positive_integer_args.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive integers: {', '.join(invalid)}")
    if args.max_segment_len < 0 or args.temporal_min_gap < 0:
        raise ValueError("--max-segment-len and --temporal-min-gap cannot be negative")
    if args.descriptor_clip <= 0.0:
        raise ValueError("--descriptor-clip must be positive")
    if args.alignment_atol < 0.0 or args.alignment_rtol < 0.0:
        raise ValueError("Alignment tolerances cannot be negative")
    weights = (args.context_weight, args.action_weight, args.physical_weight)
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Feature weights must be non-negative and have a positive sum")


def build_buffer(args: argparse.Namespace) -> None:
    _validate_args(args)
    config, data_config, meta, raw_dataset, ranges, episode_indices = _build_dataset_context(args)
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    if not raw_data_dir.is_dir():
        raise NotADirectoryError(f"Raw data directory does not exist: {raw_data_dir}")

    raw_paths = _list_raw_episode_paths(raw_data_dir)
    raw_cache: dict[int, RawEpisode] = {}
    used_raw_indices: set[int] = set()
    episode_sources: dict[int, tuple[RawEpisode, np.ndarray, int, int]] = {}
    for episode_index in episode_indices:
        start, end = ranges[episode_index]
        lerobot_states, _ = _v1._collect_physical_signals(
            raw_dataset, start, end, config.model.action_horizon
        )
        raw_episode_index, raw_episode, raw_start_offset = _match_raw_episode(
            raw_paths=raw_paths,
            raw_cache=raw_cache,
            used_raw_indices=used_raw_indices,
            preferred_raw_index=episode_index,
            lerobot_states=lerobot_states,
            quaternion_order=args.raw_quaternion_order,
            atol=args.alignment_atol,
            rtol=args.alignment_rtol,
        )
        used_raw_indices.add(raw_episode_index)
        episode_sources[episode_index] = (raw_episode, lerobot_states, raw_start_offset, raw_episode_index)
        print(
            f"Validated LeRobot episode {episode_index} -> raw episode {raw_episode_index}: "
            f"raw_frames={raw_episode.joints.shape[0]}, "
            f"training_frames={end-start}, raw_start_offset={raw_start_offset}"
        )

    if args.validate_only:
        print(f"Validated {len(episode_indices)} raw/LeRobot episode pairs; no buffer was written")
        return
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --validate-only is used")

    transformed_dataset = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
    checkpoint_dir = _v1._resolve_checkpoint_dir(config, args)
    model = _v1._load_model(config, checkpoint_dir)
    task_id = int(getattr(config.model, "cl_task_id", 0))
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)
    (output_dir / "kinematics").mkdir(parents=True, exist_ok=True)

    norm_assets_rel = Path("assets")
    if data_config.norm_stats is None:
        raise ValueError("Prototype V2 requires the old task normalization statistics")
    _normalize.save(output_dir / norm_assets_rel / args.repo_id, data_config.norm_stats)

    manifest_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    prototype_arrays: dict[str, np.ndarray] = {}
    behavior_feature_arrays: dict[str, np.ndarray] = {}
    episode_diagnostics: list[dict[str, Any]] = []
    total_source_frames = 0
    sample_id = 0
    global_segment_id = 0

    for episode_index in episode_indices:
        start, end = ranges[episode_index]
        num_frames = end - start
        raw_episode, _, raw_start_offset, raw_episode_index = episode_sources[episode_index]
        signals = _compute_motion_signals(
            raw_episode,
            num_frames,
            start_offset=raw_start_offset,
            window=args.kinematic_window,
            translation_threshold=args.translation_threshold,
            rotation_threshold=args.rotation_threshold,
            gripper_threshold=args.gripper_threshold,
            descriptor_clip=args.descriptor_clip,
        )
        segments = _build_kinematic_segments(
            signals,
            persistence_frames=args.persistence_frames,
            min_boundary_gap=args.min_boundary_gap,
            min_segment_len=args.min_segment_len,
            max_segment_len=args.max_segment_len,
        )
        active_ratio = float(np.mean(np.any(signals.modes != 0, axis=1)))
        segment_lengths = [end_index - start_index for start_index, end_index in segments]
        episode_diagnostics.append(
            {
                "episode_index": episode_index,
                "raw_episode_index": int(raw_episode_index),
                "active_motion_ratio": active_ratio,
                "segment_count": len(segments),
                "raw_start_offset": int(raw_start_offset),
                "min_segment_length": int(min(segment_lengths)),
                "median_segment_length": float(np.median(segment_lengths)),
                "max_segment_length": int(max(segment_lengths)),
            }
        )
        print(
            f"Segmented episode {episode_index}: segments={len(segments)}, "
            f"active_motion_ratio={active_ratio:.3f}, lengths={min(segment_lengths)}/"
            f"{np.median(segment_lengths):.1f}/{max(segment_lengths)}"
        )
        if active_ratio < 0.02 or active_ratio > 0.98:
            print(
                f"[W] Episode {episode_index} has an extreme active-motion ratio; "
                "inspect thresholds and kinematics diagnostics before training"
            )
        np.savez_compressed(
            output_dir / "kinematics" / f"episode_{episode_index:04d}.npz",
            end_effector_poses=raw_episode.poses[raw_start_offset : raw_start_offset + num_frames],
            grippers=raw_episode.grippers[raw_start_offset : raw_start_offset + num_frames],
            translation_delta=signals.translation,
            rotation_delta=signals.rotation,
            gripper_delta=signals.gripper_delta,
            motion_descriptors=signals.descriptors,
            motion_modes=signals.modes,
            segment_bounds=np.asarray(segments, dtype=np.int32),
            raw_start_offset=np.asarray(raw_start_offset, dtype=np.int32),
            raw_episode_index=np.asarray(raw_episode_index, dtype=np.int32),
        )
        frame_indices = list(range(start, end))
        context_features, action_tokens = _extract_model_features(
            transformed_dataset,
            frame_indices,
            model=model,
            task_id=task_id,
            batch_size=args.feature_batch_size,
            flow_time=args.flow_time,
            noise_seed=args.noise_seed,
            noise_samples=args.noise_samples,
        )
        if context_features.shape[0] != num_frames or action_tokens.shape[0] != num_frames:
            raise RuntimeError(f"Feature count mismatch in episode {episode_index}")
        total_source_frames += num_frames

        for episode_segment_index, (local_start, local_end) in enumerate(segments):
            context = context_features[local_start:local_end]
            action = _pool_local_action_features(
                action_tokens, local_start, local_end, args.action_local_horizon
            )
            physical = signals.descriptors[local_start:local_end]
            fused, components = _fuse_behavior_features(
                context,
                action,
                physical,
                context_weight=args.context_weight,
                action_weight=args.action_weight,
                physical_weight=args.physical_weight,
            )
            selection = _select_coverage_representatives(
                fused,
                top_k=args.top_k,
                temporal_min_gap=args.temporal_min_gap,
                outlier_mad_scale=args.outlier_mad_scale,
            )

            proto_prefix = f"segment_{global_segment_id:06d}"
            prototype_arrays[f"{proto_prefix}_fused"] = selection.prototype
            behavior_feature_arrays[f"{proto_prefix}_fused_candidates"] = fused.astype(np.float16)
            behavior_feature_arrays[f"{proto_prefix}_selected_offsets"] = np.asarray(
                selection.selected_indices, dtype=np.int32
            )
            for component_name, component_features in components.items():
                component_prototype = component_features.mean(axis=0)
                component_prototype /= max(float(np.linalg.norm(component_prototype)), 1e-8)
                prototype_arrays[f"{proto_prefix}_{component_name}"] = component_prototype.astype(np.float32)

            segment_name = f"ep{episode_index:04d}_seg{episode_segment_index:04d}"
            selected_records = []
            for rank, (segment_offset, coverage_gain) in enumerate(
                zip(selection.selected_indices, selection.coverage_gains, strict=True)
            ):
                episode_frame = local_start + segment_offset
                global_index = start + episode_frame
                sample = raw_dataset[global_index]
                prompt = _v1._prompt_from_sample(sample, meta.tasks, args.fixed_prompt)
                sample_rel_path = Path("samples") / f"sample_{sample_id:06d}.npz"
                _v1._save_replay_sample(
                    sample, output_dir / sample_rel_path, horizon=config.model.action_horizon
                )
                record = {
                    "sample_path": str(sample_rel_path).replace("\\", "/"),
                    "task_name": args.task_name,
                    "repo_id": args.repo_id,
                    "episode_index": episode_index,
                    "raw_episode_index": int(raw_episode_index),
                    "raw_start_offset": int(raw_start_offset),
                    "frame_index": int(episode_frame),
                    "global_index": int(global_index),
                    "segment_id": segment_name,
                    "segment_start": int(local_start),
                    "segment_end": int(local_end),
                    "rank": rank,
                    "similarity": float(fused[segment_offset] @ selection.prototype),
                    "prototype_similarity": float(fused[segment_offset] @ selection.prototype),
                    "coverage_gain": float(coverage_gain),
                    "selection_method": "robust_temporal_k_center",
                    "valid_action_steps_in_segment": int(
                        min(config.model.action_horizon, local_end - episode_frame)
                    ),
                    "prompt": prompt,
                }
                manifest_records.append(record)
                selected_records.append(record)
                sample_id += 1

            segment_modes = signals.modes[local_start:local_end]
            segment_records.append(
                {
                    "task_name": args.task_name,
                    "repo_id": args.repo_id,
                    "episode_index": episode_index,
                    "raw_episode_index": int(raw_episode_index),
                    "raw_start_offset": int(raw_start_offset),
                    "raw_episode_path": str(raw_episode.path),
                    "segment_id": segment_name,
                    "start": int(local_start),
                    "end": int(local_end),
                    "length": int(local_end - local_start),
                    "left_motion_mode": _modal_mode_name(segment_modes, 0),
                    "right_motion_mode": _modal_mode_name(segment_modes, 1),
                    "prototype_key": f"{proto_prefix}_fused",
                    "prototype_error": selection.prototype_error,
                    "coverage_error": selection.coverage_error,
                    "inlier_coverage_error": selection.inlier_coverage_error,
                    "inlier_count": selection.inlier_count,
                    "outlier_count": selection.outlier_count,
                    "effective_temporal_gap": selection.effective_temporal_gap,
                    "selected": selected_records,
                }
            )
            global_segment_id += 1

    if not manifest_records:
        raise RuntimeError("No replay samples were selected; check segmentation and selection settings")
    saved_count = len(manifest_records)
    compression_ratio = saved_count / max(total_source_frames, 1)
    np.savez_compressed(output_dir / "prototypes.npz", **prototype_arrays)
    np.savez_compressed(output_dir / "behavior_features.npz", **behavior_feature_arrays)
    _v1._write_jsonl(output_dir / "manifest.jsonl", manifest_records)
    _v1._write_jsonl(output_dir / "segments.jsonl", segment_records)

    meta_record = {
        "format": "robotwin_pi05_prototype_replay_v2",
        "task_name": args.task_name,
        "repo_id": args.repo_id,
        "train_config_name": args.train_config_name,
        "cl_task_id": task_id,
        "source_checkpoint": str(checkpoint_dir),
        "raw_data_dir": str(raw_data_dir),
        "selected_episode_indices": episode_indices,
        "episode_diagnostics": episode_diagnostics,
        "num_source_episodes": len(episode_indices),
        "source_frame_count": total_source_frames,
        "saved_frame_count": saved_count,
        "sample_count": saved_count,
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
        "segmentation": {
            "method": "dual_arm_end_effector_kinematic",
            "kinematic_window": args.kinematic_window,
            "translation_threshold_m": args.translation_threshold,
            "rotation_threshold_rad": args.rotation_threshold,
            "gripper_threshold": args.gripper_threshold,
            "persistence_frames": args.persistence_frames,
            "min_boundary_gap": args.min_boundary_gap,
            "min_segment_len": args.min_segment_len,
            "max_segment_len": args.max_segment_len,
            "quaternion_order": args.raw_quaternion_order,
        },
        "feature_space": {
            "method": "task_conditioned_context_action_physical_fusion",
            "context_weight": args.context_weight,
            "action_weight": args.action_weight,
            "physical_weight": args.physical_weight,
            "flow_time": args.flow_time,
            "noise_seed": args.noise_seed,
            "noise_samples": args.noise_samples,
            "action_local_horizon": args.action_local_horizon,
        },
        "selection": {
            "method": "robust_temporal_k_center",
            "temporal_min_gap": args.temporal_min_gap,
            "outlier_mad_scale": args.outlier_mad_scale,
        },
        "diagnostic_artifacts": {
            "kinematics_dir": "kinematics",
            "behavior_features": "behavior_features.npz",
            "segment_records": "segments.jsonl",
            "prototypes": "prototypes.npz",
        },
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta_record, file, indent=2, ensure_ascii=False)

    validation_summary = _validate_completed_buffer(output_dir)

    print("Prototype Replay V2 buffer written")
    print(f"  output_dir: {output_dir}")
    print(f"  episodes: {len(episode_indices)}")
    print(f"  source frames: {total_source_frames}")
    print(f"  segments: {len(segment_records)}")
    print(f"  saved replay frames: {saved_count}")
    print(f"  compression_ratio: {compression_ratio:.6f}")
    print(f"  output validation: passed ({validation_summary['samples']} samples)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--checkpoint-id", default="10000")
    parser.add_argument("--checkpoint-dir", default=None, help="Step checkpoint directory containing params/")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing raw episodeN.hdf5 files")
    parser.add_argument("--raw-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--random-episodes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-prompt", default=None)

    parser.add_argument("--kinematic-window", type=int, default=5)
    parser.add_argument("--translation-threshold", type=float, default=0.03)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-threshold", type=float, default=0.1)
    parser.add_argument("--descriptor-clip", type=float, default=5.0)
    parser.add_argument("--persistence-frames", type=int, default=5)
    parser.add_argument("--min-boundary-gap", type=int, default=5)
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--max-segment-len", type=int, default=0)

    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--action-weight", type=float, default=0.50)
    parser.add_argument("--physical-weight", type=float, default=0.25)
    parser.add_argument("--action-local-horizon", type=int, default=10)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--noise-samples", type=int, default=2)
    parser.add_argument("--feature-batch-size", type=int, default=8)

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temporal-min-gap", type=int, default=5)
    parser.add_argument("--outlier-mad-scale", type=float, default=4.0)
    parser.add_argument("--alignment-atol", type=float, default=1e-5)
    parser.add_argument("--alignment-rtol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    build_buffer(parse_args())
