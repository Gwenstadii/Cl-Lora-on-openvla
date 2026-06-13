"""Build an offline replay buffer with implicit-skill physical segmentation.

Pipeline:
1) Load a (possibly task-filtered) LeRobot LIBERO dataset.
2) Segment each trajectory with AtomicVLA-style physics rules
   (translation / rotation / gripper change over short chunks).
3) Extract per-frame visual features from the model visual tower.
4) Compute segment prototype (feature center), then pick Top-K closest frames.
5) Save selected frames + state/action/prompt/task_id into a buffer directory.

This script is intentionally "offline-first": it does not modify training code.
It only builds replay assets for later mixed-sampling training.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.training.config as _config


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BuildReplayBufferConfig:
    # Training config name used to resolve dataset/model defaults.
    train_config_name: str

    # Output root folder. A task-specific subfolder is created inside.
    output_dir: str = "./replay_buffers"
    overwrite: bool = False

    # Path to model params for feature extraction.
    # Accepts either ".../<step>" (containing params/) or ".../params".
    # If omitted, falls back to train_config.weight_loader (typically base weights).
    model_params_path: str | None = None

    # Optional dataset root override. If None, use config.data.root.
    dataset_root: str | None = None
    # Optional task override. If None, use config.data.target_task_name.
    target_task_name: str | None = None
    max_episodes: int | None = None

    # Prototype-based frame selection.
    top_k_per_segment: int = 2

    # Segmentation hyperparameters (AtomicVLA-inspired).
    chunk_frames: int = 5
    min_segment_frames: int = 3
    min_stable_chunks: int = 2
    translation_threshold_m: float = 0.03  # 3 cm
    rotation_threshold_rad: float = 0.05
    gripper_threshold: float = 0.1

    # State parsing rules for physical motion.
    xyz_indices: tuple[int, int, int] = (0, 1, 2)
    # Use either quaternion or euler for orientation.
    quat_indices: tuple[int, int, int, int] | None = (3, 4, 5, 6)
    quat_order: Literal["xyzw", "wxyz"] = "xyzw"
    euler_indices: tuple[int, int, int] | None = None
    gripper_index: int = -1
    # Merge over-fragmented short segments into adjacent similar-motion segments.
    enable_short_segment_merge: bool = True
    short_segment_merge_frames: int = 10
    preserve_short_gripper_segments: bool = True


@dataclasses.dataclass(frozen=True)
class _ChunkMotion:
    idx: int
    end_frame: int
    dominant: str
    norm_mag: np.ndarray  # shape [7] => tx,ty,tz,rx,ry,rz,grip
    raw_mag: np.ndarray   # same order, but in physical units
    grip_delta: float


@dataclasses.dataclass
class _FrameRecord:
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    task: str


_AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz", "grip")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text or "unknown"


def _decode_task(task: Any) -> str:
    if isinstance(task, bytes):
        return task.decode("utf-8")
    return str(task)


def _extract_value(data: dict[str, Any], candidates: list[tuple[str, ...]]) -> Any:
    for path in candidates:
        cur: Any = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    paths = [".".join(p) for p in candidates]
    raise KeyError(f"Cannot find any candidate path in sample: {paths}")


def _to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with ndim=3, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        # Handle both [0,1] and [-1,1].
        lo, hi = float(arr.min()), float(arr.max())
        if lo >= -1.01 and hi <= 1.01:
            if lo < 0.0:
                arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)

    # CHW -> HWC if needed.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _quat_to_euler(quat: np.ndarray, order: Literal["xyzw", "wxyz"]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Quaternion must have shape [4], got {q.shape}")
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q

    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _state_to_pose_components(
    state: np.ndarray,
    *,
    xyz_indices: tuple[int, int, int],
    quat_indices: tuple[int, int, int, int] | None,
    quat_order: Literal["xyzw", "wxyz"],
    euler_indices: tuple[int, int, int] | None,
    gripper_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    xyz = s[list(xyz_indices)]
    if quat_indices is not None:
        rpy = _quat_to_euler(s[list(quat_indices)], quat_order)
    elif euler_indices is not None:
        rpy = s[list(euler_indices)]
    else:
        rpy = np.zeros(3, dtype=np.float64)
    grip = float(s[gripper_index])
    return xyz, rpy, grip


def _compute_chunk_motions(states: np.ndarray, cfg: BuildReplayBufferConfig) -> list[_ChunkMotion]:
    if len(states) <= cfg.chunk_frames:
        return []

    motions: list[_ChunkMotion] = []
    for i in range(0, len(states) - cfg.chunk_frames):
        s0 = states[i]
        s1 = states[i + cfg.chunk_frames]
        xyz0, rpy0, g0 = _state_to_pose_components(
            s0,
            xyz_indices=cfg.xyz_indices,
            quat_indices=cfg.quat_indices,
            quat_order=cfg.quat_order,
            euler_indices=cfg.euler_indices,
            gripper_index=cfg.gripper_index,
        )
        xyz1, rpy1, g1 = _state_to_pose_components(
            s1,
            xyz_indices=cfg.xyz_indices,
            quat_indices=cfg.quat_indices,
            quat_order=cfg.quat_order,
            euler_indices=cfg.euler_indices,
            gripper_index=cfg.gripper_index,
        )
        d_xyz = xyz1 - xyz0
        d_rpy = _wrap_to_pi(rpy1 - rpy0)
        d_g = g1 - g0

        raw_mag = np.asarray(
            [
                abs(d_xyz[0]),
                abs(d_xyz[1]),
                abs(d_xyz[2]),
                abs(d_rpy[0]),
                abs(d_rpy[1]),
                abs(d_rpy[2]),
                abs(d_g),
            ],
            dtype=np.float64,
        )
        norm_mag = np.asarray(
            [
                raw_mag[0] / max(cfg.translation_threshold_m, 1e-12),
                raw_mag[1] / max(cfg.translation_threshold_m, 1e-12),
                raw_mag[2] / max(cfg.translation_threshold_m, 1e-12),
                raw_mag[3] / max(cfg.rotation_threshold_rad, 1e-12),
                raw_mag[4] / max(cfg.rotation_threshold_rad, 1e-12),
                raw_mag[5] / max(cfg.rotation_threshold_rad, 1e-12),
                raw_mag[6] / max(cfg.gripper_threshold, 1e-12),
            ],
            dtype=np.float64,
        )
        if float(np.max(norm_mag)) < 1.0:
            dominant = "idle"
        else:
            dominant = _AXIS_NAMES[int(np.argmax(norm_mag))]
        motions.append(
            _ChunkMotion(
                idx=i,
                end_frame=min(i + cfg.chunk_frames, len(states) - 1),
                dominant=dominant,
                norm_mag=norm_mag.astype(np.float32),
                raw_mag=raw_mag.astype(np.float32),
                grip_delta=float(d_g),
            )
        )
    return motions


def _find_boundaries(motions: list[_ChunkMotion], cfg: BuildReplayBufferConfig) -> list[int]:
    if not motions:
        return []
    boundaries: set[int] = set()
    labels = [m.dominant for m in motions]

    # Force boundary on gripper transitions.
    for m in motions:
        if abs(m.grip_delta) >= cfg.gripper_threshold:
            boundaries.add(m.end_frame)

    # Boundary on dominant-mode changes with short stability check.
    for i in range(1, len(motions)):
        if labels[i] == labels[i - 1]:
            continue
        cand = labels[i]
        stable = True
        for j in range(cfg.min_stable_chunks):
            k = min(i + j, len(motions) - 1)
            if labels[k] != cand:
                stable = False
                break
        if stable:
            boundaries.add(motions[i].end_frame)
    return sorted(boundaries)


def _build_segments(num_frames: int, boundaries: list[int], _min_segment_frames: int) -> list[tuple[int, int]]:
    if num_frames <= 0:
        return []
    b = sorted({x for x in boundaries if 0 < x < num_frames})
    starts = [0, *b]
    ends = [*b, num_frames]
    segments: list[tuple[int, int]] = []
    for s, e in zip(starts, ends, strict=True):
        # IMPORTANT: do not drop short segments here.
        # Short segments should be merged (or explicitly preserved if they are gripper events),
        # otherwise the timeline gets holes before merge.
        if e > s:
            segments.append((s, e))  # [s, e)
    if not segments:
        segments.append((0, num_frames))
    return segments

def _segment_motion_stats(
    states: np.ndarray,
    cfg: BuildReplayBufferConfig,
    seg_s: int,
    seg_e: int,
) -> tuple[np.ndarray, float]:
    # Returns:
    #   motion_vec: [6] absolute motion sums for tx,ty,tz,rx,ry,rz
    #   grip_event_score: scalar gripper event strength, robust to long-segment accumulation
    #   (max of per-step delta and end-to-end delta)
    if seg_e - seg_s <= 1:
        return np.zeros(6, dtype=np.float64), 0.0

    motion = np.zeros(6, dtype=np.float64)
    _, _, first_g = _state_to_pose_components(
        states[seg_s],
        xyz_indices=cfg.xyz_indices,
        quat_indices=cfg.quat_indices,
        quat_order=cfg.quat_order,
        euler_indices=cfg.euler_indices,
        gripper_index=cfg.gripper_index,
    )
    max_step_grip_delta = 0.0
    last_g = first_g
    for i in range(seg_s, seg_e - 1):
        xyz0, rpy0, g0 = _state_to_pose_components(
            states[i],
            xyz_indices=cfg.xyz_indices,
            quat_indices=cfg.quat_indices,
            quat_order=cfg.quat_order,
            euler_indices=cfg.euler_indices,
            gripper_index=cfg.gripper_index,
        )
        xyz1, rpy1, g1 = _state_to_pose_components(
            states[i + 1],
            xyz_indices=cfg.xyz_indices,
            quat_indices=cfg.quat_indices,
            quat_order=cfg.quat_order,
            euler_indices=cfg.euler_indices,
            gripper_index=cfg.gripper_index,
        )
        d_xyz = np.abs(xyz1 - xyz0)
        d_rpy = np.abs(_wrap_to_pi(rpy1 - rpy0))
        motion[:3] += d_xyz
        motion[3:] += d_rpy
        step_grip_delta = abs(g1 - g0)
        max_step_grip_delta = max(max_step_grip_delta, step_grip_delta)
        last_g = g1
    end_to_end_grip_delta = abs(last_g - first_g)
    grip_event_score = max(max_step_grip_delta, end_to_end_grip_delta)
    return motion, float(grip_event_score)


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return -1.0
    return float(np.dot(a, b) / max(na * nb, eps))


def _merge_short_segments(
    segments: list[tuple[int, int]],
    states: np.ndarray,
    cfg: BuildReplayBufferConfig,
) -> list[tuple[int, int]]:
    if not cfg.enable_short_segment_merge or len(segments) <= 1:
        return segments
    if cfg.short_segment_merge_frames <= 0:
        return segments

    merged = [tuple(x) for x in segments]

    while True:
        changed = False
        for i, (seg_s, seg_e) in enumerate(merged):
            seg_len = seg_e - seg_s
            if seg_len >= cfg.short_segment_merge_frames:
                continue

            seg_motion, seg_grip_mag = _segment_motion_stats(states, cfg, seg_s, seg_e)
            if cfg.preserve_short_gripper_segments and seg_grip_mag >= cfg.gripper_threshold:
                # Keep short but meaningful gripper-open/close style segments.
                continue

            has_left = i > 0
            has_right = i < len(merged) - 1
            if not has_left and not has_right:
                continue

            left_score = None
            right_score = None
            left_len = -1
            right_len = -1

            if has_left:
                ls, le = merged[i - 1]
                left_motion, _ = _segment_motion_stats(states, cfg, ls, le)
                left_score = _cosine_sim(seg_motion, left_motion)
                left_len = le - ls
            if has_right:
                rs, re = merged[i + 1]
                right_motion, _ = _segment_motion_stats(states, cfg, rs, re)
                right_score = _cosine_sim(seg_motion, right_motion)
                right_len = re - rs

            merge_to_left = False
            if has_left and has_right:
                if left_score > right_score:
                    merge_to_left = True
                elif right_score > left_score:
                    merge_to_left = False
                else:
                    # Tie-break by neighbor segment length for stability.
                    merge_to_left = left_len >= right_len
            else:
                merge_to_left = has_left

            if merge_to_left:
                ls, _ = merged[i - 1]
                merged[i - 1] = (ls, seg_e)
                del merged[i]
            else:
                _, re = merged[i + 1]
                merged[i + 1] = (seg_s, re)
                del merged[i]

            changed = True
            break

        if not changed:
            break

    return merged




def _extract_frame_record(sample: dict[str, Any], fallback_task: str) -> _FrameRecord:
    image = _extract_value(
        sample,
        [
            ("observation/image",),
            ("image",),
            ("observation", "image"),
        ],
    )
    wrist = _extract_value(
        sample,
        [
            ("observation/wrist_image",),
            ("wrist_image",),
            ("observation", "wrist_image"),
        ],
    )
    state = _extract_value(
        sample,
        [
            ("observation/state",),
            ("state",),
            ("observation", "state"),
        ],
    )
    action = _extract_value(
        sample,
        [
            ("actions",),
            ("action",),
            ("observation/actions",),
            ("observation", "actions"),
        ],
    )
    task_value = sample.get("task", fallback_task)
    task = _decode_task(task_value)
    return _FrameRecord(
        image=_to_hwc_uint8(image),
        wrist_image=_to_hwc_uint8(wrist),
        state=np.asarray(state, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        task=task,
    )


def _make_model_observation(frame: _FrameRecord) -> _model.Observation:
    base = frame.image
    wrist = frame.wrist_image
    right = np.zeros_like(base)
    obs_dict = {
        "image": {
            "base_0_rgb": base[None, ...],
            "left_wrist_0_rgb": wrist[None, ...],
            "right_wrist_0_rgb": right[None, ...],
        },
        "image_mask": {
            "base_0_rgb": np.asarray([True], dtype=np.bool_),
            "left_wrist_0_rgb": np.asarray([True], dtype=np.bool_),
            "right_wrist_0_rgb": np.asarray([False], dtype=np.bool_),
        },
        "state": frame.state[None, ...],
    }
    obs = _model.Observation.from_dict(obs_dict)
    return _model.preprocess_observation(None, obs, train=False)


def _extract_visual_feature(model: Any, frame: _FrameRecord) -> np.ndarray:
    obs = _make_model_observation(frame)
    if not hasattr(model, "PaliGemma") or not hasattr(model.PaliGemma, "img"):
        raise TypeError("Model does not expose PaliGemma.img; this script currently supports pi0/pi0.5 style models.")

    pooled_list = []
    weight_list = []
    for cam_name, image in obs.images.items():
        tokens, _ = model.PaliGemma.img(image, train=False)
        pooled = jnp.mean(jnp.asarray(tokens, dtype=jnp.float32), axis=1)  # [1, d]
        mask = jnp.asarray(obs.image_masks[cam_name], dtype=jnp.float32).reshape(-1, 1)  # [1,1]
        pooled_list.append(pooled * mask)
        weight_list.append(mask)

    feat_sum = jnp.sum(jnp.stack(pooled_list, axis=0), axis=0)   # [1, d]
    weight_sum = jnp.sum(jnp.stack(weight_list, axis=0), axis=0) # [1, 1]
    feat = feat_sum / jnp.maximum(weight_sum, 1e-6)
    feat = jnp.asarray(feat[0], dtype=jnp.float32)
    return np.asarray(jax.device_get(feat), dtype=np.float32)


def _load_feature_model(train_cfg: _config.TrainConfig, model_params_path: str | None) -> Any:
    if model_params_path:
        path = pathlib.Path(model_params_path)
        if path.is_dir() and (path / "params").exists():
            path = path / "params"
        LOGGER.info("Loading model params from: %s", path)
        params = _model.restore_params(path, dtype=jnp.bfloat16)
        model = train_cfg.model.load(params, remove_extra_params=True)
    else:
        LOGGER.warning(
            "model_params_path is not set; fallback to train_config.weight_loader. "
            "For continual replay, pass the old-task checkpoint params explicitly."
        )
        model_shape = nnx.eval_shape(train_cfg.model.create, jax.random.key(0))
        _, state = nnx.split(model_shape)
        template_params = state.to_pure_dict()
        params = train_cfg.weight_loader.load(template_params)
        model = train_cfg.model.load(params, remove_extra_params=True)

    if hasattr(model, "eval"):
        model.eval()
    return model


def _get_episode_task(dataset: Any, episode_idx: int) -> str:
    from_idx = int(dataset.episode_data_index["from"][episode_idx].item())
    sample = dataset[from_idx]
    return _decode_task(sample.get("task", ""))


def _select_episode_indices(dataset: Any, target_task_name: str | None) -> list[int]:
    episode_indices: list[int] = []
    for i in range(int(dataset.num_episodes)):
        task = _get_episode_task(dataset, i)
        if target_task_name is None or task == target_task_name:
            episode_indices.append(i)
    return episode_indices


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _build_action_chunk(actions: list[np.ndarray], idx: int, horizon: int) -> np.ndarray:
    chunk = []
    last = actions[-1]
    for k in range(horizon):
        j = idx + k
        chunk.append(actions[j] if j < len(actions) else last)
    return np.stack(chunk, axis=0).astype(np.float32)


def main(cfg: BuildReplayBufferConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.set_printoptions(precision=4, suppress=True)

    train_cfg = _config.get_config(cfg.train_config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.repo_id is None:
        raise ValueError("Data config has no repo_id; cannot build replay buffer.")

    dataset_root = cfg.dataset_root if cfg.dataset_root is not None else data_cfg.root
    target_task_name = cfg.target_task_name if cfg.target_task_name is not None else data_cfg.target_task_name
    task_slug = _slugify(target_task_name if target_task_name else "all_tasks")

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
    (out_dir / "prototypes").mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset repo_id=%s root=%s", data_cfg.repo_id, dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(data_cfg.repo_id, root=dataset_root)
    episode_indices = _select_episode_indices(dataset, target_task_name)
    if cfg.max_episodes is not None:
        episode_indices = episode_indices[: cfg.max_episodes]
    if not episode_indices:
        raise ValueError(f"No episodes found for target_task_name={target_task_name!r}.")
    LOGGER.info("Selected episodes: %d / %d", len(episode_indices), int(dataset.num_episodes))

    model = _load_feature_model(train_cfg, cfg.model_params_path)

    task_to_id: dict[str, int] = {}
    sample_manifest_path = out_dir / "manifest.jsonl"
    segment_manifest_path = out_dir / "segments.jsonl"

    sample_count = 0
    segment_count = 0
    skipped_short_episodes = 0

    with sample_manifest_path.open("w", encoding="utf-8") as sample_f, segment_manifest_path.open(
        "w", encoding="utf-8"
    ) as segment_f:
        for ep_idx in tqdm.tqdm(episode_indices, desc="Episodes"):
            ep_from = int(dataset.episode_data_index["from"][ep_idx].item())
            ep_to = int(dataset.episode_data_index["to"][ep_idx].item())
            num_frames = ep_to - ep_from
            if num_frames <= max(cfg.chunk_frames, cfg.min_segment_frames):
                skipped_short_episodes += 1
                continue

            first_task = _get_episode_task(dataset, ep_idx)
            frames: list[_FrameRecord] = []
            for gidx in range(ep_from, ep_to):
                sample = dataset[gidx]
                frames.append(_extract_frame_record(sample, fallback_task=first_task))

            states = np.stack([f.state for f in frames], axis=0)
            actions = [f.action for f in frames]
            task = frames[0].task if frames else first_task
            task_id = task_to_id.setdefault(task, len(task_to_id))
            prompt = task

            motions = _compute_chunk_motions(states, cfg)
            boundaries = _find_boundaries(motions, cfg)
            segments = _build_segments(num_frames, boundaries, cfg.min_segment_frames)
            segments = _merge_short_segments(segments, states, cfg)
            if not segments:
                continue

            feature_cache: dict[int, np.ndarray] = {}

            for local_seg_idx, (seg_s, seg_e) in enumerate(segments):
                frame_ids = list(range(seg_s, seg_e))
                if not frame_ids:
                    continue

                seg_feats = []
                for fid in frame_ids:
                    if fid not in feature_cache:
                        feature_cache[fid] = _extract_visual_feature(model, frames[fid])
                    seg_feats.append(feature_cache[fid])

                feat_mat = np.stack(seg_feats, axis=0).astype(np.float32)  # [n, d]
                feat_norm = _l2_normalize(feat_mat)
                proto = _l2_normalize(np.mean(feat_norm, axis=0, keepdims=True))[0]  # [d]
                cosine = feat_norm @ proto  # [n]

                k = min(cfg.top_k_per_segment, len(frame_ids))
                if k <= 0:
                    continue
                top_idx = np.argpartition(-cosine, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(-cosine[top_idx])]

                seg_id = segment_count
                proto_path = out_dir / "prototypes" / f"segment_{seg_id:08d}.npy"
                np.save(proto_path, proto.astype(np.float32))

                seg_record = {
                    "segment_id": seg_id,
                    "episode_index": ep_idx,
                    "task": task,
                    "task_id": task_id,
                    "segment_start": seg_s,
                    "segment_end": seg_e,
                    "num_frames": seg_e - seg_s,
                    "selected_local_indices": [int(i) for i in top_idx.tolist()],
                    "selected_episode_frame_indices": [int(frame_ids[i]) for i in top_idx.tolist()],
                    "prototype_path": str(proto_path.relative_to(out_dir)),
                    "local_segment_index": local_seg_idx,
                }
                segment_f.write(json.dumps(seg_record, ensure_ascii=False) + "\n")

                for rank, i_local in enumerate(top_idx.tolist()):
                    ep_frame_idx = frame_ids[i_local]
                    g_frame_idx = ep_from + ep_frame_idx
                    sample_id = sample_count
                    sample_path = out_dir / "samples" / f"sample_{sample_id:08d}.npz"
                    frame = frames[ep_frame_idx]
                    action_chunk = _build_action_chunk(actions, ep_frame_idx, horizon=train_cfg.model.action_horizon)

                    np.savez_compressed(
                        sample_path,
                        image=frame.image,
                        wrist_image=frame.wrist_image,
                        state=frame.state.astype(np.float32),
                        action=frame.action.astype(np.float32),
                        action_chunk=action_chunk.astype(np.float32),
                        task=np.asarray(task),
                        prompt=np.asarray(prompt),
                        task_id=np.asarray(task_id, dtype=np.int32),
                        segment_id=np.asarray(seg_id, dtype=np.int32),
                        local_segment_index=np.asarray(local_seg_idx, dtype=np.int32),
                        episode_index=np.asarray(ep_idx, dtype=np.int32),
                        episode_frame_index=np.asarray(ep_frame_idx, dtype=np.int32),
                        global_frame_index=np.asarray(g_frame_idx, dtype=np.int32),
                        segment_start=np.asarray(seg_s, dtype=np.int32),
                        segment_end=np.asarray(seg_e, dtype=np.int32),
                        segment_rank=np.asarray(rank, dtype=np.int32),
                        cosine_to_prototype=np.asarray(cosine[i_local], dtype=np.float32),
                    )

                    sample_record = {
                        "sample_id": sample_id,
                        "sample_path": str(sample_path.relative_to(out_dir)),
                        "task": task,
                        "task_id": task_id,
                        "prompt": prompt,
                        "episode_index": ep_idx,
                        "episode_frame_index": ep_frame_idx,
                        "global_frame_index": g_frame_idx,
                        "segment_id": seg_id,
                        "segment_start": seg_s,
                        "segment_end": seg_e,
                        "segment_rank": rank,
                        "cosine_to_prototype": float(cosine[i_local]),
                    }
                    sample_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                    sample_count += 1

                segment_count += 1

    meta = {
        "train_config_name": cfg.train_config_name,
        "repo_id": data_cfg.repo_id,
        "dataset_root": dataset_root,
        "target_task_name": target_task_name,
        "model_params_path": cfg.model_params_path,
        "segmentation": {
            "chunk_frames": cfg.chunk_frames,
            "min_segment_frames": cfg.min_segment_frames,
            "min_stable_chunks": cfg.min_stable_chunks,
            "translation_threshold_m": cfg.translation_threshold_m,
            "rotation_threshold_rad": cfg.rotation_threshold_rad,
            "gripper_threshold": cfg.gripper_threshold,
            "xyz_indices": cfg.xyz_indices,
            "quat_indices": cfg.quat_indices,
            "quat_order": cfg.quat_order,
            "euler_indices": cfg.euler_indices,
            "gripper_index": cfg.gripper_index,
            "enable_short_segment_merge": cfg.enable_short_segment_merge,
            "short_segment_merge_frames": cfg.short_segment_merge_frames,
            "preserve_short_gripper_segments": cfg.preserve_short_gripper_segments,
        },
        "selection": {
            "top_k_per_segment": cfg.top_k_per_segment,
        },
        "stats": {
            "num_selected_episodes": len(episode_indices),
            "num_skipped_short_episodes": skipped_short_episodes,
            "num_segments": segment_count,
            "num_samples": sample_count,
            "num_tasks": len(task_to_id),
            "task_to_id": task_to_id,
        
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Replay buffer build complete.")
    LOGGER.info("Output dir: %s", out_dir)
    LOGGER.info("Segments: %d | Samples: %d | Tasks: %d", segment_count, sample_count, len(task_to_id))


if __name__ == "__main__":
    tyro.cli(main)

