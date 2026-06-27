"""
OpenVLA: offline prototype replay buffer builder with implicit-skill physical segmentation.

Pipeline:
1) Load a LIBERO RLDS dataset (optionally task-filtered).
2) Segment each trajectory with physics rules (translation / rotation / gripper).
3) Extract per-frame visual features from the loaded model's vision backbone.
4) Compute segment prototype (L2-normalized mean feature), select Top-K closest frames.
5) Save selected frames + state / action_chunk / prompt / task_id into a buffer directory.

Port from PI scripts/build_replay_buffer.py, adapted for OpenVLA (SigLIP vision backbone, RLDS data).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Prevent TF from competing with PyTorch for GPU memory.
tf.config.set_visible_devices([], 'GPU')

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BuildReplayBufferConfig:
    # ---- Model & checkpoint ----
    vla_path: str = "/root/autodl-tmp/models/openvla-7b"
    cl_lora_checkpoint_dir: Optional[str] = None

    # ---- Dataset ----
    data_root_dir: str = "/root/autodl-tmp/modified_libero_rlds"
    dataset_name: str = "libero_spatial_no_noops"
    # When None, capture ALL tasks in the dataset (branch-3 full-dataset mode).
    target_task_name: Optional[str] = None

    # ---- Output ----
    output_dir: str = "/root/autodl-tmp/replay_buffers/taskA_prototypes"
    overwrite: bool = False

    # ---- Sample selection ----
    top_k_per_segment: int = 2
    max_episodes: Optional[int] = None
    action_horizon: int = 8  # NUM_ACTIONS_CHUNK

    # ---- Segmentation ----
    chunk_frames: int = 5
    min_segment_frames: int = 3
    min_stable_chunks: int = 2
    translation_threshold_m: float = 0.03
    rotation_threshold_rad: float = 0.05
    gripper_threshold: float = 0.1

    # ---- LIBERO state indices ----
    xyz_indices: Tuple[int, int, int] = (0, 1, 2)
    euler_indices: Tuple[int, int, int] = (3, 4, 5)
    gripper_index: int = 6

    # ---- Short-segment merge ----
    enable_short_segment_merge: bool = True
    short_segment_merge_frames: int = 10
    preserve_short_gripper_segments: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz", "grip")


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _state_to_pose_components(
    state: np.ndarray, cfg: BuildReplayBufferConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    xyz = s[list(cfg.xyz_indices)]
    rpy = s[list(cfg.euler_indices)]
    grip = float(s[cfg.gripper_index])
    return xyz, rpy, grip


# ---------------------------------------------------------------------------
# Motion & segmentation
# ---------------------------------------------------------------------------

def _compute_chunk_motions(states: np.ndarray, cfg: BuildReplayBufferConfig) -> list:
    if len(states) <= cfg.chunk_frames:
        return []

    motions = []
    for i in range(0, len(states) - cfg.chunk_frames):
        xyz0, rpy0, g0 = _state_to_pose_components(states[i], cfg)
        xyz1, rpy1, g1 = _state_to_pose_components(states[i + cfg.chunk_frames], cfg)

        d_xyz = xyz1 - xyz0
        d_rpy = _wrap_to_pi(rpy1 - rpy0)
        d_g = g1 - g0

        raw_mag = np.array([
            abs(d_xyz[0]), abs(d_xyz[1]), abs(d_xyz[2]),
            abs(d_rpy[0]), abs(d_rpy[1]), abs(d_rpy[2]),
            abs(d_g),
        ], dtype=np.float64)

        norm_mag = raw_mag / np.array(
            [cfg.translation_threshold_m] * 3 +
            [cfg.rotation_threshold_rad] * 3 +
            [cfg.gripper_threshold]
        )

        dominant = (
            _AXIS_NAMES[int(np.argmax(norm_mag))]
            if np.max(norm_mag) >= 1.0 else "idle"
        )

        motions.append({
            "end_frame": min(i + cfg.chunk_frames, len(states) - 1),
            "dominant": dominant,
            "norm_mag": norm_mag.astype(np.float32),
            "grip_delta": float(d_g),
        })
    return motions


def _find_boundaries(motions: list, cfg: BuildReplayBufferConfig) -> list:
    if not motions:
        return []
    boundaries: set = set()
    labels = [m["dominant"] for m in motions]

    for m in motions:
        if abs(m["grip_delta"]) >= cfg.gripper_threshold:
            boundaries.add(m["end_frame"])

    for i in range(1, len(motions)):
        if labels[i] != labels[i - 1]:
            cand = labels[i]
            stable = True
            for j in range(cfg.min_stable_chunks):
                k = min(i + j, len(motions) - 1)
                if labels[k] != cand:
                    stable = False
                    break
            if stable:
                boundaries.add(motions[i]["end_frame"])
    return sorted(boundaries)


def _build_segments(num_frames: int, boundaries: list, min_seg: int) -> list:
    if num_frames <= 0:
        return []
    b = sorted({x for x in boundaries if 0 < x < num_frames})
    starts = [0] + b
    ends = b + [num_frames]
    segments = [(s, e) for s, e in zip(starts, ends) if e - s >= min_seg]
    if not segments:
        segments.append((0, num_frames))
    return segments


# ---------------------------------------------------------------------------
# Short-segment merge (ported from PI)
# ---------------------------------------------------------------------------

def _segment_motion_stats(
    states: np.ndarray, cfg: BuildReplayBufferConfig,
    seg_s: int, seg_e: int,
) -> Tuple[np.ndarray, float]:
    if seg_e - seg_s <= 1:
        return np.zeros(6, dtype=np.float64), 0.0

    motion = np.zeros(6, dtype=np.float64)
    _, _, first_g = _state_to_pose_components(states[seg_s], cfg)
    max_step_grip_delta = 0.0
    last_g = first_g
    for i in range(seg_s, seg_e - 1):
        xyz0, rpy0, g0 = _state_to_pose_components(states[i], cfg)
        xyz1, rpy1, g1 = _state_to_pose_components(states[i + 1], cfg)
        d_xyz = np.abs(xyz1 - xyz0)
        d_rpy = np.abs(_wrap_to_pi(rpy1 - rpy0))
        motion[:3] += d_xyz
        motion[3:] += d_rpy
        max_step_grip_delta = max(max_step_grip_delta, abs(g1 - g0))
        last_g = g1
    end_to_end_grip = abs(last_g - first_g)
    grip_event_score = max(max_step_grip_delta, end_to_end_grip)
    return motion, float(grip_event_score)


def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return -1.0
    return float(np.dot(a, b) / max(na * nb, eps))


def _merge_short_segments(
    segments: list, states: np.ndarray, cfg: BuildReplayBufferConfig,
) -> list:
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


# ---------------------------------------------------------------------------
# Action chunk building
# ---------------------------------------------------------------------------

def _build_action_chunk(all_actions: list, frame_idx: int, horizon: int) -> np.ndarray:
    chunk = []
    last = all_actions[-1]
    for k in range(horizon):
        j = frame_idx + k
        chunk.append(all_actions[j] if j < len(all_actions) else last)
    return np.stack(chunk, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_visual_feature(model, processor, image: np.ndarray, device) -> np.ndarray:
    pil_img = Image.fromarray(image).convert("RGB")
    inputs = processor(text="dummy", images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    patch_features = model.vision_backbone(pixel_values)
    pooled = patch_features.mean(dim=1).squeeze(0).cpu().to(torch.float32).numpy()
    return pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = BuildReplayBufferConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- 1. Resolve CL-LoRA config (rank / alpha / shared_depth) ----
    lora_rank = 16
    alpha = 16.0
    shared_split_ratio = 8.0 / 32.0  # default: shared_depth=8

    if cfg.cl_lora_checkpoint_dir is not None:
        config_path = os.path.join(cfg.cl_lora_checkpoint_dir, "cl_lora_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cl_config = json.load(f)
            lora_rank = cl_config.get("lora_rank", 16)
            alpha = cl_config.get("alpha", float(lora_rank))
            shared_split_ratio = cl_config.get("shared_split_ratio", 8.0 / 32.0)
            LOGGER.info("Loaded CL-LoRA config from checkpoint: rank=%d alpha=%.1f shared_ratio=%.4f",
                        lora_rank, alpha, shared_split_ratio)
        else:
            LOGGER.warning("cl_lora_config.json not found in %s, using defaults.", cfg.cl_lora_checkpoint_dir)

    # ---- 2. Prepare output dir ----
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
    (out_dir / "prototypes").mkdir(parents=True, exist_ok=True)

    # ---- 3. Load model ----
    LOGGER.info("Loading OpenVLA-7B model skeleton...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)

    # Inject CL-LoRA
    from cl_lora import inject_cl_lora_into_model
    vla = inject_cl_lora_into_model(
        vla, rank=lora_rank, alpha=alpha, dropout=0.0,
        shared_split_ratio=shared_split_ratio,
        orthogonal_init=True, freeze_a=True, use_block_scale=True,
    )

    # Load CL-LoRA adapter weights if available
    if cfg.cl_lora_checkpoint_dir is not None:
        adapter_path = os.path.join(cfg.cl_lora_checkpoint_dir, "cl_lora_adapter.pt")
        if os.path.exists(adapter_path):
            cl_state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
            missing, unexpected = vla.load_state_dict(cl_state_dict, strict=False)
            LOGGER.info("Loaded CL-LoRA adapter: %d keys (missing=%d, unexpected=%d)",
                        len(cl_state_dict), len(missing), len(unexpected))
        else:
            LOGGER.warning("cl_lora_adapter.pt not found at %s", adapter_path)
    vla.eval()

    # ---- 4. Load RLDS dataset ----
    LOGGER.info("Loading RLDS dataset: %s", cfg.dataset_name)
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    dataset = builder.as_dataset(split='all')

    segment_manifest = out_dir / "segments.jsonl"
    sample_manifest = out_dir / "manifest.jsonl"

    segment_count = 0
    sample_count = 0

    with segment_manifest.open("w", encoding="utf-8") as seg_f, \
         sample_manifest.open("w", encoding="utf-8") as samp_f:

        for ep_idx, episode in enumerate(tqdm.tqdm(dataset, desc="Episodes")):
            steps = list(episode['steps'])
            if not steps:
                continue

            # ---- Task filtering ----
            step_zero = steps[0]
            if 'language_instruction' in step_zero:
                raw_inst = step_zero['language_instruction']
            elif 'natural_language_instruction' in step_zero.get('observation', {}):
                raw_inst = step_zero['observation']['natural_language_instruction']
            else:
                step_keys = list(step_zero.keys())
                obs_keys = list(step_zero.get('observation', {}).keys())
                raise KeyError(
                    f"Cannot find language instruction! "
                    f"Step keys: {step_keys}, Observation keys: {obs_keys}"
                )

            if hasattr(raw_inst, 'numpy'):
                raw_inst = raw_inst.numpy()
            language_instruction = raw_inst.decode('utf-8') if isinstance(raw_inst, bytes) else str(raw_inst)

            # Target task filter (None = capture all)
            if cfg.target_task_name is not None and cfg.target_task_name not in language_instruction:
                continue

            num_frames = len(steps)
            if num_frames <= cfg.chunk_frames:
                continue

            # Extract states array for segmentation
            states = np.array([
                step['observation']['state'].numpy() for step in steps
            ])

            # Extract actions list for chunk building
            all_actions = [step['action'].numpy() for step in steps]

            # ---- Segmentation ----
            motions = _compute_chunk_motions(states, cfg)
            boundaries = _find_boundaries(motions, cfg)
            segments = _build_segments(num_frames, boundaries, cfg.min_segment_frames)
            segments = _merge_short_segments(segments, states, cfg)

            if cfg.max_episodes is not None and ep_idx >= cfg.max_episodes:
                break

            # ---- Per-segment feature extraction & top-K selection ----
            feature_cache: dict = {}

            for local_seg_idx, (seg_s, seg_e) in enumerate(segments):
                frame_ids = list(range(seg_s, seg_e))
                if not frame_ids:
                    continue

                seg_feats = []
                for fid in frame_ids:
                    if fid not in feature_cache:
                        img = steps[fid]['observation']['image'].numpy()
                        feature_cache[fid] = _extract_visual_feature(vla, processor, img, device)
                    seg_feats.append(feature_cache[fid])

                feat_mat = np.stack(seg_feats, axis=0).astype(np.float32)
                feat_norm = feat_mat / (np.linalg.norm(feat_mat, axis=-1, keepdims=True) + 1e-8)
                proto = np.mean(feat_norm, axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-8)

                cosine = feat_norm @ proto
                k = min(cfg.top_k_per_segment, len(frame_ids))
                if k <= 0:
                    continue
                top_idx = np.argpartition(-cosine, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(-cosine[top_idx])]

                # Save prototype
                proto_path = out_dir / "prototypes" / f"segment_{segment_count:08d}.npy"
                np.save(proto_path, proto.astype(np.float32))

                # Segment record
                seg_record = {
                    "segment_id": segment_count,
                    "episode_index": ep_idx,
                    "task": language_instruction,
                    "num_frames": seg_e - seg_s,
                    "segment_start": seg_s,
                    "segment_end": seg_e,
                    "selected_episode_frame_indices": [int(frame_ids[i]) for i in top_idx.tolist()],
                    "local_segment_index": local_seg_idx,
                    "prototype_path": str(proto_path.relative_to(out_dir)),
                }
                seg_f.write(json.dumps(seg_record, ensure_ascii=False) + "\n")

                # Save key frames
                for rank, i_local in enumerate(top_idx.tolist()):
                    ep_frame_idx = frame_ids[i_local]
                    step_data = steps[ep_frame_idx]

                    action_chunk = _build_action_chunk(all_actions, ep_frame_idx, cfg.action_horizon)

                    np.savez_compressed(
                        out_dir / "samples" / f"sample_{sample_count:08d}.npz",
                        image=step_data['observation']['image'].numpy(),
                        state=step_data['observation']['state'].numpy(),
                        action=step_data['action'].numpy(),
                        action_chunk=action_chunk,
                        task=language_instruction,
                    )

                    samp_record = {
                        "sample_id": sample_count,
                        "sample_path": str((out_dir / "samples" / f"sample_{sample_count:08d}.npz").relative_to(out_dir)),
                        "task": language_instruction,
                        "episode_index": ep_idx,
                        "episode_frame_index": ep_frame_idx,
                        "segment_id": segment_count,
                        "segment_start": seg_s,
                        "segment_end": seg_e,
                        "segment_rank": rank,
                        "cosine_to_prototype": float(cosine[i_local]),
                    }
                    samp_f.write(json.dumps(samp_record, ensure_ascii=False) + "\n")
                    sample_count += 1

                segment_count += 1

    if sample_count == 0:
        raise RuntimeError(
            f"Built 0 replay samples! Check:\n"
            f"  1. target_task_name matches data: '{cfg.target_task_name}'\n"
            f"  2. dataset path: {cfg.data_root_dir}/{cfg.dataset_name}\n"
            f"  3. min_segment_frames ({cfg.min_segment_frames}) is not too large"
        )

    # ---- Write meta ----
    meta = {
        "dataset_name": cfg.dataset_name,
        "data_root_dir": cfg.data_root_dir,
        "target_task_name": cfg.target_task_name,
        "cl_lora_checkpoint_dir": cfg.cl_lora_checkpoint_dir,
        "lora_rank": lora_rank,
        "alpha": alpha,
        "shared_split_ratio": shared_split_ratio,
        "action_horizon": cfg.action_horizon,
        "segmentation": {
            "chunk_frames": cfg.chunk_frames,
            "min_segment_frames": cfg.min_segment_frames,
            "min_stable_chunks": cfg.min_stable_chunks,
            "translation_threshold_m": cfg.translation_threshold_m,
            "rotation_threshold_rad": cfg.rotation_threshold_rad,
            "gripper_threshold": cfg.gripper_threshold,
            "enable_short_segment_merge": cfg.enable_short_segment_merge,
            "short_segment_merge_frames": cfg.short_segment_merge_frames,
            "preserve_short_gripper_segments": cfg.preserve_short_gripper_segments,
        },
        "selection": {
            "top_k_per_segment": cfg.top_k_per_segment,
        },
        "stats": {
            "num_segments": segment_count,
            "num_samples": sample_count,
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Replay buffer build complete!")
    LOGGER.info("Output dir: %s", out_dir)
    LOGGER.info("Segments: %d | Samples: %d", segment_count, sample_count)


if __name__ == "__main__":
    main()
