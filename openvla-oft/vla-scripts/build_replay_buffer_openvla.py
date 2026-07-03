"""
Prototype Replay v2 buffer builder for OpenVLA + LIBERO.

Migrated from PI0.5 RoboTwin prototype replay v2.  Three core upgrades over v1:
  1. End-effector kinematic implicit segmentation (single-arm LIBERO)
  2. Action-conditioned multimodal behavior representation (vision + physical motion)
  3. Intra-segment temporal de-duplicated max-coverage selection (MAD + greedy k-center)

Reference: RoboTwin-main/policy/pi05/scripts/build_robotwin_prototype_buffer_v2.py

Expected usage (single GPU, RTX 5090):
  python vla-scripts/build_replay_buffer_openvla.py \
    --vla-path /root/autodl-tmp/models/openvla-7b \
    --cl-lora-path $LOGS_ROOT/cl_lora_v26_taskA--6000_chkpt/cl_lora_adapter.pt \
    --data-root-dir /root/autodl-tmp/modified_libero_rlds \
    --dataset-name libero_spatial_no_noops \
    --output-dir /root/autodl-tmp/replay_buffers/prototype_v2/taskA_ep10_top3 \
    --num-episodes 10 --top-k 3
"""

import argparse
import dataclasses
import json
import math
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from cl_lora import inject_cl_lora_into_model

tf.config.set_visible_devices([], "GPU")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReplayBufferConfig:
    vla_path: str = "/root/autodl-tmp/models/openvla-7b"
    cl_lora_path: str = ""
    data_root_dir: str = "/root/autodl-tmp/modified_libero_rlds"
    dataset_name: str = "libero_spatial_no_noops"
    output_dir: str = "/root/autodl-tmp/replay_buffers/prototype_v2"

    # Episode selection
    num_episodes: int = 10
    top_k: int = 3

    # CL-LoRA injection params (must match training)
    lora_rank: int = 32
    shared_depth: int = 8

    # Kinematic segmentation
    kinematic_window: int = 5
    translation_threshold: float = 0.03        # metres
    rotation_threshold: float = 0.05           # radians
    gripper_threshold: float = 0.1
    min_segment_frames: int = 5
    descriptor_clip: float = 5.0

    # Coverage selection
    temporal_min_gap: int = 5
    outlier_mad_scale: float = 2.0

    # Feature fusion weights
    vision_weight: float = 0.5
    physical_weight: float = 0.5

    overwrite: bool = False

    # LIBERO state indices  (7D: xyz(3) + axisangle(3) + gripper(1))
    xyz_indices: Tuple[int, ...] = (0, 1, 2)
    rot_indices: Tuple[int, ...] = (3, 4, 5)
    gripper_index: int = 6


# ---------------------------------------------------------------------------
# Motion signal computation (PI v2  §3)
# ---------------------------------------------------------------------------

def _compute_motion_signals(
    states: np.ndarray,
    cfg: ReplayBufferConfig,
) -> Dict[str, np.ndarray]:
    """Compute per-frame translation / rotation / gripper changes over a forward window.

    Parameters
    ----------
    states : np.ndarray  shape [T, 7]
        LIBERO state vectors.

    Returns
    -------
    dict with keys:
      descriptors  [T, 8]   tx,ty,tz,rx,ry,rz,grip, activity_norm  (clipped + smoothed)
      modes        [T] int  motion-mode ID per frame
    """
    T = states.shape[0]
    window = cfg.kinematic_window
    trans_th = cfg.translation_threshold
    rot_th = cfg.rotation_threshold
    grip_th = cfg.gripper_threshold

    translation = np.zeros((T, 3), dtype=np.float32)
    rotation = np.zeros((T, 3), dtype=np.float32)
    gripper_delta = np.zeros(T, dtype=np.float32)

    for t in range(T):
        future = min(t + window, T - 1)
        translation[t] = states[future, :3] - states[t, :3]
        rotation[t] = _wrap_angle(states[future, 3:6] - states[t, 3:6])
        gripper_delta[t] = states[future, 6] - states[t, 6]

    # Scale + clip
    scaled = np.column_stack([
        translation / trans_th,
        rotation / rot_th,
        gripper_delta[:, None] / grip_th,
    ])  # [T, 7]

    # Moving-average smooth
    smoothed = np.stack([
        _moving_average(scaled[:, i], window) for i in range(7)
    ], axis=-1)
    smoothed = np.clip(smoothed, -cfg.descriptor_clip, cfg.descriptor_clip)

    # Per-frame motion mode
    modes = np.zeros(T, dtype=np.int32)
    mode_names = ["tx", "ty", "tz", "rx", "ry", "rz", "grip"]
    for t in range(T):
        comp = int(np.argmax(np.abs(smoothed[t])))
        if abs(smoothed[t, comp]) >= 1.0:
            modes[t] = 1 + 2 * comp + int(smoothed[t, comp] < 0.0)  # PI mode encoding
        else:
            modes[t] = 0  # idle

    activity = np.linalg.norm(smoothed, axis=-1)
    current_gripper = np.clip(states[:, 6:7], -1.0, 1.0)

    descriptors = np.concatenate([smoothed, current_gripper, activity[:, None]], axis=-1).astype(np.float32)

    return {
        "descriptors": descriptors,       # [T, 9]
        "modes": modes,                    # [T]
        "translation": translation,        # [T, 3]
        "rotation": rotation,              # [T, 3]
        "gripper_delta": gripper_delta,    # [T]
    }


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _moving_average(seq: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(seq, kernel, mode="same")


# ---------------------------------------------------------------------------
# Kinematic segmentation  (PI v2  §3)
# ---------------------------------------------------------------------------

def _run_length_encode_modes(modes: np.ndarray) -> List[Tuple[int, int, int]]:
    """Encode motion-mode sequence as (start, end, mode) runs."""
    if len(modes) == 0:
        return []
    runs = []
    start = 0
    current = int(modes[0])
    for i in range(1, len(modes)):
        if int(modes[i]) != current:
            runs.append((start, i, current))
            start = i
            current = int(modes[i])
    runs.append((start, len(modes), current))
    return runs


def _build_kinematic_segments(
    modes: np.ndarray,
    descriptors: np.ndarray,
    cfg: ReplayBufferConfig,
) -> List[Tuple[int, int]]:
    """Build segment boundaries from motion-mode runs, merge short segments."""
    runs = _run_length_encode_modes(modes)
    segments = [(s, e) for s, e, _m in runs if e - s >= 2]  # drop 1-frame runs

    if not segments:
        return [(0, len(modes))]  # single segment as fallback

    # Merge short segments by descriptor similarity
    segments = _merge_short_segments(segments, descriptors, cfg.min_segment_frames)
    return segments


def _merge_short_segments(
    segments: List[Tuple[int, int]],
    descriptors: np.ndarray,
    min_len: int,
) -> List[Tuple[int, int]]:
    """Iteratively merge the shortest segment into its most similar neighbour."""
    segments = list(segments)
    while len(segments) > 1:
        short_idx = next(
            (i for i, (s, e) in enumerate(segments) if e - s < min_len), None
        )
        if short_idx is None:
            break
        start, end = segments[short_idx]
        choices = []
        if short_idx > 0:
            d = _descriptor_distance(descriptors, segments[short_idx], segments[short_idx - 1])
            choices.append((d, -1))
        if short_idx + 1 < len(segments):
            d = _descriptor_distance(descriptors, segments[short_idx], segments[short_idx + 1])
            choices.append((d, 1))
        if not choices:
            break
        _, direction = min(choices, key=lambda x: x[0])
        if direction < 0:
            prev_s, _ = segments[short_idx - 1]
            segments[short_idx - 1 : short_idx + 1] = [(prev_s, end)]
        else:
            _, next_e = segments[short_idx + 1]
            segments[short_idx : short_idx + 2] = [(start, next_e)]
    return segments


def _mean_descriptor(descriptors: np.ndarray, seg: Tuple[int, int]) -> np.ndarray:
    s, e = seg
    feat = descriptors[s:e].mean(axis=0)
    norm = float(np.linalg.norm(feat))
    return feat / max(norm, 1e-8)


def _descriptor_distance(
    descriptors: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]
) -> float:
    return 1.0 - float(np.dot(_mean_descriptor(descriptors, a), _mean_descriptor(descriptors, b)))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_vision_features(
    model, processor, images: List[np.ndarray], device: torch.device
) -> np.ndarray:
    """Batch extract vision features from OpenVLA DinoSigLIP backbone.

    Returns
    -------
    features  [N, embed_dim]  L2-normalised.
    """
    features = []
    for img in images:
        pil = Image.fromarray(img).convert("RGB")
        inputs = processor(text="dummy", images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        patch_feat = model.vision_backbone(pixel_values)  # [1, P, D]
        pooled = patch_feat.mean(dim=1).squeeze(0).cpu().to(torch.float32).numpy()
        features.append(pooled)
    feats = np.stack(features, axis=0)
    feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    return feats


def _extract_physical_descriptors(
    motions: Dict[str, np.ndarray], seg: Tuple[int, int]
) -> np.ndarray:
    """Compute a physical motion descriptor vector inside a segment.

    Returns
    -------
    desc  [1, 9]   mean/std/max of: |translation|, |rotation|, |gripper_delta|
    """
    s, e = seg
    if e <= s:
        return np.zeros((1, 9), dtype=np.float32)

    trans_norm = np.linalg.norm(motions["translation"][s:e], axis=-1)
    rot_norm = np.linalg.norm(motions["rotation"][s:e], axis=-1)
    grip_abs = np.abs(motions["gripper_delta"][s:e])

    stats = []
    for series in [trans_norm, rot_norm, grip_abs]:
        stats.extend([np.mean(series), np.std(series), np.max(series)])

    return np.array(stats, dtype=np.float32).reshape(1, -1)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return mat / norm


def _fuse_features(
    vision: np.ndarray,
    physical: np.ndarray,
    cfg: ReplayBufferConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fuse normalised vision and physical features with sqrt(weight) scaling.

    Returns
    -------
    fused       [N, D_v + D_p]
    components  {"vision": [N, D_v], "physical": [N, D_p]}
    """
    weights = np.array([cfg.vision_weight, cfg.physical_weight], dtype=np.float32)
    weights /= weights.sum()

    norm_vision = _normalize_rows(vision.astype(np.float32))
    norm_physical = _normalize_rows(physical.astype(np.float32))

    fused = np.concatenate(
        [
            math.sqrt(float(weights[0])) * norm_vision,
            math.sqrt(float(weights[1])) * norm_physical,
        ],
        axis=-1,
    )
    return _normalize_rows(fused), {"vision": norm_vision, "physical": norm_physical}


# ---------------------------------------------------------------------------
# Coverage selection  (PI v2  §5)
# ---------------------------------------------------------------------------

def _coverage_error(features: np.ndarray, selected: List[int]) -> float:
    if not selected:
        return 1.0
    best_sim = np.max(features @ features[selected].T, axis=1)
    return float(np.mean(1.0 - best_sim))


def _select_coverage_representatives(
    features: np.ndarray,
    cfg: ReplayBufferConfig,
) -> Dict[str, Any]:
    """Select top-K frames via MAD outlier filtering + greedy k-center + temporal gap."""
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    n = features.shape[0]
    top_k = min(max(1, cfg.top_k), n)

    similarity = features @ features.T
    medoid = int(np.argmax(similarity.mean(axis=1)))
    dist_to_medoid = 1.0 - similarity[:, medoid]

    # MAD outlier filtering
    inliers = np.arange(n, dtype=np.int32)
    outlier_count = 0
    if n > top_k + 2 and cfg.outlier_mad_scale > 0.0:
        median = float(np.median(dist_to_medoid))
        mad = float(np.median(np.abs(dist_to_medoid - median)))
        if mad > 1e-8:
            cutoff = median + cfg.outlier_mad_scale * 1.4826 * mad
            inliers = np.flatnonzero(dist_to_medoid <= cutoff).astype(np.int32)
            if inliers.size < top_k:
                inliers = np.argsort(dist_to_medoid)[:top_k].astype(np.int32)
            outlier_count = n - inliers.size

    # Recompute medoid within inliers
    inlier_sim = similarity[np.ix_(inliers, inliers)]
    medoid = int(inliers[int(np.argmax(inlier_sim.mean(axis=1)))])

    # Effective temporal gap
    if top_k > 1 and inliers.size > 0:
        span = int(inliers.max() - inliers.min())
        feasible_gap = span // (top_k - 1)
        effective_gap = min(max(0, cfg.temporal_min_gap), feasible_gap)
    else:
        effective_gap = 0

    selected = [medoid]
    gains = [_coverage_error(features, []) - _coverage_error(features, selected)]
    gap = effective_gap
    while len(selected) < top_k:
        remaining = [int(i) for i in inliers if int(i) not in selected]
        valid = [i for i in remaining if all(abs(i - s) >= gap for s in selected)]
        while not valid and gap > 0:
            gap -= 1
            valid = [i for i in remaining if all(abs(i - s) >= gap for s in selected)]
        if not valid:
            break
        min_dists = {i: float(np.min(1.0 - similarity[i, selected])) for i in valid}
        chosen = max(valid, key=lambda i: (min_dists[i], -dist_to_medoid[i], -i))
        prev_err = _coverage_error(features, selected)
        selected.append(chosen)
        gains.append(prev_err - _coverage_error(features, selected))

    prototype = features[inliers].mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    proto_err = float(np.mean(1.0 - features @ prototype))

    return {
        "selected_indices": selected,
        "prototype": prototype.astype(np.float32),
        "prototype_error": proto_err,
        "coverage_error": _coverage_error(features, selected),
        "inlier_count": int(inliers.size),
        "outlier_count": outlier_count,
        "effective_temporal_gap": gap,
        "coverage_gains": gains,
    }


# ---------------------------------------------------------------------------
# Buffer build loop
# ---------------------------------------------------------------------------

def _get_language_instruction(step) -> str:
    if "language_instruction" in step:
        raw = step["language_instruction"]
    elif "natural_language_instruction" in step.get("observation", {}):
        raw = step["observation"]["natural_language_instruction"]
    else:
        raise KeyError(f"Cannot find language instruction.  Step keys: {list(step.keys())}")
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def build_buffer(cfg: ReplayBufferConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(cfg.output_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not cfg.overwrite:
            raise FileExistsError(f"{out_dir} not empty.  Pass --overwrite.")
        shutil.rmtree(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print("[1/5] Loading OpenVLA model ...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    if cfg.cl_lora_path:
        shared_ratio = max(1, cfg.shared_depth) / 32
        vla = inject_cl_lora_into_model(
            vla, rank=cfg.lora_rank, alpha=float(cfg.lora_rank),
            shared_split_ratio=shared_ratio, orthogonal_init=True,
            freeze_a=True, use_block_scale=True,
        )
        sd = torch.load(cfg.cl_lora_path, map_location="cpu", weights_only=True)
        vla.load_state_dict(sd, strict=False)
        print(f"  Loaded CL-LoRA adapter from {cfg.cl_lora_path}")
    vla.eval()

    # ---- RLDS dataset ----
    print(f"[2/5] Loading RLDS dataset: {cfg.dataset_name}")
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    ds = builder.as_dataset(split="all")

    # Collect all episodes for filtering
    episodes = []
    for ep_idx, episode in enumerate(ds):
        steps = list(episode["steps"])
        episodes.append((ep_idx, steps))
        if len(episodes) >= cfg.num_episodes:
            break

    if not episodes:
        raise RuntimeError("No episodes found in dataset")

    print(f"[3/5] Processing {len(episodes)} episodes ...")
    segment_count = 0
    sample_count = 0
    all_diagnostics = []

    seg_manifest = out_dir / "segments.jsonl"
    samp_manifest = out_dir / "manifest.jsonl"

    with seg_manifest.open("w", encoding="utf-8") as seg_f, \
         samp_manifest.open("w", encoding="utf-8") as samp_f:

        for ep_idx, steps in tqdm.tqdm(episodes, desc="Episodes"):
            # Parse
            lang = _get_language_instruction(steps[0])
            T = len(steps)
            if T < cfg.kinematic_window + cfg.min_segment_frames:
                continue

            states = np.array([s["observation"]["state"].numpy() for s in steps], dtype=np.float32)
            images = [s["observation"]["image"].numpy() for s in steps]

            # --- Step A: kinematic segmentation ---
            motions = _compute_motion_signals(states, cfg)
            segments = _build_kinematic_segments(motions["modes"], motions["descriptors"], cfg)

            if not segments:
                continue

            # --- Step B: per-frame vision features ---
            vis_feats = _extract_vision_features(vla, processor, images, device)

            for seg_idx, (seg_s, seg_e) in enumerate(segments):
                frame_ids = list(range(seg_s, seg_e))
                seg_vis = vis_feats[frame_ids]

                # Physical descriptor per segment
                phys_desc = _extract_physical_descriptors(motions, (seg_s, seg_e))
                phys_tiled = np.tile(phys_desc, (len(frame_ids), 1))

                # Fuse
                fused, comps = _fuse_features(seg_vis, phys_tiled, cfg)

                # Coverage selection
                sel = _select_coverage_representatives(fused, cfg)
                chosen_frames = [frame_ids[i] for i in sel["selected_indices"]]

                # Save prototype
                proto_path = out_dir / "diagnostics" / f"prototype_{segment_count:06d}.npy"
                proto_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(proto_path, sel["prototype"])

                # Segment record
                seg_rec = {
                    "segment_id": segment_count,
                    "episode_index": ep_idx,
                    "task": lang,
                    "num_frames": seg_e - seg_s,
                    "motion_mode": int(motions["modes"][seg_s:seg_e].mean()),
                    "selected_frame_indices": chosen_frames,
                    "prototype_error": sel["prototype_error"],
                    "coverage_error": sel["coverage_error"],
                    "inlier_count": sel["inlier_count"],
                    "outlier_count": sel["outlier_count"],
                }
                seg_f.write(json.dumps(seg_rec, ensure_ascii=False) + "\n")

                # Save replay samples
                for rank, fidx in enumerate(chosen_frames):
                    step_data = steps[fidx]
                    sp = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                    np.savez_compressed(
                        sp,
                        image=step_data["observation"]["image"].numpy(),
                        state=step_data["observation"]["state"].numpy(),
                        action=step_data["action"].numpy(),
                        task=lang,
                    )
                    samp_rec = {
                        "sample_id": sample_count,
                        "sample_path": str(sp.relative_to(out_dir)),
                        "task": lang,
                        "episode_index": ep_idx,
                        "episode_frame_index": fidx,
                        "segment_id": segment_count,
                        "coverage_gain": sel["coverage_gains"][rank] if rank < len(sel["coverage_gains"]) else 0.0,
                    }
                    samp_f.write(json.dumps(samp_rec, ensure_ascii=False) + "\n")
                    sample_count += 1

                segment_count += 1

            # Per-episode diagnostics
            all_diagnostics.append({
                "episode": ep_idx,
                "task": lang,
                "num_frames": T,
                "num_segments": len(segments),
                "active_motion_ratio": float((motions["modes"] != 0).mean()),
                "segment_lengths": [e - s for s, e in segments],
            })

    if sample_count == 0:
        raise RuntimeError(
            f"Built 0 samples! Check dataset path or thresholds.\n"
            f"  dataset: {cfg.data_root_dir}/{cfg.dataset_name}\n"
            f"  min_segment_frames: {cfg.min_segment_frames}"
        )

    # ---- meta.json ----
    source_frames = sum(d["num_frames"] for d in all_diagnostics)
    meta = {
        "format": "openvla_prototype_replay_v2",
        "dataset": cfg.dataset_name,
        "saved_replay_samples": sample_count,
        "source_frames": source_frames,
        "compression_ratio": float(sample_count) / max(1, source_frames),
        "num_episodes": cfg.num_episodes,
        "num_segments": segment_count,
        "top_k": cfg.top_k,
        "feature_weights": {"vision": cfg.vision_weight, "physical": cfg.physical_weight},
        "segmentation": {
            "window": cfg.kinematic_window,
            "trans_threshold": cfg.translation_threshold,
            "rot_threshold": cfg.rotation_threshold,
            "grip_threshold": cfg.gripper_threshold,
            "min_segment": cfg.min_segment_frames,
        },
        "selection": {
            "temporal_min_gap": cfg.temporal_min_gap,
            "outlier_mad_scale": cfg.outlier_mad_scale,
        },
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # ---- diagnostics.jsonl ----
    with (out_dir / "diagnostics.jsonl").open("w") as f:
        for d in all_diagnostics:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[5/5] Done!  {segment_count} segments → {sample_count} samples saved → {out_dir}")
    print(f"  compression ratio: {meta['compression_ratio']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prototype Replay v2 buffer builder (OpenVLA)")
    parser.add_argument("--vla-path", default="/root/autodl-tmp/models/openvla-7b")
    parser.add_argument("--cl-lora-path", default="")
    parser.add_argument("--data-root-dir", default="/root/autodl-tmp/modified_libero_rlds")
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/replay_buffers/prototype_v2/taskA")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--shared-depth", type=int, default=8)
    parser.add_argument("--kinematic-window", type=int, default=5)
    parser.add_argument("--translation-threshold", type=float, default=0.03)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-threshold", type=float, default=0.1)
    parser.add_argument("--min-segment-frames", type=int, default=5)
    parser.add_argument("--temporal-min-gap", type=int, default=5)
    parser.add_argument("--outlier-mad-scale", type=float, default=2.0)
    parser.add_argument("--vision-weight", type=float, default=0.5)
    parser.add_argument("--physical-weight", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    cfg = ReplayBufferConfig(
        vla_path=args.vla_path,
        cl_lora_path=args.cl_lora_path,
        data_root_dir=args.data_root_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        top_k=args.top_k,
        lora_rank=args.lora_rank,
        shared_depth=args.shared_depth,
        kinematic_window=args.kinematic_window,
        translation_threshold=args.translation_threshold,
        rotation_threshold=args.rotation_threshold,
        gripper_threshold=args.gripper_threshold,
        min_segment_frames=args.min_segment_frames,
        temporal_min_gap=args.temporal_min_gap,
        outlier_mad_scale=args.outlier_mad_scale,
        vision_weight=args.vision_weight,
        physical_weight=args.physical_weight,
        overwrite=args.overwrite,
    )
    build_buffer(cfg)


if __name__ == "__main__":
    main()
