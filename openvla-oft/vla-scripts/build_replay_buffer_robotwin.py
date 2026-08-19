"""
build_replay_buffer_robotwin.py — Prototype Replay v2 buffer builder for OpenVLA + RoboTwin (bimanual ALOHA).

基于 build_replay_buffer_openvla.py (LIBERO 版) 改造, 适配 RoboTwin:
  - 双臂 14D 关节 proprio (JOINT_BIMANUAL): 物理分割用 左7+右7+双夹爪 关节差分
  - 3 相机 (head / left_wrist / right_wrist): 选帧特征用 head 图, sample 存 3 图
  - 数据经 prismatic make_single_dataset 加载 → action/proprio 已按训练侧归一化,
    与 RLDSBatchTransform 输出同分布, replay 训练 loss 尺度一致
  - sample npz 额外存 dataset_name (每任务 buffer 自带, replay_dataset 直接使用)

用法 (服务器, openvla-oft 目录下):
  python vla-scripts/build_replay_buffer_robotwin.py \
    --vla-path $VLA_PATH \
    --data-root-dir datasets/rlds \
    --dataset-name aloha_handover_mic_clean \
    --output-dir $LOGS_ROOT/replay_buffers/taskA \
    --num-episodes 10 --top-k 3 --overwrite
"""

import argparse
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

from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ReplayBufferConfig:
    vla_path: str = "/mnt/data/pengshengdi/models/openvla-7b"
    data_root_dir: str = "datasets/rlds"
    dataset_name: str = "aloha_handover_mic_clean"
    output_dir: str = "/mnt/data/pengshengdi/LOGS-RT/replay_buffers/taskA"
    stats_path: str = ""                 # checkpoint 的 dataset_statistics.json (归一化 action/proprio 用)
    num_episodes: int = 10
    top_k: int = 3

    # 双臂关节空间分割阈值 (proprio 已归一化到 [-1,1])
    kinematic_window: int = 5
    joint_threshold: float = 0.05      # 关节差分阈值 (归一化空间)
    gripper_threshold: float = 0.1
    min_segment_frames: int = 5
    descriptor_clip: float = 5.0

    # 覆盖选择
    temporal_min_gap: int = 5
    outlier_mad_scale: float = 2.0

    # 特征融合权重
    vision_weight: float = 0.5
    physical_weight: float = 0.5

    overwrite: bool = False


# ---------------------------------------------------------------------------
# 运动信号 (双臂关节空间)
# ---------------------------------------------------------------------------

def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _moving_average(seq: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(seq, kernel, mode="same")


def _compute_motion_signals_bimanual(
    proprio: np.ndarray,
    cfg: ReplayBufferConfig,
) -> Dict[str, np.ndarray]:
    """Per-frame joint-space motion signals for bimanual 14D proprio [T, 14].

    左臂 = 0:7, 右臂 = 7:14, 夹爪 = 6 和 13。
    """
    T = proprio.shape[0]
    window = cfg.kinematic_window
    jt = cfg.joint_threshold
    gt = cfg.gripper_threshold

    left_delta = np.zeros((T, 7), dtype=np.float32)
    right_delta = np.zeros((T, 7), dtype=np.float32)
    grip_delta = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        future = min(t + window, T - 1)
        left_delta[t] = _wrap_angle(proprio[future, 0:7] - proprio[t, 0:7])
        right_delta[t] = _wrap_angle(proprio[future, 7:14] - proprio[t, 7:14])
        grip_delta[t] = proprio[future, [6, 13]] - proprio[t, [6, 13]]

    scaled = np.concatenate([
        left_delta / jt,        # [T, 7]
        right_delta / jt,       # [T, 7]
        grip_delta / gt,        # [T, 2]
    ], axis=-1)                 # [T, 16]

    smoothed = np.stack([
        _moving_average(scaled[:, i], window) for i in range(16)
    ], axis=-1)
    smoothed = np.clip(smoothed, -cfg.descriptor_clip, cfg.descriptor_clip)

    # 运动模式: 16 个分量中最大者
    modes = np.zeros(T, dtype=np.int32)
    for t in range(T):
        comp = int(np.argmax(np.abs(smoothed[t])))
        if abs(smoothed[t, comp]) >= 1.0:
            modes[t] = 1 + 2 * comp + int(smoothed[t, comp] < 0.0)
        else:
            modes[t] = 0

    activity = np.linalg.norm(smoothed, axis=-1)
    current_gripper = np.clip(proprio[:, [6, 13]], -1.0, 1.0)

    descriptors = np.concatenate(
        [smoothed, current_gripper, activity[:, None]], axis=-1
    ).astype(np.float32)  # [T, 19]

    return {
        "descriptors": descriptors,
        "modes": modes,
        "left_delta": left_delta,
        "right_delta": right_delta,
        "gripper_delta": grip_delta,
    }


# ---------------------------------------------------------------------------
# 运动分割 (与原版相同)
# ---------------------------------------------------------------------------

def _run_length_encode_modes(modes: np.ndarray) -> List[Tuple[int, int, int]]:
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


def _mean_descriptor(descriptors: np.ndarray, seg: Tuple[int, int]) -> np.ndarray:
    s, e = seg
    feat = descriptors[s:e].mean(axis=0)
    norm = float(np.linalg.norm(feat))
    return feat / max(norm, 1e-8)


def _descriptor_distance(descriptors: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return 1.0 - float(np.dot(_mean_descriptor(descriptors, a), _mean_descriptor(descriptors, b)))


def _merge_short_segments(segments: List[Tuple[int, int]], descriptors: np.ndarray, min_len: int):
    segments = list(segments)
    while len(segments) > 1:
        short_idx = next((i for i, (s, e) in enumerate(segments) if e - s < min_len), None)
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
            segments[short_idx - 1:short_idx + 1] = [(prev_s, end)]
        else:
            _, next_e = segments[short_idx + 1]
            segments[short_idx:short_idx + 2] = [(start, next_e)]
    return segments


def _build_kinematic_segments(modes: np.ndarray, descriptors: np.ndarray, cfg: ReplayBufferConfig):
    runs = _run_length_encode_modes(modes)
    segments = [(s, e) for s, e, _m in runs if e - s >= 2]
    if not segments:
        return [(0, len(modes))]
    return _merge_short_segments(segments, descriptors, cfg.min_segment_frames)


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_vision_features(model, processor, images: List[np.ndarray], device: torch.device) -> np.ndarray:
    """head 图逐帧过冻结视觉骨干, 返回 L2 归一化特征 [N, embed_dim]."""
    features = []
    for img in images:
        pil = Image.fromarray(img).convert("RGB")
        inputs = processor(text="dummy", images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        patch_feat = model.vision_backbone(pixel_values)
        pooled = patch_feat.mean(dim=1).squeeze(0).cpu().to(torch.float32).numpy()
        features.append(pooled)
    feats = np.stack(features, axis=0)
    feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    return feats


def _extract_physical_descriptors(motions: Dict[str, np.ndarray], seg: Tuple[int, int]) -> np.ndarray:
    """3 组序列 (左臂/右臂/夹爪 活动范数) × 3 stats (mean/std/max) = [1, 9]."""
    s, e = seg
    if e <= s:
        return np.zeros((1, 9), dtype=np.float32)
    left_norm = np.linalg.norm(motions["left_delta"][s:e], axis=-1)
    right_norm = np.linalg.norm(motions["right_delta"][s:e], axis=-1)
    grip_norm = np.linalg.norm(motions["gripper_delta"][s:e], axis=-1)
    stats = []
    for series in [left_norm, right_norm, grip_norm]:
        stats.extend([np.mean(series), np.std(series), np.max(series)])
    return np.array(stats, dtype=np.float32).reshape(1, -1)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return mat / norm


def _fuse_features(vision: np.ndarray, physical: np.ndarray, cfg: ReplayBufferConfig):
    weights = np.array([cfg.vision_weight, cfg.physical_weight], dtype=np.float32)
    weights /= weights.sum()
    fused = np.concatenate([
        math.sqrt(float(weights[0])) * _normalize_rows(vision.astype(np.float32)),
        math.sqrt(float(weights[1])) * _normalize_rows(physical.astype(np.float32)),
    ], axis=-1)
    return _normalize_rows(fused)


# ---------------------------------------------------------------------------
# 覆盖选择 (与原版相同)
# ---------------------------------------------------------------------------

def _coverage_error(features: np.ndarray, selected: List[int]) -> float:
    if not selected:
        return 1.0
    best_sim = np.max(features @ features[selected].T, axis=1)
    return float(np.mean(1.0 - best_sim))


def _select_coverage_representatives(features: np.ndarray, cfg: ReplayBufferConfig) -> Dict[str, Any]:
    features = _normalize_rows(np.asarray(features, dtype=np.float32))
    n = features.shape[0]
    top_k = min(max(1, cfg.top_k), n)

    similarity = features @ features.T
    medoid = int(np.argmax(similarity.mean(axis=1)))
    dist_to_medoid = 1.0 - similarity[:, medoid]

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

    inlier_sim = similarity[np.ix_(inliers, inliers)]
    medoid = int(inliers[int(np.argmax(inlier_sim.mean(axis=1)))])

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
# 语言指令 / 归一化 (与训练侧 BOUNDS 公式一致)
# ---------------------------------------------------------------------------

def _get_language_instruction(step: Any) -> str:
    """从 step 探测语言指令 (兼容多种字段形态)."""
    candidates = []
    if isinstance(step, dict):
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
    raise KeyError(f"Cannot find language instruction. Step keys: {list(step.keys()) if isinstance(step, dict) else type(step)}")


def _normalize_bounds(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """与 prismatic normalize_action_and_proprio (BOUNDS) 完全一致的归一化."""
    values = np.asarray(values, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    out = 2.0 * (values - low) / (high - low + 1e-8) - 1.0
    out = np.clip(out, -1.0, 1.0)
    zeros_mask = low == high
    out[..., zeros_mask] = 0.0
    return out.astype(np.float32)


def _load_normalization_stats(stats_path: str, dataset_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """从 checkpoint 的 dataset_statistics.json 取 action/proprio 的 min/max."""
    if not stats_path or not os.path.isfile(stats_path):
        print(f"[WARN] stats_path 未提供或不存在: {stats_path!r} —— action/proprio 不做归一化!")
        return None, None
    with open(stats_path, "r") as f:
        stats = json.load(f)
    if dataset_name not in stats:
        print(f"[WARN] dataset_statistics.json 里没有 {dataset_name} 的统计! keys={sorted(stats.keys())}")
        return None, None
    ds_stats = stats[dataset_name]
    action_stats = {k: np.asarray(v, dtype=np.float64) for k, v in ds_stats.get("action", {}).items()}
    proprio_stats = {k: np.asarray(v, dtype=np.float64) for k, v in ds_stats.get("proprio", {}).items()}
    if "min" not in action_stats or "max" not in action_stats:
        print(f"[WARN] action stats 缺 min/max: keys={list(action_stats.keys())}")
        return None, None
    print(f"[OK] 归一化统计已加载: action min/max (dim={action_stats['min'].shape}), "
          f"proprio min/max (dim={proprio_stats.get('min', np.array([])).shape})")
    return action_stats, proprio_stats


def _normalize_action(action: np.ndarray, action_stats: Optional[Dict]) -> np.ndarray:
    if action_stats is None:
        return action.astype(np.float32)
    return _normalize_bounds(action, action_stats["min"], action_stats["max"])


def _normalize_proprio(proprio: np.ndarray, proprio_stats: Optional[Dict]) -> np.ndarray:
    if proprio_stats is None:
        return proprio.astype(np.float32)
    return _normalize_bounds(proprio, proprio_stats["min"], proprio_stats["max"])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_buffer(cfg: ReplayBufferConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = pathlib.Path(cfg.output_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not cfg.overwrite:
            raise FileExistsError(f"{out_dir} not empty.  Pass --overwrite.")
        shutil.rmtree(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

    # ---- [1/5] 模型 ----
    print("[1/5] Loading OpenVLA model (frozen vision backbone for feature extraction) ...")
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device)
    vla.eval()

    # ---- [2/5] RoboTwin RLDS 数据集 (tfds episode 级, 与 LIBERO 原版一致) ----
    print(f"[2/5] Loading RLDS dataset: {cfg.dataset_name}")
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    ds = builder.as_dataset(split="all")

    # 归一化统计 (来自 checkpoint 的 dataset_statistics.json, 与训练侧同分布)
    action_stats, proprio_stats = _load_normalization_stats(cfg.stats_path, cfg.dataset_name)

    # ---- [3/5] 逐 episode 处理 ----
    print(f"[3/5] Processing up to {cfg.num_episodes} episodes ...")
    segment_count = 0
    sample_count = 0
    all_diagnostics = []

    seg_manifest = out_dir / "segments.jsonl"
    samp_manifest = out_dir / "manifest.jsonl"

    with seg_manifest.open("w", encoding="utf-8") as seg_f, \
         samp_manifest.open("w", encoding="utf-8") as samp_f:

        for ep_idx, episode in enumerate(ds):
            if ep_idx >= cfg.num_episodes:
                break

            steps = list(episode["steps"])
            T = len(steps)
            if T < cfg.kinematic_window + cfg.min_segment_frames:
                continue

            # --- 字段探测 (打印一次帮助排错) ---
            if ep_idx == 0:
                obs_keys = list(steps[0]["observation"].keys()) if "observation" in steps[0] else []
                step_keys = list(steps[0].keys())
                print(f"  [INFO] step keys: {step_keys}")
                print(f"  [INFO] observation keys: {obs_keys}")

            lang = _get_language_instruction(steps[0])
            print(f"  episode {ep_idx}: T={T}, task={lang!r}")

            # state: observation["state"] [14]  (JOINT_BIMANUAL: 左7+右7)
            states_raw = np.array([s["observation"]["state"].numpy() for s in steps], dtype=np.float32)
            if states_raw.ndim != 2 or states_raw.shape[1] != 14:
                raise RuntimeError(f"state 维度异常: {states_raw.shape} (期望 [T, 14])")
            proprio = _normalize_proprio(states_raw, proprio_stats)          # [T, 14] 归一化

            # 图像: head = image, 双腕 = left/right_wrist_image
            obs0 = steps[0]["observation"]
            if "image" not in obs0 or "left_wrist_image" not in obs0 or "right_wrist_image" not in obs0:
                raise RuntimeError(f"observation 缺少相机字段, 实际 keys: {list(obs0.keys())}")
            head_imgs = np.array([s["observation"]["image"].numpy() for s in steps], dtype=np.uint8)
            left_imgs = np.array([s["observation"]["left_wrist_image"].numpy() for s in steps], dtype=np.uint8)
            right_imgs = np.array([s["observation"]["right_wrist_image"].numpy() for s in steps], dtype=np.uint8)

            # 动作: 归一化 + 真实未来 chunk [T, 25, 14]
            actions_raw = np.array([s["action"].numpy() for s in steps], dtype=np.float32)
            if actions_raw.ndim != 2 or actions_raw.shape[1] != 14:
                raise RuntimeError(f"action 维度异常: {actions_raw.shape} (期望 [T, 14])")
            actions_norm = _normalize_action(actions_raw, action_stats)      # [T, 14]
            action_chunks = np.repeat(actions_norm[:, None, :], NUM_ACTIONS_CHUNK, axis=1)  # [T, 25, 14]
            for t in range(T):
                end = min(t + NUM_ACTIONS_CHUNK, T)
                action_chunks[t, : end - t] = actions_norm[t:end]

            # --- Step A: 运动分割 (双臂关节) ---
            motions = _compute_motion_signals_bimanual(proprio, cfg)
            segments = _build_kinematic_segments(motions["modes"], motions["descriptors"], cfg)
            if not segments:
                continue

            # --- Step B: 逐帧视觉特征 (head 图) ---
            vis_feats = _extract_vision_features(vla, processor, list(head_imgs), device)

            for seg_idx, (seg_s, seg_e) in enumerate(segments):
                frame_ids = list(range(seg_s, seg_e))
                seg_vis = vis_feats[frame_ids]

                phys_desc = _extract_physical_descriptors(motions, (seg_s, seg_e))
                phys_tiled = np.tile(phys_desc, (len(frame_ids), 1))
                fused = _fuse_features(seg_vis, phys_tiled, cfg)

                sel = _select_coverage_representatives(fused, cfg)
                chosen_frames = [frame_ids[i] for i in sel["selected_indices"]]

                proto_path = out_dir / "diagnostics" / f"prototype_{segment_count:06d}.npy"
                proto_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(proto_path, sel["prototype"])

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

                # 保存回放样本 (3 相机 + 归一化 proprio/action chunk + task)
                for rank, fidx in enumerate(chosen_frames):
                    sp = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                    np.savez_compressed(
                        sp,
                        image_primary=head_imgs[fidx],
                        left_wrist_image=left_imgs[fidx],
                        right_wrist_image=right_imgs[fidx],
                        proprio=proprio[fidx],
                        action=action_chunks[fidx],       # [25, 14] 归一化真实 chunk
                        task=lang,
                        dataset_name=cfg.dataset_name,
                    )
                    samp_rec = {
                        "sample_id": sample_count,
                        "sample_path": str(sp.relative_to(out_dir)),
                        "task": lang,
                        "episode_index": ep_idx,
                        "episode_frame_index": int(fidx),
                        "segment_id": segment_count,
                        "coverage_gain": sel["coverage_gains"][rank] if rank < len(sel["coverage_gains"]) else 0.0,
                    }
                    samp_f.write(json.dumps(samp_rec, ensure_ascii=False) + "\n")
                    sample_count += 1

                segment_count += 1

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
            f"  dataset: {cfg.data_root_dir}/{cfg.dataset_name}"
        )

    # ---- meta.json ----
    source_frames = sum(d["num_frames"] for d in all_diagnostics)
    meta = {
        "format": "openvla_prototype_replay_v2_robotwin",
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
            "joint_threshold": cfg.joint_threshold,
            "gripper_threshold": cfg.gripper_threshold,
            "min_segment": cfg.min_segment_frames,
        },
        "selection": {
            "temporal_min_gap": cfg.temporal_min_gap,
            "outlier_mad_scale": cfg.outlier_mad_scale,
        },
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    with (out_dir / "diagnostics.jsonl").open("w") as f:
        for d in all_diagnostics:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n[5/5] Done!  {segment_count} segments → {sample_count} samples saved → {out_dir}")
    print(f"  compression ratio: {meta['compression_ratio']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prototype Replay v2 buffer builder (OpenVLA + RoboTwin)")
    parser.add_argument("--vla-path", default="/mnt/data/pengshengdi/models/openvla-7b")
    parser.add_argument("--data-root-dir", default="datasets/rlds")
    parser.add_argument("--dataset-name", default="aloha_handover_mic_clean")
    parser.add_argument("--output-dir", default="/mnt/data/pengshengdi/LOGS-RT/replay_buffers/taskA")
    parser.add_argument("--stats-path", default="",
                        help="checkpoint 的 dataset_statistics.json 路径 (归一化 action/proprio, 必须与训练同源)")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--kinematic-window", type=int, default=5)
    parser.add_argument("--joint-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-threshold", type=float, default=0.1)
    parser.add_argument("--min-segment-frames", type=int, default=5)
    parser.add_argument("--temporal-min-gap", type=int, default=5)
    parser.add_argument("--outlier-mad-scale", type=float, default=2.0)
    parser.add_argument("--vision-weight", type=float, default=0.5)
    parser.add_argument("--physical-weight", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    cfg = ReplayBufferConfig()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    build_buffer(cfg)


if __name__ == "__main__":
    main()
