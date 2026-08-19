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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm
from PIL import Image

from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.rlds import make_single_dataset
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE

tf = None  # make_single_dataset 内部使用 tf; 这里不直接依赖


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ReplayBufferConfig:
    vla_path: str = "/mnt/data/pengshengdi/models/openvla-7b"
    data_root_dir: str = "datasets/rlds"
    dataset_name: str = "aloha_handover_mic_clean"
    output_dir: str = "/mnt/data/pengshengdi/LOGS-RT/replay_buffers/taskA"
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
# 语言指令
# ---------------------------------------------------------------------------

def _get_language_instruction(step: Any) -> str:
    instr = step["task"]["language_instruction"]
    if hasattr(instr, "numpy"):
        instr = instr.numpy()
    if isinstance(instr, np.ndarray):
        instr = instr.item() if instr.size == 1 else instr[0]
    if isinstance(instr, bytes):
        instr = instr.decode("utf-8")
    return str(instr)


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

    # ---- [2/5] RoboTwin RLDS 轨迹 (与训练同管线: 3 相机 + proprio 归一化) ----
    print(f"[2/5] Loading RLDS dataset: {cfg.dataset_name}")
    # 与 RLDSDataset.__init__ 相同的 aloha 配置
    mixture_spec = [(cfg.dataset_name, 1.0)]
    per_dataset_kwargs, _weights = get_oxe_dataset_kwargs_and_weights(
        cfg.data_root_dir,
        mixture_spec,
        load_camera_views=("primary", "left_wrist", "right_wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )
    traj_transform_kwargs = dict(
        window_size=1,
        future_action_window_size=NUM_ACTIONS_CHUNK - 1,
        skip_unlabeled=True,
        goal_relabeling_strategy="uniform",
    )
    frame_transform_kwargs = dict(
        resize_size=(224, 224),
        num_parallel_calls=8,
    )
    ds = make_single_dataset(
        per_dataset_kwargs[0],
        train=False,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )

    # ---- [3/5] 逐 episode 处理 ----
    print(f"[3/5] Processing up to {cfg.num_episodes} episodes ...")
    segment_count = 0
    sample_count = 0
    all_diagnostics = []

    seg_manifest = out_dir / "segments.jsonl"
    samp_manifest = out_dir / "manifest.jsonl"

    with seg_manifest.open("w", encoding="utf-8") as seg_f, \
         samp_manifest.open("w", encoding="utf-8") as samp_f:

        for ep_idx, batch in enumerate(ds.as_numpy_iterator()):
            if ep_idx >= cfg.num_episodes:
                break

            obs = batch["observation"]
            lang = _get_language_instruction(batch)
            proprio = np.asarray(obs["proprio"], dtype=np.float32)          # [T, 14] 归一化
            head_imgs = np.asarray(obs["image_primary"])                    # [T, H, W, 3]
            T = head_imgs.shape[0]
            if T < cfg.kinematic_window + cfg.min_segment_frames:
                continue
            print(f"  episode {ep_idx}: T={T}, task={lang!r}")

            # 动作 chunk: [T, chunk, 14] 或 [T, 14]
            action_raw = np.asarray(batch["action"], dtype=np.float32)
            if action_raw.ndim == 2:
                action_raw = np.repeat(action_raw[:, None, :], NUM_ACTIONS_CHUNK, axis=1)

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

                # 保存回放样本 (3 相机 + 归一化 proprio/action + task)
                for rank, fidx in enumerate(chosen_frames):
                    sp = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                    np.savez_compressed(
                        sp,
                        image_primary=head_imgs[fidx],
                        left_wrist_image=obs["left_wrist_image"][fidx],
                        right_wrist_image=obs["right_wrist_image"][fidx],
                        proprio=proprio[fidx],
                        action=action_raw[fidx],          # [chunk, 14] 归一化
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
