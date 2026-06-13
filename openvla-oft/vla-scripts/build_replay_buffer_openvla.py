"""
OpenVLA 专属：基于隐式物理技能切分的原型回放构建脚本。
适用于单卡 RTX 5090 环境，仅通过视觉骨干网络进行轻量化特征提取。
"""

import os
import json
import logging
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from PIL import Image

# 导入 OpenVLA 相关依赖
from transformers import AutoProcessor, AutoModelForVision2Seq
from cl_lora import inject_cl_lora_into_model

# 配置 TF 不使用 GPU，防止与 PyTorch 抢占显存
tf.config.set_visible_devices([], 'GPU')

LOGGER = logging.getLogger(__name__)

class BuildReplayBufferConfig:
    # 基础模型与 CL-LoRA 权重路径
    vla_path = "/root/autodl-tmp/models/openvla-7b"
    cl_lora_path = "/root/autodl-tmp/LOGS/cllora_pure_taskA_4k--4000_chkpt/cl_lora_adapter.pt"
    
    # RLDS 数据集路径
    data_root_dir = "/root/autodl-tmp/modified_libero_rlds"
    dataset_name = "libero_spatial_no_noops"
    target_task_name = "pick up the black bowl next to the cookie box and place it on the plate"
    
    output_dir = "/root/autodl-tmp/replay_buffers/taskA_prototypes"
    
    # 提取参数
    top_k_per_segment = 2
    chunk_frames = 5
    min_segment_frames = 3
    translation_threshold_m = 0.03
    rotation_threshold_rad = 0.05
    gripper_threshold = 0.1
    
    # LIBERO 状态维度解析 (根据 get_libero_env: [xyz(3) + axisangle(3) + gripper(2)])
    xyz_indices = (0, 1, 2)
    euler_indices = (3, 4, 5) # 近似使用 axisangle 作为旋转度量
    gripper_index = 6

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def _compute_chunk_motions(states: np.ndarray, cfg: BuildReplayBufferConfig) -> list:
    """计算连续帧之间的物理运动量"""
    if len(states) <= cfg.chunk_frames:
        return []

    motions = []
    for i in range(0, len(states) - cfg.chunk_frames):
        s0, s1 = states[i], states[i + cfg.chunk_frames]
        
        xyz0, rpy0, g0 = s0[list(cfg.xyz_indices)], s0[list(cfg.euler_indices)], s0[cfg.gripper_index]
        xyz1, rpy1, g1 = s1[list(cfg.xyz_indices)], s1[list(cfg.euler_indices)], s1[cfg.gripper_index]
        
        d_xyz = xyz1 - xyz0
        d_rpy = _wrap_to_pi(rpy1 - rpy0)
        d_g = g1 - g0

        raw_mag = np.array([abs(d_xyz[0]), abs(d_xyz[1]), abs(d_xyz[2]), 
                            abs(d_rpy[0]), abs(d_rpy[1]), abs(d_rpy[2]), abs(d_g)])
        
        norm_mag = raw_mag / np.array([cfg.translation_threshold_m]*3 + 
                                      [cfg.rotation_threshold_rad]*3 + 
                                      [cfg.gripper_threshold])
        
        dominant = ["tx", "ty", "tz", "rx", "ry", "rz", "grip"][int(np.argmax(norm_mag))] if np.max(norm_mag) >= 1.0 else "idle"
        
        motions.append({"end_frame": i + cfg.chunk_frames, "dominant": dominant, "grip_delta": float(d_g)})
    return motions

def _find_boundaries(motions: list, cfg: BuildReplayBufferConfig) -> list:
    """基于物理启发式规则寻找轨迹切分点"""
    boundaries = set()
    labels = [m["dominant"] for m in motions]

    for m in motions:
        if abs(m["grip_delta"]) >= cfg.gripper_threshold:
            boundaries.add(m["end_frame"])

    for i in range(1, len(motions)):
        if labels[i] != labels[i - 1]:
            # 简单去抖动
            if i + 2 < len(motions) and labels[i] == labels[i+1] == labels[i+2]:
                boundaries.add(motions[i]["end_frame"])
                
    return sorted(boundaries)

@torch.no_grad()
def _extract_visual_feature(model, processor, image: np.ndarray, device) -> np.ndarray:
    """
    轻量化特征提取：仅使用 OpenVLA 的 Vision Backbone (SigLIP + DINOv2)
    避开 7B LLaMA 解码器，彻底解决 OOM 问题。
    """
    pil_img = Image.fromarray(image).convert("RGB")
    inputs = processor(text="dummy", images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    
    # 这一行之前可能被不小心删掉了，它负责将图像输入视觉骨干网络
    patch_features = model.vision_backbone(pixel_values) # Shape: [1, num_patches, embed_dim]
    
    # 这是我们刚才修改的行，加入了 .to(torch.float32) 解决 NumPy 不兼容 BFloat16 的问题
    pooled_feature = patch_features.mean(dim=1).squeeze(0).cpu().to(torch.float32).numpy()
    
    return pooled_feature

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = BuildReplayBufferConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备输出目录
    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "prototypes").mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("正在加载 OpenVLA-7B 模型骨架...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    
    LOGGER.info("正在注入 Task A 的 CL-LoRA 权重...")
    vla = inject_cl_lora_into_model(vla, rank=32, alpha=32.0, dropout=0.0, shared_split_ratio=0.25)
    cl_state_dict = torch.load(cfg.cl_lora_path, map_location="cpu")
    vla.load_state_dict(cl_state_dict, strict=False)
    vla.eval()

    # 2. 读取 RLDS 数据集
    LOGGER.info(f"正在加载 RLDS 数据集: {cfg.dataset_name}")
    builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_root_dir)
    dataset = builder.as_dataset(split='all')
    
    segment_manifest = out_dir / "segments.jsonl"
    sample_manifest = out_dir / "manifest.jsonl"
    
    segment_count = 0
    sample_count = 0

    with segment_manifest.open("w", encoding="utf-8") as seg_f, sample_manifest.open("w", encoding="utf-8") as samp_f:
        for ep_idx, episode in enumerate(tqdm.tqdm(dataset, desc="Processing Episodes")):
            steps = list(episode['steps'])
            
            # 任务过滤
            step_zero = steps[0]
            if 'language_instruction' in step_zero:
                raw_inst = step_zero['language_instruction']
            elif 'natural_language_instruction' in step_zero['observation']:
                raw_inst = step_zero['observation']['natural_language_instruction']
            else:
                step_keys = list(step_zero.keys())
                obs_keys = list(step_zero['observation'].keys())
                raise KeyError(f"找不到语言指令！\nStep 包含的键: {step_keys}\nObservation 包含的键: {obs_keys}")
            
            # 转为 numpy 对象（如果是 Tensor 的话）
            if hasattr(raw_inst, 'numpy'):
                raw_inst = raw_inst.numpy()
                
            # 兼容 bytes 和 str 两种情况
            if isinstance(raw_inst, bytes):
                language_instruction = raw_inst.decode('utf-8')
            else:
                language_instruction = str(raw_inst)

            if cfg.target_task_name not in language_instruction:
                continue

            num_frames = len(steps)
            if num_frames < cfg.chunk_frames:
                continue

            # 提取整个 episode 的状态序列用于切分
            states = np.array([step['observation']['state'].numpy() for step in steps])
            
            motions = _compute_chunk_motions(states, cfg)
            boundaries = _find_boundaries(motions, cfg)
            
            b_list = sorted({x for x in boundaries if 0 < x < num_frames})
            starts = [0] + b_list
            ends = b_list + [num_frames]
            segments = [(s, e) for s, e in zip(starts, ends) if e - s >= cfg.min_segment_frames]
            
            if not segments:
                continue

            # 按 Segment 处理
            for local_seg_idx, (seg_s, seg_e) in enumerate(segments):
                frame_ids = list(range(seg_s, seg_e))
                seg_feats = []
                
                # 特征提取
                for fid in frame_ids:
                    img = steps[fid]['observation']['image'].numpy()
                    feat = _extract_visual_feature(vla, processor, img, device)
                    seg_feats.append(feat)
                
                feat_mat = np.stack(seg_feats, axis=0)
                feat_norm = feat_mat / np.linalg.norm(feat_mat, axis=-1, keepdims=True)
                proto = np.mean(feat_norm, axis=0)
                proto = proto / np.linalg.norm(proto)
                
                # 余弦相似度找 Top-K
                cosine = feat_norm @ proto
                k = min(cfg.top_k_per_segment, len(frame_ids))
                top_idx = np.argsort(-cosine)[:k]
                
                # 保存 Prototype
                proto_path = out_dir / "prototypes" / f"segment_{segment_count:08d}.npy"
                np.save(proto_path, proto.astype(np.float32))
                
                # 记录 Segment Meta
                seg_record = {
                    "segment_id": segment_count, "episode_index": ep_idx,
                    "task": language_instruction, "num_frames": seg_e - seg_s,
                    "selected_episode_frame_indices": [int(frame_ids[i]) for i in top_idx.tolist()]
                }
                seg_f.write(json.dumps(seg_record, ensure_ascii=False) + "\n")
                
                # 保存关键帧及其信息
                for rank, i_local in enumerate(top_idx.tolist()):
                    ep_frame_idx = frame_ids[i_local]
                    step_data = steps[ep_frame_idx]
                    
                    sample_path = out_dir / "samples" / f"sample_{sample_count:08d}.npz"
                    np.savez_compressed(
                        sample_path,
                        image=step_data['observation']['image'].numpy(),
                        state=step_data['observation']['state'].numpy(),
                        action=step_data['action'].numpy(),
                        task=language_instruction
                    )
                    
                    samp_record = {
                        "sample_id": sample_count,
                        "sample_path": str(sample_path.relative_to(out_dir)),
                        "task": language_instruction,
                        "episode_index": ep_idx,
                        "episode_frame_index": ep_frame_idx,
                        "cosine_to_prototype": float(cosine[i_local])
                    }
                    samp_f.write(json.dumps(samp_record, ensure_ascii=False) + "\n")
                    sample_count += 1
                
                segment_count += 1

    LOGGER.info(f"原型回放池构建完成！Segments: {segment_count} | Samples: {sample_count}")
    LOGGER.info(f"输出目录: {out_dir}")

if __name__ == "__main__":
    main()