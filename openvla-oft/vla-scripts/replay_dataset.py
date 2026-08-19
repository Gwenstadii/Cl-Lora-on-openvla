import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSBatchTransform

class PrototypeReplayDataset(Dataset):
    def __init__(self, replay_dir: str, batch_transform: RLDSBatchTransform):
        self.replay_dir = replay_dir
        self.batch_transform = batch_transform
        self.samples = []
        
        # 读取 manifest.jsonl
        manifest_path = os.path.join(replay_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                self.samples.append(record)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Manifest at {manifest_path} is empty. "
                "The replay buffer may have been built with zero matching samples. "
                "Check that the buffer build script's target_task_name matches the dataset."
            )

        print(f"[Replay Dataset] Successfully loaded {len(self.samples)} prototype frames from {replay_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        npz_path = os.path.join(self.replay_dir, record["sample_path"])
        
        # 1. 加载 npz 数据 (此时是 Numpy 数组)
        data = np.load(npz_path)
        # 兼容两种 key 命名: 新版 (robotwin) 用 image_primary, 旧版 (libero) 用 image
        if "image_primary" in data:
            img_array = data["image_primary"]  # 原形状: (H, W, 3)  head 相机
            left_wrist = data.get("left_wrist_image", None)   # (H, W, 3) 或 None
            right_wrist = data.get("right_wrist_image", None) # (H, W, 3) 或 None
        else:
            img_array = data["image"]          # LIBERO 单图兼容
            left_wrist = None
            right_wrist = None
        action = data["action"]    # [chunk, dim] 已归一化 (robotwin: [25,14]; libero 旧版: [8,7])
        task_desc = str(data["task"]).encode('utf-8')
        dataset_name = str(data.get("dataset_name", "libero_spatial_no_noops"))
        
        # 2. 动作维度对齐 (OpenVLA 期望 Action Chunk 维度 [NUM_ACTIONS_CHUNK, ACTION_DIM])
        if len(action.shape) == 1:
            action = np.tile(action, (NUM_ACTIONS_CHUNK, 1))
        elif action.shape[0] != NUM_ACTIONS_CHUNK:
            # 旧 buffer (LIBERO chunk=8) 与当前平台 chunk 不一致时, 沿时间维广播
            action = np.tile(action[0:1], (NUM_ACTIONS_CHUNK, 1)) if action.shape[0] > 0 else action
            
        # 3. 🚨 核心修复：不要转成 PIL Image！
        # 增加 window_size 维度，模拟 RLDS 切块后的输出 [1, H, W, 3]
        img_with_window = np.expand_dims(img_array, axis=0)
        
        # 4. 构造 Dummy 轨迹字典
        observation = {
            "image_primary": img_with_window,
        }
        if left_wrist is not None and right_wrist is not None:
            observation["left_wrist_image"] = np.expand_dims(left_wrist, axis=0)
            observation["right_wrist_image"] = np.expand_dims(right_wrist, axis=0)
        # 以防万一你的配置里开启了 proprio
        if "proprio" in data:
            observation["proprio"] = np.expand_dims(data["proprio"], axis=0)
        elif "state" in data:
            observation["proprio"] = np.expand_dims(data["state"], axis=0)

        dummy_step = {
            "observation": observation,
            "task": {
                "language_instruction": task_desc
            },
            "action": action,
            "dataset_name": dataset_name,
        }
        
        # 让 OpenVLA 自己的 Transform 去做最后的清洗
        processed_batch = self.batch_transform(dummy_step)
        return processed_batch
