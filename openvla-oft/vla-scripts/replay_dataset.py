import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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
        img_array = data["image"]  # 原形状: (H, W, 3)
        action = data["action"]    # 原形状: (7,) 
        task_desc = str(data["task"]).encode('utf-8')
        
        # 2. 动作维度对齐 (OpenVLA 期望 Action Chunk 维度 [8, 7])
        if len(action.shape) == 1:
            action = np.tile(action, (8, 1))
            
        # 3. 🚨 核心修复：不要转成 PIL Image！
        # 增加 window_size 维度，模拟 RLDS 切块后的输出 [1, H, W, 3]
        img_with_window = np.expand_dims(img_array, axis=0) 
        
        # 4. 构造 Dummy 轨迹字典
        dummy_step = {
            "observation": {
                "image_primary": img_with_window,
            },
            "task": {
                "language_instruction": task_desc
            },
            "action": action,
            "dataset_name": "libero_spatial_no_noops" 
        }
        
        # 以防万一你的配置里开启了 proprio
        if "state" in data:
            dummy_step["observation"]["proprio"] = np.expand_dims(data["state"], axis=0)
            
        # 让 OpenVLA 自己的 Transform 去做最后的清洗
        processed_batch = self.batch_transform(dummy_step)
        return processed_batch