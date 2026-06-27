"""
PrototypeReplayDataset: reads offline prototype replay buffer .npz samples
and converts them to the format expected by RLDSBatchTransform for training.

Compatible with buffers built by:
  - build_replay_buffer_openvla.py  (prototype-based, saves action_chunk)
  - build_uniform_replay_buffer.py  (uniform, saves action_chunk)
  - Legacy buffers (single action, tile fallback)
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PrototypeReplayDataset(Dataset):
    def __init__(self, replay_dir: str, batch_transform, min_samples: int = 0):
        self.replay_dir = replay_dir
        self.batch_transform = batch_transform

        manifest_path = os.path.join(replay_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        self._records = records
        self._base_len = len(records)
        # Ensure len >= batch_size to avoid DataLoader crash
        self._len = max(self._base_len, int(min_samples))

        print(f"[Replay Dataset] Loaded {self._base_len} prototype frames from {replay_dir}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        rec = self._records[idx % self._base_len]
        npz_path = os.path.join(self.replay_dir, rec["sample_path"])

        data = np.load(npz_path, allow_pickle=False)

        # --- Image ---
        img_array = data["image"]  # (H, W, 3)

        # --- Action chunk ---
        if "action_chunk" in data:
            action = np.asarray(data["action_chunk"], dtype=np.float32)
        elif "action" in data:
            # Legacy buffer: single action, tile to [action_horizon, 7]
            action = np.asarray(data["action"], dtype=np.float32)
            if action.ndim == 1:
                action = np.tile(action, (8, 1))
        else:
            raise KeyError(f"Neither 'action_chunk' nor 'action' found in {npz_path}")

        # --- Task description ---
        task_raw = data.get("task", rec.get("task", ""))
        if isinstance(task_raw, np.ndarray):
            task_raw = str(task_raw)
        task_desc = task_raw.encode("utf-8") if isinstance(task_raw, str) else task_raw

        # --- Dataset name ---
        dataset_name = rec.get("dataset_name", "libero_spatial_no_noops")

        # --- Construct dummy step matching RLDS format ---
        # RLDS batch format expected by RLDSBatchTransform:
        #   action: [window_size + future_action_window_size, 7] = [8, 7]
        #   observation["image_primary"]: [window_size, H, W, 3] = [1, H, W, 3]
        #   task["language_instruction"]: bytes
        #   dataset_name: str
        img_with_window = np.expand_dims(img_array, axis=0)  # [1, H, W, 3]

        dummy_step = {
            "observation": {
                "image_primary": img_with_window,
            },
            "task": {
                "language_instruction": task_desc,
            },
            "action": action,  # [8, 7] (or [action_horizon, 7])
            "dataset_name": dataset_name,
        }

        # Optional: proprio state
        if "state" in data:
            dummy_step["observation"]["proprio"] = np.expand_dims(
                np.asarray(data["state"], dtype=np.float32), axis=0
            )

        processed_batch = self.batch_transform(dummy_step)
        return processed_batch
