#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
from pathlib import Path

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, cl_task_id=None, eval_repo_id=None):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        checkpoint_dir = Path(f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}")
        specified_path = checkpoint_dir / "assets"
        entries = os.listdir(specified_path)
        assets_id = entries[0]
        if eval_repo_id is not None and str(eval_repo_id).strip() not in ("", "None", "null"):
            assets_id = str(eval_repo_id)

        config = _config.get_config(self.train_config_name)
        sample_kwargs = None
        if cl_task_id is not None and str(cl_task_id).strip() not in ("", "None", "null"):
            sample_kwargs = {"task_id": int(cl_task_id)}
        norm_stats = None
        if eval_repo_id is not None and str(eval_repo_id).strip() not in ("", "None", "null"):
            norm_stats = self._load_eval_norm_stats(config, checkpoint_dir, assets_id)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
            sample_kwargs=sample_kwargs,
            norm_stats=norm_stats,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    def _load_eval_norm_stats(self, config, checkpoint_dir: Path, asset_id: str):
        candidate_dirs = [
            checkpoint_dir / "assets" / asset_id,
            Path(config.assets_dirs) / asset_id,
        ]
        for assets_base in (Path("./assets"), Path("policy/pi05/assets")):
            for assets_root in sorted(assets_base.glob("*")):
                candidate_dirs.append(assets_root / asset_id)

        for norm_dir in candidate_dirs:
            if (norm_dir / "norm_stats.json").exists():
                return _normalize.load(norm_dir)
        searched = "\n".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(f"Cannot find norm stats for eval_repo_id={asset_id}. Searched:\n{searched}")

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
