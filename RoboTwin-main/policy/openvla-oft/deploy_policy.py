import os
import sys
import json
import glob
import numpy as np
import torch
from dataclasses import dataclass

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)

sys.path.insert(0, "/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft/vla-scripts")


@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = True
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    eval_task_id: int = 0


def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation"]["head_camera"]["rgb"],
        "left_wrist_image": obs["observation"]["left_camera"]["rgb"],
        "right_wrist_image": obs["observation"]["right_camera"]["rgb"],
        "state": obs["joint_action"]["vector"],
        "instruction": obs["language"],
    }


class Model:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

        # CL-LoRA detection
        cl_config_path = os.path.join(cfg.pretrained_checkpoint, "cl_lora_config.json")
        is_cl = os.path.exists(cl_config_path)
        cl_cfg = {}
        if is_cl:
            with open(cl_config_path, "r") as f:
                cl_cfg = json.load(f)
            from cl_lora import inject_cl_lora_into_model, load_task_bank, inject_cl_lora_into_action_head
            self._inj = inject_cl_lora_into_model
            self._inj_ah = inject_cl_lora_into_action_head
            self._lb = load_task_bank

        # Load base VLA (from base model if CL-LoRA, without FiLM first)
        base_cfg = InferenceConfig(
            pretrained_checkpoint="/root/autodl-tmp/models/openvla-7b" if is_cl else cfg.pretrained_checkpoint,
            use_l1_regression=cfg.use_l1_regression,
            use_diffusion=cfg.use_diffusion,
            use_film=False if is_cl else cfg.use_film,  # FiLM applied after CL-LoRA injection
            use_proprio=cfg.use_proprio,
            num_images_in_input=cfg.num_images_in_input,
            unnorm_key=cfg.unnorm_key,
            lora_rank=cl_cfg.get("lora_rank", 32) if is_cl else cfg.lora_rank,
        )
        self.vla = get_vla(base_cfg)

        # CL-LoRA injection into backbone
        if is_cl:
            r = cl_cfg.get("lora_rank", 32); s = cl_cfg.get("shared_split_ratio", 0.5)
            fst = cl_cfg.get("first_lora_layer", 0)
            print(f"[CL-LoRA] inject rank={r} shared={s} first_lora={fst}")
            self.vla = self._inj(
                self.vla, rank=r, alpha=float(r),
                shared_split_ratio=s, first_lora_layer=fst,
                orthogonal_init=cl_cfg.get("orthogonal_init", True),
                freeze_a=cl_cfg.get("freeze_a", True),
                use_block_scale=cl_cfg.get("use_block_scale", True),
            )
            adp = os.path.join(cfg.pretrained_checkpoint, "cl_lora_adapter.pt")
            if os.path.exists(adp):
                sd = torch.load(adp, map_location="cpu", weights_only=True)
                self.vla.load_state_dict(sd, strict=False)
                print("[CL-LoRA] loaded adapter")

            # Apply FiLM after CL-LoRA injection and load vision_backbone from checkpoint
            if cfg.use_film:
                from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
                self.vla.vision_backbone = FiLMedPrismaticVisionBackbone(
                    vision_backbone=self.vla.vision_backbone,
                    llm_dim=self.vla.llm_dim,
                ).to(dtype=torch.bfloat16)
                vb_pattern = os.path.join(cfg.pretrained_checkpoint, "vision_backbone--*_checkpoint.pt")
                vb_files = sorted(glob.glob(vb_pattern))
                if vb_files:
                    vb_sd = torch.load(vb_files[-1], map_location="cpu", weights_only=True)
                    self.vla.vision_backbone.to("cuda")
                    self.vla.vision_backbone.load_state_dict(vb_sd, strict=False)
                    print("[CL-LoRA] loaded vision_backbone (FiLM)")
            self.vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

            # Load dataset statistics (needed for action unnormalization during inference)
            ds_file = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
            if os.path.exists(ds_file):
                from prismatic.training.train_utils import load_dataset_statistics
                self.vla.norm_stats = load_dataset_statistics(ds_file)
                print("[CL-LoRA] loaded dataset statistics")

        # Processor: use base model path (not checkpoint) for CL-LoRA
        proc_cfg = InferenceConfig(
            pretrained_checkpoint="/root/autodl-tmp/models/openvla-7b" if is_cl else cfg.pretrained_checkpoint,
            use_l1_regression=cfg.use_l1_regression,
            use_film=False, num_images_in_input=cfg.num_images_in_input,
            unnorm_key=cfg.unnorm_key,
        )
        self.processor = get_processor(proc_cfg)

        # Action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            if is_cl and fst > 0:
                # Action head needs CL-LoRA injection
                ah_cfg = InferenceConfig(
                    pretrained_checkpoint=cfg.pretrained_checkpoint,
                    use_l1_regression=cfg.use_l1_regression,
                    use_film=cfg.use_film, num_images_in_input=cfg.num_images_in_input,
                    unnorm_key=cfg.unnorm_key,
                )
                raw_ah = get_action_head(ah_cfg, self.vla.llm_dim)
                self.action_head = self._inj_ah(
                    raw_ah, rank=r, alpha=float(r),
                    orthogonal_init=cl_cfg.get("orthogonal_init", True),
                    freeze_a=cl_cfg.get("freeze_a", True),
                    use_block_scale=cl_cfg.get("use_block_scale", True),
                )
                # Reload weights
                pattern = os.path.join(cfg.pretrained_checkpoint, "action_head--*_checkpoint.pt")
                files = sorted(glob.glob(pattern))
                if files:
                    state = torch.load(files[-1], map_location="cpu", weights_only=True)
                    self.action_head.load_state_dict(state, strict=False)
            else:
                ah_cfg = InferenceConfig(
                    pretrained_checkpoint=cfg.pretrained_checkpoint,
                    use_l1_regression=cfg.use_l1_regression,
                    use_film=cfg.use_film, num_images_in_input=cfg.num_images_in_input,
                    unnorm_key=cfg.unnorm_key,
                )
                self.action_head = get_action_head(ah_cfg, self.vla.llm_dim)

        # Proprio projector (for CL-LoRA: init directly, no checkpoint file exists)
        self.proprio_projector = None
        if cfg.use_proprio:
            if is_cl:
                from prismatic.models.projectors import ProprioProjector
                self.proprio_projector = ProprioProjector(
                    llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
                ).to(dtype=torch.bfloat16).to("cuda")
            else:
                pp_cfg = InferenceConfig(
                    pretrained_checkpoint=cfg.pretrained_checkpoint,
                    use_l1_regression=cfg.use_l1_regression,
                    use_film=False, num_images_in_input=cfg.num_images_in_input,
                    unnorm_key=cfg.unnorm_key,
                )
                self.proprio_projector = get_proprio_projector(pp_cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Task bank
        if is_cl and cfg.eval_task_id > 0:
            bp = os.path.join(cfg.pretrained_checkpoint, f"task_{cfg.eval_task_id}_bank.pt")
            if os.path.exists(bp):
                self._lb(self.vla, self.action_head, bp)
                print(f"[CL-LoRA] loaded task {cfg.eval_task_id} bank")

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        actions = get_vla_action(
            cfg=self.cfg, vla=self.vla, processor=self.processor,
            obs=obs, task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions


def get_model(usr_args: dict):
    config_args = {
        "pretrained_checkpoint": usr_args["checkpoint_path"],
        "use_l1_regression": usr_args.get("use_l1_regression", True),
        "use_diffusion": usr_args.get("use_diffusion", False),
        "use_film": usr_args.get("use_film", True),
        "use_proprio": usr_args.get("use_proprio", True),
        "load_in_8bit": usr_args.get("load_in_8bit", False),
        "load_in_4bit": usr_args.get("load_in_4bit", False),
        "num_images_in_input": usr_args.get("num_images_in_input", 3),
        "center_crop": usr_args.get("center_crop", True),
        "unnorm_key": usr_args["unnorm_key"],
        "num_open_loop_steps": usr_args.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
        "lora_rank": usr_args.get("lora_rank", 32),
        "eval_task_id": usr_args.get("eval_task_id", 0),
    }
    return Model(InferenceConfig(**config_args))


def reset_model(model=None):
    pass


def eval(TASK_ENV, model: Model, observation: dict):
    observation["language"] = TASK_ENV.get_instruction()
    actions = model.get_action(observation)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
