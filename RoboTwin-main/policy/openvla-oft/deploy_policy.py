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

# vla-scripts 目录（含 cl_lora.py）相对本文件定位，兼容不同服务器；找不到时回退旧路径
_vla_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "openvla-oft", "vla-scripts"))
if not os.path.isdir(_vla_scripts_dir):
    _vla_scripts_dir = "/root/autodl-tmp/openvla-oft/Cl-Lora-on-openvla/openvla-oft/vla-scripts"
sys.path.insert(0, _vla_scripts_dir)

# 基座模型路径可用环境变量覆盖（新服务器 export OPENVLA_BASE_PATH=/path/to/openvla-7b）
BASE_MODEL_PATH = os.environ.get("OPENVLA_BASE_PATH", "/root/autodl-tmp/models/openvla-7b")


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
            from cl_lora import inject_cl_lora_into_model, load_task_bank
            self._inj = inject_cl_lora_into_model
            self._lb = load_task_bank

        # Load base VLA (from base model if CL-LoRA, without FiLM first)
        base_cfg = InferenceConfig(
            pretrained_checkpoint=BASE_MODEL_PATH if is_cl else cfg.pretrained_checkpoint,
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
                with open(ds_file, "r") as f:
                    self.vla.norm_stats = json.load(f)
                print("[CL-LoRA] loaded dataset statistics")

        # Processor: use base model path (not checkpoint) for CL-LoRA
        proc_cfg = InferenceConfig(
            pretrained_checkpoint=BASE_MODEL_PATH if is_cl else cfg.pretrained_checkpoint,
            use_l1_regression=cfg.use_l1_regression,
            use_film=False, num_images_in_input=cfg.num_images_in_input,
            unnorm_key=cfg.unnorm_key,
        )
        self.processor = get_processor(proc_cfg)

        # Action head (get_action_head already handles CL-LoRA injection if first_lora_layer>0)
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            ah_cfg = InferenceConfig(
                pretrained_checkpoint=cfg.pretrained_checkpoint,
                use_l1_regression=cfg.use_l1_regression,
                use_film=cfg.use_film, num_images_in_input=cfg.num_images_in_input,
                unnorm_key=cfg.unnorm_key,
            )
            self.action_head = get_action_head(ah_cfg, self.vla.llm_dim)

        # Proprio projector (load from checkpoint if available)
        self.proprio_projector = None
        if cfg.use_proprio:
            from prismatic.models.projectors import ProprioProjector
            self.proprio_projector = ProprioProjector(
                llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
            ).to(dtype=torch.bfloat16).to("cuda")
            # Load trained weights from CL-LoRA checkpoint
            pp_pattern = os.path.join(cfg.pretrained_checkpoint, "proprio_projector--*_checkpoint.pt")
            pp_files = sorted(glob.glob(pp_pattern))
            if pp_files:
                pp_sd = torch.load(pp_files[-1], map_location="cpu", weights_only=True)
                self.proprio_projector.load_state_dict(pp_sd)
                print("[CL-LoRA] loaded proprio_projector")

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


_FIXED_PROMPTS = {
    "handover_mic": "Pick up the handheld microphone and hand it over",
    "grab_roller": "Grab the smooth wooden roller with both arms",
    "stack_bowls_two": "Stack the small smooth brown-rimmed bowl directly over the smooth bowl with glossy finish",
    "open_laptop": "Raise the lid of the rectangular laptop with hinge",
}

def eval(TASK_ENV, model: Model, observation: dict):
    # Use fixed prompt matching training data (not random eval instructions)
    task_name = TASK_ENV.task_name if hasattr(TASK_ENV, 'task_name') else ""
    instruction = _FIXED_PROMPTS.get(task_name, TASK_ENV.get_instruction())
    observation["language"] = instruction
    actions = model.get_action(observation)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
