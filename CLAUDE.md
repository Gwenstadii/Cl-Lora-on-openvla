# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a continual-learning fine-tuning system for OpenVLA-7B, a Vision-Language-Action robot model. It implements **CL-LoRA** (Continual Learning LoRA with orthogonal init, frozen A matrices, and block-scale gating) plus **prototype replay** to mitigate catastrophic forgetting across sequential LIBERO manipulation tasks.

## Commands

```bash
# Install (editable)
cd openvla-oft && pip install -e .

# Flash Attention (install AFTER editable install)
pip install "flash-attn==2.5.5" --no-build-isolation

# Standard LoRA fine-tuning (single GPU)
cd openvla-oft
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ../LOGS/standard_lora \
  --lora_rank 16 --batch_size 8 --learning_rate 5e-4 \
  --use_l1_regression --num_steps 4000

# CL-LoRA fine-tuning
torchrun --standalone --nproc_per_node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ../modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ../LOGS/cl_lora \
  --use_cl_lora --shared_depth 16 --orthogonal_init --freeze_a \
  --use_block_scale --clip_weight 1.0 --lora_rank 32 --num_steps 4000

# LIBERO evaluation
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla --pretrained_checkpoint ../LOGS/cl_lora/checkpoint-4000 \
  --task_suite_name libero_spatial --use_cl_lora

# Build prototype replay buffer
python vla-scripts/build_replay_buffer_openvla.py

# Merge LoRA weights offline
python vla-scripts/merge_lora_weights_and_save.py

# Inspect RLDS dataset structure
python vla-scripts/check_rlds_keys.py
```

## Architecture

The model pipeline is: **DinoSigLIP (fused ViT vision encoder) → MLP projector → LLaMa-2 7B (32 layers) → action head (L1 regression or DDIM diffusion)**. FiLM conditioning modulates vision features with language embeddings.

### Package: `prismatic/` (installed as `prismatic`)

OpenVLA's core library, structured as a proper Python package:

| Subpackage | Purpose |
|---|---|
| `prismatic.conf` | Model configs (`models.py`, `vla.py`) |
| `prismatic.extern.hf` | HuggingFace wrappers: `configuration_prismatic.py`, `modeling_prismatic.py`, `processing_prismatic.py` |
| `prismatic.models.backbones.vision` | ViT backbones (SigLIP, DINOv2, DinoSigLIP fusion) |
| `prismatic.models.backbones.llm` | LLM backbones (LLaMa-2, Mistral, Phi) with prompt builders |
| `prismatic.models.vlms.prismatic` | `PrismaticVLM` — vision→projector→LLM forward pass |
| `prismatic.models.vlas.openvla` | `OpenVLA` class — adds action tokenization + prediction on top of PrismaticVLM |
| `prismatic.models.action_heads` | `L1RegressionActionHead` and `DiffusionActionHead` (DDIM) |
| `prismatic.models.film_vit_wrapper` | `FiLMedPrismaticVisionBackbone` — language-conditioned vision modulation |
| `prismatic.models.projectors` | `ProprioProjector`, `NoisyActionProjector` |
| `prismatic.vla.action_tokenizer` | Action discretization → token mapping → continuous reconstruction |
| `prismatic.vla.constants` | Platform constants (action dims, proprio dims per LIBERO/ALOHA/BRIDGE) |
| `prismatic.vla.datasets` | RLDS dataset pipeline (OXE configs, transforms, mixtures) |
| `prismatic.training` | DDP/FSDP strategies, metrics, loss functions |
| `prismatic.util` | Batching, data utils, torch utils |

### Key scripts: `vla-scripts/`

| File | Purpose |
|---|---|
| `finetune.py` | Main training script. `FinetuneConfig` dataclass controls ~50 hyperparams. Supports standard LoRA and CL-LoRA via `--use_cl_lora`. |
| `cl_lora.py` | `CLLoRALinear` module and `inject_cl_lora_into_model()`. Replaces attention (q/k/v/o) and FFN (gate/up/down) linear layers in LlamaDecoderLayer with CL-LoRA variants. |
| `deploy.py` | FastAPI model server exposing `/act` endpoint for ALOHA real-robot inference. |
| `build_replay_buffer_openvla.py` | Offline replay buffer construction: physical trajectory segmentation → vision feature extraction → prototype computation → top-K keyframe selection → `.npz` storage. |
| `replay_dataset.py` | PyTorch Dataset loader for replay `.npz` samples. |

### Source-code directories (three branches)

- **`vla-scripts/`** — primary development target. Contains the current unified CL-LoRA + prototype replay implementation for OpenVLA.
- **`cllora代码/`** — early CL-LoRA experiments with a standalone `dataset.py` (RLDS loading/normalization) and `openvla_utils.py` (model loading/inference helpers). The `dataset.py` here is the reference for RLDS data pipeline details.
- **`90成功率代码/`** — standard LoRA baseline that achieves 90%+ success on LIBERO-Spatial (overfit baseline, no continual learning).
- **`pi0.5代码/`** — reference implementation from which CL-LoRA + prototype replay concepts were ported (JAX/Flax, PI0.5 model). `lora.py` is the canonical reference for orthogonal init / freeze_A / block_scale logic. `train_cl_lora.py` is the reference for teacher-model KD + replay loss combination.

### CL-LoRA design

CL-LoRA replaces `nn.Linear` with `CLLoRALinear` across all Attention (q_proj, k_proj, v_proj, o_proj) and FFN (gate_proj, up_proj, down_proj) layers of the 32 LlamaDecoderLayers:

- **Shared layers** (0 to `shared_depth-1`): LoRA-A is orthogonally initialized then frozen (`requires_grad=False`), protecting old knowledge. LoRA-B is zero-initialized and trainable.
- **Specific layers** (`shared_depth` to 31): Both LoRA-A and LoRA-B are trainable. Each layer has a learnable `block_scale` parameter gated by `1.0 + 0.5 * tanh(w)`.

Forward: `result = Wx + scaling * block_scale * B @ A @ x`

### Prototype replay pipeline

1. **Build buffer** (`build_replay_buffer_openvla.py`): Load old-task RLDS data → segment trajectories by physical motion (translation/rotation/gripper delta thresholds) → merge short segments → extract vision features via frozen vision backbone → compute segment prototypes (normalized mean feature) → select top-K frames closest to prototype → save as `.npz`.
2. **Train with replay**: At each training step, mix current-task batches with replay batches from the buffer. Total loss = `L_task + λ_kd * L_KD + λ_replay * L_replay`. Teacher model is a snapshot of only the trainable CL-LoRA parameters (not full model).
3. **Continual learning loop**: Train task A → build replay buffer for task A → train task B with task A's buffer → build buffer for task B → train task C with buffers A+B, etc.

### Evaluation

- `experiments/robot/libero/run_libero_eval.py` — LIBERO simulation evaluation across task suites (spatial/object/goal/10/90). Records success rates and video.
- `experiments/robot/aloha/run_aloha_eval.py` — Real-robot ALOHA evaluation with client-server architecture (model runs on GPU server, robot client calls `/act`).
- `experiments/robot/openvla_utils.py` — Shared evaluation utilities (model loading, inference pipeline).

### Datasets: `modified_libero_rlds/`

Pre-processed LIBERO datasets in RLDS format. Each subdirectory (`libero_spatial_no_noops/`, `libero_object_no_noops/`, etc.) contains RLDS metadata. The `_no_noops` variants filter out no-op actions. Actual `.tfrecord` data is gitignored.

## Important conventions

- **Transformers fork**: This project depends on `moojink/transformers-openvla-oft.git` (a custom transformers fork with bidirectional attention for parallel decoding). Do not replace with upstream transformers.
- **No HF token needed**: `openvla/openvla-7b` is a public model on HuggingFace Hub.
- **Multi-GPU**: Use `torchrun --nproc_per_node N` for DDP/FSDP. The training script auto-detects distributed state via `accelerate.PartialState`.
- **All training config** lives in the `FinetuneConfig` dataclass in `finetune.py`. CL-LoRA specific flags (`--use_cl_lora`, `--shared_depth`, `--orthogonal_init`, `--freeze_a`, `--use_block_scale`, `--clip_weight`) are parsed from CLI via `draccus`.
- **Checkpoint format**: Training saves per-module checkpoints (`<module_name>--<step>_checkpoint.pt`). DDP `module.` prefix is stripped during loading.
- **Gitignore**: Model weights (`*.safetensors`, `*.pt` with training checkpoints), datasets (`*.tfrecord`), logs, and conda envs are excluded from version control.
