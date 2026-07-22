#!/usr/bin/env python
"""Push a SmolVLA SFT checkpoint + a generated model card to a public HF repo.

Uploads a LeRobot ``pretrained_model`` dir — the inference files
``SmolVLAPolicy.from_pretrained`` reads (config.json + model.safetensors, which carry
the baked normalization stats). It EXCLUDES ``train_config.json`` (LeRobot's full
training-pipeline config dump — it embeds absolute server paths ``dataset.root`` /
``output_dir`` and is NOT read at inference) plus the optimizer / scheduler / rng
training state.

Run in the lerobot env (needs HF_TOKEN). Usage:

    python tools/smolvla_sft/push_to_hf.py \
        --ckpt <output>/checkpoints/last/pretrained_model \
        --repo IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-clutter-joint-2cam \
        --title "Clutter-Pickup" --task clutter \
        --data-repo IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam \
        --frames 901520 --epochs 2 --steps 28173 --batch 64 --external-cam left
"""

import argparse

from huggingface_hub import HfApi

# Not needed for inference; excluded to keep the repo clean + leak-free. ★
# train_config.json is LeRobot's full TrainPipelineConfig dump — it embeds absolute
# server paths (dataset.root, output_dir; verified in lerobot 0.5.1 configs/train.py)
# and SmolVLAPolicy.from_pretrained never reads it, so it MUST be dropped.
_IGNORE = [
    "train_config.json",   # leaks dataset.root + output_dir; not read at inference
    "training_state/*",
    "optimizer*",
    "scheduler*",
    "rng_state*",
    "*.tmp",
]

_CARD = """---
license: apache-2.0
base_model: lerobot/smolvla_base
library_name: lerobot
pipeline_tag: robotics
tags: [robotics, vla, smolvla, lerobot, manipulation, maniguard, franka]
---

# SmolVLA - {title} (joint, 2-cam)

HuggingFace LeRobot **SmolVLA** fine-tuned on the ManiGuard **{task}** base task (sim
Franka Panda). Part of the ManiGuard VLA benchmark - SmolVLA vs pi0.5 vs GR00T on the
same task families with identical data, cameras, and controller.

## Model
- **Base:** [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) - SmolVLM2 vision-language backbone + flow-matching action expert
- **Embodiment:** Franka Panda, **8-D joint** state/action (7 arm joints + 1 gripper), padded to SmolVLA's 32-D
- **Cameras (2):** `observation.images.top` (overview = `{external_cam}` view) + `observation.images.wrist` (256x256)
- **Action:** absolute joint targets (fed straight to a JointController at eval); 50-step chunk
- **Tuning:** SmolVLA default - vision encoder **frozen**, train the action expert (**no LoRA**)

## Training
- 8-GPU accelerate DDP, global batch {batch}, {steps} steps (~{epochs} epochs over {frames:,} frames)
- Data: [{data_repo}](https://huggingface.co/datasets/{data_repo}) - the 5-cam datagen dataset, prepared to a 2-cam standard-keyed copy (H.264 videos)

## Usage
Load with `SmolVLAPolicy.from_pretrained("{repo}")` from [LeRobot](https://github.com/huggingface/lerobot). The checkpoint carries the normalization stats.

> WARNING - Convention (must match at eval): joint-space JointController (absolute joint targets) + 2 cameras (`observation.images.top` = the `{external_cam}` overview, `observation.images.wrist`). A mismatched controller, camera set, or overview view silently feeds an out-of-distribution input.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LeRobot pretrained_model dir to upload")
    ap.add_argument("--repo", required=True, help="target HF model repo (org/name)")
    ap.add_argument("--title", required=True, help='card title, e.g. "Clutter-Pickup"')
    ap.add_argument("--task", required=True, help="task slug, e.g. clutter")
    ap.add_argument("--data-repo", required=True, help="source HF dataset repo")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--external-cam", default="left", help="which overview fed observation.images.top")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.ckpt,
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=_IGNORE,
        commit_message=f"SmolVLA SFT on {args.task} ({args.epochs} epochs, {args.steps} steps)",
    )
    card = _CARD.format(
        title=args.title,
        task=args.task,
        data_repo=args.data_repo,
        frames=args.frames,
        epochs=args.epochs,
        steps=args.steps,
        batch=args.batch,
        external_cam=args.external_cam,
        repo=args.repo,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="model card",
    )
    print("PUSHED", args.repo)


if __name__ == "__main__":
    main()
