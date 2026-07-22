# ManiGuard SmolVLA SFT (lerobot v0.5.1 fork)

Fine-tunes HuggingFace LeRobot **SmolVLA** on the 6 ManiGuard base-task families, the
SmolVLA half of the ManiGuard VLA benchmark (SmolVLA vs pi0.5 vs GR00T on identical
data, cameras, and controller). Self-contained: this repo **is** lerobot v0.5.1 (pinned)
plus a thin `maniguard_sft/` layer, so you clone one public repo, point at the shared
datagen datasets, and push public checkpoints — no ManiGuard checkout needed.

- **Model:** `lerobot/smolvla_base` (SmolVLM2 backbone + flow-matching action expert).
  Tuning = SmolVLA defaults (verified in `SmolVLAConfig`): **`freeze_vision_encoder=True`
  + `train_expert_only=True`** → only the action expert trains, the VLM is frozen (no LoRA).
- **Inputs (benchmark parity):** 2 cameras (`observation.images.top` = the `left`
  overview + `observation.images.wrist`), 8-D joint state/action (padded to 32), absolute
  JointController, 50-step chunk (SmolVLA native).
- **Data:** the SAME model-agnostic datagen LeRobot datasets the openpi/gr00t tracks use —
  `datagen-<fam>-v1-joint-5cam` (H.264 GOP10). A one-time **prepare** step remaps them to
  a 2-cam, standard-keyed copy (below).

## Why pin 0.5.1 (not 0.3.3)

8-GPU training needs `accelerate`, which lerobot added by **0.5.x** — 0.3.3's `lerobot-train`
is single-GPU only. 0.5.1's `scripts/lerobot_train.py` auto-detects distributed mode and
wraps the policy in DDP with `find_unused_parameters=True` (handles the frozen VLM).

## Env

```bash
uv sync --extra smolvla                 # or: pip install -e '.[smolvla]'
pip install accelerate                  # multi-GPU launcher (usually pulled by lerobot)
# bare containers also need FFmpeg on the path (lerobot decodes video via pyav/torchcodec)
export HF_TOKEN=...  WANDB_API_KEY=...
```

## Run (per family)

```bash
# --data-root = the shared read-only root holding IDEAS-Lab-Northwestern/datagen-*-v1-joint-5cam
# (e.g. the openpi track's outputs; NOT re-downloaded).
TAG=-yanZ bash tools/smolvla_sft/run_all.sh \
  --data-root /path/to/shared/lerobot --family clutter
```

Per family: **prepare** a 2-cam copy → **train** ~2 epochs on 8 cards → **push** to
`IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-<fam>-joint-2cam<TAG>`.

- **Steps** = `ceil(frames * 2 / (BATCH * GPUS))` — `--batch_size` is **per-GPU**
  (effective global batch = `BATCH*GPUS`), so the divisor includes `GPUS`.
- Knobs (env): `BATCH` (per-GPU, default 64) · `GPUS` (8) · `WORKERS` (8/gpu) · `TAG`.
  On an H200, raise `BATCH` to fill VRAM (max card util), then scale LR via
  `run_sft.sh ... -- --optimizer.lr <val>`. Smoke ~100 steps first to read util + VRAM.

## The prepare step (2-cam physical copy — required)

`lerobot-train` classifies features by on-disk standard key prefix and its load-time
`--rename_map` renames only **observations, not the action key** (lerobot 0.5.1
`processor/rename_processor.py`). So `prepare_dataset.py` makes a v2.1→v2.1 **key remap**
of the read-only 5-cam source: `image_<cam>`→`observation.images.top`,
`wrist_image`→`observation.images.wrist`, `state`→`observation.state`, `actions`→`action`;
drops the 3 unused overviews + `actions_commanded`. Videos are copied as-is (no transcode);
state/action stats are unchanged (only the keys move). Source stays untouched.

## ⚠️ Verify on the SFT box before the first real run

The scripts are wired from source-verified 0.5.1 internals, but confirm end-to-end on one
family:
1. **prepare** clutter → `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('id', root='<prepped>')"` loads without error; `meta/info.json` features are the 4 standard keys.
2. **smoke** ~100 steps (small `--steps`) on 8 cards → all cards busy, VRAM fits, loss descends.
3. Check `lerobot-train --help` matches run_sft.sh's flags for your exact install.
