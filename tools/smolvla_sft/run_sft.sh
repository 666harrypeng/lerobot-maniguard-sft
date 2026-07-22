#!/usr/bin/env bash
# Fine-tune SmolVLA on a ManiGuard 2-cam joint LeRobot dataset, MULTI-GPU.
#
# Wraps LeRobot's native `lerobot-train` (this fork pins lerobot v0.5.1). SmolVLA is
# LeRobot-native: the policy's features come straight from the dataset's standard
# `observation.*` / `action` keys, so the dataset must already be a 2-cam,
# standard-keyed copy -- run tools/smolvla_sft/prepare_dataset.py first.
#
# Recipe = SmolVLA defaults (verified in lerobot 0.5.1 SmolVLAConfig):
#   freeze_vision_encoder=True + train_expert_only=True  -> only the action expert
#   trains, the SmolVLM2 backbone is frozen (no LoRA; same "freeze VLM" strategy as
#   the GR00T path). Warm-start from lerobot/smolvla_base.
#
# --- 8-GPU config (verified in lerobot 0.5.1 scripts/lerobot_train.py) --------------
#   * launch with `accelerate launch --multi_gpu --num_processes <GPUS>`; lerobot_train
#     auto-detects distributed mode and wraps the policy in DDP with
#     find_unused_parameters=True (handles the frozen VLM automatically, line 178).
#   * `--batch_size` is PER-PROCESS (per GPU); effective global batch = batch x GPUS
#     (lerobot_train.py line 351-353). So run_all's step count divides by GPUS.
#   * SmolVLA has a training preset (optimizer + cosine schedule). For a larger
#     effective batch, scale LR up (override via `-- --optimizer.lr <val>`).
#
# Prereqs (once per shell, in the lerobot env):
#   export HF_TOKEN=...  WANDB_API_KEY=...     # base-model pull + online logs
#   # bare containers also need FFmpeg on the path for lerobot's video decode
#
# Usage:
#   bash tools/smolvla_sft/run_sft.sh --dataset <prepared_dir> --repo-id <id> \
#        --output <ckpt_dir> --steps <N> [--batch 64] [--gpus 8] [--workers 8] \
#        [--exp-name clutter] [--save-freq 5000] [-- <extra lerobot-train args>...]
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-base-joint-2cam}"

DATASET=""; REPO_ID=""; OUTPUT=""; STEPS=20000; BATCH=64; GPUS=8; WORKERS=8; SAVE_FREQ=5000; EXP_NAME=""
EXTRA=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset)   DATASET="$2"; shift 2 ;;
        --repo-id)   REPO_ID="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --steps)     STEPS="$2"; shift 2 ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --gpus)      GPUS="$2"; shift 2 ;;
        --workers)   WORKERS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --exp-name)  EXP_NAME="$2"; shift 2 ;;
        --)          shift; EXTRA=("$@"); break ;;
        -h|--help)   sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -n "$DATASET" ] || { echo "Missing --dataset" >&2; exit 1; }
[ -n "$REPO_ID" ] || { echo "Missing --repo-id" >&2; exit 1; }
[ -n "$OUTPUT" ]  || { echo "Missing --output" >&2; exit 1; }
DATASET="$(realpath -m "$DATASET")"; OUTPUT="$(realpath -m "$OUTPUT")"
[ -f "$DATASET/meta/info.json" ] || { echo "ERROR: $DATASET/meta/info.json missing -- run prepare_dataset.py first." >&2; exit 1; }
EXP_NAME="${EXP_NAME:-$(basename "$OUTPUT")}"
[ -n "${WANDB_API_KEY:-}" ] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ]      || { echo "ERROR: HF_TOKEN unset (base-model $BASE_MODEL pull)." >&2; exit 1; }
command -v lerobot-train >/dev/null 2>&1 || { echo "ERROR: lerobot-train not on PATH -- pip install -e '.[smolvla]'." >&2; exit 1; }
# Do NOT pre-create $OUTPUT: lerobot-train's validate() refuses a pre-existing output_dir
# (unless --resume). Only ensure its PARENT exists (the symlinked runs/ dir); lerobot
# creates the fresh output_dir itself.
mkdir -p "$(dirname "$OUTPUT")"

# cap per-worker math threads so WORKERS x GPUS dataloader procs don't oversubscribe CPU.
: "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS

# GPUS>1 -> accelerate DDP (batch is per-GPU); GPUS==1 -> plain single-process.
# MASTER_PORT (env, default 29500) -> accelerate rendezvous port. Set a DIFFERENT value
# (e.g. 29555) when running CONCURRENTLY with another distributed job (e.g. a GR00T run
# on the same box uses 29500) to avoid "address already in use".
if [ "$GPUS" -gt 1 ]; then
    LAUNCH=(accelerate launch --multi_gpu --num_processes "$GPUS" --main_process_port "${MASTER_PORT:-29500}" --mixed_precision bf16 "$(command -v lerobot-train)")
else
    LAUNCH=(lerobot-train)
fi

echo "[run_sft] exp=$EXP_NAME gpus=$GPUS batch/gpu=$BATCH -> effective=$((BATCH*GPUS)) steps=$STEPS workers=$WORKERS save_freq=$SAVE_FREQ"
echo "[run_sft] dataset=$DATASET (repo_id=$REPO_ID)  base=$BASE_MODEL  wandb=$WANDB_PROJECT"

"${LAUNCH[@]}" \
    --policy.path="$BASE_MODEL" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --num_workers="$WORKERS" \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$OUTPUT" \
    --job_name="$EXP_NAME" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[run_sft] training done -> $OUTPUT"
