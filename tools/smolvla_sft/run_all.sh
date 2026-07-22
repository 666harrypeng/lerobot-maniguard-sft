#!/usr/bin/env bash
# End-to-end SmolVLA SFT over the 6 ManiGuard-Bench families on the finalized datagen
# v1 datasets. Run one family (--family <fam>) or all serially (--all). Per family:
#   1. locate the model-agnostic 5-cam LeRobot dataset in the SHARED read-only root
#      (reused across the openpi / gr00t / smolvla tracks -- NOT re-downloaded);
#   2. build a SmolVLA-ready 2-cam, standard-keyed copy (prepare_dataset.py);
#   3. train ~2 epochs on GPUS cards (accelerate DDP) via run_sft.sh;
#   4. push the checkpoint + a model card to HF.
#
# Self-contained: prepare_dataset.py imports maniguard_sft from THIS repo root (it puts
# the root on sys.path), so no separate package install is needed.
#
# Compute knobs via env: BATCH (per-GPU) GPUS WORKERS TAG WANDB_PROJECT. Steps derive
# from each dataset's frame count as ceil(frames * EPOCHS / (BATCH * GPUS)) -- the
# effective global batch is BATCH*GPUS, so the divisor includes GPUS.
#
# Usage:
#   export HF_TOKEN=...  WANDB_API_KEY=...
#   TAG=-yanZ bash tools/smolvla_sft/run_all.sh --data-root <shared_lerobot_root> --family clutter
#   TAG=-yanZ bash tools/smolvla_sft/run_all.sh --data-root <root> --all
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

ORG="IDEAS-Lab-Northwestern"
EPOCHS=2
BATCH="${BATCH:-64}"          # PER-GPU batch (effective = BATCH*GPUS)
GPUS="${GPUS:-8}"
WORKERS="${WORKERS:-8}"       # per-GPU dataloader workers
TAG="${TAG:-}"                # e.g. TAG=-yanZ -> HF model repo suffix
RUN_ROOT="$REPO_ROOT/outputs/smolvla_sft"

# family -> total frame count (the only per-dataset value; steps derive from it).
declare -A FRAMES=(
  [clutter]=901520  [cabinet]=4172962  [stack]=2652083
  [jar]=946870      [lid]=1055142      [dusty]=1879498
)
# family -> overview view fed to observation.images.top. MUST match the openpi / gr00t
# datagen-v1 SFT (all use image_left) for benchmark parity.
declare -A EXTERNAL_CAM=(
  [clutter]=left  [cabinet]=left  [stack]=left
  [jar]=left      [lid]=left      [dusty]=left
)
ORDER=(clutter cabinet stack jar lid dusty)

# --- args: [--data-root <dir>] [--family <fam> | --all] ---
DATA_ROOT_ARG=""; SELECT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root) [[ -n "${2:-}" ]] || { echo "Usage: $0 --data-root <dir>" >&2; exit 1; }; DATA_ROOT_ARG="$2"; shift 2 ;;
    --family)    [[ -n "${2:-}" ]] || { echo "Usage: $0 --family <fam>" >&2; exit 1; }; SELECT="$2"; shift 2 ;;
    --all)       SELECT="all"; shift ;;
    -h|--help)   sed -n '1,25p' "$0"; exit 0 ;;
    *) echo "Usage: $0 [--data-root <dir>] [--family <fam> | --all]" >&2; exit 1 ;;
  esac
done
[[ -n "$SELECT" ]] || SELECT="all"
# dataset root precedence: --data-root > $MANIGUARD_SFT_DATA_ROOT > error
SFT_DATA_ROOT="${DATA_ROOT_ARG:-${MANIGUARD_SFT_DATA_ROOT:-}}"
[[ -n "$SFT_DATA_ROOT" ]] || { echo "ERROR: pass --data-root <shared LeRobot root holding $ORG/datagen-*-v1-joint-5cam>." >&2; exit 1; }

# --- pre-flight ---
[[ -n "${HF_TOKEN:-}" ]]      || { echo "ERROR: HF_TOKEN unset (base model pull + checkpoint push)." >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "ERROR: WANDB_API_KEY unset (online training logs)." >&2; exit 1; }
command -v lerobot-train >/dev/null 2>&1 || { echo "ERROR: lerobot-train not on PATH -- pip install -e '.[smolvla]'." >&2; exit 1; }

run_family() {
  local fam="$1"
  local frames="${FRAMES[$fam]:-}"
  [[ -n "$frames" ]] || { echo "ERROR: unknown family '$fam' (want: ${ORDER[*]})." >&2; exit 1; }
  local cam="${EXTERNAL_CAM[$fam]:-left}"
  local data_repo="$ORG/datagen-$fam-v1-joint-5cam"
  local prep_repo="$ORG/datagen-$fam-v1-joint-2cam"                       # id stamped into the prepared copy
  local model_repo="$ORG/smolvla-base-datagen-v1-$fam-joint-2cam$TAG"
  local exp_name="smolvla-base_datagen_v1_${fam}_joint_2cam"
  local src="$SFT_DATA_ROOT/$data_repo"
  local prepped="$RUN_ROOT/data/${fam}_smolvla"
  local out="$RUN_ROOT/runs/$fam"
  local steps=$(( (frames * EPOCHS + BATCH * GPUS - 1) / (BATCH * GPUS) ))   # ÷ (BATCH*GPUS): effective global batch
  local save_freq=$(( steps / 5 > 0 ? steps / 5 : 1 ))

  echo "==================== $fam ===================="
  echo "[run_all] data=$data_repo frames=$frames epochs=$EPOCHS batch/gpu=$BATCH gpus=$GPUS -> effective=$((BATCH*GPUS)) steps=$steps (cam=$cam)"

  # 1. locate the shared read-only 5-cam dataset (must already be present; not pulled here)
  [[ -f "$src/meta/info.json" ]] || { echo "ERROR: dataset not found at $src (expected the datagen 5-cam LeRobot dir)." >&2; exit 1; }
  echo "[run_all] dataset present (read-only source): $src"

  # 2. SmolVLA-ready 2-cam standard-keyed copy (idempotent; skips if already complete)
  python tools/smolvla_sft/prepare_dataset.py \
    --src "$src" --out "$prepped" --repo-id "$prep_repo" --external-cam "$cam"

  # 3. train ~EPOCHS epochs, GPUS-card accelerate DDP (online wandb)
  bash tools/smolvla_sft/run_sft.sh \
    --dataset "$prepped" --repo-id "$prep_repo" --output "$out" \
    --steps "$steps" --batch "$BATCH" --gpus "$GPUS" --workers "$WORKERS" \
    --save-freq "$save_freq" --exp-name "$exp_name"

  # 4. push. LeRobot writes <out>/checkpoints/<step>/pretrained_model + a `last` symlink.
  local ckpt="$out/checkpoints/last/pretrained_model"
  if [[ ! -d "$ckpt" ]]; then
    ckpt="$(ls -dt "$out"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$ckpt" && -d "$ckpt" ]] || { echo "ERROR: no pretrained_model checkpoint under $out/checkpoints." >&2; exit 1; }
  python tools/smolvla_sft/push_to_hf.py \
    --ckpt "$ckpt" --repo "$model_repo" --title "${fam^}" --task "$fam" \
    --data-repo "$data_repo" --frames "$frames" --epochs "$EPOCHS" \
    --steps "$steps" --batch "$((BATCH*GPUS))" --external-cam "$cam"
  echo "[run_all] $fam DONE -> $model_repo"
}

if [ "$SELECT" = "all" ]; then fams=("${ORDER[@]}"); else fams=("$SELECT"); fi
for fam in "${fams[@]}"; do run_family "$fam"; done
