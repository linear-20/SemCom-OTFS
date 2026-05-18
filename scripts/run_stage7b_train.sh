#!/usr/bin/env bash
set -euo pipefail

MODE="pilot"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
OUTPUT_TAG=""
RESUME_CHECKPOINT=""
DRY_RUN=0

CODEBOOK_SIZE=256
BATCH_SIZE=""
STEPS=""
EMBED_DIM=128
NUM_HEADS=4
SELF_ATTN_LAYERS=2
DROPOUT=0.1
USE_PACKED_LOCAL=0
USE_CHANNEL_FEATURES=0
MAX_CHANNEL_PATHS=3
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0

SNR_MIN=20
SNR_MAX=30
CHANNEL_MODE="channel"
NUM_PATHS=3
MAX_DELAY_SAMPLES=3
MAX_DOPPLER_HZ=500
SAMPLE_RATE=15.36e6
FADING="rayleigh"
FIXED_CHANNEL=0
NO_AWGN=0
EVAL_EVERY=""
EVAL_BATCHES=4
SAVE_EVERY=""
SEED=0

usage() {
    cat <<'EOF'
Stage 7B DD Token Perceiver receiver training launcher.

Usage:
  bash scripts/run_stage7b_train.sh [options]

Modes:
  --mode pilot   1000 steps, batch size 4, save every 250 steps
  --mode train   5000 steps, batch size 8, save every 500 steps
  --mode custom  provide your own --steps / --batch-size / output tag

Common options:
  --python-bin PATH
  --device cuda|cpu
  --steps N
  --batch-size N
  --output-tag NAME
  --resume-checkpoint PATH
  --dry-run
  --codebook-size N
  --embed-dim N
  --num-heads N
  --self-attn-layers N
  --use-packed-local
  --use-channel-features
  --max-channel-paths N
  --lr FLOAT
  --snr-min FLOAT
  --snr-max FLOAT
  --channel-mode channel|identity
  --max-delay-samples FLOAT
  --max-doppler-hz FLOAT
  --fading rayleigh|rician|fixed
  --fixed-channel
  --no-awgn
  --eval-every N
  --eval-batches N
  --save-every N
  --seed N

Examples:
  bash scripts/run_stage7b_train.sh --mode pilot
  bash scripts/run_stage7b_train.sh --mode train --batch-size 8
  bash scripts/run_stage7b_train.sh --mode custom --steps 2000 --batch-size 4 --output-tag cb256_bs4_2k_seed0
  bash scripts/run_stage7b_train.sh --mode custom --steps 2000 --resume-checkpoint outputs/stage7b_receiver_train/cb256_bs4_1k_seed0/receiver_checkpoint.pt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --output-tag) OUTPUT_TAG="$2"; shift 2 ;;
        --resume-checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --codebook-size) CODEBOOK_SIZE="$2"; shift 2 ;;
        --embed-dim) EMBED_DIM="$2"; shift 2 ;;
        --num-heads) NUM_HEADS="$2"; shift 2 ;;
        --self-attn-layers) SELF_ATTN_LAYERS="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --use-packed-local) USE_PACKED_LOCAL=1; shift ;;
        --use-channel-features) USE_CHANNEL_FEATURES=1; shift ;;
        --max-channel-paths) MAX_CHANNEL_PATHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
        --snr-min) SNR_MIN="$2"; shift 2 ;;
        --snr-max) SNR_MAX="$2"; shift 2 ;;
        --channel-mode) CHANNEL_MODE="$2"; shift 2 ;;
        --num-paths) NUM_PATHS="$2"; shift 2 ;;
        --max-delay-samples) MAX_DELAY_SAMPLES="$2"; shift 2 ;;
        --max-doppler-hz) MAX_DOPPLER_HZ="$2"; shift 2 ;;
        --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
        --fading) FADING="$2"; shift 2 ;;
        --fixed-channel) FIXED_CHANNEL=1; shift ;;
        --no-awgn) NO_AWGN=1; shift ;;
        --eval-every) EVAL_EVERY="$2"; shift 2 ;;
        --eval-batches) EVAL_BATCHES="$2"; shift 2 ;;
        --save-every) SAVE_EVERY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    pilot)
        [[ -z "$STEPS" ]] && STEPS=1000
        [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=4
        [[ -z "$EVAL_EVERY" ]] && EVAL_EVERY=100
        [[ -z "$SAVE_EVERY" ]] && SAVE_EVERY=250
        [[ -z "$OUTPUT_TAG" ]] && OUTPUT_TAG="cb${CODEBOOK_SIZE}_bs${BATCH_SIZE}_${STEPS}step_seed${SEED}"
        ;;
    train)
        [[ -z "$STEPS" ]] && STEPS=5000
        [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=8
        [[ -z "$EVAL_EVERY" ]] && EVAL_EVERY=250
        [[ -z "$SAVE_EVERY" ]] && SAVE_EVERY=500
        [[ -z "$OUTPUT_TAG" ]] && OUTPUT_TAG="cb${CODEBOOK_SIZE}_bs${BATCH_SIZE}_${STEPS}step_seed${SEED}"
        ;;
    custom)
        [[ -z "$STEPS" ]] && STEPS=1000
        [[ -z "$BATCH_SIZE" ]] && BATCH_SIZE=4
        [[ -z "$EVAL_EVERY" ]] && EVAL_EVERY=100
        [[ -z "$SAVE_EVERY" ]] && SAVE_EVERY=250
        [[ -z "$OUTPUT_TAG" ]] && OUTPUT_TAG="custom_cb${CODEBOOK_SIZE}_bs${BATCH_SIZE}_${STEPS}step_seed${SEED}"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage7b_receiver_train/${OUTPUT_TAG}"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_dd_token_perceiver_receiver.py"

mkdir -p "$OUTPUT_DIR"

ARGS=(
    "$TRAIN_SCRIPT"
    --output-dir "$OUTPUT_DIR"
    --codebook-size "$CODEBOOK_SIZE"
    --token-shape 16 16
    --symbols-per-token 4
    --dd-shape 32 32
    --cp-len 4
    --batch-size "$BATCH_SIZE"
    --num-steps "$STEPS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --grad-clip "$GRAD_CLIP"
    --embed-dim "$EMBED_DIM"
    --num-heads "$NUM_HEADS"
    --self-attn-layers "$SELF_ATTN_LAYERS"
    --dropout "$DROPOUT"
    --max-channel-paths "$MAX_CHANNEL_PATHS"
    --snr-db-min "$SNR_MIN"
    --snr-db-max "$SNR_MAX"
    --channel-mode "$CHANNEL_MODE"
    --num-paths "$NUM_PATHS"
    --sample-rate "$SAMPLE_RATE"
    --max-delay-samples "$MAX_DELAY_SAMPLES"
    --max-doppler-hz "$MAX_DOPPLER_HZ"
    --fading "$FADING"
    --device "$DEVICE"
    --eval-every "$EVAL_EVERY"
    --eval-batches "$EVAL_BATCHES"
    --save-every "$SAVE_EVERY"
    --seed "$SEED"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$FIXED_CHANNEL" -eq 1 ]]; then
    ARGS+=(--fixed-channel)
fi

if [[ "$NO_AWGN" -eq 1 ]]; then
    ARGS+=(--no-awgn)
fi

if [[ "$USE_PACKED_LOCAL" -eq 1 ]]; then
    ARGS+=(--use-packed-local)
fi

if [[ "$USE_CHANNEL_FEATURES" -eq 1 ]]; then
    ARGS+=(--use-channel-features)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    ARGS+=(--dry-run)
fi

echo "Project root:      $PROJECT_ROOT"
echo "Output dir:        $OUTPUT_DIR"
echo "Mode:              $MODE"
echo "Device:            $DEVICE"
echo "Steps:             $STEPS"
echo "Batch size:        $BATCH_SIZE"
echo "Codebook size:     $CODEBOOK_SIZE"
echo "Model:             embed_dim=$EMBED_DIM heads=$NUM_HEADS self_layers=$SELF_ATTN_LAYERS packed_local=$USE_PACKED_LOCAL channel_features=$USE_CHANNEL_FEATURES max_channel_paths=$MAX_CHANNEL_PATHS"
echo "Channel:           mode=$CHANNEL_MODE paths=$NUM_PATHS delay=$MAX_DELAY_SAMPLES doppler=$MAX_DOPPLER_HZ snr=$SNR_MIN..$SNR_MAX fading=$FADING fixed=$FIXED_CHANNEL no_awgn=$NO_AWGN"
echo "Eval/Save:         every $EVAL_EVERY / $SAVE_EVERY steps, eval_batches=$EVAL_BATCHES"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-<none>}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "${ARGS[@]}"
