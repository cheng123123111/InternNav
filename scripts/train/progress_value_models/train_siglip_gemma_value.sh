#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/root/InternNav-boundary-pilot"}
cd "$ROOT_DIR"

TORCHRUN=${TORCHRUN:-"/root/.conda/envs/internvln39/bin/torchrun"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-23845}
DATA=${DATA:-"/root/dualvln_gated_boundary_aux_v24_20260720/train_joint_v22_gated_boundary_aux_blocks.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"/root/dualvln_siglip_gemma_value_v25_20260720/checkpoints/full"}
MAX_STEPS=${MAX_STEPS:-811}
MAX_SAMPLES=${MAX_SAMPLES:-0}

export HF_HOME=${HF_HOME:-"/vepfs-C区/qiancheng/hf-cache"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"/vepfs-C区/qiancheng/hf-cache/hub"}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  scripts/train/progress_value_models/train_siglip_gemma_value.py \
  --data "$DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-frames 8 \
  --visual-grid 4 \
  --value-bins 201 \
  --batch-size 4 \
  --block-size 4 \
  --max-samples "$MAX_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --learning-rate 1e-5 \
  --head-learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --seed 83 \
  --train-language
