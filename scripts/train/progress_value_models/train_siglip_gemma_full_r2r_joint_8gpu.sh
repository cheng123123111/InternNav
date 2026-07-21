#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/root/InternNav-boundary-pilot"}
cd "$ROOT_DIR"

TORCHRUN=${TORCHRUN:-"/root/.conda/envs/internvln39/bin/torchrun"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-23855}
DATA=${DATA:-"/vepfs-C区/qiancheng/dualvln_siglip_gemma_full_r2r_joint_v2_20260721/train_full_r2r_failure_x4_blocks.jsonl"}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"/vepfs-C区/qiancheng/dualvln_siglip_gemma_value_v25_20260720/checkpoints/full_epoch1/value_model.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"/vepfs-C区/qiancheng/dualvln_siglip_gemma_full_r2r_joint_v2_20260721/checkpoints/full_finetune_epoch1"}

BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}

export HF_HOME=${HF_HOME:-"/vepfs-C区/qiancheng/hf-cache"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"/vepfs-C区/qiancheng/hf-cache/hub"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"

"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  scripts/train/progress_value_models/train_siglip_gemma_value.py \
  --data "$DATA" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --max-frames 8 \
  --visual-grid 4 \
  --value-bins 201 \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --block-size 4 \
  --max-samples "$MAX_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --vision-learning-rate 1e-6 \
  --learning-rate 5e-6 \
  --head-learning-rate 5e-5 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --num-workers 4 \
  --seed 83 \
  --train-language \
  --train-vision \
  --vision-gradient-checkpointing \
  --fp32-master-weights
