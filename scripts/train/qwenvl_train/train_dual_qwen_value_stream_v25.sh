#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/root/InternNav-boundary-pilot"}
cd "$ROOT_DIR"

TORCHRUN=${TORCHRUN:-"/root/.conda/envs/internvln39/bin/torchrun"}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-23835}
VALUE_STREAM_DEPTH=${VALUE_STREAM_DEPTH:-28}
VALUE_STREAM_DIM=${VALUE_STREAM_DIM:-512}
VALUE_STREAM_FFN_DIM=${VALUE_STREAM_FFN_DIM:-2048}
VALUE_STREAM_HEADS=${VALUE_STREAM_HEADS:-8}
RUN_NAME=${RUN_NAME:-"InternVLA-N1-QwenValueStreamV25-D${VALUE_STREAM_DEPTH}"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/root/dualvln_qwen_value_stream_v25_20260720"}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/checkpoints/$RUN_NAME"}
SYSTEM2_CKPT=${SYSTEM2_CKPT:-"/vepfs-B区/vlnce/InternNav/checkpoints/InternVLA-N1-DualVLN"}
PROGRESS_JSONL_USE=${PROGRESS_JSONL_USE:-"/root/dualvln_gated_boundary_aux_v24_20260720/train_joint_v22_gated_boundary_aux_blocks.jsonl"}

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_STEPS=${MAX_STEPS:-811}
MAX_SAMPLES=${MAX_SAMPLES:-0}
LEARNING_RATE=${LEARNING_RATE:-8e-5}
SEED=${SEED:-83}
PROGRESS_AUX_TEMPORAL_TOKENS=${PROGRESS_AUX_TEMPORAL_TOKENS:-False}

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-"/vepfs-C区/qiancheng/hf-cache"}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"/vepfs-C区/qiancheng/hf-cache/hub"}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}

mkdir -p "$OUTPUT_DIR"

"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  internnav/trainer/internvla_n1_trainer.py \
  --model_name_or_path "$SYSTEM2_CKPT" \
  --vln_dataset_use "" \
  --progress_jsonl_use "$PROGRESS_JSONL_USE" \
  --progress_jsonl_max_frames 8 \
  --progress_jsonl_rebuild_history True \
  --progress_jsonl_include_action_history False \
  --progress_jsonl_preserve_order True \
  --progress_jsonl_max_samples "$MAX_SAMPLES" \
  --progress_aux_temporal_tokens "$PROGRESS_AUX_TEMPORAL_TOKENS" \
  --progress_aux_causal_prior False \
  --progress_aux_target_mode completion_success \
  --progress_aux_completion_mode scalar \
  --progress_aux_num_queries 4 \
  --progress_aux_in_qwen_queries False \
  --progress_aux_value_stream True \
  --progress_aux_value_stream_depth "$VALUE_STREAM_DEPTH" \
  --progress_aux_value_stream_dim "$VALUE_STREAM_DIM" \
  --progress_aux_value_stream_ffn_dim "$VALUE_STREAM_FFN_DIM" \
  --progress_aux_value_stream_heads "$VALUE_STREAM_HEADS" \
  --progress_aux_query_specific_readout True \
  --progress_aux_gated_aux_fusion False \
  --progress_aux_qwen_backprop_fp32 False \
  --enable_progress_aux True \
  --progress_aux_loss_weight 1.0 \
  --progress_aux_query_loss_weight 0.15 \
  --progress_aux_completion_loss_weight 1.0 \
  --progress_aux_success_loss_weight 1.5 \
  --progress_aux_rank_loss_weight 1.0 \
  --progress_aux_rank_margin 0.15 \
  --progress_aux_only_trainable True \
  --progress_aux_save_only True \
  --data_flatten False \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm False \
  --num_history 8 \
  --data_augmentation False \
  --resize_h 384 \
  --resize_w 384 \
  --sample_step 4 \
  --num_future_steps 4 \
  --predict_step_num 32 \
  --pixel_goal_only True \
  --system1 nextdit_async \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir True \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --max_pixels 313600 \
  --min_pixels 3136 \
  --eval_strategy no \
  --save_strategy no \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --model_max_length 8192 \
  --gradient_checkpointing False \
  --bf16 True \
  --dataloader_num_workers 4 \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  --report_to none
