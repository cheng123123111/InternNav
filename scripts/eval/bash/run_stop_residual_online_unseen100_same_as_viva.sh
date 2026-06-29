#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/root-code/InternNav

PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/internvln39/bin/python}"
OUT_DIR="${OUT_DIR:-/vepfs-C/qiancheng/stop_residual_online_unseen100_same_as_viva_20260629}"
LOG="$OUT_DIR/run.log"

mkdir -p "$OUT_DIR"

export PYTHONPATH=/data/projects/root-code/InternNav:${PYTHONPATH:-}
export LD_PRELOAD="/lib/x86_64-linux-gnu/libEGL.so.1:/lib/x86_64-linux-gnu/libOpenGL.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0${LD_PRELOAD:+:${LD_PRELOAD}}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT:-29658}" \
  -m scripts.eval.eval \
  --config scripts/eval/configs/habitat_dual_system_progress_unseen100_same_as_viva_stopresidual_cfg.py \
  2>&1 | tee "$LOG"
