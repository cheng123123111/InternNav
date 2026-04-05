#!/usr/bin/env bash
set -euo pipefail

# Force Habitat-Sim to use the host NVIDIA GL/EGL stack instead of conda's
# libEGL/libGLdispatch copies. Without this, headless renderer initialization
# can fail with "cannot retrieve OpenGL version" on server machines.
export LD_PRELOAD="/lib/x86_64-linux-gnu/libEGL.so.1:/lib/x86_64-linux-gnu/libOpenGL.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0${LD_PRELOAD:+:${LD_PRELOAD}}"

CONFIG="scripts/eval/configs/habitat_dual_system_cfg_local.py"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-2333}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CONFIG_BASENAME=$(basename "$CONFIG" .py)
LOG_DIR="logs/${CONFIG_BASENAME}"
mkdir -p "$LOG_DIR"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  scripts/eval/eval.py \
    --config "$CONFIG" \
  2>&1 | tee "$LOG_DIR/torchrun.log"
