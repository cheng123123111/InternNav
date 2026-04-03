#!/usr/bin/env bash
set -euo pipefail

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
