#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/code/InternNav"
TRAJ_ROOT_BASE="/dataset-vln/vln_ce/traj_data"
OUT_BASE="/dataset-vln/InternNav/stop_data/full_history5_manifest_only"

DATASETS=(
  "envdrop"
  "r2r"
  "r2r_v1-3"
  "rxr"
)

mkdir -p "${OUT_BASE}"

cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  traj_root="${TRAJ_ROOT_BASE}/${dataset}"
  out_dir="${OUT_BASE}/${dataset}"
  manifest_path="${out_dir}/${dataset}_stop_manifest_history5_full.jsonl"
  enriched_path="${out_dir}/${dataset}_stop_manifest_history5_full_elements.jsonl"
  log_path="${out_dir}/build_and_enrich.log"

  mkdir -p "${out_dir}"

  {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] dataset=${dataset} step=build start"
    python scripts/data_collect/build_stop_manifest_from_traj_tar.py \
      --traj-root "${traj_root}" \
      --output "${manifest_path}" \
      --negative-offsets 8,16 \
      --history-frames 5 \
      --manifest-only

    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] dataset=${dataset} step=enrich start"
    python scripts/data_collect/enrich_stop_manifest_with_elements.py \
      --input "${manifest_path}" \
      --output "${enriched_path}" \
      --max-elements 4

    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] dataset=${dataset} step=done"
  } 2>&1 | tee "${log_path}"
done
