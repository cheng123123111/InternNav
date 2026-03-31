# VLN-CE Run Notes

This file records the exact commands and local config changes used for:

1. Full `val_unseen` evaluation without video output
2. A 20-episode video-producing run covering all `val_unseen` scenes

## 1. Full Evaluation Run

Goal: run the latest Habitat dual-system VLN-CE evaluation on the full `val_unseen` split with reduced GPU memory pressure.

### Command

```bash
cd /mnt/data/0923_Interndata/InternNav
CONDA_NO_PLUGINS=true conda run -n internvln39 \
python -u scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_tuned_cfg.py
```

### Log and outputs

- Log: `logs/habitat/latest_dual_system_tuned_stdout.log`
- Output dir: `logs/habitat/test_dual_system_tuned`

### Local modifications used by this run

#### A. Habitat config compatibility update

File:

- `scripts/eval/configs/vln_r2r.yaml`

Changes:

- Switched from old Habitat schema:
  - `/habitat/simulator/agents@habitat.simulator.agents.main_agent: rgbd_agent`
- To current installed Habitat schema:
  - `/habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent: rgbd_agent`
- Added task action defaults:
  - `look_up`
  - `look_down`
- Removed obsolete simulator keys not accepted by the local Habitat version:
  - `tilt_angle`
  - `action_space_config`
- Removed per-action `agent_index` fields
- Pointed dataset and scene paths to the local machine:
  - `scenes_dir: /mnt/data/VL-LN-Bench/scene_datasets/`
  - `data_path: /mnt/data/0923_Interndata/vln_ce/raw_data/r2r/{split}/{split}.json.gz`

#### B. Tuned full-run config

File:

- `scripts/eval/configs/habitat_dual_system_tuned_cfg.py`

This config was created from the baseline `scripts/eval/configs/habitat_dual_system_cfg.py`.

Changes relative to the baseline:

- `resize_w: 384 -> 320`
- `resize_h: 384 -> 320`
- `max_new_tokens: 1024 -> 256`
- `output_path: ./logs/habitat/test_dual_system_tuned`
- `save_video: False`
- `vis_debug: False`

Rationale:

- Reduce GPU memory use while preserving the same dual-system model and main inference logic.

## 2. 20-Episode Video Run

Goal: run a small VLN-CE evaluation with the same tuned inference settings, but:

- only 20 episodes
- cover all 11 `val_unseen` scenes
- output videos

### Step 1. Generate the 20-episode subset

Command:

```bash
cd /mnt/data/0923_Interndata/InternNav
python scripts/eval/create_vlnce_subset.py
```

Generated file:

- `/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_cover20.json.gz`

Generator script:

- `scripts/eval/create_vlnce_subset.py`

Behavior:

- Reads `val_unseen.json.gz`
- Selects 20 episodes
- Guarantees coverage across all 11 scenes
- Uses a round-robin scene selection strategy after selecting one episode per scene

### Step 2. Run the 20-episode video config

Command:

```bash
cd /mnt/data/0923_Interndata/InternNav
CONDA_NO_PLUGINS=true conda run -n internvln39 \
python -u scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_video20_cfg.py
```

### Log and outputs

- Log: `logs/habitat/video20_stdout.log`
- Output dir: `logs/habitat/video20`

### Local modifications used by this run

#### A. Video subset Habitat YAML

File:

- `scripts/eval/configs/vln_r2r_video20.yaml`

Behavior:

- Uses the same current-Habitat-compatible schema as the patched full-run config
- Points to:
  - `scenes_dir: /mnt/data/VL-LN-Bench/scene_datasets/`
  - `data_path: /mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_cover20.json.gz`

#### B. Video run config

File:

- `scripts/eval/configs/habitat_dual_system_video20_cfg.py`

Changes relative to the baseline dual-system config:

- Keeps tuned memory-saving inference settings:
  - `resize_w = 320`
  - `resize_h = 320`
  - `max_new_tokens = 256`
  - `num_history = 8`
- Enables video outputs:
  - `save_video = True`
  - `vis_debug = True`
- Uses dedicated output paths:
  - `output_path = ./logs/habitat/video20`
  - `vis_debug_path = ./logs/habitat/video20/vis_debug`

Video behavior:

- `vis_debug=True` writes per-episode debug MP4 files
- `save_video=True` writes success-case visualization videos

## Files Added During This Work

- `scripts/eval/configs/habitat_dual_system_tuned_cfg.py`
- `scripts/eval/configs/habitat_dual_system_video20_cfg.py`
- `scripts/eval/configs/vln_r2r_video20.yaml`
- `scripts/eval/create_vlnce_subset.py`
- `RUN_NOTES.md`

## File Modified During This Work

- `scripts/eval/configs/vln_r2r.yaml`
