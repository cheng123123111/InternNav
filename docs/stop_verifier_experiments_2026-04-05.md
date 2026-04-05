# Stop Verifier Experiments (2026-04-05)

This note summarizes the code and experiment changes made on branch `qwen_should_stop_verifier`.

## What Changed

- Fixed headless Habitat rendering on the server by forcing the system NVIDIA GL stack in `scripts/eval/bash/torchrun_habitat_eval.sh`.
- Added pure front-view evaluation support so saved videos can exclude top-down map overlays when needed.
- Extended the stop verifier path in `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` to support:
  - history-frame verification
  - legacy LongCLIP stop heads
  - `cross_attn` stop heads
  - `grounding_v4` stop heads
- Added fallback attention handling for environments where `flash_attn` is missing.
- Added a local `depth_camera_filtering.py` compatibility shim used by this workspace.

## Data / Training Pipeline

- Improved `scripts/data_collect/build_stop_manifest_from_traj_tar.py` so corrupt or incomplete tar episodes are skipped instead of crashing the build.
- Added front-video stop dataset builder:
  - `scripts/data_collect/build_stop_manifest_from_front_videos.py`
- Added manifest merge / monitor / run helpers:
  - `scripts/data_collect/merge_stop_manifests.py`
  - `scripts/data_collect/monitor_full_vlnce_stop_history5.py`
  - `scripts/data_collect/monitor_full_vlnce_stop_history5_manifest_only.py`
  - `scripts/data_collect/run_full_vlnce_stop_history5.sh`
  - `scripts/data_collect/run_full_vlnce_stop_history5_manifest_only.sh`
- Refined stop phrase extraction in `scripts/data_collect/stop_alignment_utils.py` for cases such as:
  - `stop when ...`
  - `stop once ...`
  - weak deictic endings like `that's where you will wait`

## Stop Model Work

- Expanded `scripts/data_collect/train_stop_element_longclip_history.py` from the original small MLP head into a configurable trainer that now supports:
  - `mlp`
  - `cross_attn`
  - `grounding_v4`
- Added support for:
  - patch-level visual tokens
  - temporal modeling over history frames
  - multi-element / relation / action text inputs
  - stop and uncertainty queries
  - contrastive supervision
  - worker-based image loading and larger batch training

## Evaluation Configs Added

- Pure front-video configs:
  - `scripts/eval/configs/habitat_dual_system_first10_frontvideo_cfg.py`
  - `scripts/eval/configs/habitat_dual_system_valunseen500_frontvideo_cfg.py`
  - `scripts/eval/configs/habitat_dual_system_valunseen500_frontvideo_longclipstop_cfg.py`
- Smoke / verifier configs:
  - `scripts/eval/configs/habitat_dual_system_ep1_groundingv4stop_cfg.py`
- Failure-case rerender configs:
  - `scripts/eval/configs/habitat_dual_system_fail2success39_localvideo_cfg.py`
  - `scripts/eval/configs/habitat_dual_system_fail2success39_longclipstop_cfg.py`
- Dataset subset configs:
  - `scripts/eval/configs/vln_r2r_val_unseen_500.json.gz`
  - `scripts/eval/configs/vln_r2r_val_unseen_500.yaml`
  - `scripts/eval/configs/vln_r2r_val_unseen_fail2success39.json.gz`
  - `scripts/eval/configs/vln_r2r_val_unseen_fail2success39.yaml`

## Key Experiment Results

### 500-Episode Pure Front Baseline

- success: `0.6200`
- spl: `0.5556`
- oracle success: `0.6920`
- navigation error: `4.0389`

### 500-Episode Pure Front + Stop Interception

Using a stop model trained on the clean `val_unseen500_front` stop dataset:

- success: `0.5560`
- spl: `0.4821`
- oracle success: `0.7380`
- navigation error: `4.6706`

Interpretation:

- the verifier prevented some premature stops
- oracle success increased
- final success still decreased because regressions outnumbered recoveries

### Fail-to-Success Rerender Set

We rerendered the 39 previously observed rescue candidates with top-down maps for easier visual comparison.

On that rerendered subset:

- baseline success: `0.4872`
- stop success: `0.6410`

When aligned by rerendered episode outcome:

- rescued (`baseline fail -> stop success`): `10`
- regressed (`baseline success -> stop fail`): `4`

The bundled comparison package is archived outside the repo in:

- `/dataset-vln/InternNav/archives/fail2success10_bundle_2026-04-05.tar.gz`

## Current Caveat

The stop verifier can help in some cases, but it is not yet a net-positive intervention on the larger 500-episode front-view benchmark. The next step should focus on larger clean stop data and stronger validation before wider rollout.
