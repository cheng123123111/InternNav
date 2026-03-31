# N1-Like Data Collection Notes

## Goal

This document records the reconstruction of an `InternData-N1`-like data pipeline on local Habitat/MP3D scenes, including:

- short-trajectory data collection
- N1-style conversion
- instruction generation
- visualization
- training smoke tests against the local `InternVLA-N1-DualVLN` codebase

It also records the issues encountered, the fixes applied, and the user-requested optimizations that changed the pipeline.

## Relevant Paths

- Repo root: `/mnt/data/0923_Interndata/InternNav`
- Habitat env: `/mnt/data/.conda/envs/habitat`
- InternNav train/eval env: `/mnt/data/.conda/envs/internvln39`
- MP3D scenes: `/mnt/data/0923_Interndata/VL-LN-Bench/scene_datasets/mp3d`
- Official N1 sample data used for comparison:
  `/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample`

## Scripts Added Or Updated

### Collection / conversion / instruction generation

- `scripts/data_collect/collect_habitat_n1_like.py`
- `scripts/data_collect/convert_raw_to_n1_like.py`
- `scripts/data_collect/generate_vln_instructions.py`
- `scripts/data_collect/apply_generated_instructions.py`

### Smoke tests

- `scripts/data_collect/smoke_test_n1_dataset.py`
- `scripts/data_collect/smoke_test_n1_forward.py`

### Visualization

- `/mnt/data/0923_Interndata/visualize_n1_episode.py`
- `/mnt/data/0923_Interndata/visualize_official_n1_sample.py`

### Model / dataset compatibility fixes

- `internnav/dataset/internvla_n1_lerobot_dataset.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1.py`

## What The Reconstructed Pipeline Does

### 1. Collect a short expert trajectory in Habitat

`collect_habitat_n1_like.py`:

- samples random navigable start/goal pairs from the MP3D navmesh
- keeps only short tasks within a geodesic range
- follows a shortest path
- records per-frame:
  - `action`
  - `pose.125cm_30deg`
  - `goal.125cm_30deg`
  - `relative_goal_frame_id.125cm_30deg`
  - RGB
  - depth

Current default tuning:

- `forward_step = 0.05`
- `min_geodesic = 2.0`
- `max_geodesic = 6.0`

These were changed after comparing local official N1 samples and finding that:

- the initial step size was too coarse
- the original tasks were too long

### 2. Convert the raw trajectory to an N1-like training layout

`convert_raw_to_n1_like.py` writes:

- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/info.json`
- `data/chunk-000/episode_xxxxxx.parquet`
- `data/chunk-000/episode_xxxxxx.json`
- `videos/chunk-000/observation.images.rgb.125cm_0deg/...`
- `videos/chunk-000/observation.images.rgb.125cm_30deg/...`
- `videos/chunk-000/observation.images.depth.125cm_30deg/...`

The `125cm_0deg` RGB directory is currently a compatibility alias copied from the only collected RGB stream, because the N1 loader expects that path name.

### 3. Generate N1-style instructions

`generate_vln_instructions.py`:

- extracts keyframes from trajectory geometry
- builds segment contact sheets
- asks `gpt-4o` to generate:
  - `fine_grained_instructions`
  - `revised_fine_grained_instructions`
  - `long_instruction`

The final format is aligned with the N1 training expectation by joining multiple instructions with:

- `"<INSTRUCTION_SEP>"`

Then `apply_generated_instructions.py` writes these generated instructions back into:

- `meta/episodes.jsonl`
- `meta/tasks.jsonl`

### 4. Visualize the data

`visualize_n1_episode.py` renders:

- left: RGB frames
- right: Habitat-style global topdown map
- overlay: current pixel goal
- header: generated long instruction and sub-instructions

`visualize_official_n1_sample.py` renders official local N1 examples for comparison.

## Data Structure

### Raw collection layout

For one episode:

- `raw/episode_000000/meta.json`
- `raw/episode_000000/frames.json`
- `raw/episode_000000/rgb_125cm_30deg/*.jpg`
- `raw/episode_000000/depth_125cm_30deg/*.png`

`frames.json` stores the important supervision:

- `action`
- `pose.125cm_30deg`
- `goal.125cm_30deg`
- `relative_goal_frame_id.125cm_30deg`

### Converted N1-like layout

For one scene:

- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `data/chunk-000/episode_000000.parquet`
- `data/chunk-000/episode_000000.json`
- `videos/chunk-000/...`

### Meaning of the key fields

- `pose.125cm_30deg`
  - camera pose for the current frame
- `goal.125cm_30deg`
  - pixel goal on the current image plane
- `relative_goal_frame_id.125cm_30deg`
  - the future frame offset used to define the pixel goal

Current pixel-goal rule:

- search forward within a fixed horizon
- take the farthest visible future frame that can still be projected into the current image

This is a simplified approximation of the paper’s “farthest visible pixel goal” idea.

## Official N1 Comparison

We compared against local official N1 samples under:

- `/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample`

Observed properties:

- many trajectories have only 1 sub-instruction
- some have 2 or 3
- typical sub-clip length is around `~75-100` frames
- instructions often use:
  - strong landmark grounding
  - side relations
  - an explicit final stop target

Representative comparison videos created:

- `/mnt/data/0923_Interndata/n1_compare/office_3_traj_68.mp4`
- `/mnt/data/0923_Interndata/n1_compare/room_0_traj_65.mp4`
- `/mnt/data/0923_Interndata/n1_compare/apartment_2_traj_89.mp4`

## User-Driven Optimizations Applied

The following changes were made in response to iterative inspection:

### Shorter tasks

Requested:

- tasks should be short
- trajectory distance should be reduced

Applied:

- added `--min-geodesic`
- added `--max-geodesic`
- lowered task distances for the short-task pipeline

### Denser frame sampling

Requested:

- move closer to the apparent frame density of local official N1 samples

Applied:

- reduced `forward_step` from `0.25` to `0.05`

### Instruction style

Requested:

- do not include explicit meters in template instructions
- summary must include an explicit stop condition
- segment count should be reduced
- summary should be shorter and closer to official N1

Applied:

- removed explicit meter counts from template instructions
- changed keyframe segmentation to be coarser
- added segment merging
- changed summary prompt to aggressively merge repeated forward motions and landmarks
- forced explicit stop in the final summary

### Video / visualization

Requested:

- add instruction to the visualization
- show a real topdown map, not a fake blank canvas

Applied:

- visualizer now supports instruction JSON input
- right-hand map now uses Habitat topdown rendering

## Problems Encountered And Fixes

### 1. Pixel goal projection looked wrong or missing

Symptoms:

- projected goal point was missing
- some trajectories appeared visually reversed

Root causes:

- pose / camera forward convention was inconsistent
- a pose-based projection version had the sign wrong
- a topdown visualization version used the wrong axis pair

Fixes:

- switched back to render-camera style projection for goal generation
- corrected map plotting from `(x, y)` to `(x, z)`
- verified that `relative_goal_frame_id=16` was usually visible, so the issue was not horizon size

### 2. Topdown map was blank or misleading

Symptoms:

- right-hand visualization was empty or not comparable to Habitat eval videos

Fix:

- replaced the custom blank trajectory panel with Habitat-style global topdown rendering

### 3. Instructions were too long and too fragmented

Symptoms:

- too many sub-segments
- summary was much longer than official N1 style

Fixes:

- changed keyframe extraction to be less sensitive
- merged short segments
- changed the summary prompt to remove repeated forward phrases and repeated landmarks

### 4. Left / right errors

Symptoms:

- generated instructions sometimes swapped left and right
- turn direction could be mistrusted

Findings:

- turn direction derived from geometry is more reliable than the LLM’s landmark-side narration
- landmark-side narration from images alone is still a weak point

Fixes applied:

- corrected geometric turn-direction extraction
- added prompt constraints:
  - explicit geometric turn hint is authoritative
  - summary must preserve sub-instruction turn directions
  - do not invent side relations unless clearly supported

Current status:

- turn-direction stability is improved
- landmark-side narration is still imperfect and remains the main quality weakness

### 5. Training loader failed due to missing dependencies

Symptoms:

- `decord` missing
- `pyarrow` missing

Fixes:

- smoke tests use small module stubs for optional video-only imports
- added JSON fallback in `internvla_n1_lerobot_dataset.py` so training smoke tests do not require `pyarrow`

### 6. Converted data failed against the N1 loader

Symptoms:

- loader looked for `rgb.125cm_0deg`, but collected data only had `rgb.125cm_30deg`

Fix:

- converter now also creates a compatibility `rgb.125cm_0deg` directory

### 7. Forward failed because `t_s_pos` was missing

Symptoms:

- `NoneType` error in `internvla_n1.py`

Root cause:

- the smoke test used generic turn/stop samples
- `t_s_pos` is only created for trajectory/pixel-goal samples

Fix:

- reran forward smoke test with `pixel_goal_only=True`

### 8. Forward failed because of bf16 / float32 mismatch

Symptoms:

- `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same`

Root cause:

- System1 trajectory image path was normalized in float32 while the model branch was bf16

Fix:

- in `internvla_n1.py`, cast the normalized System1 RGB tensor to the backbone dtype before passing it into `rgb_model`

## Training Smoke Test Result

The reconstructed dataset was tested against the actual N1 code.

### Dataset smoke test

Script:

- `scripts/data_collect/smoke_test_n1_dataset.py`

Result:

- dataset construction succeeded
- collator batching succeeded

Observed result:

- `dataset_len = 413`
- batch tensors were created successfully

### Model forward smoke test

Script:

- `scripts/data_collect/smoke_test_n1_forward.py`

Result:

- local `InternVLA-N1-DualVLN` checkpoint loaded on GPU
- one real forward pass succeeded on the collected data

Observed result:

- `loss 0.1687498688697815`
- `forward_ok`

This confirms:

- the reconstructed data is not only format-compatible
- it can pass through the actual N1 model forward path

## Commands Used

### Collect short trajectories

```bash
export DISPLAY=:0
/mnt/data/.conda/envs/habitat/bin/python \
  /mnt/data/0923_Interndata/InternNav/scripts/data_collect/collect_habitat_n1_like.py \
  --scene /mnt/data/0923_Interndata/VL-LN-Bench/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb \
  --output-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10 \
  --episodes 10 \
  --forward-step 0.05 \
  --min-geodesic 2.0 \
  --max-geodesic 6.0
```

### Convert to N1-like structure

```bash
python /mnt/data/0923_Interndata/InternNav/scripts/data_collect/convert_raw_to_n1_like.py \
  --raw-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10 \
  --scene-id 2azQ1b91cZZ \
  --output-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted
```

### Generate instructions

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
export OPENAI_MODEL=gpt-4o

python /mnt/data/0923_Interndata/InternNav/scripts/data_collect/generate_vln_instructions.py \
  --raw-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10 \
  --scene-id 2azQ1b91cZZ \
  --episode-index 0
```

### Apply generated instructions into converted metadata

```bash
python /mnt/data/0923_Interndata/InternNav/scripts/data_collect/apply_generated_instructions.py \
  --raw-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10 \
  --scene-id 2azQ1b91cZZ \
  --converted-root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted
```

### Visualize a generated sample

```bash
/mnt/data/.conda/envs/habitat/bin/python \
  /mnt/data/0923_Interndata/visualize_n1_episode.py \
  --root /mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted \
  --scene 2azQ1b91cZZ \
  --scene-path /mnt/data/0923_Interndata/VL-LN-Bench/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb \
  --episode 0 \
  --instruction-json /mnt/data/0923_Interndata/tmp_collect_n1_like_short10/2azQ1b91cZZ/raw/episode_000000/generated_instructions_v7.json \
  --output /mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted/2azQ1b91cZZ_episode_0_v7_instruction.mp4
```

### Dataset smoke test

```bash
conda activate internvln39
cd /mnt/data/0923_Interndata/InternNav
python scripts/data_collect/smoke_test_n1_dataset.py
```

### Forward smoke test

```bash
conda activate internvln39
cd /mnt/data/0923_Interndata/InternNav
python scripts/data_collect/smoke_test_n1_forward.py
```

## Current Output Directories

### Generated short-task dataset

- raw: `/mnt/data/0923_Interndata/tmp_collect_n1_like_short10`
- converted: `/mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted`

### Example generated videos

- `/mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted/2azQ1b91cZZ_episode_0_v7_instruction.mp4`
- `/mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted/2azQ1b91cZZ_episode_1_v7_instruction.mp4`
- ...
- `/mnt/data/0923_Interndata/tmp_collect_n1_like_short10_converted/2azQ1b91cZZ_episode_9_v7_instruction.mp4`

## Remaining Gaps

The following are still approximations rather than full official reproduction:

- expert planner is shortest-path based, not the full ESDF + optimized pipeline described in the paper
- pixel-goal rule is a simplified visible-future-frame heuristic
- landmark-side description is still less reliable than official N1
- instruction generation uses `gpt-4o`, not the exact paper stack
- `125cm_0deg` RGB is currently a compatibility alias, not a separately rendered stream

## Recommended Next Steps

1. Replace compatibility `125cm_0deg` RGB with a true second rendered stream.
2. Improve landmark-side estimation using explicit geometric checks or view-conditioned projection.
3. Expand the short dataset beyond one scene.
4. Add a minimal one-step training script, not only forward smoke tests.
