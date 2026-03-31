# VLN-CE Failure Analysis

This note summarizes the failed VLN-CE episodes from the `video20` run using the previous DualVLN checkpoint.

## Run setup

- Config: [habitat_dual_system_video20_cfg.py](/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/habitat_dual_system_video20_cfg.py)
- Habitat YAML: [vln_r2r_video20.yaml](/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/vln_r2r_video20.yaml)
- Model checkpoint: `checkpoints/InternVLA-N1-DualVLN`
- Output dir: [video20](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20)
- Aggregate result: [result.json](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/result.json)

## Overall result

- Episodes: `20`
- `SR = 0.45`
- `SPL = 0.4166`
- `OS = 0.55`
- `NE = 5.0085`

## How to read failures

I grouped failures into three coarse modes from the recorded metrics:

- `Timed out / wandered without reaching goal`
  - `steps = 501`
  - `os = 0`
  - usually means the agent never got near the goal region.
- `Reached vicinity but failed to stop precisely`
  - `os = 1`, `success = 0`
  - usually means it got near the goal at some point but never stopped correctly.
- `Early wrong turn or premature stop far from goal`
  - short or medium trajectory, large `ne`, `os = 0`
  - usually means early route deviation or stop error.
- `Drifted off route and ended far from goal`
  - longer trajectory, large `ne`, `os = 0`
  - usually means route tracking degraded after some correct progress.

## Failed episodes

### 1. Timeout and never reached the goal

- Scene `8194nk5LbLH`, episode `220`
  - Instruction: `Start in the middle of the large room head towards the door that leads to the outside, turn left and head up the stairs, take a hard left, stop when you are inside the workout room.`
  - Metrics: `steps=501`, `ne=21.871`, `os=0`, `success=0`
  - Debug video: [8194nk5LbLH_0220.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/8194nk5LbLH_0220.mp4)

### 2. Drifted off route after partial progress

- Scene `8194nk5LbLH`, episode `221`
  - Instruction: `Walk towards the door and take a left. Go up the stairs. At the top of the stairs take a left into the fitness room. In the fitness room stop after you pass the door.`
  - Metrics: `steps=371`, `ne=18.733`, `os=0`, `success=0`
  - Debug video: [8194nk5LbLH_0221.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/8194nk5LbLH_0221.mp4)

### 3. Early wrong turn or early stop

- Scene `EU6Fwq7SyZv`, episode `127`
  - Instruction: `Turn around 180 degrees. Go through the open door to the loft. Once through the doorway, turn to the left and walk towards the far side of the loft. Stop once you are just past the opening to the stairs.`
  - Metrics: `steps=96`, `ne=5.186`, `os=0`, `success=0`
  - Debug video: [EU6Fwq7SyZv_0127.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/EU6Fwq7SyZv_0127.mp4)

- Scene `EU6Fwq7SyZv`, episode `128`
  - Instruction: `Turn around and leave the bathroom. Go into the bedroom. Turn left and go down the hallway to the left. You'll stop by the big cabinet on the left.`
  - Metrics: `steps=48`, `ne=5.349`, `os=0`, `success=0`
  - Debug video: [EU6Fwq7SyZv_0128.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/EU6Fwq7SyZv_0128.mp4)

- Scene `TbHJrupSAjP`, episode `4`
  - Instruction: `Walk up the flight of stairs to the top. Turn and follow the railing into the bedroom area. Walk towards the closet.`
  - Metrics: `steps=67`, `ne=7.316`, `os=0`, `success=0`
  - Debug video: [TbHJrupSAjP_0004.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/TbHJrupSAjP_0004.mp4)

- Scene `TbHJrupSAjP`, episode `5`
  - Instruction: `Go up the stairs, and then turn around and go through the door past the stairs. Walk up to the bed, then turn to the right and walk into the doorway. Stop there.`
  - Metrics: `steps=63`, `ne=7.186`, `os=0`, `success=0`
  - Debug video: [TbHJrupSAjP_0005.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/TbHJrupSAjP_0005.mp4)

- Scene `oLBMNvg9in8`, episode `28`
  - Instruction: `Exit the room through the door on the right. Walk past the stairs and the bathroom and enter the room with the cardboard boxes. Stop by the doorway on the left that leads to the living room.`
  - Metrics: `steps=91`, `ne=6.131`, `os=0`, `success=0`
  - Debug video: [oLBMNvg9in8_0028.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/oLBMNvg9in8_0028.mp4)

- Scene `oLBMNvg9in8`, episode `29`
  - Instruction: `Turn around and go through the single door on the right. Go down the hallway. You'll go all the way into the room and stand on the rug and wait.`
  - Metrics: `steps=91`, `ne=8.862`, `os=0`, `success=0`
  - Debug video: [oLBMNvg9in8_0029.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/oLBMNvg9in8_0029.mp4)

- Scene `pLe4wQe7qrG`, episode `413`
  - Instruction: `Go through the archway to the right, and walk to the podium.`
  - Metrics: `steps=59`, `ne=6.334`, `os=0`, `success=0`
  - Debug video: [pLe4wQe7qrG_0413.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/pLe4wQe7qrG_0413.mp4)

### 4. Got near the goal but did not stop correctly

- Scene `Z6MFQCViBuw`, episode `206`
  - Instruction: `Turn around. Walk down the long hallway until you reach the room with the red canopy bed.`
  - Metrics: `steps=89`, `ne=3.171`, `os=1`, `success=0`
  - Debug video: [Z6MFQCViBuw_0206.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/Z6MFQCViBuw_0206.mp4)
  - Interpretation: the agent likely reached the right room or close to it, but did not satisfy the success stop condition.

- Scene `pLe4wQe7qrG`, episode `412`
  - Instruction: `Walk past altar book stands. Wait under wooden rafter.`
  - Metrics: `steps=501`, `ne=2.227`, `os=1`, `success=0`
  - Debug video: [pLe4wQe7qrG_0412.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/pLe4wQe7qrG_0412.mp4)
  - Interpretation: this is the clearest `near goal but did not stop correctly` failure. It timed out while staying relatively close.

## Best failure examples to inspect first

If you want a quick cross-section, start with these four:

- [8194nk5LbLH_0220.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/8194nk5LbLH_0220.mp4)
  - clear long-horizon failure, total drift and timeout.
- [EU6Fwq7SyZv_0128.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/EU6Fwq7SyZv_0128.mp4)
  - short-horizon failure, likely early orientation mistake.
- [Z6MFQCViBuw_0206.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/Z6MFQCViBuw_0206.mp4)
  - near-goal failure, useful for stop-analysis.
- [pLe4wQe7qrG_0412.mp4](/mnt/data/0923_Interndata/InternNav/logs/habitat/video20/vis_debug/epoch_0/pLe4wQe7qrG_0412.mp4)
  - another near-goal timeout, useful for inspecting stop behavior.

## Notes

- Successful episodes additionally have `vis_0/<scene>/<episode>.mp4`.
- Failed episodes in this run are best inspected through `vis_debug/epoch_0/*.mp4`, because that directory covers every episode.
- The failure categorization above is metric-driven. If needed, a second pass can be done by manually reviewing the videos and labeling each failure as:
  - initial heading error
  - wrong branch after waypoint
  - looping
  - premature stop
  - no-stop near goal
