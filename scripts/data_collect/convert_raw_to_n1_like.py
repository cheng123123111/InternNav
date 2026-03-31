#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root) / args.scene_id / "raw"
    out_scene = Path(args.output_root) / args.scene_id
    (out_scene / "meta").mkdir(parents=True, exist_ok=True)
    (out_scene / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_scene / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg").mkdir(parents=True, exist_ok=True)
    (out_scene / "videos" / "chunk-000" / "observation.images.rgb.125cm_30deg").mkdir(parents=True, exist_ok=True)
    (out_scene / "videos" / "chunk-000" / "observation.images.depth.125cm_30deg").mkdir(parents=True, exist_ok=True)

    tasks = {}
    episodes_meta = []
    episodes_stats = []

    for ep_dir in sorted(raw_root.glob("episode_*")):
        meta = json.loads((ep_dir / "meta.json").read_text())
        frames = json.loads((ep_dir / "frames.json").read_text())
        ep_idx = meta["episode_index"]
        task = meta["instruction"]

        if task not in tasks:
            tasks[task] = len(tasks)

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": len(frames)})
        episodes_stats.append({"episode_index": ep_idx, "length": len(frames)})

        rows = []
        for fr in frames:
            rows.append(
                {
                    "action": fr["action"],
                    "pose.125cm_30deg": fr["pose.125cm_30deg"],
                    "goal.125cm_30deg": fr["goal.125cm_30deg"],
                    "relative_goal_frame_id.125cm_30deg": fr["relative_goal_frame_id.125cm_30deg"],
                    "timestamp": fr["timestamp"],
                    "frame_index": fr["frame_index"],
                    "episode_index": ep_idx,
                    "index": fr["frame_index"],
                    "task_index": tasks[task],
                }
            )

            rgb_src = Path(fr["rgb_path"])
            depth_src = Path(fr["depth_path"])
            rgb_dst = out_scene / "videos" / "chunk-000" / "observation.images.rgb.125cm_30deg" / (
                f"episode_{ep_idx:06d}_{fr['frame_index']}.jpg"
            )
            rgb_horizon_dst = out_scene / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg" / (
                f"episode_{ep_idx:06d}_{fr['frame_index']}.jpg"
            )
            depth_dst = out_scene / "videos" / "chunk-000" / "observation.images.depth.125cm_30deg" / (
                f"episode_{ep_idx:06d}_{fr['frame_index']}.png"
            )
            shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(rgb_src, rgb_horizon_dst)
            shutil.copy2(depth_src, depth_dst)

        df = pd.DataFrame(rows)
        df.to_parquet(out_scene / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet", index=False)
        (out_scene / "data" / "chunk-000" / f"episode_{ep_idx:06d}.json").write_text(json.dumps(rows))

    with open(out_scene / "meta" / "tasks.jsonl", "w") as f:
        for task, idx in tasks.items():
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(out_scene / "meta" / "episodes.jsonl", "w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with open(out_scene / "meta" / "episodes_stats.jsonl", "w") as f:
        for row in episodes_stats:
            f.write(json.dumps(row) + "\n")

    info = {
        "scene_id": args.scene_id,
        "total_episodes": len(episodes_meta),
        "total_tasks": len(tasks),
        "format": "minimal_n1_like_v0",
    }
    (out_scene / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    print(out_scene)


if __name__ == "__main__":
    main()
