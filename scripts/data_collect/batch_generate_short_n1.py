#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd, env=None):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--converted-root", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--forward-step", type=float, default=0.05)
    parser.add_argument("--min-geodesic", type=float, default=2.0)
    parser.add_argument("--max-geodesic", type=float, default=6.0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    raw_root = Path(args.raw_root)
    converted_root = Path(args.converted_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    converted_root.mkdir(parents=True, exist_ok=True)

    collect_cmd = [
        "/mnt/data/.conda/envs/habitat/bin/python",
        "/mnt/data/0923_Interndata/InternNav/scripts/data_collect/collect_habitat_n1_like.py",
        "--scene", args.scene,
        "--output-root", str(raw_root),
        "--episodes", str(args.episodes),
        "--forward-step", str(args.forward_step),
        "--min-geodesic", str(args.min_geodesic),
        "--max-geodesic", str(args.max_geodesic),
    ]
    collect_env = os.environ.copy()
    collect_env["DISPLAY"] = ":0"
    run(collect_cmd, env=collect_env)

    convert_cmd = [
        "python",
        "/mnt/data/0923_Interndata/InternNav/scripts/data_collect/convert_raw_to_n1_like.py",
        "--raw-root", str(raw_root),
        "--scene-id", args.scene_id,
        "--output-root", str(converted_root),
    ]
    run(convert_cmd)

    for episode_idx in range(args.episodes):
        instruction_json = raw_root / args.scene_id / "raw" / f"episode_{episode_idx:06d}" / "generated_instructions_v7.json"
        generate_env = os.environ.copy()
        generate_env["OPENAI_API_KEY"] = api_key
        generate_env["OPENAI_BASE_URL"] = args.base_url
        generate_env["OPENAI_MODEL"] = args.model
        generate_cmd = [
            "python",
            "/mnt/data/0923_Interndata/InternNav/scripts/data_collect/generate_vln_instructions.py",
            "--raw-root", str(raw_root),
            "--scene-id", args.scene_id,
            "--episode-index", str(episode_idx),
            "--output", str(instruction_json),
        ]
        run(generate_cmd, env=generate_env)

        video_out = converted_root / f"{args.scene_id}_episode_{episode_idx}_v7_instruction.mp4"
        visualize_cmd = [
            "/mnt/data/.conda/envs/habitat/bin/python",
            "/mnt/data/0923_Interndata/visualize_n1_episode.py",
            "--root", str(converted_root),
            "--scene", args.scene_id,
            "--scene-path", args.scene,
            "--episode", str(episode_idx),
            "--instruction-json", str(instruction_json),
            "--output", str(video_out),
        ]
        run(visualize_cmd)

    print("DONE")
    print("RAW_ROOT", raw_root)
    print("CONVERTED_ROOT", converted_root)


if __name__ == "__main__":
    main()
