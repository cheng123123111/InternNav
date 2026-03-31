#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--converted-root", required=True)
    parser.add_argument("--instruction-file", default="generated_instructions_v7.json")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_scene = Path(args.raw_root) / args.scene_id / "raw"
    conv_scene = Path(args.converted_root) / args.scene_id
    episodes_path = conv_scene / "meta" / "episodes.jsonl"
    tasks_path = conv_scene / "meta" / "tasks.jsonl"

    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    task_to_index = {}
    next_task_idx = 0

    updated_episodes = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        instr_path = raw_scene / f"episode_{ep_idx:06d}" / args.instruction_file
        if instr_path.exists():
            payload = json.loads(instr_path.read_text())
            instructions = payload.get("revised_fine_grained_instructions", []) + [payload["long_instruction"]]
            task_string = "<INSTRUCTION_SEP>".join(instructions)
        else:
            task_string = ep["tasks"][0]

        if task_string not in task_to_index:
            task_to_index[task_string] = next_task_idx
            next_task_idx += 1

        updated_episodes.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_string],
                "length": ep["length"],
            }
        )

    updated_tasks = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_index.items(), key=lambda x: x[1])]

    episodes_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in updated_episodes) + "\n")
    tasks_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in updated_tasks) + "\n")
    print(episodes_path)
    print(tasks_path)


if __name__ == "__main__":
    main()
