import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a scene-balanced VLN-CE subset with a target number of episodes."
    )
    parser.add_argument(
        "--input",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz",
        help="Path to the source json.gz file.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_bal50.json.gz",
        help="Path to the output subset json.gz file.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50,
        help="Number of episodes to keep.",
    )
    return parser.parse_args()


def load_episodes(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"] if isinstance(data, dict) else data
    return data, episodes


def get_scene_name(scene_id: str) -> str:
    return scene_id.split("/")[-1].replace(".glb", "")


def select_balanced_episodes(episodes, target_count: int):
    grouped = defaultdict(list)
    for ep in episodes:
        grouped[get_scene_name(ep["scene_id"])].append(ep)

    scenes = sorted(grouped)
    per_scene_target = target_count // len(scenes)
    remainder = target_count % len(scenes)
    selected = []

    for idx, scene in enumerate(scenes):
        scene_eps = sorted(grouped[scene], key=lambda ep: int(ep["episode_id"]))
        take = per_scene_target + (1 if idx < remainder else 0)
        if take <= 0:
            continue
        if take >= len(scene_eps):
            selected.extend(scene_eps)
            continue

        if take == 1:
            indices = [len(scene_eps) // 2]
        else:
            indices = []
            for i in range(take):
                pos = round(i * (len(scene_eps) - 1) / (take - 1))
                indices.append(pos)
            indices = sorted(set(indices))
            while len(indices) < take:
                for cand in range(len(scene_eps)):
                    if cand not in indices:
                        indices.append(cand)
                        if len(indices) == take:
                            break
            indices = sorted(indices[:take])
        selected.extend(scene_eps[i] for i in indices)

    return sorted(selected, key=lambda ep: (get_scene_name(ep["scene_id"]), int(ep["episode_id"])))


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    data, episodes = load_episodes(input_path)

    selected = select_balanced_episodes(episodes, args.target_count)
    if len(selected) < args.target_count:
        raise RuntimeError(
            f"Only selected {len(selected)} episodes, fewer than target_count={args.target_count}"
        )

    if isinstance(data, dict):
        data["episodes"] = selected
        out = data
    else:
        out = selected

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(selected)} episodes to {output_path}")
    for ep in selected:
        scene = get_scene_name(ep["scene_id"])
        instruction = ep["instruction"]["instruction_text"]
        print(scene, ep["episode_id"], instruction)


if __name__ == "__main__":
    main()
