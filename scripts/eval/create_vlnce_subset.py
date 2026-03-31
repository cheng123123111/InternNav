import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small VLN-CE subset covering all scenes.")
    parser.add_argument(
        "--input",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz",
        help="Path to the source VLN-CE json.gz file.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_cover20.json.gz",
        help="Path to the output subset json.gz file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of episodes to keep in the subset.",
    )
    return parser.parse_args()


def load_dataset(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, list):
        episodes = data
    else:
        raise ValueError(f"Unsupported dataset format in {path}")
    return data, episodes


def select_episodes(episodes, count: int):
    by_scene = defaultdict(list)
    for ep in episodes:
        scene = ep["scene_id"].split("/")[-2]
        by_scene[scene].append(ep)

    scenes = sorted(by_scene)
    if count < len(scenes):
        raise ValueError(f"Requested count={count} is smaller than scene count={len(scenes)}")

    selected = []
    used = set()

    # First pass: guarantee at least one episode per scene.
    for scene in scenes:
        ep = sorted(by_scene[scene], key=lambda x: int(x["episode_id"]))[0]
        selected.append(ep)
        used.add((scene, int(ep["episode_id"])))

    # Second pass: round-robin across scenes for the remaining slots.
    scene_lists = {scene: sorted(by_scene[scene], key=lambda x: int(x["episode_id"])) for scene in scenes}
    indices = {scene: 1 for scene in scenes}
    while len(selected) < count:
        progressed = False
        for scene in scenes:
            eps = scene_lists[scene]
            while indices[scene] < len(eps):
                ep = eps[indices[scene]]
                indices[scene] += 1
                key = (scene, int(ep["episode_id"]))
                if key in used:
                    continue
                selected.append(ep)
                used.add(key)
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break

    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)} episodes, expected {count}")
    return selected


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data, episodes = load_dataset(input_path)
    selected = select_episodes(episodes, args.count)

    if isinstance(data, dict) and "episodes" in data:
        data["episodes"] = selected
        out_obj = data
    else:
        out_obj = selected

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(out_obj, f)

    scene_counts = defaultdict(int)
    for ep in selected:
        scene_counts[ep["scene_id"].split("/")[-2]] += 1

    print(f"Wrote {len(selected)} episodes to {output_path}")
    for scene in sorted(scene_counts):
        print(f"{scene}: {scene_counts[scene]}")


if __name__ == "__main__":
    main()
