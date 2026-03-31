import collections
import gzip
import json
from pathlib import Path


SOURCE = Path("/mnt/data/0923_Interndata/VL-LN-Bench/raw_data/mp3d/val_unseen/val_unseen_iign.json.gz")
TARGET = Path("/mnt/data/0923_Interndata/VL-LN-Bench/raw_data/mp3d/val_unseen/val_unseen_iign_cover50.json.gz")
TARGET_COUNT = 50


def scene_name(scene_id: str) -> str:
    parts = scene_id.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return scene_id


def main() -> None:
    with gzip.open(SOURCE, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    by_scene = collections.defaultdict(list)
    for episode in episodes:
        by_scene[scene_name(episode["scene_id"])].append(episode)

    scenes = sorted(by_scene)
    selected = []
    round_idx = 0

    # Round-robin selection keeps the subset spread across scenes.
    while len(selected) < min(TARGET_COUNT, len(episodes)):
        added = False
        for scene in scenes:
            items = by_scene[scene]
            if round_idx < len(items):
                selected.append(items[round_idx])
                added = True
                if len(selected) >= TARGET_COUNT:
                    break
        if not added:
            break
        round_idx += 1

    output = {
        "episodes": selected,
        "category_to_task_category_id": data["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": data["category_to_scene_annotation_category_id"],
    }

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(TARGET, "wt") as f:
        json.dump(output, f)

    counts = collections.Counter(scene_name(e["scene_id"]) for e in selected)
    print(f"wrote {len(selected)} episodes to {TARGET}")
    for scene, count in sorted(counts.items()):
        print(f"{scene}\t{count}")


if __name__ == "__main__":
    main()
