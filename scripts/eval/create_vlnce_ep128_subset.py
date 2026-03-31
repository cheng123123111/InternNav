import gzip
import json
from pathlib import Path


INPUT_PATH = Path("/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_cover20.json.gz")
OUTPUT_PATH = Path("/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_ep128.json.gz")
EPISODE_ID = 128


def main():
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"] if isinstance(data, dict) else data
    selected = [ep for ep in episodes if int(ep["episode_id"]) == EPISODE_ID]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one episode {EPISODE_ID}, found {len(selected)}")

    if isinstance(data, dict):
        data["episodes"] = selected
        out = data
    else:
        out = selected

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out, f)

    ep = selected[0]
    print(ep["scene_id"], ep["episode_id"], ep["instruction"]["instruction_text"])
    print(f"Wrote 1 episode to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
