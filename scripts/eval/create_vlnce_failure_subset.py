import argparse
import gzip
import json
from pathlib import Path


DEFAULT_EPISODES = [
    220,  # timeout, far away
    128,  # early wrong turn
    206,  # near-goal but no successful stop
    412,  # timeout near goal
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a VLN-CE subset from explicit episode ids.")
    parser.add_argument(
        "--input",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_cover20.json.gz",
        help="Path to the source json.gz file.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/data/0923_Interndata/vln_ce/raw_data/r2r/val_unseen/val_unseen_fail4.json.gz",
        help="Path to the output subset json.gz file.",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="*",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episode ids to keep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["episodes"] if isinstance(data, dict) else data
    wanted = set(args.episode_ids)
    selected = [ep for ep in episodes if int(ep["episode_id"]) in wanted]

    if len(selected) != len(wanted):
        found = {int(ep["episode_id"]) for ep in selected}
        missing = sorted(wanted - found)
        raise RuntimeError(f"Missing requested episode ids: {missing}")

    if isinstance(data, dict):
        data["episodes"] = selected
        out = data
    else:
        out = selected

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(out, f)

    for ep in selected:
        scene = ep["scene_id"].split("/")[-2]
        print(scene, ep["episode_id"], ep["instruction"]["instruction_text"])
    print(f"Wrote {len(selected)} episodes to {output_path}")


if __name__ == "__main__":
    main()
