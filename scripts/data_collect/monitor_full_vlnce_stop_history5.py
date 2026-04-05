import argparse
from pathlib import Path


DATASETS = ["envdrop", "r2r", "r2r_v1-3", "rxr"]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_jpgs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="/dataset-vln/InternNav/stop_data/full_history5",
        help="Base directory for full history5 stop data outputs.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    total_rows = 0
    total_est_episodes = 0
    total_frames = 0

    print(f"base_dir: {base_dir}")
    for dataset in DATASETS:
        manifest_path = base_dir / dataset / f"{dataset}_stop_manifest_history5_full.jsonl"
        frames_root = base_dir / dataset / "frames_history5_full"
        rows = count_lines(manifest_path)
        est_episodes = rows // 3
        frames = count_jpgs(frames_root)
        total_rows += rows
        total_est_episodes += est_episodes
        total_frames += frames
        print(
            f"{dataset:8s} rows={rows:6d} est_episodes={est_episodes:6d} frames={frames:7d}"
        )

    print(
        f"TOTAL    rows={total_rows:6d} est_episodes={total_est_episodes:6d} frames={total_frames:7d}"
    )


if __name__ == "__main__":
    main()
