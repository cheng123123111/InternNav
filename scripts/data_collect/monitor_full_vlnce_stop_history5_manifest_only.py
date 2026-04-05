import argparse
from pathlib import Path


DATASETS = ["envdrop", "r2r", "r2r_v1-3", "rxr"]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="/dataset-vln/InternNav/stop_data/full_history5_manifest_only",
        help="Base directory for full history5 manifest-only stop data outputs.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    total_rows = 0
    total_est_episodes = 0

    print(f"base_dir: {base_dir}")
    for dataset in DATASETS:
        manifest_path = base_dir / dataset / f"{dataset}_stop_manifest_history5_full.jsonl"
        enriched_path = base_dir / dataset / f"{dataset}_stop_manifest_history5_full_elements.jsonl"
        rows = count_lines(manifest_path)
        enriched_rows = count_lines(enriched_path)
        est_episodes = rows // 3
        total_rows += rows
        total_est_episodes += est_episodes
        print(
            f"{dataset:8s} rows={rows:6d} est_episodes={est_episodes:6d} enriched_rows={enriched_rows:6d}"
        )

    print(f"TOTAL    rows={total_rows:6d} est_episodes={total_est_episodes:6d}")


if __name__ == "__main__":
    main()
