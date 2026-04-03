import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor Habitat progress.json and print cumulative metrics every N episodes."
    )
    parser.add_argument("--progress", required=True, help="Path to progress.json")
    parser.add_argument("--every", type=int, default=10, help="Print cumulative summary every N finished episodes")
    parser.add_argument("--target", type=int, default=None, help="Stop after this many finished episodes are observed")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval in seconds")
    return parser.parse_args()


def load_entries(path: Path):
    entries = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize(entries):
    total = len(entries)
    if total == 0:
        return None

    success = sum(float(e.get("success", 0.0)) for e in entries)
    spl = sum(float(e.get("spl", 0.0)) for e in entries)
    os_ = sum(float(e.get("os", 0.0)) for e in entries)
    ne = sum(float(e.get("ne", 0.0)) for e in entries)
    failed = [
        f"{e.get('scene_id', 'unknown')}_{int(e.get('episode_id', -1)):04d}"
        for e in entries
        if float(e.get("success", 0.0)) < 1.0
    ]
    return {
        "count": total,
        "success_rate": success / total,
        "spl": spl / total,
        "os": os_ / total,
        "ne": ne / total,
        "failed": failed,
    }


def print_summary(summary):
    print(
        "[progress] "
        f"done={summary['count']} "
        f"success_rate={summary['success_rate']:.4f} "
        f"spl={summary['spl']:.4f} "
        f"os={summary['os']:.4f} "
        f"ne={summary['ne']:.4f}"
    )
    if summary["failed"]:
        print("[progress] failed_episodes=" + ", ".join(summary["failed"]))
    else:
        print("[progress] failed_episodes=none")


def main():
    args = parse_args()
    progress_path = Path(args.progress)
    last_bucket = 0

    while True:
        entries = load_entries(progress_path)
        done = len(entries)
        bucket = done // args.every

        if bucket > last_bucket:
            summary = summarize(entries)
            if summary is not None:
                print_summary(summary)
            last_bucket = bucket

        if args.target is not None and done >= args.target:
            if done % args.every != 0:
                summary = summarize(entries)
                if summary is not None:
                    print_summary(summary)
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
