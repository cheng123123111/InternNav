import argparse
import json
from pathlib import Path


def choose_instruction(task_text: str) -> str:
    parts = [p.strip() for p in task_text.split("<INSTRUCTION_SEP>") if p.strip()]
    return parts[-1] if parts else task_text.strip()


def resolve_image_path(scene_root: Path, episode_index: int, frame_index: int) -> Path:
    candidates = [
        scene_root / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg" / f"episode_{episode_index:06d}_{frame_index}.jpg",
        scene_root / "videos" / "chunk-000" / "observation.images.rgb.125cm_30deg" / f"episode_{episode_index:06d}_{frame_index}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-last-k", type=int, default=2)
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--near-negative-gap", type=int, default=6)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    positives = 0
    negatives = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for scene_root in sorted([p for p in data_root.iterdir() if p.is_dir()]):
            episodes_path = scene_root / "meta" / "episodes.jsonl"
            if not episodes_path.exists():
                continue
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    ep = json.loads(line)
                    episode_index = int(ep["episode_index"])
                    instruction = choose_instruction(ep["tasks"][0])
                    data_json = scene_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.json"
                    if not data_json.exists():
                        continue
                    records = json.loads(data_json.read_text(encoding="utf-8"))
                    if not records:
                        continue
                    length = len(records)
                    pos_start = max(0, length - args.positive_last_k)
                    neg_end = max(0, pos_start - args.near_negative_gap)
                    for frame_index in range(length):
                        label = None
                        if frame_index >= pos_start:
                            label = 1
                        elif frame_index < neg_end and frame_index % args.negative_stride == 0:
                            label = 0
                        if label is None:
                            continue
                        image_path = resolve_image_path(scene_root, episode_index, frame_index)
                        sample = {
                            "scene_id": scene_root.name,
                            "episode_index": episode_index,
                            "frame_index": frame_index,
                            "instruction": instruction,
                            "image_path": str(image_path),
                            "label": label,
                        }
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
                        positives += int(label == 1)
                        negatives += int(label == 0)

    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "num_samples": count,
                "positives": positives,
                "negatives": negatives,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
