import argparse
import json
from pathlib import Path

import cv2

from scripts.data_collect.stop_alignment_utils import (
    extract_stop_object_phrase,
    extract_stop_phrase,
    is_usable_stop_object_phrase,
)


def resolve_video_path(eval_root: Path, scene_id: str, episode_id: int) -> Path:
    return eval_root / "vis_debug" / "epoch_0" / f"{scene_id}_{episode_id:04d}.mp4"


def resolve_frame_dir(output_root: Path, scene_id: str, episode_id: int) -> Path:
    return output_root / "frames" / scene_id / f"{episode_id:04d}"


def should_label_positive(row: dict, frame_idx: int, length: int, positive_last_k: int, near_goal_threshold: float):
    success = float(row.get("success", 0.0))
    oracle_success = float(row.get("os", 0.0))
    ne = float(row.get("ne", 1e9))
    if success >= 1.0:
        return frame_idx >= max(0, length - positive_last_k)
    if oracle_success >= 1.0 and ne <= near_goal_threshold:
        return frame_idx >= max(0, length - positive_last_k)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-last-k", type=int, default=4)
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--near-negative-gap", type=int, default=6)
    parser.add_argument("--near-goal-threshold", type=float, default=3.0)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    progress_path = Path(args.progress_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.joinpath("frames").mkdir(parents=True, exist_ok=True)

    rows = []
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_episodes > 0:
        rows = rows[: args.max_episodes]

    count = 0
    positives = 0
    negatives = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            scene_id = row["scene_id"]
            episode_id = int(row["episode_id"])
            instruction = row.get("episode_instruction", "").strip()
            stop_phrase = extract_stop_phrase(instruction)
            stop_object_phrase = extract_stop_object_phrase(instruction)
            usable_stop_object_phrase = is_usable_stop_object_phrase(stop_object_phrase)
            video_path = resolve_video_path(eval_root, scene_id, episode_id)
            if not video_path.exists():
                skipped += 1
                continue

            frame_dir = resolve_frame_dir(output_path.parent, scene_id, episode_id)
            frame_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped += 1
                continue

            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()

            length = len(frames)
            if length == 0:
                skipped += 1
                continue

            pos_start = max(0, length - args.positive_last_k)
            neg_end = max(0, pos_start - args.near_negative_gap)

            for frame_idx, frame in enumerate(frames):
                label = None
                if should_label_positive(row, frame_idx, length, args.positive_last_k, args.near_goal_threshold):
                    label = 1
                elif frame_idx < neg_end and frame_idx % args.negative_stride == 0:
                    label = 0
                if label is None:
                    continue

                frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
                if not frame_path.exists() or args.overwrite:
                    cv2.imwrite(str(frame_path), frame)

                sample = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "frame_index": frame_idx,
                    "frame_count": length,
                    "instruction": instruction,
                    "stop_phrase": stop_phrase,
                    "stop_object_phrase": stop_object_phrase,
                    "usable_stop_object_phrase": usable_stop_object_phrase,
                    "image_path": str(frame_path),
                    "label": label,
                    "success": float(row.get("success", 0.0)),
                    "oracle_success": float(row.get("os", 0.0)),
                    "distance_to_goal": float(row.get("ne", -1.0)),
                    "steps": int(row.get("steps", -1)),
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
                "skipped_episodes": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
