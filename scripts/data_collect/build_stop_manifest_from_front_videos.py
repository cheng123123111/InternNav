import argparse
import json
from pathlib import Path

import cv2

from scripts.data_collect.stop_alignment_utils import (
    extract_stop_object_phrase,
    extract_stop_phrase,
    is_usable_stop_object_phrase,
)


def read_progress_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "episode_id" not in row:
                continue
            rows[int(row["episode_id"])] = row
    return rows


def extract_frames(video_path: Path, frame_indices: list[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved = []
    wanted = set(frame_indices)
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx in wanted:
            out_path = out_dir / f"{video_path.stem}_frame_{idx:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append((idx, str(out_path)))
        idx += 1
        ok, frame = cap.read()
    cap.release()
    saved.sort(key=lambda x: x[0])
    return [path for _, path in saved], idx


def choose_window(frame_count: int, history_frames: int, mode: str):
    if frame_count <= 0:
        return []
    if mode == "final":
        start = max(0, frame_count - history_frames)
        end = frame_count
    else:
        end = min(frame_count, history_frames)
        start = 0
        if frame_count > history_frames * 2:
            end = history_frames
    return list(range(start, end))


def build_row(meta: dict, frame_paths: list[str], frame_index: int, frame_count: int, label: int, sample_type: str, video_path: Path):
    instruction = str(meta["episode_instruction"]).strip()
    stop_phrase = extract_stop_phrase(instruction)
    stop_object_phrase = extract_stop_object_phrase(instruction)
    usable_stop_object_phrase = is_usable_stop_object_phrase(stop_object_phrase)
    return {
        "scene_id": meta["scene_id"],
        "episode_id": int(meta["episode_id"]),
        "instruction": instruction,
        "stop_phrase": stop_phrase,
        "stop_object_phrase": stop_object_phrase,
        "usable_stop_object_phrase": usable_stop_object_phrase,
        "image_path": frame_paths[-1] if frame_paths else "",
        "source_tar": "",
        "member_name": str(video_path),
        "frame_index": int(frame_index),
        "frame_count": int(frame_count),
        "declared_length": int(meta.get("steps", frame_count)),
        "frame_paths": frame_paths,
        "history_frame_count": len(frame_paths),
        "label": int(label),
        "sample_type": sample_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", required=True)
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--history-frames", type=int, default=5)
    args = parser.parse_args()

    progress = read_progress_jsonl(Path(args.progress))
    videos_root = Path(args.videos_root)
    output_path = Path(args.output)
    frames_root = Path(args.frames_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for status_dir, label in [("success", 1), ("failed", 0)]:
        for video_path in sorted((videos_root / status_dir).glob("*/*.mp4")):
            episode_id = int(video_path.stem)
            meta = progress.get(episode_id)
            if meta is None:
                continue

            final_indices = choose_window(int(meta.get("steps", 0)) or 0, args.history_frames, "final")
            final_dir = frames_root / ("positive_final" if label == 1 else "negative_final_fail") / meta["scene_id"] / f"{episode_id:04d}"
            final_frame_paths, decoded_count = extract_frames(video_path, final_indices, final_dir)
            if final_frame_paths:
                rows.append(
                    build_row(
                        meta,
                        final_frame_paths,
                        frame_index=final_indices[-1],
                        frame_count=decoded_count,
                        label=label,
                        sample_type="positive_final" if label == 1 else "negative_final_fail",
                        video_path=video_path,
                    )
                )

            if label == 1:
                early_indices = choose_window(decoded_count, args.history_frames, "early")
                early_dir = frames_root / "negative_early_success" / meta["scene_id"] / f"{episode_id:04d}"
                early_frame_paths, _ = extract_frames(video_path, early_indices, early_dir)
                if early_frame_paths and early_indices != final_indices:
                    rows.append(
                        build_row(
                            meta,
                            early_frame_paths,
                            frame_index=early_indices[-1],
                            frame_count=decoded_count,
                            label=0,
                            sample_type="negative_early_success",
                            video_path=video_path,
                        )
                    )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(output_path),
        "count": len(rows),
        "pos": sum(int(r["label"]) for r in rows),
        "neg": len(rows) - sum(int(r["label"]) for r in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
