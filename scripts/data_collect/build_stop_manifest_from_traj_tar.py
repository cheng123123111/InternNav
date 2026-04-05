import argparse
import ast
import json
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.data_collect.stop_alignment_utils import (
    extract_stop_object_phrase,
    extract_stop_phrase,
    is_usable_stop_object_phrase,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stop-alignment manifest from VLN-CE trajectory tarballs."
    )
    parser.add_argument(
        "--traj-root",
        required=True,
        help="Root directory containing scene tarballs, e.g. /dataset-vln/vln_ce/traj_data/r2r_v1-3",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest jsonl path.",
    )
    parser.add_argument(
        "--frames-root",
        default="",
        help="Optional output directory for extracted frame images.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=-1,
        help="Limit the number of scene tarballs processed.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=-1,
        help="Limit the number of episodes processed across all tarballs.",
    )
    parser.add_argument(
        "--negative-offsets",
        default="8,16",
        help="Comma-separated frame offsets before the last frame to use as negatives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite extracted frame images if they already exist.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=1,
        help="Number of recent frames to keep for each sample. Use 5 for short history aggregation.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write manifest metadata and tar-backed frame references. Do not extract image files.",
    )
    return parser.parse_args()


def parse_instruction(task_payload: str) -> str:
    try:
        parsed = ast.literal_eval(task_payload)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except (ValueError, SyntaxError):
        pass
    return str(task_payload).strip()


def load_single_jsonl_member(tar: tarfile.TarFile, member_name: str) -> dict:
    with tar.extractfile(member_name) as f:
        line = f.readline().decode("utf-8").strip()
    return json.loads(line) if line else {}


def has_member(tar: tarfile.TarFile, member_name: str) -> bool:
    try:
        tar.getmember(member_name)
        return True
    except KeyError:
        return False


def list_rgb_members(tar: tarfile.TarFile, episode_root: str) -> list[str]:
    prefix = f"{episode_root}/videos/chunk-000/observation.images.rgb/"
    members = [
        m.name
        for m in tar.getmembers()
        if m.isfile() and m.name.startswith(prefix) and m.name.lower().endswith(".jpg")
    ]
    return sorted(members, key=lambda name: int(Path(name).stem))


def save_member_bytes(tar: tarfile.TarFile, member_name: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tar.extractfile(member_name) as src, output_path.open("wb") as dst:
        dst.write(src.read())


def make_frame_ref(tar_path: Path, member_name: str) -> str:
    return f"{tar_path}:::{member_name}"


def build_history_frame_paths(
    tar: tarfile.TarFile,
    tar_path: Path,
    rgb_members: list[str],
    target_idx: int,
    scene_id: str,
    episode_id: str,
    label_dir: str,
    frames_root: Path,
    history_frames: int,
    overwrite: bool,
    manifest_only: bool,
) -> list[str]:
    start_idx = max(0, target_idx - history_frames + 1)
    history_paths = []
    for idx in range(start_idx, target_idx + 1):
        member_name = rgb_members[idx]
        if manifest_only:
            history_paths.append(make_frame_ref(tar_path, member_name))
        else:
            history_path = frames_root / label_dir / scene_id / f"{episode_id}_frame_{idx:03d}.jpg"
            save_member_bytes(tar, member_name, history_path, overwrite)
            history_paths.append(str(history_path))
    return history_paths


def iter_episode_roots(tar: tarfile.TarFile) -> list[str]:
    roots = set()
    for member in tar.getmembers():
        path = Path(member.name)
        if len(path.parts) >= 3 and path.parts[2] == "meta":
            roots.add("/".join(path.parts[:2]))
    return sorted(roots)


def build_rows_for_episode(
    tar: tarfile.TarFile,
    tar_path: Path,
    episode_root: str,
    frames_root: Path,
    negative_offsets: list[int],
    overwrite: bool,
    history_frames: int,
    manifest_only: bool,
) -> list[dict]:
    tasks_name = f"{episode_root}/meta/tasks.jsonl"
    episodes_name = f"{episode_root}/meta/episodes.jsonl"
    if not has_member(tar, tasks_name) or not has_member(tar, episodes_name):
        return []

    tasks = load_single_jsonl_member(tar, tasks_name)
    episodes = load_single_jsonl_member(tar, episodes_name)
    rgb_members = list_rgb_members(tar, episode_root)
    if not rgb_members:
        return []

    scene_id, episode_id = episode_root.split("/")[:2]
    instruction = parse_instruction(tasks.get("task", ""))
    stop_phrase = extract_stop_phrase(instruction)
    stop_object_phrase = extract_stop_object_phrase(instruction)
    usable_stop_object_phrase = is_usable_stop_object_phrase(stop_object_phrase)
    frame_count = len(rgb_members)
    declared_length = int(episodes.get("length", frame_count))

    rows = []
    positive_idx = frame_count - 1
    positive_path = frames_root / "positive" / scene_id / f"{episode_id}_last.jpg"
    positive_ref = make_frame_ref(tar_path, rgb_members[positive_idx])
    if not manifest_only:
        save_member_bytes(tar, rgb_members[positive_idx], positive_path, overwrite)
    positive_history_paths = build_history_frame_paths(
        tar=tar,
        tar_path=tar_path,
        rgb_members=rgb_members,
        target_idx=positive_idx,
        scene_id=scene_id,
        episode_id=episode_id,
        label_dir="positive_history",
        frames_root=frames_root,
        history_frames=history_frames,
        overwrite=overwrite,
        manifest_only=manifest_only,
    )
    rows.append(
        {
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "instruction": instruction,
            "stop_phrase": stop_phrase,
            "stop_object_phrase": stop_object_phrase,
            "usable_stop_object_phrase": usable_stop_object_phrase,
            "image_path": positive_ref if manifest_only else str(positive_path),
            "source_tar": str(tar_path),
            "member_name": rgb_members[positive_idx],
            "frame_index": positive_idx,
            "frame_count": frame_count,
            "declared_length": declared_length,
            "frame_paths": positive_history_paths,
            "history_frame_count": len(positive_history_paths),
            "label": 1,
            "sample_type": "last_frame_positive",
        }
    )

    used_negative_indices = set()
    for offset in negative_offsets:
        neg_idx = positive_idx - offset
        if neg_idx < 0 or neg_idx in used_negative_indices:
            continue
        used_negative_indices.add(neg_idx)
        negative_path = frames_root / "negative" / scene_id / f"{episode_id}_frame_{neg_idx:03d}.jpg"
        negative_ref = make_frame_ref(tar_path, rgb_members[neg_idx])
        if not manifest_only:
            save_member_bytes(tar, rgb_members[neg_idx], negative_path, overwrite)
        negative_history_paths = build_history_frame_paths(
            tar=tar,
            tar_path=tar_path,
            rgb_members=rgb_members,
            target_idx=neg_idx,
            scene_id=scene_id,
            episode_id=episode_id,
            label_dir="negative_history",
            frames_root=frames_root,
            history_frames=history_frames,
            overwrite=overwrite,
            manifest_only=manifest_only,
        )
        rows.append(
            {
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "instruction": instruction,
                "stop_phrase": stop_phrase,
                "stop_object_phrase": stop_object_phrase,
                "usable_stop_object_phrase": usable_stop_object_phrase,
                "image_path": negative_ref if manifest_only else str(negative_path),
                "source_tar": str(tar_path),
                "member_name": rgb_members[neg_idx],
                "frame_index": neg_idx,
                "frame_count": frame_count,
                "declared_length": declared_length,
                "frame_paths": negative_history_paths,
                "history_frame_count": len(negative_history_paths),
                "label": 0,
                "sample_type": f"same_traj_negative_offset_{offset}",
            }
        )

    return rows


def main():
    args = parse_args()
    traj_root = Path(args.traj_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root) if args.frames_root else output_path.parent / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    negative_offsets = [int(x) for x in args.negative_offsets.split(",") if x.strip()]
    tar_paths = sorted(traj_root.glob("*.tar.gz"))
    if args.max_scenes > 0:
        tar_paths = tar_paths[: args.max_scenes]

    total_rows = 0
    total_episodes = 0
    positives = 0
    negatives = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r:gz") as tar:
                for episode_root in iter_episode_roots(tar):
                    rows = build_rows_for_episode(
                        tar=tar,
                        tar_path=tar_path,
                        episode_root=episode_root,
                        frames_root=frames_root,
                        negative_offsets=negative_offsets,
                        overwrite=args.overwrite,
                        history_frames=args.history_frames,
                        manifest_only=args.manifest_only,
                    )
                    if not rows:
                        continue
                    for row in rows:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_rows += 1
                        positives += int(row["label"] == 1)
                        negatives += int(row["label"] == 0)
                    total_episodes += 1
                    if args.max_episodes > 0 and total_episodes >= args.max_episodes:
                        break
            if args.max_episodes > 0 and total_episodes >= args.max_episodes:
                break

    print(
        json.dumps(
            {
                "traj_root": str(traj_root),
                "manifest": str(output_path),
                "frames_root": str(frames_root),
                "episodes": total_episodes,
                "samples": total_rows,
                "positives": positives,
                "negatives": negatives,
                "negative_offsets": negative_offsets,
                "history_frames": args.history_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
