#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def load_progress(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[(str(row["scene_id"]), int(row["episode_id"]))] = row
    return rows


def load_verify_rows(online_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(online_dir / "stop_residual_verify_rank*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def mean(rows: list[dict], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        default="/vepfs-C/qiancheng/native_progress_unseen100_same_as_viva_20260629",
    )
    parser.add_argument(
        "--online-dir",
        default="/vepfs-C/qiancheng/stop_residual_online_unseen100_same_as_viva_20260629",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    online_dir = Path(args.online_dir)
    base = load_progress(baseline_dir / "progress.json")
    online = load_progress(online_dir / "progress.json")
    keys = sorted(set(base) & set(online))

    compare_path = online_dir / "paired_compare.csv"
    with compare_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "episode_id",
                "baseline_success",
                "online_success",
                "baseline_spl",
                "online_spl",
                "baseline_os",
                "online_os",
                "baseline_ne",
                "online_ne",
                "baseline_steps",
                "online_steps",
                "status",
                "instruction",
            ],
        )
        writer.writeheader()
        for key in keys:
            b = base[key]
            o = online[key]
            bs = float(b.get("success", 0.0))
            os_ = float(o.get("success", 0.0))
            if bs < 0.5 <= os_:
                status = "rescued"
            elif bs >= 0.5 > os_:
                status = "harmed"
            elif os_ >= 0.5:
                status = "both_success"
            else:
                status = "both_fail"
            writer.writerow(
                {
                    "scene_id": key[0],
                    "episode_id": key[1],
                    "baseline_success": bs,
                    "online_success": os_,
                    "baseline_spl": float(b.get("spl", 0.0)),
                    "online_spl": float(o.get("spl", 0.0)),
                    "baseline_os": float(b.get("os", 0.0)),
                    "online_os": float(o.get("os", 0.0)),
                    "baseline_ne": float(b.get("ne", 0.0)),
                    "online_ne": float(o.get("ne", 0.0)),
                    "baseline_steps": int(b.get("steps", -1)),
                    "online_steps": int(o.get("steps", -1)),
                    "status": status,
                    "instruction": o.get("episode_instruction", b.get("episode_instruction", "")),
                }
            )

    verify = load_verify_rows(online_dir)
    rejected = [row for row in verify if not bool(row.get("accept"))]
    accepted = [row for row in verify if bool(row.get("accept"))]
    near_rejected = [
        row for row in rejected
        if row.get("distance_to_goal") is not None and float(row["distance_to_goal"]) <= 3.0
    ]
    far_rejected = [
        row for row in rejected
        if row.get("distance_to_goal") is not None and float(row["distance_to_goal"]) > 3.0
    ]

    statuses = {}
    for key in keys:
        b = float(base[key].get("success", 0.0))
        o = float(online[key].get("success", 0.0))
        if b < 0.5 <= o:
            status = "rescued"
        elif b >= 0.5 > o:
            status = "harmed"
        elif o >= 0.5:
            status = "both_success"
        else:
            status = "both_fail"
        statuses[status] = statuses.get(status, 0) + 1

    summary = {
        "baseline_progress": str(baseline_dir / "progress.json"),
        "online_progress": str(online_dir / "progress.json"),
        "paired_compare": str(compare_path),
        "episodes_compared": len(keys),
        "baseline": {
            "sr": mean(list(base.values()), "success"),
            "spl": mean(list(base.values()), "spl"),
            "os": mean(list(base.values()), "os"),
            "ne": mean(list(base.values()), "ne"),
            "episodes": len(base),
        },
        "online": {
            "sr": mean(list(online.values()), "success"),
            "spl": mean(list(online.values()), "spl"),
            "os": mean(list(online.values()), "os"),
            "ne": mean(list(online.values()), "ne"),
            "episodes": len(online),
        },
        "paired_status": statuses,
        "stop_residual_verify": {
            "total": len(verify),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejected_far_wrong_distance_gt_3m": len(far_rejected),
            "rejected_near_possible_mistake_distance_le_3m": len(near_rejected),
            "errors": sum(1 for row in verify if row.get("error")),
            "accepted_final_prob_mean": mean(accepted, "final_prob"),
            "rejected_final_prob_mean": mean(rejected, "final_prob"),
            "accepted_distance_mean": mean(accepted, "distance_to_goal"),
            "rejected_distance_mean": mean(rejected, "distance_to_goal"),
        },
    }
    summary_path = online_dir / "online_stop_residual_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
