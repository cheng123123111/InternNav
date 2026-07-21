#!/usr/bin/env python3
"""Evaluate the independent SigLIP + Gemma navigation value model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "scripts" / "train" / "progress_value_models"
sys.path.insert(0, str(TRAIN_DIR))

from train_siglip_gemma_value import (  # noqa: E402
    ProgressCollator,
    ProgressRows,
    SiglipGemmaValueModel,
    move_batch,
)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 2:
        return math.nan
    pred_rank = rankdata(prediction)
    target_rank = rankdata(target)
    if pred_rank.std() == 0 or target_rank.std() == 0:
        return math.nan
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return math.nan
    ranks = rankdata(scores)
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    return {
        "n": int(len(prediction)),
        "mae": float(np.abs(error).mean()) if len(error) else math.nan,
        "rmse": float(np.sqrt(np.square(error).mean())) if len(error) else math.nan,
        "spearman": spearman(prediction, target),
    }


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    prediction = scores >= 0.5
    target = labels >= 0.5
    tp = int((prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {
        "n": int(len(scores)),
        "positive": int(target.sum()),
        "negative": int((~target).sum()),
        "accuracy": (tp + tn) / max(1, len(scores)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(1.0e-12, precision + recall),
        "auc": auc(scores, target),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SiglipGemmaValueModel(
        vision_model=str(package["vision_model"]),
        language_model=str(package["language_model"]),
        max_frames=int(package["max_frames"]),
        visual_grid=int(package["visual_grid"]),
        value_bins=int(package["value_bins"]),
        train_language=False,
    )
    missing, unexpected = model.load_state_dict(package["state_dict"], strict=False)
    bad_missing = [name for name in missing if not name.startswith("vision.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={bad_missing[:10]} unexpected={unexpected[:10]}"
        )

    device = torch.device("cuda")
    model.to(device).eval()
    dataset = ProgressRows(
        args.data,
        max_frames=int(package["max_frames"]),
        max_samples=args.max_samples,
    )
    collator = ProgressCollator(
        str(package["vision_model"]), str(package["language_model"])
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    pixel_values=batch["pixel_values"],
                    frame_counts=batch["frame_counts"],
                )
            values = output["value"].float().cpu().tolist()
            successes = torch.sigmoid(output["success_logits"].float()).cpu().tolist()
            for index, row_id in enumerate(batch["row_ids"]):
                predictions.append(
                    {
                        "row_id": row_id,
                        "value": values[index],
                        "success": successes[index],
                        "completion_target": float(batch["completion"][index].cpu()),
                        "success_target": float(batch["success"][index].cpu()),
                        "completion_mask": float(batch["completion_mask"][index].cpu()),
                        "success_mask": float(batch["success_mask"][index].cpu()),
                    }
                )

    completion_rows = [row for row in predictions if row["completion_mask"] > 0]
    success_rows = [row for row in predictions if row["success_mask"] > 0]
    metrics = {
        "n": len(predictions),
        "completion": regression_metrics(
            np.asarray([row["value"] for row in completion_rows]),
            np.asarray([row["completion_target"] for row in completion_rows]),
        ),
        "success": binary_metrics(
            np.asarray([row["success"] for row in success_rows]),
            np.asarray([row["success_target"] for row in success_rows]),
        ),
        "checkpoint": str(args.checkpoint),
        "jsonl": str(args.data),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
