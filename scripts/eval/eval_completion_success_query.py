#!/usr/bin/env python3
"""Evaluate the two-output completion/success Query head."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from internnav.dataset.internvla_n1_lerobot_dataset import (  # noqa: E402
    DataCollatorForSupervisedDataset,
    ProgressJsonlDataset,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import (  # noqa: E402
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.trainer.internvla_n1_argument import DataArguments, ModelArguments  # noqa: E402


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


def spearman(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return math.nan
    pred_rank = rankdata(pred)
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
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = pred - target
    return {
        "n": int(len(pred)),
        "mae": float(np.abs(error).mean()) if len(error) else math.nan,
        "rmse": float(np.sqrt(np.square(error).mean())) if len(error) else math.nan,
        "spearman": spearman(pred, target),
        "pred_mean": float(pred.mean()) if len(pred) else math.nan,
        "target_mean": float(target.mean()) if len(target) else math.nan,
    }


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = scores >= 0.5
    gold = labels >= 0.5
    tp = int((pred & gold).sum())
    tn = int((~pred & ~gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {
        "n": int(len(scores)),
        "positive": int(gold.sum()),
        "negative": int((~gold).sum()),
        "accuracy": (tp + tn) / max(1, len(scores)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(1.0e-12, precision + recall),
        "auc": auc(scores, gold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_head(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    package = torch.load(path, map_location="cpu")
    if bool(package.get("progress_aux_value_stream", False)):
        state = package.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("Value-stream checkpoint has no state_dict")
        missing, unexpected = model.load_state_dict(state, strict=False)
        prefix = "progress_value_stream."
        bad_missing = [key for key in missing if key.startswith(prefix)]
        bad_unexpected = [key for key in unexpected if key.startswith(prefix)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Value-stream mismatch: missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}"
            )
        return package
    if bool(package.get("progress_aux_in_qwen_queries", False)):
        state = package.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("In-Qwen progress checkpoint has no state_dict")
        missing, unexpected = model.load_state_dict(state, strict=False)
        prefixes = (
            "model.progress_latent_queries",
            "progress_in_qwen_temporal_conditioner.",
            "progress_in_qwen_readout.",
        )
        bad_missing = [key for key in missing if key.startswith(prefixes)]
        bad_unexpected = [key for key in unexpected if key.startswith(prefixes)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"In-Qwen query mismatch: missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}"
            )
        return package
    state = package.get("progress_aux_head", package)
    normalized = {
        key if key.startswith("progress_aux_head.") else f"progress_aux_head.{key}": value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    bad_missing = [key for key in missing if key.startswith("progress_aux_head.")]
    bad_unexpected = [key for key in unexpected if key.startswith("progress_aux_head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Head mismatch: missing={bad_missing[:5]} unexpected={bad_unexpected[:5]}"
        )
    return package


def evaluate_pairs(path: Path | None, predictions: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"n": 0, "accuracy": math.nan}
    good = 0
    total = 0
    by_type = Counter()
    good_by_type = Counter()
    by_failure_type = Counter()
    good_by_failure_type = Counter()
    by_failure_phase = Counter()
    good_by_failure_phase = Counter()
    by_target = Counter()
    good_by_target = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair = json.loads(line)
            preferred = predictions.get(str(pair.get("preferred_row_id")))
            rejected = predictions.get(str(pair.get("rejected_row_id")))
            if preferred is None or rejected is None:
                continue
            target = str(pair.get("ranking_target") or pair.get("pair_type") or "")
            if target == "segment_progress":
                dim = 3
                target_name = "segment_progress"
            elif target == "boundary":
                dim = 4
                target_name = "boundary"
            elif target == "query_global_completion":
                dim = 2
                target_name = "query_global_completion"
            elif "completion" in target:
                dim = 0
                target_name = "completion"
            else:
                dim = 1
                target_name = "success"
            correct = preferred[dim] > rejected[dim]
            pair_type = str(pair.get("pair_type") or target)
            failure_type = str(pair.get("failure_type") or "none")
            failure_phase = str(pair.get("failure_phase") or "none")
            total += 1
            good += int(correct)
            by_type[pair_type] += 1
            good_by_type[pair_type] += int(correct)
            by_failure_type[failure_type] += 1
            good_by_failure_type[failure_type] += int(correct)
            by_failure_phase[failure_phase] += 1
            good_by_failure_phase[failure_phase] += int(correct)
            by_target[target_name] += 1
            good_by_target[target_name] += int(correct)
    return {
        "n": total,
        "accuracy": good / max(1, total),
        "by_type": {
            key: {"n": value, "accuracy": good_by_type[key] / max(1, value)}
            for key, value in by_type.items()
        },
        "by_failure_type": {
            key: {"n": value, "accuracy": good_by_failure_type[key] / max(1, value)}
            for key, value in by_failure_type.items()
        },
        "by_failure_phase": {
            key: {"n": value, "accuracy": good_by_failure_phase[key] / max(1, value)}
            for key, value in by_failure_phase.items()
        },
        "by_target": {
            key: {"n": value, "accuracy": good_by_target[key] / max(1, value)}
            for key, value in by_target.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/vepfs-B区/vlnce/InternNav/checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--head-path", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--rebuild-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reconstruct causal history frames from the current-frame directory.",
    )
    parser.add_argument(
        "--disable-temporal-input",
        action="store_true",
        help="Evaluate without the auxiliary action/pose temporal token branch.",
    )
    parser.add_argument(
        "--temporal-ablation",
        choices=("none", "frame_index", "pose", "action", "context_order", "only_frame_index"),
        default="none",
        help="Zero selected temporal-token channels for leakage and reliance audits.",
    )
    parser.add_argument("--resize", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = torch.load(args.head_path, map_location="cpu")
    target_mode = str(package.get("progress_aux_target_mode", ""))
    if target_mode not in (
        "completion_success",
        "segment_completion_success",
        "gated_boundary_completion_success",
    ):
        raise ValueError(
            "The supplied checkpoint is not a completion/success Query head"
        )
    boundary_aware = target_mode in (
        "segment_completion_success",
        "gated_boundary_completion_success",
    )

    config = InternVLAN1ModelConfig.from_pretrained(args.base_model, trust_remote_code=True)
    config.enable_progress_aux = True
    config.progress_aux_target_mode = target_mode
    config.progress_aux_completion_mode = str(
        package.get("progress_aux_completion_mode", "scalar")
    )
    config.progress_aux_dim = int(package.get("progress_aux_dim", 2))
    config.progress_aux_num_queries = int(package.get("progress_aux_num_queries", 4))
    config.progress_aux_in_qwen_queries = bool(
        package.get("progress_aux_in_qwen_queries", False)
    )
    config.progress_aux_value_stream = bool(
        package.get("progress_aux_value_stream", False)
    )
    config.progress_aux_value_stream_depth = int(
        package.get("progress_aux_value_stream_depth", 8) or 8
    )
    config.progress_aux_value_stream_dim = int(
        package.get("progress_aux_value_stream_dim", 512) or 512
    )
    config.progress_aux_value_stream_ffn_dim = int(
        package.get("progress_aux_value_stream_ffn_dim", 2048) or 2048
    )
    config.progress_aux_value_stream_heads = int(
        package.get("progress_aux_value_stream_heads", 8) or 8
    )
    config.progress_aux_query_specific_readout = bool(
        package.get("progress_aux_query_specific_readout", False)
    )
    config.progress_aux_gated_aux_fusion = bool(
        package.get("progress_aux_gated_aux_fusion", False)
    )
    config.progress_aux_in_qwen_temporal_dim = int(
        package.get("progress_aux_in_qwen_temporal_dim", 256) or 256
    )
    config.progress_aux_in_qwen_readout_dim = int(
        package.get("progress_aux_in_qwen_readout_dim", 512) or 512
    )
    config.progress_aux_temporal_input_dim = int(package.get("progress_aux_temporal_input_dim", 16))
    config.progress_aux_temporal_max_tokens = int(
        package.get("progress_aux_temporal_max_tokens", args.max_frames)
    )
    config.progress_aux_temporal_layers = int(package.get("progress_aux_temporal_layers", 1))
    config.progress_aux_causal_feature_dim = int(
        package.get("progress_aux_causal_feature_dim", 0) or 0
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation=args.attn_implementation,
    )
    model_args = ModelArguments(
        model_name_or_path=args.base_model,
        system1=getattr(config, "system1", "nextdit_async"),
        n_query=int(getattr(config, "n_query", 4)),
    )
    base_model = model.get_model()
    if getattr(base_model, "latent_queries", None) is None:
        base_model.initialize_vision_modules(model_args=model_args)
    load_head(model, args.head_path)
    if config.progress_aux_in_qwen_queries:
        if model.progress_in_qwen_temporal_conditioner is not None:
            model.progress_in_qwen_temporal_conditioner.float()
        model.progress_in_qwen_readout.float()
    elif config.progress_aux_value_stream:
        model.progress_value_stream.float()
    else:
        model.progress_aux_head.float()
    model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
    )
    data_args = DataArguments(
        progress_jsonl_use=str(args.jsonl),
        progress_jsonl_max_frames=args.max_frames,
        progress_jsonl_max_samples=args.max_samples,
        progress_jsonl_rebuild_history=args.rebuild_history,
        progress_jsonl_include_action_history=bool(
            package.get("progress_jsonl_include_action_history", False)
        ),
        progress_aux_temporal_tokens=not args.disable_temporal_input,
        progress_aux_causal_prior=config.progress_aux_causal_feature_dim > 0,
        progress_aux_target_mode=target_mode,
        progress_aux_num_queries=config.progress_aux_num_queries,
        progress_aux_in_qwen_queries=config.progress_aux_in_qwen_queries,
        progress_aux_value_stream=config.progress_aux_value_stream,
        enable_progress_aux=True,
        resize_h=args.resize,
        resize_w=args.resize,
    )
    data_args.model_type = "internvla-n1"
    data_args.image_processor = AutoProcessor.from_pretrained(args.base_model).image_processor
    data_args.transform_train = v2.Resize((args.resize, args.resize))
    dataset = ProgressJsonlDataset(tokenizer=tokenizer, data_args=data_args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            progress_query_token_length=(
                config.progress_aux_num_queries if config.progress_aux_in_qwen_queries else 0
            ),
        ),
        pin_memory=device.type == "cuda",
    )

    all_pred = []
    all_query_pred = []
    all_pred_stage = []
    all_ordinal_probs = []
    all_stage_class_probs = []
    all_label = []
    all_mask = []
    rows = []
    cursor = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            batch_size = int(batch["input_ids"].shape[0])
            rows.extend(dataset.rows[cursor : cursor + batch_size])
            cursor += batch_size
            batch = move_batch(batch, device)
            temporal_tokens = batch.get("progress_temporal_tokens")
            if temporal_tokens is not None and args.temporal_ablation != "none":
                temporal_tokens = temporal_tokens.clone()
                channel_groups = {
                    "pose": tuple(range(0, 6)),
                    "frame_index": (6, 7),
                    "action": tuple(range(8, 15)),
                    "context_order": (15,),
                }
                if args.temporal_ablation == "only_frame_index":
                    keep = set(channel_groups["frame_index"])
                    drop = [index for index in range(temporal_tokens.shape[-1]) if index not in keep]
                else:
                    drop = list(channel_groups[args.temporal_ablation])
                temporal_tokens[..., drop] = 0.0
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch.get("position_ids"),
                labels=batch["labels"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                t_s_pos=batch.get("t_s_pos"),
                progress_s_pos=batch.get("progress_s_pos"),
                progress_temporal_tokens=temporal_tokens,
                progress_temporal_mask=batch.get("progress_temporal_mask"),
                progress_causal_features=batch.get("progress_causal_features"),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                skip_lm_head=True,
            )
            if config.progress_aux_in_qwen_queries or config.progress_aux_value_stream:
                logits = outputs.progress_aux_logits
                query_logits = outputs.progress_aux_query_logits
            else:
                hidden = outputs.hidden_states[-1]
                memory_mask = model._build_progress_aux_memory_mask(
                    hidden_states=hidden,
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                logits, query_logits = model.progress_aux_head(
                    hidden,
                    attention_mask=batch["attention_mask"],
                    memory_mask=memory_mask,
                    temporal_tokens=temporal_tokens,
                    temporal_mask=batch.get("progress_temporal_mask"),
                    causal_features=batch.get("progress_causal_features"),
                )
            batch_pred = torch.sigmoid(logits[:, :2])
            if config.progress_aux_completion_mode == "categorical4":
                if query_logits.shape[-1] < 4:
                    raise ValueError("categorical4 checkpoint requires four progress queries")
                stage_probs = torch.softmax(query_logits[:, :4], dim=1)
                stage_values = torch.arange(
                    4, device=stage_probs.device, dtype=stage_probs.dtype
                ) / 3.0
                batch_pred = batch_pred.clone()
                batch_pred[:, 0] = (stage_probs * stage_values[None, :]).sum(dim=1)
                all_pred_stage.append(stage_probs.argmax(dim=1).cpu().numpy())
                all_stage_class_probs.append(stage_probs.cpu().numpy())
                all_ordinal_probs.append(
                    np.full((batch_size, 3), np.nan, dtype=np.float32)
                )
            elif config.progress_aux_completion_mode == "ordinal4":
                if query_logits.shape[-1] < 3:
                    raise ValueError("ordinal4 checkpoint requires at least three progress queries")
                ordinal_probs = torch.sigmoid(query_logits[:, :3])
                batch_pred = batch_pred.clone()
                batch_pred[:, 0] = ordinal_probs.mean(dim=1)
                all_pred_stage.append((ordinal_probs >= 0.5).sum(dim=1).cpu().numpy())
                all_ordinal_probs.append(ordinal_probs.cpu().numpy())
                all_stage_class_probs.append(
                    np.full((batch_size, 4), np.nan, dtype=np.float32)
                )
            else:
                all_pred_stage.append(np.full(batch_size, -1, dtype=np.int64))
                all_ordinal_probs.append(
                    np.full((batch_size, 3), np.nan, dtype=np.float32)
                )
                all_stage_class_probs.append(
                    np.full((batch_size, 4), np.nan, dtype=np.float32)
                )
            all_pred.append(batch_pred.cpu().numpy())
            all_query_pred.append(torch.sigmoid(query_logits[:, :4]).cpu().numpy())
            all_label.append(batch["progress_aux_labels"].float().cpu().numpy())
            all_mask.append(batch["progress_aux_mask"].float().cpu().numpy())

    pred = np.concatenate(all_pred)
    query_pred = np.concatenate(all_query_pred)
    pred_stage = np.concatenate(all_pred_stage)
    ordinal_probs = np.concatenate(all_ordinal_probs)
    stage_class_probs = np.concatenate(all_stage_class_probs)
    label = np.concatenate(all_label)
    mask = np.concatenate(all_mask)
    completion_valid = mask[:, 0] > 0
    success_valid = mask[:, 1] > 0
    strict_success = success_valid & ((label[:, 1] <= 0.25) | (label[:, 1] >= 0.75))
    supervision_rows = [row.get("query_supervision") or {} for row in rows]
    failure_phase = np.asarray(
        [str(item.get("failure_phase") or "unknown") for item in supervision_rows]
    )
    refined_failure_type = np.asarray(
        [
            str(item.get("refined_failure_type") or item.get("primary_failure_type") or "unknown")
            for item in supervision_rows
        ]
    )
    failure_binary = np.asarray(
        [float(item.get("failure_binary_target", 0.0)) for item in supervision_rows],
        dtype=np.float32,
    )
    failure_eval = np.asarray(
        [float(item.get("failure_eval_mask", 0.0)) > 0 for item in supervision_rows],
        dtype=bool,
    )
    clean_completion = completion_valid & np.isin(failure_phase, ["clean", "recovery"])
    failed_completion = completion_valid & np.char.startswith(failure_phase.astype(str), "failure_")
    trusted_target = np.asarray(
        [float((row.get("query_supervision") or {}).get("trusted_completion_target", 0.0)) for row in rows]
    )
    trusted_pred = pred[:, 0] * pred[:, 1]
    landmark_aux = np.asarray(
        [bool(item.get("landmark_aux_used", False)) for item in supervision_rows],
        dtype=bool,
    )
    metrics = {
        "n": int(len(rows)),
        "completion": regression_metrics(pred[completion_valid, 0], label[completion_valid, 0]),
        "completion_clean": regression_metrics(
            pred[clean_completion, 0], label[clean_completion, 0]
        ),
        "completion_failed": regression_metrics(
            pred[failed_completion, 0], label[failed_completion, 0]
        ),
        "completion_clean_landmark_aux": regression_metrics(
            pred[clean_completion & landmark_aux, 0],
            label[clean_completion & landmark_aux, 0],
        ),
        "completion_clean_without_landmark_aux": regression_metrics(
            pred[clean_completion & ~landmark_aux, 0],
            label[clean_completion & ~landmark_aux, 0],
        ),
        "route_health": regression_metrics(
            pred[success_valid, 1], label[success_valid, 1]
        ),
        "success_all": binary_metrics(pred[success_valid, 1], label[success_valid, 1]),
        "success_strict": binary_metrics(pred[strict_success, 1], label[strict_success, 1]),
        "trusted_completion": regression_metrics(
            trusted_pred[success_valid], trusted_target[success_valid]
        ),
        "baseline": {
            "completion_constant_0_5_mae": float(
                np.abs(0.5 - label[completion_valid, 0]).mean()
            ) if bool(completion_valid.any()) else math.nan,
            "success_constant_0_5_balanced_accuracy": 0.5,
        },
        "by_failure_type": {},
        "by_phase": {},
        "by_route_state": {},
        "by_primary_failure_type": {},
        "by_failure_phase": {},
        "by_refined_failure_type": {},
        "failure_detection_by_type": {},
        "failure_detection_by_phase": {},
        "phase_summary": {},
    }
    if bool(failure_eval.any()):
        metrics["failure_detection"] = binary_metrics(
            1.0 - pred[failure_eval, 1], failure_binary[failure_eval]
        )
    else:
        metrics["failure_detection"] = binary_metrics(
            np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
        )

    if boundary_aware:
        segment_valid = mask[:, 2] > 0
        boundary_valid = mask[:, 3] > 0
        metrics["query_diagnostics"] = {
            "global_completion": regression_metrics(
                query_pred[completion_valid, 0], label[completion_valid, 0]
            ),
            "segment_progress": regression_metrics(
                query_pred[segment_valid, 1], label[segment_valid, 2]
            ),
            "boundary": binary_metrics(
                query_pred[boundary_valid, 2], label[boundary_valid, 3]
            ),
            "route_health": binary_metrics(
                query_pred[success_valid, 3], label[success_valid, 1]
            ),
        }

    for phase in sorted(set(failure_phase.tolist())):
        phase_mask = failure_phase == phase
        phase_completion = phase_mask & completion_valid
        phase_health = phase_mask & success_valid
        metrics["phase_summary"][phase] = {
            "n": int(phase_mask.sum()),
            "completion_pred_mean": float(pred[phase_completion, 0].mean())
            if bool(phase_completion.any())
            else math.nan,
            "completion_target_mean": float(label[phase_completion, 0].mean())
            if bool(phase_completion.any())
            else math.nan,
            "health_pred_mean": float(pred[phase_health, 1].mean())
            if bool(phase_health.any())
            else math.nan,
            "health_target_mean": float(label[phase_health, 1].mean())
            if bool(phase_health.any())
            else math.nan,
            "failure_pred_mean": float((1.0 - pred[phase_health, 1]).mean())
            if bool(phase_health.any())
            else math.nan,
        }

    clean_failure_eval = failure_eval & (failure_binary < 0.5)
    for name in sorted(set(refined_failure_type.tolist())):
        typed_failure = failure_eval & (failure_binary >= 0.5) & (refined_failure_type == name)
        typed_eval = clean_failure_eval | typed_failure
        if not bool(typed_failure.any()) or not bool(clean_failure_eval.any()):
            continue
        metrics["failure_detection_by_type"][name] = binary_metrics(
            1.0 - pred[typed_eval, 1], failure_binary[typed_eval]
        )
    for phase in sorted(set(failure_phase.tolist())):
        phased_failure = (
            failure_eval
            & (failure_binary >= 0.5)
            & (failure_phase == phase)
        )
        phased_eval = clean_failure_eval | phased_failure
        if not bool(phased_failure.any()) or not bool(clean_failure_eval.any()):
            continue
        metrics["failure_detection_by_phase"][phase] = binary_metrics(
            1.0 - pred[phased_eval, 1], failure_binary[phased_eval]
        )

    predictions_by_id = {}
    detailed = []
    for index, row in enumerate(rows):
        supervision = row.get("query_supervision") or {}
        rid = str(row.get("row_id"))
        predictions_by_id[rid] = (
            float(pred[index, 0]),
            float(pred[index, 1]),
            float(query_pred[index, 0]),
            float(query_pred[index, 1]),
            float(query_pred[index, 2]),
            float(query_pred[index, 3]),
        )
        detailed.append(
            {
                "row_id": rid,
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "pilot_group": row.get("pilot_group"),
                "trajectory_id": supervision.get("trajectory_id"),
                "failure_type": supervision.get("failure_type"),
                "failure_phase": supervision.get("failure_phase"),
                "route_state": supervision.get("route_state"),
                "primary_failure_type": supervision.get("primary_failure_type"),
                "refined_failure_type": supervision.get("refined_failure_type"),
                "failure_binary_target": supervision.get("failure_binary_target"),
                "failure_eval_mask": supervision.get("failure_eval_mask"),
                "landmark_aux_used": supervision.get("landmark_aux_used"),
                "coarse_target": supervision.get("coarse_target"),
                "coarse_label_source": supervision.get("coarse_label_source"),
                "coarse_stage": supervision.get("coarse_stage"),
                "pred_completion": float(pred[index, 0]),
                "pred_stage": int(pred_stage[index]) if pred_stage[index] >= 0 else None,
                "ordinal_threshold_probs": (
                    [float(value) for value in ordinal_probs[index]]
                    if np.isfinite(ordinal_probs[index]).all()
                    else None
                ),
                "stage_class_probs": (
                    [float(value) for value in stage_class_probs[index]]
                    if np.isfinite(stage_class_probs[index]).all()
                    else None
                ),
                "target_completion": float(label[index, 0]),
                "completion_mask": float(mask[index, 0]),
                "pred_success": float(pred[index, 1]),
                "target_success": float(label[index, 1]),
                "success_mask": float(mask[index, 1]),
                "query_global_completion": float(query_pred[index, 0]),
                "query_segment_progress": (
                    float(query_pred[index, 1])
                    if boundary_aware
                    else None
                ),
                "query_boundary": (
                    float(query_pred[index, 2])
                    if boundary_aware
                    else None
                ),
                "query_route_health": (
                    float(query_pred[index, 3])
                    if boundary_aware
                    else None
                ),
                "target_segment_progress": (
                    float(label[index, 2])
                    if boundary_aware
                    else None
                ),
                "segment_progress_mask": (
                    float(mask[index, 2])
                    if boundary_aware
                    else None
                ),
                "target_boundary": (
                    float(label[index, 3])
                    if boundary_aware
                    else None
                ),
                "boundary_mask": (
                    float(mask[index, 3])
                    if boundary_aware
                    else None
                ),
                "pred_trusted_completion": float(trusted_pred[index]),
                "target_trusted_completion": float(trusted_target[index]),
            }
        )

    for group_key, output_key in (
        ("failure_type", "by_failure_type"),
        ("failure_phase", "by_phase"),
        ("route_state", "by_route_state"),
        ("primary_failure_type", "by_primary_failure_type"),
        ("failure_phase", "by_failure_phase"),
        ("refined_failure_type", "by_refined_failure_type"),
    ):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            value = str((row.get("query_supervision") or {}).get(group_key, "unknown"))
            groups[value].append(index)
        for name, indices in sorted(groups.items()):
            idx = np.asarray(indices, dtype=np.int64)
            cur_completion = idx[completion_valid[idx]]
            cur_success = idx[strict_success[idx]]
            metrics[output_key][name] = {
                "n": len(indices),
                "completion": regression_metrics(
                    pred[cur_completion, 0], label[cur_completion, 0]
                )
                if len(cur_completion)
                else None,
                "success_strict": binary_metrics(
                    pred[cur_success, 1], label[cur_success, 1]
                )
                if len(cur_success)
                else None,
            }

    metrics["pair_ranking"] = evaluate_pairs(args.pairs, predictions_by_id)
    metrics["checkpoint"] = {
        "base_model": args.base_model,
        "head_path": str(args.head_path),
        "metadata": {
            key: package.get(key)
            for key in (
                "progress_aux_arch",
                "progress_aux_target_mode",
                "progress_aux_completion_mode",
                "progress_aux_dim",
                "progress_aux_num_queries",
                "progress_aux_causal_feature_dim",
                "progress_aux_query_specific_readout",
                "progress_aux_gated_aux_fusion",
            )
        },
    }
    checkpoint_state = package.get("state_dict") or {}
    aux_gate = checkpoint_state.get("progress_in_qwen_readout.aux_gate")
    if aux_gate is not None:
        metrics["checkpoint"]["aux_gate"] = float(aux_gate.reshape(-1)[0])
        metrics["checkpoint"]["aux_gate_tanh"] = float(
            torch.tanh(aux_gate.float().reshape(-1)[0])
        )
    metrics["jsonl"] = str(args.jsonl)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in detailed:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
