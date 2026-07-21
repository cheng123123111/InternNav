#!/usr/bin/env python3
"""Train an independent SigLIP-So400M + Gemma-3-270M navigation value model."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel


def clip01(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=4096)
def frame_candidates(frame_dir: str, base_prefix: str, suffix: str) -> tuple[tuple[int, str], ...]:
    pattern = re.compile(rf"^{re.escape(base_prefix)}(\d+){re.escape(suffix)}$")
    candidates = []
    for name in os.listdir(frame_dir):
        frame_match = pattern.match(name)
        if frame_match:
            candidates.append((int(frame_match.group(1)), os.path.join(frame_dir, name)))
    candidates.sort()
    return tuple(candidates)


def rebuild_history(image_paths: list[str], max_frames: int) -> list[str]:
    if not image_paths or len(image_paths) >= max_frames:
        return image_paths[-max_frames:]
    current = str(image_paths[-1])
    match = re.match(r"^(.*frame_)(\d+)(\.[^./]+)$", current)
    if not match or ":::" in current:
        return image_paths[-max_frames:]
    current_idx = int(match.group(2))
    frame_dir = os.path.dirname(current)
    base_prefix = os.path.basename(match.group(1))
    suffix = match.group(3)
    if current_idx <= 0 or not os.path.isdir(frame_dir):
        return image_paths[-max_frames:]
    candidates = [
        item
        for item in frame_candidates(frame_dir, base_prefix, suffix)
        if item[0] <= current_idx
    ]
    history = [item for item in candidates if item[0] < current_idx]
    count = min(max_frames - 1, len(history))
    if count <= 0:
        return image_paths[-max_frames:]
    positions = np.unique(np.linspace(0, len(history) - 1, count, dtype=np.int32))
    rebuilt = [history[int(index)][1] for index in positions] + [current]
    return rebuilt[-max_frames:]


class ProgressRows(Dataset):
    def __init__(self, path: Path, max_frames: int = 8, max_samples: int = 0):
        with path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_paths = row.get("image_paths") or row.get("frame_paths") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        preserve_explicit_history = bool(
            row.get("preserve_explicit_history")
            or (row.get("model_inputs") or {}).get("preserve_explicit_history")
        )
        image_paths = list(image_paths)
        if preserve_explicit_history:
            image_paths = image_paths[-self.max_frames :]
        else:
            image_paths = rebuild_history(image_paths, self.max_frames)
        labels = row.get("progress_labels") or {}
        completion = clip01(labels.get("completion_target"))
        success = clip01(labels.get("success_target"))
        completion_mask = clip01(labels.get("completion_mask"), 1.0)
        success_mask = clip01(labels.get("success_mask"), 1.0)
        weight = max(0.0, float(labels.get("sample_weight", 1.0) or 1.0))
        return {
            "row_id": str(row.get("row_id", index)),
            "instruction": str(row.get("instruction") or ""),
            "image_paths": image_paths,
            "completion": completion,
            "success": success,
            "completion_mask": completion_mask,
            "success_mask": success_mask,
            "weight": weight,
        }


class BlockDistributedSampler(Sampler[int]):
    """Shard contiguous ranking blocks without splitting a block across ranks."""

    def __init__(self, size: int, block_size: int, rank: int, world_size: int):
        if size % block_size != 0:
            raise ValueError(f"Dataset size {size} is not divisible by block size {block_size}")
        self.blocks = list(range(size // block_size))[rank::world_size]
        self.block_size = int(block_size)

    def __iter__(self) -> Iterable[int]:
        for block in self.blocks:
            start = block * self.block_size
            yield from range(start, start + self.block_size)

    def __len__(self) -> int:
        return len(self.blocks) * self.block_size


class ProgressCollator:
    def __init__(self, vision_model: str, language_model: str):
        self.image_processor = SiglipImageProcessor.from_pretrained(vision_model)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        flat_images = []
        frame_counts = []
        for sample in samples:
            images = [Image.open(path).convert("RGB") for path in sample["image_paths"]]
            if not images:
                raise ValueError(f"No image for row {sample['row_id']}")
            flat_images.extend(images)
            frame_counts.append(len(images))
        pixels = self.image_processor(images=flat_images, return_tensors="pt")["pixel_values"]
        prompts = [
            (
                "Navigation instruction: "
                f"{sample['instruction']}\n"
                "Use the ordered visual observations to evaluate current task progress."
            )
            for sample in samples
        ]
        text = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        return {
            "input_ids": text["input_ids"],
            "text_attention_mask": text["attention_mask"],
            "pixel_values": pixels,
            "frame_counts": torch.tensor(frame_counts, dtype=torch.long),
            "completion": torch.tensor([sample["completion"] for sample in samples]),
            "success": torch.tensor([sample["success"] for sample in samples]),
            "completion_mask": torch.tensor(
                [sample["completion_mask"] for sample in samples]
            ),
            "success_mask": torch.tensor([sample["success_mask"] for sample in samples]),
            "sample_weight": torch.tensor([sample["weight"] for sample in samples]),
            "row_ids": [sample["row_id"] for sample in samples],
        }


class SiglipGemmaValueModel(nn.Module):
    def __init__(
        self,
        vision_model: str,
        language_model: str,
        max_frames: int = 8,
        visual_grid: int = 4,
        value_bins: int = 201,
        train_language: bool = True,
        train_vision: bool = False,
        vision_gradient_checkpointing: bool = False,
        fp32_master_weights: bool = False,
    ):
        super().__init__()
        self.vision_model_name = vision_model
        self.language_model_name = language_model
        self.max_frames = int(max_frames)
        self.visual_grid = int(visual_grid)
        self.value_bins = int(value_bins)
        pretrained_dtype = torch.float32 if fp32_master_weights else torch.bfloat16
        self.vision = SiglipVisionModel.from_pretrained(
            vision_model,
            torch_dtype=pretrained_dtype,
        )
        language_causal_model = AutoModelForCausalLM.from_pretrained(
            language_model,
            torch_dtype=pretrained_dtype,
        )
        self.language = language_causal_model.model
        vision_dim = int(self.vision.config.hidden_size)
        language_dim = int(self.language.config.hidden_size)
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, language_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(language_dim, language_dim),
        )
        self.frame_position = nn.Parameter(torch.empty(self.max_frames, language_dim))
        self.vision_start = nn.Parameter(torch.empty(1, language_dim))
        self.vision_end = nn.Parameter(torch.empty(1, language_dim))
        self.value_query = nn.Parameter(torch.empty(1, language_dim))
        self.value_norm = nn.LayerNorm(language_dim)
        self.value_head = nn.Linear(language_dim, self.value_bins)
        self.success_head = nn.Linear(language_dim, 1)
        self.register_buffer(
            "bin_centers",
            torch.linspace(0.0, 1.0, self.value_bins),
            persistent=True,
        )
        self.train_vision = bool(train_vision)
        self.vision.requires_grad_(self.train_vision)
        self.language.requires_grad_(bool(train_language))
        if bool(vision_gradient_checkpointing) and self.train_vision:
            self.vision.gradient_checkpointing_enable()
        if not self.train_vision:
            self.vision.eval()
        self.language.config.use_cache = False
        self._reset_new_parameters()

    def _reset_new_parameters(self) -> None:
        for module in (self.visual_projector, self.value_norm, self.value_head, self.success_head):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.normal_(child.weight, mean=0.0, std=0.02)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)
        for parameter in (
            self.frame_position,
            self.vision_start,
            self.vision_end,
            self.value_query,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_vision:
            self.vision.eval()
        return self

    def _compress_patches(self, patches: torch.Tensor) -> torch.Tensor:
        side = int(round(math.sqrt(patches.shape[1])))
        if side * side != patches.shape[1]:
            raise ValueError(f"Expected square SigLIP patch grid, got {patches.shape[1]} tokens")
        feature_map = patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], side, side
        )
        pooled = F.adaptive_avg_pool2d(
            feature_map, (self.visual_grid, self.visual_grid)
        )
        return pooled.flatten(2).transpose(1, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        frame_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        grad_context = torch.enable_grad() if self.train_vision else torch.no_grad()
        with grad_context:
            vision_output = self.vision(
                pixel_values=pixel_values.to(dtype=torch.bfloat16),
                return_dict=True,
            )
        local_tokens = self._compress_patches(vision_output.last_hidden_state)
        global_token = vision_output.pooler_output.unsqueeze(1)
        visual_tokens = self.visual_projector(
            torch.cat([global_token, local_tokens], dim=1)
        )
        text_tokens = self.language.get_input_embeddings()(input_ids)

        sequences = []
        offset = 0
        for batch_index, count_tensor in enumerate(frame_counts):
            count = int(count_tensor.item())
            valid_text = text_tokens[batch_index, text_attention_mask[batch_index].bool()]
            frames = visual_tokens[offset : offset + count]
            offset += count
            frames = frames + self.frame_position[:count, None, :]
            frames = frames.flatten(0, 1)
            sequences.append(
                torch.cat(
                    [
                        valid_text,
                        self.vision_start,
                        frames,
                        self.vision_end,
                        self.value_query,
                    ],
                    dim=0,
                )
            )

        max_length = max(sequence.shape[0] for sequence in sequences)
        inputs_embeds = sequences[0].new_zeros(
            len(sequences), max_length, sequences[0].shape[-1]
        )
        attention_mask = torch.zeros(
            len(sequences), max_length, device=inputs_embeds.device, dtype=torch.long
        )
        query_positions = []
        for index, sequence in enumerate(sequences):
            length = sequence.shape[0]
            inputs_embeds[index, :length] = sequence
            attention_mask[index, :length] = 1
            query_positions.append(length - 1)

        output = self.language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        query_positions = torch.tensor(
            query_positions, device=inputs_embeds.device, dtype=torch.long
        )
        batch_indices = torch.arange(len(sequences), device=inputs_embeds.device)
        value_features = self.value_norm(
            output.last_hidden_state[batch_indices, query_positions]
        )
        value_logits = self.value_head(value_features)
        success_logits = self.success_head(value_features).squeeze(-1)
        value = (
            torch.softmax(value_logits.float(), dim=-1)
            * self.bin_centers.float().unsqueeze(0)
        ).sum(dim=-1)
        return {
            "value_logits": value_logits,
            "success_logits": success_logits,
            "value": value,
        }


def ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    target_margin: float = 0.15,
    prediction_margin: float = 0.08,
) -> torch.Tensor | None:
    valid = mask > 0
    if int(valid.sum().item()) < 2:
        return None
    prediction = prediction[valid]
    target = target[valid]
    weight = weight[valid]
    target_diff = target[:, None] - target[None, :]
    prediction_diff = prediction[:, None] - prediction[None, :]
    pair_mask = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    pair_mask &= target_diff.abs() >= target_margin
    if not bool(pair_mask.any().item()):
        return None
    desired = target_diff.sign() * prediction_diff
    pair_weight = torch.sqrt(weight[:, None] * weight[None, :])
    loss = F.softplus((prediction_margin - desired[pair_mask]) * 10.0) / 10.0
    return (loss * pair_weight[pair_mask]).sum() / pair_weight[pair_mask].sum().clamp(min=1.0)


def compute_loss(output: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    completion = batch["completion"].float()
    success = batch["success"].float()
    completion_mask = batch["completion_mask"].float()
    success_mask = batch["success_mask"].float()
    weight = batch["sample_weight"].float()

    bin_target = torch.round(completion * (output["value_logits"].shape[-1] - 1)).long()
    value_ce = F.cross_entropy(output["value_logits"].float(), bin_target, reduction="none")
    value_denom = (completion_mask * weight).sum().clamp(min=1.0)
    value_loss = (value_ce * completion_mask * weight).sum() / value_denom
    success_bce = F.binary_cross_entropy_with_logits(
        output["success_logits"].float(), success, reduction="none"
    )
    success_denom = (success_mask * weight).sum().clamp(min=1.0)
    success_loss = (success_bce * success_mask * weight).sum() / success_denom
    loss = value_loss + 1.5 * success_loss

    completion_rank = ranking_loss(
        output["value"], completion, completion_mask, weight
    )
    success_rank = ranking_loss(
        torch.sigmoid(output["success_logits"].float()), success, success_mask, weight
    )
    rank_terms = [term for term in (completion_rank, success_rank) if term is not None]
    rank_loss = torch.stack(rank_terms).mean() if rank_terms else loss.new_zeros(())
    loss = loss + rank_loss
    metrics = {
        "loss": float(loss.detach()),
        "value_ce": float(value_loss.detach()),
        "success_bce": float(success_loss.detach()),
        "rank": float(rank_loss.detach()),
        "value_mae": float(
            ((output["value"] - completion).abs() * completion_mask).sum().detach()
            / completion_mask.sum().clamp(min=1.0)
        ),
    }
    return loss, metrics


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--language-model", default="unsloth/gemma-3-270m-it")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--visual-grid", type=int, default=4)
    parser.add_argument("--value-bins", type=int, default=201)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--train-language",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vision-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fp32-master-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    dataset = ProgressRows(args.data, max_frames=args.max_frames, max_samples=args.max_samples)
    sampler = BlockDistributedSampler(
        len(dataset), args.block_size, rank=rank, world_size=world_size
    )
    collator = ProgressCollator(args.vision_model, args.language_model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    model = SiglipGemmaValueModel(
        vision_model=args.vision_model,
        language_model=args.language_model,
        max_frames=args.max_frames,
        visual_grid=args.visual_grid,
        value_bins=args.value_bins,
        train_language=args.train_language,
        train_vision=args.train_vision,
        vision_gradient_checkpointing=args.vision_gradient_checkpointing,
        fp32_master_weights=args.fp32_master_weights,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "init_checkpoint": str(args.init_checkpoint),
                        "missing_keys": len(missing),
                        "unexpected_keys": len(unexpected),
                    }
                ),
                flush=True,
            )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    vision_parameters = [parameter for name, parameter in trainable if name.startswith("vision.")]
    language_parameters = [parameter for name, parameter in trainable if name.startswith("language.")]
    head_parameters = [
        parameter
        for name, parameter in trainable
        if not name.startswith("language.") and not name.startswith("vision.")
    ]
    parameter_groups = []
    if vision_parameters:
        parameter_groups.append({"params": vision_parameters, "lr": args.vision_learning_rate})
    if language_parameters:
        parameter_groups.append({"params": language_parameters, "lr": args.learning_rate})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.head_learning_rate})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    available_steps = len(loader) // accumulation_steps
    total_steps = available_steps if args.max_steps <= 0 else min(available_steps, args.max_steps)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr(step, total_steps, warmup_steps),
    )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            json.dumps(
                {
                    "rows": len(dataset),
                    "steps": total_steps,
                    "world_size": world_size,
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                }
            ),
            flush=True,
        )

    model.train()
    last_metrics = {}
    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    for micro_step, batch in enumerate(loader):
        if update_step >= total_steps:
            break
        batch = move_batch(batch, device)
        should_update = (micro_step + 1) % accumulation_steps == 0
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not should_update
            else contextlib.nullcontext()
        )
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    pixel_values=batch["pixel_values"],
                    frame_counts=batch["frame_counts"],
                )
                loss, last_metrics = compute_loss(output, batch)
                scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
        if not should_update:
            continue
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
        if rank == 0 and (update_step == 1 or update_step % 10 == 0):
            print(json.dumps({"step": update_step, **last_metrics}), flush=True)

    peak_memory = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        ],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "peak_allocated_gib": float(peak_memory[0].item() / (1024**3)),
                    "peak_reserved_gib": float(peak_memory[1].item() / (1024**3)),
                }
            ),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        state = {name: tensor.detach().cpu() for name, tensor in unwrapped.state_dict().items()}
        torch.save(
            {
                "architecture": "siglip_so400m_gemma3_270m_distributional_value_v1",
                "vision_model": args.vision_model,
                "language_model": args.language_model,
                "max_frames": args.max_frames,
                "visual_grid": args.visual_grid,
                "value_bins": args.value_bins,
                "train_language": args.train_language,
                "train_vision": args.train_vision,
                "data": str(args.data),
                "steps": total_steps,
                "state_dict": state,
                "last_metrics": last_metrics,
            },
            args.output_dir / "value_model.pt",
        )
        with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
