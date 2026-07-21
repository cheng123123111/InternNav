#!/usr/bin/env python3
"""Build block-aligned full R2R completion + repeated failure training data."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def prepare_expert_blocks(
    rows: list[dict[str, Any]], block_size: int
) -> tuple[list[list[dict[str, Any]]], int]:
    prepared = []
    for row in rows:
        item = copy.deepcopy(row)
        labels = item.setdefault("progress_labels", {})
        labels["completion_mask"] = 1.0
        # Failure blocks already contain balanced healthy/failed states. Let the
        # full expert set supervise completion without overwhelming that head.
        labels["success_mask"] = 0.0
        supervision = item.setdefault("query_supervision", {})
        supervision["completion_mask"] = 1.0
        supervision["success_mask"] = 0.0
        prepared.append(item)

    padded = 0
    while len(prepared) % block_size:
        duplicate = copy.deepcopy(prepared[-1])
        duplicate["row_id"] = f"{duplicate.get('row_id', 'expert')}::block_pad_{padded}"
        prepared.append(duplicate)
        padded += 1

    blocks = chunks(prepared, block_size)
    for block_index, block in enumerate(blocks):
        block_id = f"full_r2r_expert::{block_index:06d}"
        for role, row in enumerate(block):
            row["joint_rank_block"] = {
                "block_id": block_id,
                "role": f"expert_{role}",
                "batch_size_contract": block_size,
            }
    return blocks, padded


def prepare_failure_blocks(
    rows: list[dict[str, Any]], block_size: int, repeats: int
) -> list[list[dict[str, Any]]]:
    if len(rows) % block_size:
        raise ValueError("Failure rows must already be divisible into complete blocks")
    source_blocks = chunks(rows, block_size)
    output = []
    for repeat in range(repeats):
        for block_index, block in enumerate(source_blocks):
            copied = copy.deepcopy(block)
            block_id = f"failure_x{repeat}::{block_index:06d}"
            for row in copied:
                row["row_id"] = f"{row.get('row_id', 'failure')}::repeat_{repeat}"
                rank_block = row.setdefault("joint_rank_block", {})
                rank_block["block_id"] = block_id
                rank_block["batch_size_contract"] = block_size
            output.append(copied)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-repeats", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    expert_rows = read_jsonl(args.expert)
    failure_rows = read_jsonl(args.failure)
    expert_blocks, expert_padding = prepare_expert_blocks(expert_rows, args.block_size)
    failure_blocks = prepare_failure_blocks(
        failure_rows, args.block_size, args.failure_repeats
    )
    blocks = expert_blocks + failure_blocks
    random.Random(args.seed).shuffle(blocks)

    world_padding_blocks = 0
    while len(blocks) % args.world_size:
        duplicate = copy.deepcopy(expert_blocks[world_padding_blocks % len(expert_blocks)])
        block_id = f"world_pad::{world_padding_blocks:04d}"
        for role, row in enumerate(duplicate):
            row["row_id"] = f"{row.get('row_id', 'expert')}::{block_id}"
            row["joint_rank_block"] = {
                "block_id": block_id,
                "role": f"world_pad_{role}",
                "batch_size_contract": args.block_size,
            }
        blocks.append(duplicate)
        world_padding_blocks += 1

    output_rows = [row for block in blocks for row in block]
    write_jsonl(args.output, output_rows)
    summary = {
        "expert_source_rows": len(expert_rows),
        "failure_source_rows": len(failure_rows),
        "failure_repeats": args.failure_repeats,
        "expert_padding_rows": expert_padding,
        "world_padding_blocks": world_padding_blocks,
        "blocks": len(blocks),
        "rows": len(output_rows),
        "block_size": args.block_size,
        "world_size": args.world_size,
        "completion_supervised": sum(
            float(row["progress_labels"].get("completion_mask", 0.0)) > 0
            for row in output_rows
        ),
        "success_supervised": sum(
            float(row["progress_labels"].get("success_mask", 0.0)) > 0
            for row in output_rows
        ),
        "output": str(args.output),
    }
    with args.output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
