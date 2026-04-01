import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


STOP_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Judge whether the current view already matches the final stopping location of the task."
)


def load_manifest(path: Path, max_samples: int = -1):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def make_batch_inputs(processor, rows, device: torch.device):
    texts = []
    images = []
    for row in rows:
        prompt = STOP_PROMPT.replace("<instruction>", row["instruction"])
        messages = [{"role": "user", "content": [{"type": "image", "image": row["image"]}, {"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.append(row["image"])
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
    return inputs, image_grid_thw


def batched(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def prepare_rows(rows):
    prepared = []
    for row in rows:
        item = dict(row)
        item["image"] = Image.open(row["image_path"]).convert("RGB")
        prepared.append(item)
    return prepared


def evaluate(model, processor, rows, device, batch_size, feature_batch_size):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_rows in batched(rows, batch_size):
            stop_features = extract_stop_features_microbatched(
                model,
                processor,
                batch_rows,
                device,
                feature_batch_size,
            )
            logits = model.get_model().stop_head(stop_features.float()).squeeze(-1)
            label = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), label)
            total_loss += float(loss.item()) * len(batch_rows)
            preds = (torch.sigmoid(logits) >= 0.5).long().tolist()
            correct += sum(int(pred == row["label"]) for pred, row in zip(preds, batch_rows))
            total += len(batch_rows)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1), "count": total}


def extract_stop_features_microbatched(model, processor, batch_rows, device, feature_batch_size):
    feature_batch_size = max(int(feature_batch_size), 1)
    features = []
    with torch.no_grad():
        for i in range(0, len(batch_rows), feature_batch_size):
            sub_rows = batch_rows[i : i + feature_batch_size]
            inputs, image_grid_thw = make_batch_inputs(processor, sub_rows, device)
            sub_features = model.extract_stop_features(inputs.input_ids, inputs.pixel_values, image_grid_thw)
            features.append(sub_features)
    return torch.cat(features, dim=0)


def compute_pos_weight(rows):
    pos = sum(int(row["label"]) for row in rows)
    neg = max(len(rows) - pos, 1)
    pos = max(pos, 1)
    return float(neg / pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=1)
    parser.add_argument("--pos-weight", type=float, default=-1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.reset_stop_head()
    model.train()
    model.get_model().stop_head.float()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_model().stop_head.parameters():
        p.requires_grad = True

    rows = prepare_rows(load_manifest(Path(args.manifest), args.max_samples))
    split = max(1, int(len(rows) * args.train_ratio))
    train_rows = rows[:split]
    val_rows = rows[split:] if split < len(rows) else rows[: min(len(rows), 32)]
    pos_weight = args.pos_weight if args.pos_weight > 0 else compute_pos_weight(train_rows)
    pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.get_model().stop_head.parameters(), lr=args.lr)
    log_path = output_dir / "train_log.jsonl"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_rows in batched(train_rows, args.batch_size):
            stop_features = extract_stop_features_microbatched(
                model,
                processor,
                batch_rows,
                device,
                args.feature_batch_size,
            )
            logits = model.get_model().stop_head(stop_features.float()).squeeze(-1)
            label = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), label, pos_weight=pos_weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_rows)
            preds = (torch.sigmoid(logits) >= 0.5).long().tolist()
            correct += sum(int(pred == row["label"]) for pred, row in zip(preds, batch_rows))
            total += len(batch_rows)

        train_metrics = {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1), "count": total}
        val_metrics = evaluate(model, processor, val_rows, device, args.batch_size, args.feature_batch_size)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

    torch.save({"stop_head": model.get_model().stop_head.state_dict()}, output_dir / "stop_head.pt")
    processor.save_pretrained(output_dir / "processor_ref")
    tokenizer.save_pretrained(output_dir / "tokenizer_ref")


if __name__ == "__main__":
    main()
