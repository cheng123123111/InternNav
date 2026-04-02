import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from scripts.data_collect.stop_alignment_utils import build_stop_alignment_prompt


class StopPhraseHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_manifest(path: Path, max_samples: int = -1):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def prepare_rows(rows):
    prepared = []
    for row in rows:
        item = dict(row)
        item["image"] = Image.open(row["image_path"]).convert("RGB")
        prepared.append(item)
    return prepared


def batched(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def make_batch_inputs(processor, rows, device):
    texts = []
    images = []
    for row in rows:
        prompt = build_stop_alignment_prompt(row["stop_phrase"])
        messages = [{"role": "user", "content": [{"type": "image", "image": row["image"]}, {"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.append(row["image"])
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
    return inputs, image_grid_thw


def extract_features(model, processor, rows, device):
    inputs, image_grid_thw = make_batch_inputs(processor, rows, device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
    return outputs.hidden_states[-1][:, -1, :].float()


def evaluate(model, processor, head, rows, device, batch_size):
    head.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_rows in batched(rows, batch_size):
            feats = extract_features(model, processor, batch_rows, device)
            logits = head(feats).squeeze(-1)
            label = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            total_loss += float(loss.item()) * len(batch_rows)
            preds = (torch.sigmoid(logits) >= 0.5).long().tolist()
            correct += sum(int(pred == row["label"]) for pred, row in zip(preds, batch_rows))
            total += len(batch_rows)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1), "count": total}


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
    parser.add_argument("--pos-weight", type=float, default=-1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_size = int(model.config.hidden_size)
    head = StopPhraseHead(hidden_size).to(device).float()

    rows = prepare_rows(load_manifest(Path(args.manifest), args.max_samples))
    split = max(1, int(len(rows) * args.train_ratio))
    train_rows = rows[:split]
    val_rows = rows[split:] if split < len(rows) else rows[: min(len(rows), 32)]
    pos_weight = args.pos_weight if args.pos_weight > 0 else compute_pos_weight(train_rows)
    pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    log_path = output_dir / "train_log.jsonl"

    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_rows in batched(train_rows, args.batch_size):
            feats = extract_features(model, processor, batch_rows, device)
            logits = head(feats).squeeze(-1)
            label = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_rows)
            preds = (torch.sigmoid(logits) >= 0.5).long().tolist()
            correct += sum(int(pred == row["label"]) for pred, row in zip(preds, batch_rows))
            total += len(batch_rows)

        train_metrics = {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1), "count": total}
        val_metrics = evaluate(model, processor, head, val_rows, device, args.batch_size)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

    torch.save({"stop_phrase_head": head.state_dict(), "hidden_size": hidden_size}, output_dir / "stop_phrase_head.pt")
    processor.save_pretrained(output_dir / "processor_ref")


if __name__ == "__main__":
    main()
