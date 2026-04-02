import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from internnav.model.basemodel.LongCLIP.model import longclip


def load_manifest(path: Path, max_samples: int = -1):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def split_rows(rows, train_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    split = max(1, int(len(rows) * train_ratio))
    train_rows = rows[:split]
    val_rows = rows[split:] if split < len(rows) else rows[: min(len(rows), 32)]
    return train_rows, val_rows


def select_text(row: dict) -> str:
    if row.get("usable_stop_object_phrase"):
        text = row.get("stop_object_phrase") or ""
    else:
        text = row.get("stop_phrase") or row.get("instruction") or ""
    return text.strip()


def batched(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def load_image_batch(rows, preprocess, device):
    images = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        images.append(preprocess(image))
    return torch.stack(images, dim=0).to(device)


def load_text_batch(rows, device):
    texts = [select_text(row) for row in rows]
    tokens = longclip.tokenize(texts, truncate=True)
    return tokens.to(device)


def encode_batch(model, preprocess, rows, device):
    image_batch = load_image_batch(rows, preprocess, device)
    text_batch = load_text_batch(rows, device)
    image_features = model.encode_image(image_batch)
    text_features = model.encode_text(text_batch)
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    return image_features, text_features


class ContrastiveProjector(nn.Module):
    def __init__(self, dim: int, proj_dim: int):
        super().__init__()
        self.image_proj = nn.Linear(dim, proj_dim)
        self.text_proj = nn.Linear(dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, image_features, text_features):
        image_features = F.normalize(self.image_proj(image_features), dim=-1)
        text_features = F.normalize(self.text_proj(text_features), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return image_features, text_features, scale


def contrastive_loss(scale, image_features, text_features):
    logits = scale * image_features @ text_features.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t), logits


def negative_margin_loss(neg_image_features, neg_text_features, margin):
    sims = (neg_image_features * neg_text_features).sum(dim=-1)
    loss = F.relu(sims - margin).mean()
    return loss, sims


def sample_negative_batch(neg_rows, batch_size, rng):
    if not neg_rows:
        return []
    if len(neg_rows) >= batch_size:
        return rng.sample(neg_rows, batch_size)
    return [rng.choice(neg_rows) for _ in range(batch_size)]


def evaluate(model, preprocess, projector, pos_rows, neg_rows, device, batch_size, margin):
    model.eval()
    projector.eval()
    pos_sims = []
    neg_sims = []
    with torch.no_grad():
        for batch_rows in batched(pos_rows, batch_size):
            image_features, text_features = encode_batch(model, preprocess, batch_rows, device)
            image_features, text_features, _ = projector(image_features, text_features)
            pos_sims.extend((image_features * text_features).sum(dim=-1).tolist())
        for batch_rows in batched(neg_rows, batch_size):
            image_features, text_features = encode_batch(model, preprocess, batch_rows, device)
            image_features, text_features, _ = projector(image_features, text_features)
            neg_sims.extend((image_features * text_features).sum(dim=-1).tolist())

    if not pos_sims:
        return {"pos_mean": 0.0, "neg_mean": 0.0, "margin_acc": 0.0, "pos_count": 0, "neg_count": len(neg_sims)}
    threshold = margin
    pos_correct = sum(sim > threshold for sim in pos_sims)
    neg_correct = sum(sim <= threshold for sim in neg_sims)
    total = max(len(pos_sims) + len(neg_sims), 1)
    return {
        "pos_mean": sum(pos_sims) / len(pos_sims),
        "neg_mean": sum(neg_sims) / max(len(neg_sims), 1),
        "margin_acc": (pos_correct + neg_correct) / total,
        "pos_count": len(pos_sims),
        "neg_count": len(neg_sims),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--longclip-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--negative-loss-weight", type=float, default=0.25)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = longclip.load(args.longclip_path, device=device)
    if not args.train_backbone:
        for param in model.parameters():
            param.requires_grad = False
    projector = ContrastiveProjector(int(model.text_projection.shape[-1]), args.proj_dim).to(device)

    rows = load_manifest(Path(args.manifest), args.max_samples)
    rows = [row for row in rows if select_text(row)]
    pos_rows = [row for row in rows if int(row["label"]) == 1]
    neg_rows = [row for row in rows if int(row["label"]) == 0]
    train_pos, val_pos = split_rows(pos_rows, args.train_ratio, args.seed)
    train_neg, val_neg = split_rows(neg_rows, args.train_ratio, args.seed + 1)

    trainable_params = list(projector.parameters()) + [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    log_path = output_dir / "train_log.jsonl"
    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_metric = float("-inf")
    for epoch in range(args.epochs):
        rng.shuffle(train_pos)
        total_loss = 0.0
        total_contrastive = 0.0
        total_neg_loss = 0.0
        total_batches = 0
        model.train(args.train_backbone)
        projector.train()
        for pos_batch in batched(train_pos, args.batch_size):
            if len(pos_batch) < 2:
                continue
            image_features, text_features = encode_batch(model, preprocess, pos_batch, device)
            image_features, text_features, scale = projector(image_features, text_features)
            c_loss, _ = contrastive_loss(scale, image_features, text_features)

            neg_batch = sample_negative_batch(train_neg, len(pos_batch), rng)
            if neg_batch:
                neg_image_features, neg_text_features = encode_batch(model, preprocess, neg_batch, device)
                neg_image_features, neg_text_features, _ = projector(neg_image_features, neg_text_features)
                n_loss, _ = negative_margin_loss(neg_image_features, neg_text_features, args.margin)
            else:
                n_loss = torch.zeros((), device=device)

            loss = c_loss + args.negative_loss_weight * n_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_contrastive += float(c_loss.item())
            total_neg_loss += float(n_loss.item())
            total_batches += 1

        train_metrics = evaluate(model, preprocess, projector, train_pos, train_neg, device, args.batch_size, args.margin)
        val_metrics = evaluate(model, preprocess, projector, val_pos, val_neg, device, args.batch_size, args.margin)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "train_contrastive_loss": total_contrastive / max(total_batches, 1),
            "train_negative_loss": total_neg_loss / max(total_batches, 1),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "logit_scale_exp": float(projector.logit_scale.exp().clamp(max=100.0).item()),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        score = val_metrics["margin_acc"] + (val_metrics["pos_mean"] - val_metrics["neg_mean"])
        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "projector": projector.state_dict(),
                    "best_score": best_metric,
                    "longclip_path": args.longclip_path,
                    "proj_dim": args.proj_dim,
                },
                output_dir / "best_longclip_stop.pt",
            )

    torch.save(
        {
            "projector": projector.state_dict(),
            "best_score": best_metric,
            "longclip_path": args.longclip_path,
            "proj_dim": args.proj_dim,
        },
        output_dir / "last_longclip_stop.pt",
    )


if __name__ == "__main__":
    main()
