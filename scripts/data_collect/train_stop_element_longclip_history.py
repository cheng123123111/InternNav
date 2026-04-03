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
    val_rows = rows[split:] if split < len(rows) else rows[: min(len(rows), 64)]
    return train_rows, val_rows


def get_target_elements(row: dict, max_target_elements: int) -> list[str]:
    elements = [str(x).strip() for x in row.get("target_elements", []) if str(x).strip()]
    if not elements:
        fallback = (
            row.get("target_elements_text")
            or row.get("stop_object_phrase")
            or row.get("stop_phrase")
            or row.get("instruction")
            or ""
        ).strip()
        if fallback:
            elements = [fallback]
    return elements[:max_target_elements]


def get_frame_paths(row: dict, history_frames: int) -> list[str]:
    frame_paths = [str(x) for x in row.get("frame_paths", []) if str(x).strip()]
    if not frame_paths:
        single = str(row.get("image_path", "")).strip()
        frame_paths = [single] if single else []
    return frame_paths[-history_frames:]


def get_action_text(row: dict) -> str:
    return (
        str(row.get("stop_action_text") or "").strip()
        or str(row.get("stop_phrase") or "").strip()
        or str(row.get("instruction") or "").strip()
    )


def prepare_rows(rows, max_target_elements: int, history_frames: int):
    prepared = []
    for row in rows:
        item = dict(row)
        item["elements"] = get_target_elements(item, max_target_elements)
        item["frame_paths"] = get_frame_paths(item, history_frames)
        item["action_text"] = get_action_text(item)
        if not item["elements"] or not item["frame_paths"] or not item["action_text"]:
            continue
        prepared.append(item)
    return prepared


def batched(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def flatten_frame_paths(rows):
    paths = []
    owners = []
    for idx, row in enumerate(rows):
        for path in row["frame_paths"]:
            paths.append(path)
            owners.append(idx)
    return paths, owners


def flatten_texts(rows):
    texts = []
    owners = []
    for idx, row in enumerate(rows):
        for text in row["elements"]:
            texts.append(text)
            owners.append(idx)
    return texts, owners


def get_action_texts(rows):
    texts = [row["action_text"] for row in rows]
    return texts


def load_image_tensor(path: str, preprocess):
    image = Image.open(path).convert("RGB")
    return preprocess(image)


class HistoryElementLongCLIPHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.image_proj = nn.Linear(dim, proj_dim)
        self.text_proj = nn.Linear(dim, proj_dim)
        self.score_head = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def project_image(self, image_features):
        return F.normalize(self.image_proj(image_features), dim=-1)

    def project_text(self, text_features):
        return F.normalize(self.text_proj(text_features), dim=-1)

    def score_from_stats(self, stats):
        return self.score_head(stats).squeeze(-1)


def encode_batch(model, preprocess, rows, device):
    frame_paths, frame_owners = flatten_frame_paths(rows)
    images = [load_image_tensor(path, preprocess) for path in frame_paths]
    image_batch = torch.stack(images, dim=0).to(device)
    texts, text_owners = flatten_texts(rows)
    text_tokens = longclip.tokenize(texts, truncate=True).to(device)
    action_tokens = longclip.tokenize(get_action_texts(rows), truncate=True).to(device)
    image_features = model.encode_image(image_batch).float()
    text_features = model.encode_text(text_tokens).float()
    action_features = model.encode_text(action_tokens).float()
    return image_features, frame_owners, text_features, text_owners, action_features


def aggregate_history_features(projected_frame_features, frame_owners, batch_size):
    per_row_frames = [[] for _ in range(batch_size)]
    for feat, row_idx in zip(projected_frame_features, frame_owners):
        per_row_frames[row_idx].append(feat)

    history_mean = []
    history_last = []
    for feats in per_row_frames:
        frame_tensor = torch.stack(feats, dim=0)
        history_mean.append(F.normalize(frame_tensor.mean(dim=0), dim=-1))
        history_last.append(frame_tensor[-1])
    return torch.stack(history_mean, dim=0), torch.stack(history_last, dim=0)


def compute_stats(
    history_mean_proj,
    history_last_proj,
    text_proj,
    text_owners,
    action_proj,
    batch_size,
    max_target_elements,
    obj_threshold,
    coverage_temperature,
):
    per_row_history = [[] for _ in range(batch_size)]
    per_row_last = [[] for _ in range(batch_size)]
    for text_idx, row_idx in enumerate(text_owners):
        per_row_history[row_idx].append((history_mean_proj[row_idx] * text_proj[text_idx]).sum())
        per_row_last[row_idx].append((history_last_proj[row_idx] * text_proj[text_idx]).sum())

    stats = []
    raw_scores = []
    for history_sims, last_sims in zip(per_row_history, per_row_last):
        history_tensor = torch.stack(history_sims)
        last_tensor = torch.stack(last_sims)
        history_mean_sim = history_tensor.mean()
        last_mean_sim = last_tensor.mean()
        temporal_max_sim = torch.maximum(history_tensor.max(), last_tensor.max())
        soft_coverage = torch.sigmoid((history_tensor - obj_threshold) * coverage_temperature).mean()
        action_hist_sim = (history_mean_proj[len(stats)] * action_proj[len(stats)]).sum()
        action_last_sim = (history_last_proj[len(stats)] * action_proj[len(stats)]).sum()
        count_ratio = torch.tensor(
            [len(history_sims) / max(max_target_elements, 1)],
            device=history_tensor.device,
            dtype=history_tensor.dtype,
        ).squeeze(0)
        stats.append(
            torch.stack(
                [
                    last_mean_sim,
                    history_mean_sim,
                    temporal_max_sim,
                    soft_coverage,
                    action_last_sim,
                    action_hist_sim,
                    count_ratio,
                ],
                dim=0,
            )
        )
        raw_scores.append(history_mean_sim)
    return torch.stack(stats, dim=0), torch.stack(raw_scores, dim=0)


def pairwise_rank_loss(logits, labels, margin):
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    diffs = margin - pos.unsqueeze(1) + neg.unsqueeze(0)
    return F.relu(diffs).mean()


@torch.no_grad()
def evaluate(model, preprocess, head, rows, device, batch_size, max_target_elements, obj_threshold, coverage_temperature):
    model.eval()
    head.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    pos_logits = []
    neg_logits = []
    pos_raw = []
    neg_raw = []
    for batch_rows in batched(rows, batch_size):
        image_features, frame_owners, text_features, text_owners, action_features = encode_batch(
            model, preprocess, batch_rows, device
        )
        image_proj = head.project_image(image_features)
        text_proj = head.project_text(text_features)
        action_proj = head.project_text(action_features)
        history_mean_proj, history_last_proj = aggregate_history_features(image_proj, frame_owners, len(batch_rows))
        stats, raw_scores = compute_stats(
            history_mean_proj,
            history_last_proj,
            text_proj,
            text_owners,
            action_proj,
            len(batch_rows),
            max_target_elements,
            obj_threshold,
            coverage_temperature,
        )
        logits = head.score_from_stats(stats)
        labels = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += float(loss.item()) * len(batch_rows)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().tolist()
        correct += sum(int(pred == int(row["label"])) for pred, row in zip(preds, batch_rows))
        total += len(batch_rows)
        for label, logit, raw in zip(labels.tolist(), logits.tolist(), raw_scores.tolist()):
            if label > 0.5:
                pos_logits.append(logit)
                pos_raw.append(raw)
            else:
                neg_logits.append(logit)
                neg_raw.append(raw)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "count": total,
        "pos_logit_mean": sum(pos_logits) / max(len(pos_logits), 1),
        "neg_logit_mean": sum(neg_logits) / max(len(neg_logits), 1),
        "pos_raw_mean": sum(pos_raw) / max(len(pos_raw), 1),
        "neg_raw_mean": sum(neg_raw) / max(len(neg_raw), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--longclip-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--obj-threshold", type=float, default=0.25)
    parser.add_argument("--coverage-temperature", type=float, default=12.0)
    parser.add_argument("--max-target-elements", type=int, default=4)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = longclip.load(args.longclip_path, device=device)
    if not args.train_backbone:
        for param in model.parameters():
            param.requires_grad = False
    feature_dim = int(model.text_projection.shape[-1])
    head = HistoryElementLongCLIPHead(feature_dim, args.proj_dim, args.hidden_dim).to(device)

    rows = load_manifest(Path(args.manifest), args.max_samples)
    rows = prepare_rows(rows, args.max_target_elements, args.history_frames)
    train_rows, val_rows = split_rows(rows, args.train_ratio, args.seed)

    trainable_params = list(head.parameters()) + [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    config_path = output_dir / "run_config.json"
    log_path = output_dir / "train_log.jsonl"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    best_score = float("-inf")
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        model.train(args.train_backbone)
        head.train()
        total_loss = 0.0
        total_bce = 0.0
        total_rank = 0.0
        total_batches = 0

        for batch_rows in batched(train_rows, args.batch_size):
            image_features, frame_owners, text_features, text_owners, action_features = encode_batch(
                model, preprocess, batch_rows, device
            )
            image_proj = head.project_image(image_features)
            text_proj = head.project_text(text_features)
            action_proj = head.project_text(action_features)
            history_mean_proj, history_last_proj = aggregate_history_features(image_proj, frame_owners, len(batch_rows))
            stats, _ = compute_stats(
                history_mean_proj,
                history_last_proj,
                text_proj,
                text_owners,
                action_proj,
                len(batch_rows),
                args.max_target_elements,
                args.obj_threshold,
                args.coverage_temperature,
            )
            logits = head.score_from_stats(stats)
            labels = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            rank_loss = pairwise_rank_loss(logits, labels, args.rank_margin)
            loss = bce_loss + args.rank_loss_weight * rank_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_bce += float(bce_loss.item())
            total_rank += float(rank_loss.item())
            total_batches += 1

        train_metrics = evaluate(
            model,
            preprocess,
            head,
            train_rows[: min(len(train_rows), 1024)],
            device,
            args.batch_size,
            args.max_target_elements,
            args.obj_threshold,
            args.coverage_temperature,
        )
        val_metrics = evaluate(
            model,
            preprocess,
            head,
            val_rows,
            device,
            args.batch_size,
            args.max_target_elements,
            args.obj_threshold,
            args.coverage_temperature,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "train_bce_loss": total_bce / max(total_batches, 1),
            "train_rank_loss": total_rank / max(total_batches, 1),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        score = val_metrics["acc"] + (val_metrics["pos_logit_mean"] - val_metrics["neg_logit_mean"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "head": head.state_dict(),
                    "best_score": best_score,
                    "longclip_path": args.longclip_path,
                    "proj_dim": args.proj_dim,
                    "hidden_dim": args.hidden_dim,
                    "obj_threshold": args.obj_threshold,
                    "coverage_temperature": args.coverage_temperature,
                    "max_target_elements": args.max_target_elements,
                    "history_frames": args.history_frames,
                },
                output_dir / "best_longclip_stop_elements_history.pt",
            )

    torch.save(
        {
            "head": head.state_dict(),
            "best_score": best_score,
            "longclip_path": args.longclip_path,
            "proj_dim": args.proj_dim,
            "hidden_dim": args.hidden_dim,
            "obj_threshold": args.obj_threshold,
            "coverage_temperature": args.coverage_temperature,
            "max_target_elements": args.max_target_elements,
            "history_frames": args.history_frames,
        },
        output_dir / "last_longclip_stop_elements_history.pt",
    )


if __name__ == "__main__":
    main()
