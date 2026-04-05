import argparse
import json
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from internnav.model.basemodel.LongCLIP.model import longclip


TARFILE_CACHE = {}


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


def get_relation_text(row: dict) -> str:
    return str(row.get("relation_hint") or row.get("stop_phrase") or row.get("instruction") or "").strip()


def prepare_rows(rows, max_target_elements: int, history_frames: int):
    prepared = []
    for row in rows:
        item = dict(row)
        item["elements"] = get_target_elements(item, max_target_elements)
        item["frame_paths"] = get_frame_paths(item, history_frames)
        item["action_text"] = get_action_text(item)
        item["relation_text"] = get_relation_text(item)
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


def get_relation_texts(rows):
    texts = [row["relation_text"] for row in rows]
    return texts


def load_image_tensor(path: str, preprocess):
    if ":::" in path:
        tar_path, member_name = path.split(":::", 1)
        tar = TARFILE_CACHE.get(tar_path)
        if tar is None:
            tar = tarfile.open(tar_path, "r:gz")
            TARFILE_CACHE[tar_path] = tar
        with tar.extractfile(member_name) as f:
            image = Image.open(BytesIO(f.read())).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    return preprocess(image)


def load_image_tensors(paths, preprocess, max_workers: int = 8):
    if len(paths) <= 1 or max_workers <= 1:
        return [load_image_tensor(path, preprocess) for path in paths]
    worker_count = min(max_workers, len(paths))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda path: load_image_tensor(path, preprocess), paths))


@torch.no_grad()
def build_frozen_feature_cache(
    model,
    preprocess,
    rows,
    device,
    image_batch_size=64,
    text_batch_size=256,
    image_load_workers: int = 8,
):
    image_paths = sorted({path for row in rows for path in row["frame_paths"]})
    element_texts = sorted({text for row in rows for text in row["elements"]})
    action_texts = sorted({row["action_text"] for row in rows})

    image_cache = {}
    for i in range(0, len(image_paths), image_batch_size):
        batch_paths = image_paths[i : i + image_batch_size]
        image_batch = torch.stack(
            load_image_tensors(batch_paths, preprocess, max_workers=image_load_workers),
            dim=0,
        ).to(device)
        features = model.encode_image(image_batch).float().cpu()
        for path, feat in zip(batch_paths, features):
            image_cache[path] = feat

    text_cache = {}
    for i in range(0, len(element_texts), text_batch_size):
        batch_texts = element_texts[i : i + text_batch_size]
        text_tokens = longclip.tokenize(batch_texts, truncate=True).to(device)
        features = model.encode_text(text_tokens).float().cpu()
        for text, feat in zip(batch_texts, features):
            text_cache[text] = feat

    action_cache = {}
    for i in range(0, len(action_texts), text_batch_size):
        batch_texts = action_texts[i : i + text_batch_size]
        text_tokens = longclip.tokenize(batch_texts, truncate=True).to(device)
        features = model.encode_text(text_tokens).float().cpu()
        for text, feat in zip(batch_texts, features):
            action_cache[text] = feat

    return image_cache, text_cache, action_cache


def encode_batch_from_cache(rows, device, image_cache, text_cache, action_cache):
    frame_paths, frame_owners = flatten_frame_paths(rows)
    image_features = torch.stack([image_cache[path] for path in frame_paths], dim=0).to(device)
    texts, text_owners = flatten_texts(rows)
    text_features = torch.stack([text_cache[text] for text in texts], dim=0).to(device)
    action_features = torch.stack([action_cache[row["action_text"]] for row in rows], dim=0).to(device)
    return image_features, frame_owners, text_features, text_owners, action_features


def encode_image_with_patch_tokens(model, image_batch):
    visual = model.visual
    x = visual.conv1(image_batch.type(model.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    cls_feat = x[:, 0, :]
    patch_feats = x[:, 1:, :]
    return cls_feat, patch_feats


def encode_batch_grounding(model, preprocess, rows, device, image_load_workers: int = 8):
    frame_paths, frame_owners = flatten_frame_paths(rows)
    images = load_image_tensors(frame_paths, preprocess, max_workers=image_load_workers)
    image_batch = torch.stack(images, dim=0).to(device)
    frame_cls_features, frame_patch_features = encode_image_with_patch_tokens(model, image_batch)
    texts, text_owners = flatten_texts(rows)
    text_tokens = longclip.tokenize(texts, truncate=True).to(device)
    action_tokens = longclip.tokenize(get_action_texts(rows), truncate=True).to(device)
    relation_tokens = longclip.tokenize(get_relation_texts(rows), truncate=True).to(device)
    text_features = model.encode_text(text_tokens).float()
    action_features = model.encode_text(action_tokens).float()
    relation_features = model.encode_text(relation_tokens).float()
    return (
        frame_cls_features.float(),
        frame_patch_features.float(),
        frame_owners,
        text_features,
        text_owners,
        action_features,
        relation_features,
    )


class HistoryElementLongCLIPHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int, hidden_dim: int = 32, score_hidden_dims=None):
        super().__init__()
        self.image_proj = nn.Linear(dim, proj_dim)
        self.text_proj = nn.Linear(dim, proj_dim)
        if score_hidden_dims is None:
            score_hidden_dims = [hidden_dim]
        layers = []
        in_dim = 7
        for h in score_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.score_head = nn.Sequential(*layers)

    def project_image(self, image_features):
        return F.normalize(self.image_proj(image_features), dim=-1)

    def project_text(self, text_features):
        return F.normalize(self.text_proj(text_features), dim=-1)

    def score_from_stats(self, stats):
        return self.score_head(stats).squeeze(-1)


class CrossAttentionStopHead(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_proj = nn.Linear(dim, proj_dim)
        self.text_proj = nn.Linear(dim, proj_dim)
        self.frame_pos_embed = nn.Parameter(torch.randn(32, proj_dim) * 0.02)
        self.element_pos_embed = nn.Parameter(torch.randn(16, proj_dim) * 0.02)
        self.action_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.element_to_frame_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(proj_dim)
        self.norm_kv = nn.LayerNorm(proj_dim)
        self.norm_out = nn.LayerNorm(proj_dim)
        self.score_head = nn.Sequential(
            nn.Linear(proj_dim * 3 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def project_image(self, image_features):
        return F.normalize(self.image_proj(image_features), dim=-1)

    def project_text(self, text_features):
        return F.normalize(self.text_proj(text_features), dim=-1)

    def score_from_sequences(
        self,
        frame_proj,
        frame_owners,
        text_proj,
        text_owners,
        action_proj,
        batch_size,
        max_target_elements,
        obj_threshold,
        coverage_temperature,
    ):
        device = frame_proj.device
        per_row_frames = [[] for _ in range(batch_size)]
        per_row_texts = [[] for _ in range(batch_size)]
        for feat, row_idx in zip(frame_proj, frame_owners):
            per_row_frames[row_idx].append(feat)
        for feat, row_idx in zip(text_proj, text_owners):
            per_row_texts[row_idx].append(feat)

        frame_lengths = [len(x) for x in per_row_frames]
        text_lengths = [len(x) for x in per_row_texts]
        max_frames = max(frame_lengths)
        max_texts = max(text_lengths)
        frame_batch = torch.zeros(batch_size, max_frames, frame_proj.shape[-1], device=device, dtype=frame_proj.dtype)
        text_batch = torch.zeros(batch_size, max_texts, text_proj.shape[-1], device=device, dtype=text_proj.dtype)
        frame_mask = torch.ones(batch_size, max_frames, device=device, dtype=torch.bool)
        text_mask = torch.ones(batch_size, max_texts, device=device, dtype=torch.bool)
        for i, feats in enumerate(per_row_frames):
            stacked = torch.stack(feats, dim=0)
            frame_batch[i, : stacked.shape[0]] = stacked
            frame_mask[i, : stacked.shape[0]] = False
        for i, feats in enumerate(per_row_texts):
            stacked = torch.stack(feats, dim=0)
            text_batch[i, : stacked.shape[0]] = stacked
            text_mask[i, : stacked.shape[0]] = False

        frame_batch = frame_batch + self.frame_pos_embed[:max_frames].unsqueeze(0)
        text_batch = text_batch + self.element_pos_embed[:max_texts].unsqueeze(0)
        attn_out, _ = self.element_to_frame_attn(
            query=self.norm_q(text_batch),
            key=self.norm_kv(frame_batch),
            value=frame_batch,
            key_padding_mask=frame_mask,
        )
        grounded_text = self.norm_out(text_batch + attn_out)

        grounded_mean = []
        raw_scores = []
        temporal_max = []
        soft_coverages = []
        count_ratios = []
        last_frame_feats = []
        history_mean_feats = []
        for i in range(batch_size):
            t_len = text_lengths[i]
            f_len = frame_lengths[i]
            gt = grounded_text[i, :t_len]
            frames = frame_batch[i, :f_len]
            grounded_mean.append(gt.mean(dim=0))
            sims = gt @ frames.transpose(0, 1)
            raw_scores.append(sims.mean())
            temporal_max.append(sims.max())
            soft_coverages.append(torch.sigmoid((sims - obj_threshold) * coverage_temperature).mean())
            count_ratios.append(
                torch.tensor(
                    t_len / max(max_target_elements, 1),
                    device=device,
                    dtype=frame_proj.dtype,
                )
            )
            last_frame_feats.append(frames[-1])
            history_mean_feats.append(F.normalize(frames.mean(dim=0), dim=-1))

        grounded_mean = torch.stack(grounded_mean, dim=0)
        last_frame_feats = torch.stack(last_frame_feats, dim=0)
        history_mean_feats = torch.stack(history_mean_feats, dim=0)
        temporal_max = torch.stack(temporal_max, dim=0)
        soft_coverages = torch.stack(soft_coverages, dim=0)
        count_ratios = torch.stack(count_ratios, dim=0)
        raw_scores = torch.stack(raw_scores, dim=0)

        action_tokens = action_proj.unsqueeze(1) + self.action_token.expand(batch_size, -1, -1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fusion_tokens = torch.cat(
            [
                cls_tokens,
                grounded_mean.unsqueeze(1),
                history_mean_feats.unsqueeze(1),
                last_frame_feats.unsqueeze(1),
                action_tokens,
            ],
            dim=1,
        )
        fused, _ = self.fusion_attn(fusion_tokens, fusion_tokens, fusion_tokens)
        fused = self.norm_out(fusion_tokens + fused)
        cls_out = fused[:, 0, :]
        extra = torch.stack(
            [
                temporal_max,
                soft_coverages,
                count_ratios,
                raw_scores,
            ],
            dim=1,
        )
        logits = self.score_head(torch.cat([cls_out, grounded_mean, action_proj, extra], dim=1)).squeeze(-1)
        return logits, raw_scores


class GroundingQueryStopHead(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_proj = nn.Linear(dim, proj_dim)
        self.text_proj = nn.Linear(dim, proj_dim)
        self.frame_pos_embed = nn.Parameter(torch.randn(32, proj_dim) * 0.02)
        self.query_type_embed = nn.Parameter(torch.randn(4, proj_dim) * 0.02)
        self.stop_query = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.uncertainty_query = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_to_region_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.score_head = nn.Sequential(
            nn.Linear(proj_dim * 4 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def project_image(self, image_features):
        return F.normalize(self.image_proj(image_features), dim=-1)

    def project_text(self, text_features):
        return F.normalize(self.text_proj(text_features), dim=-1)

    def score_from_grounding(
        self,
        frame_cls_features,
        frame_patch_features,
        frame_owners,
        text_features,
        text_owners,
        action_features,
        relation_features,
        batch_size,
        max_target_elements,
        obj_threshold,
        coverage_temperature,
    ):
        device = frame_cls_features.device
        frame_cls_proj = self.project_image(frame_cls_features)
        frame_patch_proj = self.project_image(frame_patch_features)
        text_proj = self.project_text(text_features)
        action_proj = self.project_text(action_features)
        relation_proj = self.project_text(relation_features)

        per_row_frame_cls = [[] for _ in range(batch_size)]
        per_row_frame_patches = [[] for _ in range(batch_size)]
        per_row_texts = [[] for _ in range(batch_size)]
        for cls_feat, patch_feat, row_idx in zip(frame_cls_proj, frame_patch_proj, frame_owners):
            per_row_frame_cls[row_idx].append(cls_feat)
            per_row_frame_patches[row_idx].append(patch_feat)
        for feat, row_idx in zip(text_proj, text_owners):
            per_row_texts[row_idx].append(feat)

        logits = []
        raw_scores = []
        uncertainties = []
        reprs = []
        for i in range(batch_size):
            frame_cls = torch.stack(per_row_frame_cls[i], dim=0)
            frame_cls = frame_cls + self.frame_pos_embed[: frame_cls.shape[0]]
            temp_out, _ = self.temporal_attn(frame_cls.unsqueeze(0), frame_cls.unsqueeze(0), frame_cls.unsqueeze(0))
            temp_out = self.norm(temp_out.squeeze(0) + frame_cls)
            temporal_mean = temp_out.mean(dim=0)
            temporal_last = temp_out[-1]

            patch_tokens = []
            for t, patches in enumerate(per_row_frame_patches[i]):
                patch_tokens.append(patches + self.frame_pos_embed[t])
            region_tokens = torch.cat(patch_tokens, dim=0).unsqueeze(0)

            element_queries = torch.stack(per_row_texts[i], dim=0)
            element_queries = element_queries + self.query_type_embed[0]
            relation_query = relation_proj[i : i + 1] + self.query_type_embed[1]
            action_query = action_proj[i : i + 1] + self.query_type_embed[2]
            task_queries = torch.cat([element_queries, relation_query, action_query], dim=0).unsqueeze(0)

            grounded_queries, _ = self.query_to_region_attn(task_queries, region_tokens, region_tokens)
            grounded_queries = self.norm(task_queries + grounded_queries)

            stop_query = self.stop_query.expand(1, -1, -1) + self.query_type_embed[3].view(1, 1, -1)
            uncertainty_query = self.uncertainty_query.expand(1, -1, -1)
            fusion_tokens = torch.cat(
                [
                    stop_query,
                    uncertainty_query,
                    grounded_queries,
                    temporal_mean.view(1, 1, -1),
                    temporal_last.view(1, 1, -1),
                ],
                dim=1,
            )
            fused, _ = self.fusion_attn(fusion_tokens, fusion_tokens, fusion_tokens)
            fused = self.norm(fusion_tokens + fused)
            stop_repr = fused[:, 0, :].squeeze(0)
            uncertainty_repr = fused[:, 1, :].squeeze(0)
            grounded_mean = grounded_queries.mean(dim=1).squeeze(0)

            element_grounded = grounded_queries[:, : element_queries.shape[0], :].squeeze(0)
            element_sims = (element_grounded * element_queries).sum(dim=-1)
            raw_score = element_sims.mean()
            temporal_max = element_sims.max()
            soft_coverage = torch.sigmoid((element_sims - obj_threshold) * coverage_temperature).mean()
            count_ratio = torch.tensor(
                element_queries.shape[0] / max(max_target_elements, 1),
                device=device,
                dtype=frame_cls_proj.dtype,
            )

            score_inp = torch.cat(
                [
                    stop_repr,
                    grounded_mean,
                    temporal_mean,
                    temporal_last,
                    torch.stack([raw_score, temporal_max, soft_coverage, count_ratio], dim=0),
                ],
                dim=0,
            )
            logit = self.score_head(score_inp).squeeze(-1)
            uncertainty = F.softplus(
                self.uncertainty_head(torch.cat([uncertainty_repr, stop_repr], dim=0)).squeeze(-1)
            )
            logits.append(logit)
            raw_scores.append(raw_score)
            uncertainties.append(uncertainty)
            reprs.append(stop_repr)

        return (
            torch.stack(logits, dim=0),
            torch.stack(raw_scores, dim=0),
            torch.stack(uncertainties, dim=0),
            F.normalize(torch.stack(reprs, dim=0), dim=-1),
        )


def encode_batch(model, preprocess, rows, device, image_load_workers: int = 8):
    frame_paths, frame_owners = flatten_frame_paths(rows)
    images = load_image_tensors(frame_paths, preprocess, max_workers=image_load_workers)
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


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    labels = labels.view(-1)
    sim = embeddings @ embeddings.transpose(0, 1) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    logits_mask = ~eye
    exp_sim = torch.exp(sim) * logits_mask
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & logits_mask
    positive_count = positive_mask.sum(dim=1)
    if int((positive_count > 0).sum().item()) == 0:
        return embeddings.new_zeros(())
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_prob_pos[positive_count > 0].mean()
    return loss


def forward_stop_head(
    head,
    image_features,
    image_patch_features,
    frame_owners,
    text_features,
    text_owners,
    action_features,
    relation_features,
    batch_size,
    max_target_elements,
    obj_threshold,
    coverage_temperature,
):
    image_proj = head.project_image(image_features)
    text_proj = head.project_text(text_features)
    action_proj = head.project_text(action_features)
    if isinstance(head, CrossAttentionStopHead):
        return head.score_from_sequences(
            image_proj,
            frame_owners,
            text_proj,
            text_owners,
            action_proj,
            batch_size,
            max_target_elements,
            obj_threshold,
            coverage_temperature,
        ), None, None
    if isinstance(head, GroundingQueryStopHead):
        return head.score_from_grounding(
            image_features,
            image_patch_features,
            frame_owners,
            text_features,
            text_owners,
            action_features,
            relation_features,
            batch_size,
            max_target_elements,
            obj_threshold,
            coverage_temperature,
        )
    history_mean_proj, history_last_proj = aggregate_history_features(image_proj, frame_owners, batch_size)
    stats, raw_scores = compute_stats(
        history_mean_proj,
        history_last_proj,
        text_proj,
        text_owners,
        action_proj,
        batch_size,
        max_target_elements,
        obj_threshold,
        coverage_temperature,
    )
    logits = head.score_from_stats(stats)
    return logits, raw_scores, None, None


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
        if isinstance(head, GroundingQueryStopHead):
            (
                image_features,
                image_patch_features,
                frame_owners,
                text_features,
                text_owners,
                action_features,
                relation_features,
            ) = encode_batch_grounding(model, preprocess, batch_rows, device)
        else:
            image_features, frame_owners, text_features, text_owners, action_features = encode_batch(
                model, preprocess, batch_rows, device
            )
            image_patch_features = None
            relation_features = None
        logits, raw_scores = forward_stop_head(
            head,
            image_features,
            image_patch_features,
            frame_owners,
            text_features,
            text_owners,
            action_features,
            relation_features,
            len(batch_rows),
            max_target_elements,
            obj_threshold,
            coverage_temperature,
        )[:2]
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


@torch.no_grad()
def evaluate_cached(head, rows, device, batch_size, max_target_elements, obj_threshold, coverage_temperature, image_cache, text_cache, action_cache):
    head.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    pos_logits = []
    neg_logits = []
    pos_raw = []
    neg_raw = []
    for batch_rows in batched(rows, batch_size):
        image_features, frame_owners, text_features, text_owners, action_features = encode_batch_from_cache(
            batch_rows, device, image_cache, text_cache, action_cache
        )
        logits, raw_scores = forward_stop_head(
            head,
            image_features,
            None,
            frame_owners,
            text_features,
            text_owners,
            action_features,
            None,
            len(batch_rows),
            max_target_elements,
            obj_threshold,
            coverage_temperature,
        )[:2]
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
    parser.add_argument(
        "--score-hidden-dims",
        default="",
        help="Comma-separated hidden dims for score head MLP. Empty means use hidden-dim as a single layer.",
    )
    parser.add_argument("--head-type", choices=["mlp", "cross_attn", "grounding_v4"], default="mlp")
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.05)
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--obj-threshold", type=float, default=0.25)
    parser.add_argument("--coverage-temperature", type=float, default=12.0)
    parser.add_argument("--max-target-elements", type=int, default=4)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--cache-frozen-features", action="store_true")
    parser.add_argument("--init-head-ckpt", default="", help="Optional checkpoint path to initialize the stop head from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-load-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    score_hidden_dims = [int(x) for x in str(args.score_hidden_dims).split(",") if str(x).strip()]
    if not score_hidden_dims:
        score_hidden_dims = [args.hidden_dim]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = longclip.load(args.longclip_path, device=device)
    if not args.train_backbone:
        for param in model.parameters():
            param.requires_grad = False
    feature_dim = int(model.text_projection.shape[-1])
    if args.head_type == "cross_attn":
        head = CrossAttentionStopHead(
            feature_dim,
            args.proj_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.attn_heads,
            dropout=args.attn_dropout,
        ).to(device)
    elif args.head_type == "grounding_v4":
        head = GroundingQueryStopHead(
            feature_dim,
            args.proj_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.attn_heads,
            dropout=args.attn_dropout,
        ).to(device)
    else:
        head = HistoryElementLongCLIPHead(
            feature_dim,
            args.proj_dim,
            args.hidden_dim,
            score_hidden_dims=score_hidden_dims,
        ).to(device)
    if args.init_head_ckpt:
        ckpt = torch.load(args.init_head_ckpt, map_location="cpu")
        head.load_state_dict(ckpt["head"], strict=True)

    rows = load_manifest(Path(args.manifest), args.max_samples)
    rows = prepare_rows(rows, args.max_target_elements, args.history_frames)
    train_rows, val_rows = split_rows(rows, args.train_ratio, args.seed)

    image_cache = None
    text_cache = None
    action_cache = None
    use_feature_cache = args.cache_frozen_features and not args.train_backbone and args.head_type == "mlp"
    if use_feature_cache:
        image_cache, text_cache, action_cache = build_frozen_feature_cache(
            model,
            preprocess,
            train_rows + val_rows,
            device,
            image_load_workers=args.image_load_workers,
        )

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
            if use_feature_cache:
                image_features, frame_owners, text_features, text_owners, action_features = encode_batch_from_cache(
                    batch_rows, device, image_cache, text_cache, action_cache
                )
                image_patch_features = None
                relation_features = None
            elif isinstance(head, GroundingQueryStopHead):
                (
                    image_features,
                    image_patch_features,
                    frame_owners,
                    text_features,
                    text_owners,
                    action_features,
                    relation_features,
                ) = encode_batch_grounding(
                    model,
                    preprocess,
                    batch_rows,
                    device,
                    image_load_workers=args.image_load_workers,
                )
            else:
                image_features, frame_owners, text_features, text_owners, action_features = encode_batch(
                    model,
                    preprocess,
                    batch_rows,
                    device,
                    image_load_workers=args.image_load_workers,
                )
                image_patch_features = None
                relation_features = None
            logits, _, uncertainties, embeddings = forward_stop_head(
                head,
                image_features,
                image_patch_features,
                frame_owners,
                text_features,
                text_owners,
                action_features,
                relation_features,
                len(batch_rows),
                args.max_target_elements,
                args.obj_threshold,
                args.coverage_temperature,
            )
            labels = torch.tensor([float(row["label"]) for row in batch_rows], device=device)
            adjusted_logits = logits / (1.0 + uncertainties) if uncertainties is not None else logits
            bce_loss = F.binary_cross_entropy_with_logits(adjusted_logits, labels)
            rank_loss = pairwise_rank_loss(logits, labels, args.rank_margin)
            contrastive_loss = (
                supervised_contrastive_loss(embeddings, labels, temperature=args.contrastive_temperature)
                if embeddings is not None
                else logits.new_zeros(())
            )
            uncertainty_loss = (
                (uncertainties.mean() if uncertainties is not None else logits.new_zeros(()))
            )
            loss = (
                bce_loss
                + args.rank_loss_weight * rank_loss
                + args.contrastive_loss_weight * contrastive_loss
                + args.uncertainty_loss_weight * uncertainty_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_bce += float(bce_loss.item())
            total_rank += float(rank_loss.item())
            total_batches += 1

        if use_feature_cache:
            train_metrics = evaluate_cached(
                head,
                train_rows[: min(len(train_rows), 1024)],
                device,
                args.batch_size,
                args.max_target_elements,
                args.obj_threshold,
                args.coverage_temperature,
                image_cache,
                text_cache,
                action_cache,
            )
            val_metrics = evaluate_cached(
                head,
                val_rows,
                device,
                args.batch_size,
                args.max_target_elements,
                args.obj_threshold,
                args.coverage_temperature,
                image_cache,
                text_cache,
                action_cache,
            )
        else:
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
                    "head_type": args.head_type,
                    "attn_heads": args.attn_heads,
                    "attn_dropout": args.attn_dropout,
                    "score_hidden_dims": score_hidden_dims,
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
            "head_type": args.head_type,
            "attn_heads": args.attn_heads,
            "attn_dropout": args.attn_dropout,
            "score_hidden_dims": score_hidden_dims,
            "obj_threshold": args.obj_threshold,
            "coverage_temperature": args.coverage_temperature,
            "max_target_elements": args.max_target_elements,
            "history_frames": args.history_frames,
        },
        output_dir / "last_longclip_stop_elements_history.pt",
    )


if __name__ == "__main__":
    main()
