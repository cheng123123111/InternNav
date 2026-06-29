from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def logit(prob: float) -> float:
    prob = min(1.0 - 1e-5, max(1e-5, float(prob)))
    return math.log(prob / (1.0 - prob))


def percentile_rank(values: list[float], current: float) -> float:
    if not values:
        return 0.0
    return sum(1 for x in values if x <= current) / max(1, len(values))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.m_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem = self.m_norm(memory)
        x, _ = self.attn(self.q_norm(query), mem, mem, need_weights=False)
        query = query + x
        query = query + self.ffn(self.ffn_norm(query))
        return query


class StopResidualAdapter(nn.Module):
    """Cross-attention STOP verifier trained to output a binary STOP logit delta."""

    def __init__(
        self,
        hidden_dim: int = 17920,
        token_dim: int = 3584,
        model_dim: int = 256,
        progress_token_dim: int = 8,
        max_progress_tokens: int = 8,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_qwen_tokens = hidden_dim // token_dim
        self.qwen_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, model_dim))
        self.progress_proj = nn.Sequential(
            nn.LayerNorm(progress_token_dim),
            nn.Linear(progress_token_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.progress_type_embed = nn.Embedding(max_progress_tokens, model_dim)
        self.query = nn.Parameter(torch.randn(4, model_dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(model_dim, 8) for _ in range(2)])
        self.out = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))

    def forward(
        self,
        hidden: torch.Tensor,
        progress_tokens: torch.Tensor,
        progress_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = hidden.shape[0]
        qwen_tokens = hidden.view(bsz, self.num_qwen_tokens, self.token_dim)
        progress = self.progress_proj(progress_tokens)
        type_ids = torch.arange(progress.shape[1], device=progress.device).clamp_max(
            self.progress_type_embed.num_embeddings - 1
        )
        progress = progress + self.progress_type_embed(type_ids).unsqueeze(0)
        if progress_valid is not None:
            progress = progress * progress_valid.to(progress.dtype).unsqueeze(-1)
        memory = torch.cat([self.qwen_proj(qwen_tokens), progress], dim=1)
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        return self.out(query.mean(dim=1)).squeeze(-1)


def load_stop_residual_adapter(path: str, device: torch.device | str) -> tuple[StopResidualAdapter, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    token_dim = int(state["qwen_proj.1.weight"].shape[1])
    model_dim = int(state["qwen_proj.1.weight"].shape[0])
    progress_token_dim = int(state["progress_proj.1.weight"].shape[1])
    max_progress_tokens = int(state["progress_type_embed.weight"].shape[0])
    hidden_dim = int(ckpt.get("hidden_dim", token_dim * 5))
    model = StopResidualAdapter(
        hidden_dim=hidden_dim,
        token_dim=token_dim,
        model_dim=model_dim,
        progress_token_dim=progress_token_dim,
        max_progress_tokens=max_progress_tokens,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    meta = {
        "hidden_dim": hidden_dim,
        "token_dim": token_dim,
        "num_qwen_tokens": model.num_qwen_tokens,
        "progress_windows": list(ckpt.get("progress_windows", [10, 20, 30, 50])),
        "num_progress_tokens": int(ckpt.get("num_progress_tokens", max_progress_tokens)),
        "progress_token_dim": progress_token_dim,
        "report": ckpt.get("report", {}),
    }
    return model, meta


def build_online_progress_tokens(
    history: list[dict[str, float]],
    *,
    step_id: int,
    current_progress: float,
    qstop_prob: float,
    first_token_prob: float,
    first_token_entropy: float,
    qwen_says_stop: bool,
    windows: list[int] | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the same six progress tokens used by the offline residual adapter."""
    windows = windows or [10, 20, 30, 50]
    token_dim = 8
    num_tokens = 2 + len(windows)
    tokens = torch.zeros(1, num_tokens, token_dim, dtype=torch.float32, device=device)
    valid = torch.ones(1, num_tokens, dtype=torch.bool, device=device)

    entries = [
        {
            "step_id": int(safe_float(row.get("step_id"), 0.0)),
            "progress": max(0.0, min(1.0, safe_float(row.get("progress"), 0.0))),
        }
        for row in history
        if int(safe_float(row.get("step_id"), -1.0)) <= int(step_id)
    ]
    if not entries or entries[-1]["step_id"] != int(step_id):
        entries.append({"step_id": int(step_id), "progress": max(0.0, min(1.0, float(current_progress)))})
    entries = sorted(entries, key=lambda row: row["step_id"])
    current_index = max(i for i, row in enumerate(entries) if row["step_id"] <= int(step_id))
    prev = entries[:current_index]
    cur_p = max(0.0, min(1.0, float(current_progress)))
    prev_steps = [int(row["step_id"]) for row in prev]
    prev_vals = [float(row["progress"]) for row in prev]
    values_prefix = prev_vals + [cur_p]
    max_so_far = max(values_prefix) if values_prefix else cur_p
    stagnation = 0
    for pv in reversed(prev_vals):
        if pv > max_so_far - 1e-6:
            stagnation += 1
        else:
            break

    tokens[0, 0] = torch.tensor(
        [
            cur_p,
            max_so_far,
            cur_p - max_so_far,
            percentile_rank(values_prefix, cur_p),
            min(1.0, current_index / 50.0),
            min(1.0, stagnation / 50.0),
            0.0,
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    for wi, window in enumerate(windows, start=1):
        target = int(step_id) - int(window)
        prev_idx = None
        for j in range(len(prev_steps) - 1, -1, -1):
            if prev_steps[j] <= target:
                prev_idx = j
                break
        if prev_idx is None:
            if prev_vals:
                prev_idx = 0
            else:
                valid[0, wi] = False
        if prev_idx is None:
            prev_p = cur_p
            actual_gap = 0.0
            recent_vals = [cur_p]
        else:
            prev_p = prev_vals[prev_idx]
            actual_gap = max(1.0, float(int(step_id) - prev_steps[prev_idx]))
            recent_vals = [v for s, v in zip(prev_steps, prev_vals) if s >= int(step_id) - int(window)] + [cur_p]
        delta = cur_p - prev_p
        slope = delta / max(1.0, actual_gap)
        tokens[0, wi] = torch.tensor(
            [
                prev_p,
                delta,
                slope,
                percentile_rank(recent_vals, cur_p),
                max(recent_vals) if recent_vals else cur_p,
                cur_p - (sum(recent_vals) / max(1, len(recent_vals))),
                min(1.0, actual_gap / max(1, int(window))),
                float(valid[0, wi].item()),
            ],
            dtype=torch.float32,
            device=device,
        )

    qstop = safe_float(qstop_prob, 0.0)
    entropy = safe_float(first_token_entropy, 0.0)
    first_prob = safe_float(first_token_prob, 0.0)
    tokens[0, -1] = torch.tensor(
        [
            qstop,
            logit(qstop) / 8.0,
            entropy / 8.0,
            first_prob,
            float(bool(qwen_says_stop)),
            0.0,
            min(1.0, current_index / 50.0),
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    return tokens, valid
