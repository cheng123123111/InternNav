from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

TRAJ_TOKEN_INDEX = 151667
IMAGE_TOKEN_INDEX = 151655
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


@dataclass
class InternVLAN1CausalLMOutputWithPast(CausalLMOutputWithPast):
    progress_aux_logits: Optional[torch.FloatTensor] = None
    progress_aux_query_logits: Optional[torch.FloatTensor] = None


class ProgressAuxTemporalEncoder(nn.Module):
    """Order-preserving pose/action encoder for progress supervision."""

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        max_tokens: int = 8,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_tokens = int(max_tokens)
        while query_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.input_proj = nn.Linear(int(input_dim), query_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.max_tokens, query_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=int(num_heads),
            dim_feedforward=query_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(query_dim)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, temporal_tokens, temporal_mask=None):
        temporal_tokens = torch.nan_to_num(
            temporal_tokens.to(dtype=self.input_proj.weight.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        x = self.input_proj(temporal_tokens)
        pos = self.pos_embed[: x.shape[1]].to(device=x.device, dtype=x.dtype)
        x = x + pos.unsqueeze(0)
        if temporal_mask is None:
            temporal_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            temporal_mask = temporal_mask.to(device=x.device, dtype=torch.bool)
        empty_rows = temporal_mask.sum(dim=1) == 0
        if empty_rows.any():
            temporal_mask = torch.where(empty_rows[:, None], torch.ones_like(temporal_mask), temporal_mask)
        x = self.encoder(x, src_key_padding_mask=~temporal_mask)
        x = self.norm(x)
        return x * temporal_mask.to(dtype=x.dtype).unsqueeze(-1), temporal_mask


class ProgressAuxQueryHead(nn.Module):
    """Action-conditioned query critic for dense VLN completion supervision."""

    def __init__(
        self,
        hidden_size: int,
        output_dim: int = 10,
        num_queries: int = 10,
        query_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        temporal_input_dim: int = 0,
        temporal_max_tokens: int = 8,
        temporal_layers: int = 1,
        causal_feature_dim: int = 0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        # A single shared query can support multiple readouts.  Older checkpoints
        # keep their configured query count, while the compact V11 critic uses one.
        self.num_queries = max(1, int(num_queries))
        self.query_dim = int(query_dim)
        while self.query_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads

        self.query_embed = nn.Parameter(torch.empty(self.num_queries, self.query_dim))
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.memory_proj = nn.Linear(hidden_size, self.query_dim)
        self.global_proj = nn.Linear(hidden_size, self.query_dim)
        self.q_proj = nn.Linear(self.query_dim, self.query_dim)
        self.k_proj = nn.Linear(self.query_dim, self.query_dim)
        self.v_proj = nn.Linear(self.query_dim, self.query_dim)
        self.attn_out_proj = nn.Linear(self.query_dim, self.query_dim)
        self.temporal_encoder = None
        if int(temporal_input_dim) > 0:
            self.temporal_encoder = ProgressAuxTemporalEncoder(
                input_dim=int(temporal_input_dim),
                query_dim=self.query_dim,
                max_tokens=int(temporal_max_tokens),
                num_heads=self.num_heads,
                num_layers=int(temporal_layers),
                dropout=dropout,
            )
        self.query_ffn = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.query_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.query_dim * 2, self.query_dim),
        )
        self.query_score = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, max(64, self.query_dim // 2)),
            nn.GELU(approximate="tanh"),
            nn.Linear(max(64, self.query_dim // 2), 1),
        )
        self.final_head = nn.Sequential(
            nn.LayerNorm(self.query_dim + self.num_queries),
            nn.Linear(self.query_dim + self.num_queries, max(128, self.query_dim // 2)),
            nn.GELU(approximate="tanh"),
            nn.Linear(max(128, self.query_dim // 2), self.output_dim),
        )
        self.causal_prior = None
        if int(causal_feature_dim) > 0:
            prior_dim = max(32, int(causal_feature_dim) * 8)
            self.causal_prior = nn.Sequential(
                nn.LayerNorm(int(causal_feature_dim)),
                nn.Linear(int(causal_feature_dim), prior_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(prior_dim, 1),
            )
            nn.init.zeros_(self.causal_prior[-1].weight)
            nn.init.zeros_(self.causal_prior[-1].bias)
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

    @staticmethod
    def _align_mask_to_hidden(mask, hidden_states):
        if mask is None:
            return None
        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        seq_len = hidden_states.shape[1]
        if mask.shape[1] == seq_len:
            return mask
        if mask.shape[1] > seq_len:
            return mask[:, -seq_len:]
        pad = torch.ones(
            mask.shape[0],
            seq_len - mask.shape[1],
            device=mask.device,
            dtype=torch.bool,
        )
        return torch.cat([pad, mask], dim=1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        memory_mask=None,
        temporal_tokens=None,
        temporal_mask=None,
        causal_features=None,
        return_query_tokens: bool = False,
    ):
        if self.memory_proj.weight.dtype != torch.float32:
            self.float()

        hidden_states = torch.nan_to_num(hidden_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
        batch_size = hidden_states.shape[0]
        aligned_attention_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            aligned_attention_mask = self._align_mask_to_hidden(attention_mask, hidden_states)
        if memory_mask is None:
            if aligned_attention_mask is not None:
                memory_mask = aligned_attention_mask
            else:
                memory_mask = torch.ones(
                    hidden_states.shape[:2],
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
        else:
            memory_mask = self._align_mask_to_hidden(memory_mask, hidden_states)

        if aligned_attention_mask is not None:
            memory_mask = memory_mask & aligned_attention_mask

        fallback_mask = None
        if aligned_attention_mask is not None:
            fallback_mask = aligned_attention_mask
        empty_rows = memory_mask.sum(dim=1) == 0
        if empty_rows.any():
            if fallback_mask is not None:
                memory_mask = torch.where(empty_rows[:, None], fallback_mask, memory_mask)
            memory_mask = torch.where(
                (memory_mask.sum(dim=1) == 0)[:, None],
                torch.ones_like(memory_mask),
                memory_mask,
            )

        hidden_mask_float = memory_mask.to(dtype=torch.float32).unsqueeze(-1)
        pooled_hidden = (hidden_states * hidden_mask_float).sum(dim=1) / hidden_mask_float.sum(dim=1).clamp(min=1.0)
        memory = self.memory_proj(self.memory_norm(hidden_states))
        if self.temporal_encoder is not None and temporal_tokens is not None:
            temporal_memory, temporal_valid = self.temporal_encoder(
                temporal_tokens.to(device=hidden_states.device),
                temporal_mask.to(device=hidden_states.device) if temporal_mask is not None else None,
            )
            memory = torch.cat([memory, temporal_memory.to(device=memory.device, dtype=memory.dtype)], dim=1)
            memory_mask = torch.cat([memory_mask, temporal_valid.to(device=memory_mask.device)], dim=1)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.global_proj(pooled_hidden).unsqueeze(1)

        q = self.q_proj(queries).view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch_size, memory.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch_size, memory.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_scores = attn_scores.masked_fill(~memory_mask[:, None, None, :], -1.0e4)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.num_queries, self.query_dim)
        attn_output = self.attn_out_proj(attn_output)
        query_tokens = queries + attn_output
        query_tokens = query_tokens + self.query_ffn(query_tokens)
        query_logits = self.query_score(query_tokens).squeeze(-1)
        fused = torch.cat([query_tokens.mean(dim=1), query_logits], dim=-1)
        final_logits = self.final_head(fused)
        if self.causal_prior is not None and causal_features is not None:
            prior_logits = self.causal_prior(
                torch.nan_to_num(
                    causal_features.to(device=final_logits.device, dtype=torch.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            final_logits = torch.cat(
                [final_logits[:, :1] + prior_logits, final_logits[:, 1:]],
                dim=1,
            )
        query_logits = torch.nan_to_num(query_logits, nan=0.0, posinf=0.0, neginf=0.0)
        final_logits = torch.nan_to_num(final_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if return_query_tokens:
            return final_logits, query_logits, query_tokens
        return final_logits, query_logits


class ProgressActionAdapter(nn.Module):
    """Small cross-attention adapter that turns completion queries into action-token logit bias."""

    def __init__(
        self,
        hidden_size: int,
        query_dim: int,
        progress_dim: int = 10,
        num_actions: int = 5,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.num_actions = int(num_actions)
        while self.query_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.hidden_proj = nn.Linear(hidden_size, self.query_dim)
        self.progress_norm = nn.LayerNorm(self.query_dim)
        self.progress_value_proj = nn.Linear(int(progress_dim), self.query_dim)
        self.q_proj = nn.Linear(self.query_dim, self.query_dim)
        self.k_proj = nn.Linear(self.query_dim, self.query_dim)
        self.v_proj = nn.Linear(self.query_dim, self.query_dim)
        self.out_proj = nn.Linear(self.query_dim, self.query_dim)
        self.bias_head = nn.Sequential(
            nn.LayerNorm(self.query_dim * 2),
            nn.Linear(self.query_dim * 2, max(64, self.query_dim // 2)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(float(dropout)),
            nn.Linear(max(64, self.query_dim // 2), self.num_actions),
        )
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

    def forward(self, hidden_states, progress_query_tokens, progress_logits=None, attention_mask=None):
        if self.hidden_proj.weight.dtype != torch.float32:
            self.float()

        hidden_states = torch.nan_to_num(hidden_states.float(), nan=0.0, posinf=0.0, neginf=0.0)
        progress_query_tokens = torch.nan_to_num(
            progress_query_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        hidden_tokens = self.hidden_proj(self.hidden_norm(hidden_states))
        progress_tokens = self.progress_norm(progress_query_tokens)
        if progress_logits is not None:
            progress_values = torch.sigmoid(
                torch.nan_to_num(progress_logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
            )
            value_token = self.progress_value_proj(progress_values).unsqueeze(1)
            progress_tokens = torch.cat([progress_tokens, value_token], dim=1)

        batch_size, seq_len, _ = hidden_tokens.shape
        q = self.q_proj(hidden_tokens).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(progress_tokens).view(
            batch_size, progress_tokens.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(progress_tokens).view(
            batch_size, progress_tokens.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.query_dim)
        context = self.out_proj(context)
        action_bias = self.bias_head(torch.cat([hidden_tokens, context], dim=-1))
        if attention_mask is not None and attention_mask.ndim == 2:
            mask = attention_mask.to(device=action_bias.device, dtype=torch.bool)
            if mask.shape[1] != seq_len:
                if mask.shape[1] > seq_len:
                    mask = mask[:, -seq_len:]
                else:
                    pad = torch.ones(
                        mask.shape[0],
                        seq_len - mask.shape[1],
                        device=mask.device,
                        dtype=torch.bool,
                    )
                    mask = torch.cat([pad, mask], dim=1)
            action_bias = action_bias * mask.to(dtype=action_bias.dtype).unsqueeze(-1)
        return torch.nan_to_num(action_bias, nan=0.0, posinf=0.0, neginf=0.0)


class ProgressInQwenTemporalConditioner(nn.Module):
    """Turns ordered action/pose history into an additive full-width query condition."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        max_tokens: int = 8,
        temporal_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.temporal_encoder = ProgressAuxTemporalEncoder(
            input_dim=input_dim,
            query_dim=temporal_dim,
            max_tokens=max_tokens,
            num_heads=8,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(temporal_dim)
        self.output_proj = nn.Linear(temporal_dim, hidden_size)

    def forward(self, temporal_tokens, temporal_mask=None):
        if self.output_proj.weight.dtype != torch.float32:
            self.float()
        encoded, valid = self.temporal_encoder(temporal_tokens, temporal_mask)
        valid_float = valid.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1.0)
        return self.output_proj(self.output_norm(pooled))


class ProgressInQwenReadout(nn.Module):
    """Reads completion and success from progress queries after all Qwen layers."""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 4,
        bottleneck_dim: int = 512,
        query_specific_readout: bool = False,
        gated_aux_fusion: bool = False,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_specific_readout = bool(query_specific_readout)
        self.gated_aux_fusion = bool(gated_aux_fusion)
        if self.gated_aux_fusion and self.num_queries < 4:
            raise ValueError("gated auxiliary fusion requires four progress queries")
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.query_score = nn.Linear(
            bottleneck_dim,
            self.num_queries if self.query_specific_readout else 1,
        )
        self.pool_score = nn.Linear(bottleneck_dim, 1)
        if self.gated_aux_fusion:
            self.aux_pool_score = nn.Linear(bottleneck_dim, 1)
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(bottleneck_dim),
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.GELU(approximate="tanh"),
            )
            self.aux_gate = nn.Parameter(torch.zeros(1))
            final_input_dim = bottleneck_dim + 2
        else:
            self.aux_pool_score = None
            self.aux_proj = None
            self.register_parameter("aux_gate", None)
            final_input_dim = bottleneck_dim + self.num_queries
        self.final_head = nn.Sequential(
            nn.LayerNorm(final_input_dim),
            nn.Linear(final_input_dim, max(128, bottleneck_dim // 2)),
            nn.GELU(approximate="tanh"),
            nn.Linear(max(128, bottleneck_dim // 2), 2),
        )

    def forward(self, query_hidden_states):
        if self.query_proj.weight.dtype != torch.float32:
            self.float()
        query_hidden_states = torch.nan_to_num(
            query_hidden_states.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        query_features = F.gelu(
            self.query_proj(self.query_norm(query_hidden_states)), approximate="tanh"
        )
        raw_query_logits = self.query_score(query_features)
        if self.query_specific_readout:
            query_logits = raw_query_logits.diagonal(dim1=1, dim2=2)
        else:
            query_logits = raw_query_logits.squeeze(-1)
        if self.gated_aux_fusion:
            main_indices = torch.tensor(
                [0, 3], device=query_features.device, dtype=torch.long
            )
            aux_indices = torch.tensor(
                [1, 2], device=query_features.device, dtype=torch.long
            )
            main_features = query_features.index_select(1, main_indices)
            aux_features = query_features.index_select(1, aux_indices)
            main_weights = torch.softmax(
                self.pool_score(main_features).squeeze(-1), dim=1
            )
            aux_weights = torch.softmax(
                self.aux_pool_score(aux_features).squeeze(-1), dim=1
            )
            pooled_main = (main_features * main_weights.unsqueeze(-1)).sum(dim=1)
            pooled_aux = (aux_features * aux_weights.unsqueeze(-1)).sum(dim=1)
            fused = pooled_main + torch.tanh(self.aux_gate) * self.aux_proj(pooled_aux)
            final_input = torch.cat(
                [fused, query_logits.index_select(1, main_indices)], dim=-1
            )
        else:
            pool_weights = torch.softmax(
                self.pool_score(query_features).squeeze(-1), dim=1
            )
            pooled = (query_features * pool_weights.unsqueeze(-1)).sum(dim=1)
            final_input = torch.cat([pooled, query_logits], dim=-1)
        final_logits = self.final_head(final_input)
        return (
            torch.nan_to_num(final_logits, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(query_logits, nan=0.0, posinf=0.0, neginf=0.0),
        )


class ProgressValueStreamBlock(nn.Module):
    """One-way Value block: value tokens read Qwen, while Qwen stays unchanged."""

    def __init__(
        self,
        backbone_dim: int,
        value_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        while value_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.self_norm = nn.LayerNorm(value_dim)
        self.self_attn = nn.MultiheadAttention(
            value_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(value_dim)
        self.cross_memory_norm = nn.LayerNorm(backbone_dim)
        self.cross_attn = nn.MultiheadAttention(
            value_dim,
            num_heads,
            dropout=dropout,
            kdim=backbone_dim,
            vdim=backbone_dim,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(value_dim)
        self.ffn = nn.Sequential(
            nn.Linear(value_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, value_dim),
        )

    def forward(self, value_tokens, memory, memory_mask=None):
        if not bool(torch.isfinite(value_tokens).all().item()):
            raise FloatingPointError("ProgressValueStream entered a block with non-finite value tokens")
        query = self.self_norm(value_tokens)
        update, _ = self.self_attn(query, query, query, need_weights=False)
        if not bool(torch.isfinite(update).all().item()):
            raise FloatingPointError("ProgressValueStream self-attention produced non-finite values")
        value_tokens = value_tokens + update

        query = self.cross_query_norm(value_tokens)
        memory = self.cross_memory_norm(
            memory.to(dtype=self.cross_memory_norm.weight.dtype)
        )
        if not bool(torch.isfinite(memory).all().item()):
            raise FloatingPointError("ProgressValueStream Qwen memory contains non-finite values")
        update, _ = self.cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=(~memory_mask if memory_mask is not None else None),
            need_weights=False,
        )
        if not bool(torch.isfinite(update).all().item()):
            raise FloatingPointError("ProgressValueStream cross-attention produced non-finite values")
        value_tokens = value_tokens + update
        update = self.ffn(self.ffn_norm(value_tokens))
        if not bool(torch.isfinite(update).all().item()):
            raise FloatingPointError("ProgressValueStream FFN produced non-finite values")
        value_tokens = value_tokens + update
        return value_tokens


class ProgressValueStream(nn.Module):
    """Narrow Transformer stream coupled one-way to selected Qwen layers."""

    def __init__(
        self,
        backbone_dim: int,
        backbone_layers: int,
        depth: int = 8,
        value_dim: int = 512,
        num_queries: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        temporal_input_dim: int = 0,
        temporal_max_tokens: int = 8,
        temporal_layers: int = 1,
        query_specific_readout: bool = True,
        gated_aux_fusion: bool = True,
    ):
        super().__init__()
        depth = max(1, min(int(depth), int(backbone_layers)))
        # Qwen returns embeddings at index 0, then one state per language layer.
        selected = np.linspace(1, int(backbone_layers), num=depth, dtype=np.int64)
        self.selected_hidden_indices = tuple(int(index) for index in selected.tolist())
        self.query_embed = nn.Parameter(torch.empty(1, int(num_queries), int(value_dim)))
        self.temporal_encoder = None
        if int(temporal_input_dim) > 0:
            self.temporal_encoder = ProgressAuxTemporalEncoder(
                input_dim=int(temporal_input_dim),
                query_dim=int(value_dim),
                max_tokens=int(temporal_max_tokens),
                num_heads=int(num_heads),
                num_layers=int(temporal_layers),
            )
        self.blocks = nn.ModuleList(
            [
                ProgressValueStreamBlock(
                    backbone_dim=int(backbone_dim),
                    value_dim=int(value_dim),
                    num_heads=int(num_heads),
                    ffn_dim=int(ffn_dim),
                )
                for _ in self.selected_hidden_indices
            ]
        )
        self.final_norm = nn.LayerNorm(int(value_dim))
        self.readout = ProgressInQwenReadout(
            hidden_size=int(value_dim),
            num_queries=int(num_queries),
            bottleneck_dim=int(value_dim),
            query_specific_readout=bool(query_specific_readout),
            gated_aux_fusion=bool(gated_aux_fusion),
        )
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states,
        memory_mask=None,
        temporal_tokens=None,
        temporal_mask=None,
    ):
        if hidden_states is None:
            raise ValueError("ProgressValueStream requires Qwen output_hidden_states")
        batch_size = hidden_states[0].shape[0]
        value_tokens = self.query_embed.expand(batch_size, -1, -1)
        if self.temporal_encoder is not None and temporal_tokens is not None:
            temporal_memory, temporal_valid = self.temporal_encoder(
                temporal_tokens.to(device=value_tokens.device),
                temporal_mask.to(device=value_tokens.device) if temporal_mask is not None else None,
            )
            valid = temporal_valid.to(dtype=temporal_memory.dtype).unsqueeze(-1)
            temporal_summary = (temporal_memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            value_tokens = value_tokens + temporal_summary.unsqueeze(1)

        for block_index, (block, hidden_index) in enumerate(
            zip(self.blocks, self.selected_hidden_indices)
        ):
            memory = hidden_states[hidden_index].detach()
            value_tokens = block(value_tokens, memory, memory_mask=memory_mask)
            if not bool(torch.isfinite(value_tokens).all().item()):
                valid_counts = (
                    memory_mask.sum(dim=1).detach().cpu().tolist()
                    if memory_mask is not None
                    else None
                )
                raise FloatingPointError(
                    "ProgressValueStream produced non-finite values at "
                    f"block={block_index}, qwen_hidden={hidden_index}, "
                    f"valid_memory_tokens={valid_counts}"
                )
        value_tokens = self.final_norm(value_tokens)
        return self.readout(value_tokens)


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    model_type = "internvla_n1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        progress_query_dim = int(
            getattr(config, "progress_aux_query_dim", min(512, max(128, config.hidden_size // 8)))
        )
        self.progress_aux_head = ProgressAuxQueryHead(
            hidden_size=config.hidden_size,
            output_dim=int(getattr(config, "progress_aux_dim", 10)),
            num_queries=int(getattr(config, "progress_aux_num_queries", 10)),
            query_dim=progress_query_dim,
            num_heads=int(getattr(config, "progress_aux_num_heads", 8)),
            temporal_input_dim=int(getattr(config, "progress_aux_temporal_input_dim", 16)),
            temporal_max_tokens=int(getattr(config, "progress_aux_temporal_max_tokens", 8)),
            temporal_layers=int(getattr(config, "progress_aux_temporal_layers", 1)),
            causal_feature_dim=int(getattr(config, "progress_aux_causal_feature_dim", 0)),
        )
        self.progress_action_adapter = ProgressActionAdapter(
            hidden_size=config.hidden_size,
            query_dim=progress_query_dim,
            progress_dim=int(getattr(config, "progress_aux_dim", 10)),
            num_actions=int(getattr(config, "progress_action_num_actions", 5)),
            num_heads=int(getattr(config, "progress_aux_num_heads", 8)),
        )
        self.progress_in_qwen_temporal_conditioner = None
        self.progress_in_qwen_readout = None
        self.progress_value_stream = None
        if bool(getattr(config, "progress_aux_in_qwen_queries", False)):
            temporal_input_dim = int(getattr(config, "progress_aux_temporal_input_dim", 0) or 0)
            if temporal_input_dim > 0:
                self.progress_in_qwen_temporal_conditioner = ProgressInQwenTemporalConditioner(
                    input_dim=temporal_input_dim,
                    hidden_size=config.hidden_size,
                    max_tokens=int(getattr(config, "progress_aux_temporal_max_tokens", 8)),
                    temporal_dim=int(getattr(config, "progress_aux_in_qwen_temporal_dim", 256)),
                    num_layers=int(getattr(config, "progress_aux_temporal_layers", 1)),
                )
            self.progress_in_qwen_readout = ProgressInQwenReadout(
                hidden_size=config.hidden_size,
                num_queries=int(getattr(config, "progress_aux_num_queries", 4)),
                bottleneck_dim=int(getattr(config, "progress_aux_in_qwen_readout_dim", 512)),
                query_specific_readout=bool(
                    getattr(config, "progress_aux_query_specific_readout", False)
                ),
                gated_aux_fusion=bool(
                    getattr(config, "progress_aux_gated_aux_fusion", False)
                ),
            )
        if bool(getattr(config, "progress_aux_value_stream", False)):
            if bool(getattr(config, "progress_aux_in_qwen_queries", False)):
                raise ValueError("Value stream and in-Qwen progress queries are mutually exclusive")
            self.progress_value_stream = ProgressValueStream(
                backbone_dim=config.hidden_size,
                backbone_layers=config.num_hidden_layers,
                depth=int(getattr(config, "progress_aux_value_stream_depth", 8)),
                value_dim=int(getattr(config, "progress_aux_value_stream_dim", 512)),
                num_queries=int(getattr(config, "progress_aux_num_queries", 4)),
                num_heads=int(getattr(config, "progress_aux_value_stream_heads", 8)),
                ffn_dim=int(getattr(config, "progress_aux_value_stream_ffn_dim", 2048)),
                temporal_input_dim=int(getattr(config, "progress_aux_temporal_input_dim", 0)),
                temporal_max_tokens=int(getattr(config, "progress_aux_temporal_max_tokens", 8)),
                temporal_layers=int(getattr(config, "progress_aux_temporal_layers", 1)),
                query_specific_readout=bool(
                    getattr(config, "progress_aux_query_specific_readout", True)
                ),
                gated_aux_fusion=bool(
                    getattr(config, "progress_aux_gated_aux_fusion", True)
                ),
            )
        # Initialize weights and apply final processing
        self.post_init()
        self._init_progress_aux_head()
        self._init_progress_action_adapter()
        self._init_progress_in_qwen_modules()
        self._init_progress_value_stream()

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def _init_progress_aux_head(self):
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        if hasattr(self.progress_aux_head, "query_embed"):
            self.progress_aux_head.query_embed.data.normal_(mean=0.0, std=initializer_range)
        temporal_encoder = getattr(self.progress_aux_head, "temporal_encoder", None)
        if temporal_encoder is not None and hasattr(temporal_encoder, "pos_embed"):
            temporal_encoder.pos_embed.data.normal_(mean=0.0, std=initializer_range)
        for module in self.progress_aux_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.MultiheadAttention):
                module.in_proj_weight.data.normal_(mean=0.0, std=initializer_range)
                if module.in_proj_bias is not None:
                    module.in_proj_bias.data.zero_()
        causal_prior = getattr(self.progress_aux_head, "causal_prior", None)
        if causal_prior is not None and isinstance(causal_prior[-1], nn.Linear):
            causal_prior[-1].weight.data.zero_()
            causal_prior[-1].bias.data.zero_()

    def _init_progress_action_adapter(self):
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        for module in self.progress_action_adapter.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        final_layer = self.progress_action_adapter.bias_head[-1]
        if isinstance(final_layer, nn.Linear):
            final_layer.weight.data.zero_()
            final_layer.bias.data.zero_()

    def _init_progress_value_stream(self):
        if self.progress_value_stream is None:
            return
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        self.progress_value_stream.query_embed.data.normal_(
            mean=0.0, std=initializer_range
        )
        temporal_encoder = getattr(self.progress_value_stream, "temporal_encoder", None)
        if temporal_encoder is not None and hasattr(temporal_encoder, "pos_embed"):
            temporal_encoder.pos_embed.data.normal_(mean=0.0, std=initializer_range)
        for module in self.progress_value_stream.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    module.in_proj_weight.data.normal_(mean=0.0, std=initializer_range)
                if module.in_proj_bias is not None:
                    module.in_proj_bias.data.zero_()
                for name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                    weight = getattr(module, name, None)
                    if weight is not None:
                        weight.data.normal_(mean=0.0, std=initializer_range)
                module.out_proj.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.zero_()

    def _init_progress_in_qwen_modules(self):
        if not bool(getattr(self.config, "progress_aux_in_qwen_queries", False)):
            return
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        progress_queries = getattr(self.get_model(), "progress_latent_queries", None)
        if progress_queries is None:
            raise RuntimeError("progress_aux_in_qwen_queries requires model.progress_latent_queries")
        progress_queries.data.normal_(mean=0.0, std=initializer_range)
        modules = [self.progress_in_qwen_temporal_conditioner, self.progress_in_qwen_readout]
        for root in modules:
            if root is None:
                continue
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            temporal_encoder = getattr(root, "temporal_encoder", None)
            if temporal_encoder is not None and hasattr(temporal_encoder, "pos_embed"):
                temporal_encoder.pos_embed.data.normal_(mean=0.0, std=initializer_range)
        aux_gate = getattr(self.progress_in_qwen_readout, "aux_gate", None)
        if aux_gate is not None:
            aux_gate.data.zero_()

    def get_model(self):
        return self.model

    def _progress_action_token_ids(self):
        token_ids = getattr(self.config, "progress_action_token_ids", None)
        if token_ids is None:
            # Qwen tokenizer ids for STOP, up, left, right, look-down in the DualVLN action vocabulary.
            token_ids = [50669, 76286, 71858, 51018, 79029]
        token_ids = [int(token_id) for token_id in token_ids]
        action_dim = int(getattr(self.config, "progress_action_num_actions", 5))
        return token_ids[:action_dim]

    def _apply_progress_action_adapter(
        self,
        logits,
        hidden_states,
        labels=None,
        attention_mask=None,
        progress_temporal_tokens=None,
        progress_temporal_mask=None,
        progress_causal_features=None,
    ):
        if not bool(getattr(self.config, "enable_progress_action_adapter", False)):
            return logits
        if not hasattr(self, "progress_action_adapter") or not hasattr(self, "progress_aux_head"):
            return logits

        memory_mask = self._build_progress_aux_memory_mask(
            hidden_states=hidden_states,
            labels=labels,
            attention_mask=attention_mask,
        )
        final_logits, _, query_tokens = self.progress_aux_head(
            hidden_states,
            attention_mask=attention_mask,
            memory_mask=memory_mask,
            temporal_tokens=progress_temporal_tokens,
            temporal_mask=progress_temporal_mask,
            causal_features=progress_causal_features,
            return_query_tokens=True,
        )
        action_bias = self.progress_action_adapter(
            hidden_states=hidden_states,
            progress_query_tokens=query_tokens,
            progress_logits=final_logits,
            attention_mask=attention_mask,
        ).to(device=logits.device, dtype=logits.dtype)
        scale = float(getattr(self.config, "progress_action_logit_scale", 1.0))
        action_bias = action_bias * scale
        token_ids = self._progress_action_token_ids()
        if not token_ids:
            return logits
        token_ids_tensor = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
        logits = logits.clone()
        logits.index_copy_(-1, token_ids_tensor, logits.index_select(-1, token_ids_tensor) + action_bias)
        return logits

    def _compute_progress_action_lm_loss(self, logits, labels, progress_aux_weights=None):
        if labels is None or not bool(getattr(self.config, "enable_progress_action_adapter", False)):
            return None
        if logits.shape[1] <= 1:
            return None

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(device=shift_logits.device)
        valid_mask = shift_labels.ne(-100)
        if bool(getattr(self.config, "progress_action_loss_only_action_tokens", True)):
            action_token_mask = torch.zeros_like(valid_mask)
            for token_id in self._progress_action_token_ids():
                action_token_mask = action_token_mask | shift_labels.eq(int(token_id))
            valid_mask = valid_mask & action_token_mask
        if not bool(valid_mask.any().item()):
            return None

        flat_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        loss_weights = valid_mask.to(dtype=flat_loss.dtype)
        if progress_aux_weights is not None and progress_aux_weights.shape[0] == shift_labels.shape[0]:
            sample_weights = progress_aux_weights.to(device=flat_loss.device, dtype=flat_loss.dtype)
            loss_weights = loss_weights * sample_weights[:, None]
        denom = loss_weights.sum().clamp(min=1.0)
        return (flat_loss * loss_weights).sum() / denom

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        progress_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
        progress_aux_labels: Optional[torch.Tensor] = None,
        progress_aux_mask: Optional[torch.Tensor] = None,
        progress_aux_weights: Optional[torch.Tensor] = None,
        progress_temporal_tokens: Optional[torch.Tensor] = None,
        progress_temporal_mask: Optional[torch.Tensor] = None,
        progress_causal_features: Optional[torch.Tensor] = None,
        skip_lm_head: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.progress_value_stream is not None:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            latent_queries = self.get_model().latent_queries.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            if t_s_pos is not None:
                for batch_idx, start in enumerate(t_s_pos):
                    start = int(start)
                    if start < 0:
                        continue
                    end = start + latent_queries.shape[1]
                    inputs_embeds[batch_idx, start:end] = latent_queries[0]
            else:
                n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
                traj_idx = input_ids == TRAJ_TOKEN_INDEX
                expanded_queries = latent_queries.repeat(input_ids.shape[0], 1, 1)
                hidden_size = expanded_queries.shape[-1]
                expanded_queries = expanded_queries.contiguous().view(-1, hidden_size)
                if n_traj_tokens != 0:
                    if n_traj_tokens != expanded_queries.shape[0]:
                        raise ValueError(
                            "Trajectory query placeholders require explicit t_s_pos when extra query tokens are present"
                        )
                    inputs_embeds[traj_idx] = expanded_queries

            if (
                bool(getattr(self.config, "progress_aux_in_qwen_queries", False))
                and progress_s_pos is not None
            ):
                progress_queries = self.get_model().progress_latent_queries.repeat(
                    input_ids.shape[0], 1, 1
                )
                if (
                    self.progress_in_qwen_temporal_conditioner is not None
                    and progress_temporal_tokens is not None
                ):
                    temporal_condition = self.progress_in_qwen_temporal_conditioner(
                        progress_temporal_tokens.to(device=inputs_embeds.device),
                        progress_temporal_mask.to(device=inputs_embeds.device)
                        if progress_temporal_mask is not None
                        else None,
                    )
                    progress_queries = progress_queries + temporal_condition.to(
                        device=progress_queries.device, dtype=progress_queries.dtype
                    ).unsqueeze(1)
                progress_queries = progress_queries.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                for batch_idx, start in enumerate(progress_s_pos):
                    start = int(start)
                    if start < 0:
                        continue
                    end = start + progress_queries.shape[1]
                    inputs_embeds[batch_idx, start:end] = progress_queries[batch_idx]

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        progress_aux_logits = None
        progress_aux_query_logits = None
        if self.progress_value_stream is not None:
            memory_mask = self._build_progress_aux_memory_mask(
                hidden_states=hidden_states,
                labels=labels,
                attention_mask=attention_mask,
            )
            progress_aux_logits, progress_aux_query_logits = self.progress_value_stream(
                outputs.hidden_states,
                memory_mask=memory_mask,
                temporal_tokens=progress_temporal_tokens,
                temporal_mask=progress_temporal_mask,
            )
        elif (
            bool(getattr(self.config, "progress_aux_in_qwen_queries", False))
            and progress_s_pos is not None
        ):
            if self.progress_in_qwen_readout is None:
                raise ValueError("In-Qwen progress readout module is not initialized")
            progress_query_hidden = []
            num_progress_queries = int(getattr(self.config, "progress_aux_num_queries", 4))
            for batch_idx, start in enumerate(progress_s_pos):
                start = int(start)
                if start < 0:
                    progress_query_hidden.append(
                        hidden_states.new_zeros((num_progress_queries, hidden_states.shape[-1]))
                    )
                else:
                    progress_query_hidden.append(
                        hidden_states[batch_idx, start : start + num_progress_queries]
                    )
            progress_query_hidden = torch.stack(progress_query_hidden, dim=0)
            progress_aux_logits, progress_aux_query_logits = self.progress_in_qwen_readout(
                progress_query_hidden
            )
        if skip_lm_head:
            logits = hidden_states.new_empty((*hidden_states.shape[:2], 0))
        else:
            logits = self.lm_head(hidden_states)
            logits = self._apply_progress_action_adapter(
                logits=logits,
                hidden_states=hidden_states,
                labels=labels,
                attention_mask=attention_mask,
                progress_temporal_tokens=progress_temporal_tokens,
                progress_temporal_mask=progress_temporal_mask,
                progress_causal_features=progress_causal_features,
            )

        loss = None
        if (
            labels is not None
            and traj_images is not None
            and traj_poses is not None
            and video_frame_num is not None
        ):
            traj_hidden_states = []
            for b in range(hidden_states.shape[0]):
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            loss_mask = torch.arange(traj_images.size(1), device=self.device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.unsqueeze(1)
            valid_traj_frames = int(loss_mask.sum().item())

            if valid_traj_frames > 0 and 'nextdit' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std

                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = torch.cat([memory_tokens, traj_hidden_states], dim=1)
                else:
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = traj_hidden_states

                relative_poses = traj_poses.flatten(0, 1)
                bsz = relative_poses.shape[0]
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                target = noise - relative_poses
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                masked_loss = loss * mask
                loss = masked_loss.sum() / mask.sum() / (loss.shape[1] * loss.shape[2])
            elif valid_traj_frames > 0 and 'navdp' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    loss = masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])

            elif valid_traj_frames > 0:
                raise NotImplementedError

        progress_aux_loss = self._compute_progress_aux_loss(
            hidden_states=hidden_states,
            lm_labels=labels,
            t_s_pos=t_s_pos,
            attention_mask=attention_mask,
            progress_aux_labels=progress_aux_labels,
            progress_aux_mask=progress_aux_mask,
            progress_aux_weights=progress_aux_weights,
            progress_temporal_tokens=progress_temporal_tokens,
            progress_temporal_mask=progress_temporal_mask,
            progress_causal_features=progress_causal_features,
            progress_aux_logits=progress_aux_logits,
            progress_aux_query_logits=progress_aux_query_logits,
        )
        if progress_aux_loss is not None:
            progress_weight = float(getattr(self.config, "progress_aux_loss_weight", 0.2))
            loss = progress_aux_loss * progress_weight if loss is None else loss + progress_aux_loss * progress_weight

        progress_action_loss = None if skip_lm_head else self._compute_progress_action_lm_loss(
            logits=logits,
            labels=labels,
            progress_aux_weights=progress_aux_weights,
        )
        if progress_action_loss is not None:
            action_weight = float(getattr(self.config, "progress_action_loss_weight", 1.0))
            loss = progress_action_loss * action_weight if loss is None else loss + progress_action_loss * action_weight

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return InternVLAN1CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            progress_aux_logits=progress_aux_logits,
            progress_aux_query_logits=progress_aux_query_logits,
        )

    def _build_progress_aux_memory_mask(self, hidden_states, labels=None, attention_mask=None):
        if attention_mask is not None and attention_mask.ndim == 2:
            memory_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        else:
            memory_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)

        if labels is not None and labels.shape[:2] == hidden_states.shape[:2]:
            # Use prompt/observation tokens only. Assistant target action tokens would leak STOP labels offline.
            memory_mask = memory_mask & labels.to(device=hidden_states.device).eq(-100)

        empty_rows = memory_mask.sum(dim=1) == 0
        if empty_rows.any():
            if attention_mask is not None and attention_mask.ndim == 2:
                fallback = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            else:
                fallback = torch.ones_like(memory_mask)
            memory_mask = torch.where(empty_rows[:, None], fallback, memory_mask)
        return memory_mask

    def _rank_margin_loss(self, desired_diff, margin):
        return F.softplus((margin - desired_diff) * 10.0) / 10.0

    def _compute_progress_aux_rank_loss(self, pred, labels, mask, weights):
        label_dim = labels.shape[-1]
        margin = float(getattr(self.config, "progress_aux_rank_margin", 0.08))
        losses = []

        if label_dim == 2:
            loss_weights = []

            def add_pairwise_dimension(dim):
                valid = mask[:, dim] > 0
                if int(valid.sum().item()) < 2:
                    return
                cur_pred = pred[valid, dim]
                cur_target = labels[valid, dim]
                cur_weights = weights[valid]
                target_diff = cur_target[:, None] - cur_target[None, :]
                pred_diff = cur_pred[:, None] - cur_pred[None, :]
                pair_mask = torch.triu(
                    torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
                ) & (target_diff.abs() > margin)
                if not bool(pair_mask.any().item()):
                    return
                desired_diff = target_diff.sign() * pred_diff
                pair_weights = torch.sqrt(cur_weights[:, None] * cur_weights[None, :])
                pair_loss = self._rank_margin_loss(desired_diff[pair_mask], margin)
                losses.append(
                    (pair_loss * pair_weights[pair_mask]).sum()
                    / pair_weights[pair_mask].sum().clamp(min=1.0e-6)
                )
                if dim == 0:
                    loss_weights.append(
                        float(getattr(self.config, "progress_aux_completion_loss_weight", 1.0))
                    )
                else:
                    loss_weights.append(
                        float(getattr(self.config, "progress_aux_success_loss_weight", 1.0))
                    )

            add_pairwise_dimension(0)
            add_pairwise_dimension(1)
            if not losses:
                return None
            stacked_weights = torch.tensor(
                loss_weights,
                device=losses[0].device,
                dtype=losses[0].dtype,
            )
            return (torch.stack(losses) * stacked_weights).sum() / stacked_weights.sum().clamp(
                min=1.0e-6
            )

        if label_dim < 4:
            return None

        def add_pointwise(desired_diff, valid_mask):
            valid_mask = valid_mask.to(dtype=torch.bool)
            if not bool(valid_mask.any().item()):
                return
            cur_weights = weights[valid_mask]
            cur_loss = self._rank_margin_loss(desired_diff[valid_mask], margin)
            losses.append((cur_loss * cur_weights).sum() / cur_weights.sum().clamp(min=1.0))

        cur_future_valid = (mask[:, 0] > 0) & (mask[:, 1] > 0)
        target_delta = labels[:, 1] - labels[:, 0]
        pred_delta = pred[:, 1] - pred[:, 0]
        add_pointwise(pred_delta, cur_future_valid & (target_delta > margin))
        add_pointwise(-pred_delta, cur_future_valid & (target_delta < -margin))

        if label_dim >= 10:
            stage_valid = (mask[:, 8] > 0) & (mask[:, 9] > 0)
            stage_target_delta = labels[:, 9] - labels[:, 8]
            stage_pred_delta = pred[:, 9] - pred[:, 8]
            add_pointwise(stage_pred_delta, stage_valid & (stage_target_delta > margin))
            add_pointwise(-stage_pred_delta, stage_valid & (stage_target_delta < -margin))

        ready_mask = mask[:, 2] if label_dim > 2 else torch.zeros_like(weights)
        stop_safety_mask = mask[:, 6] if label_dim > 6 else torch.zeros_like(weights)
        stop_target_denom = (ready_mask + stop_safety_mask).clamp(min=1.0)
        stop_target = (
            labels[:, 2] * ready_mask
            + (labels[:, 6] if label_dim > 6 else labels[:, 2]) * stop_safety_mask
        ) / stop_target_denom
        stop_valid = (ready_mask + stop_safety_mask) > 0
        stop_pred = (pred[:, 2] + (pred[:, 6] if label_dim > 6 else pred[:, 2])) * 0.5

        continue_valid = mask[:, 3] > 0
        continue_target = labels[:, 3]
        continue_pred = pred[:, 3]
        action_valid = stop_valid & continue_valid
        action_delta = stop_target - continue_target
        add_pointwise(stop_pred - continue_pred, action_valid & (action_delta > margin))
        add_pointwise(continue_pred - stop_pred, action_valid & (action_delta < -margin))

        if label_dim > 5:
            risk_valid = (mask[:, 4] > 0) | (mask[:, 5] > 0)
            risk_target = torch.maximum(
                torch.where(mask[:, 4] > 0, labels[:, 4], torch.zeros_like(weights)),
                torch.where(mask[:, 5] > 0, labels[:, 5], torch.zeros_like(weights)),
            )
            risk_pred = torch.maximum(pred[:, 4], pred[:, 5])
            add_pointwise(risk_pred - stop_pred, risk_valid & stop_valid & (risk_target > 0.5) & (stop_target < 0.5))

        if label_dim > 7:
            return_valid = mask[:, 7] > 0
            low_return_mask = return_valid & (labels[:, 7] <= 0.25)
            high_return_mask = return_valid & (labels[:, 7] >= 0.50)
            if bool(low_return_mask.any().item()) and bool(high_return_mask.any().item()):
                pair_diff = pred[:, 7][high_return_mask][:, None] - pred[:, 7][low_return_mask][None, :]
                pair_weights = torch.sqrt(weights[high_return_mask][:, None] * weights[low_return_mask][None, :])
                pair_loss = self._rank_margin_loss(pair_diff, margin)
                losses.append((pair_loss * pair_weights).sum() / pair_weights.sum().clamp(min=1.0))

            progress_valid = mask[:, 0] > 0
            positive_progress_mask = progress_valid & low_return_mask & (labels[:, 0] >= 0.70)
            failure_progress_mask = progress_valid & high_return_mask
            if bool(positive_progress_mask.any().item()) and bool(failure_progress_mask.any().item()):
                pair_diff = pred[:, 0][positive_progress_mask][:, None] - pred[:, 0][failure_progress_mask][None, :]
                pair_weights = torch.sqrt(
                    weights[positive_progress_mask][:, None] * weights[failure_progress_mask][None, :]
                )
                pair_loss = self._rank_margin_loss(pair_diff, margin)
                losses.append((pair_loss * pair_weights).sum() / pair_weights.sum().clamp(min=1.0))

        pos_mask = stop_valid & (stop_target >= 0.7)
        neg_mask = stop_valid & (stop_target <= 0.3)
        if label_dim > 5:
            neg_mask = neg_mask & ((labels[:, 3] > margin) | (labels[:, 4] > 0.3) | (labels[:, 5] > 0.3))
        if bool(pos_mask.any().item()) and bool(neg_mask.any().item()):
            pair_diff = stop_pred[pos_mask][:, None] - stop_pred[neg_mask][None, :]
            pair_weights = torch.sqrt(weights[pos_mask][:, None] * weights[neg_mask][None, :])
            pair_loss = self._rank_margin_loss(pair_diff, margin)
            losses.append((pair_loss * pair_weights).sum() / pair_weights.sum().clamp(min=1.0))

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compute_progress_aux_loss(
        self,
        hidden_states,
        lm_labels=None,
        t_s_pos=None,
        attention_mask=None,
        progress_aux_labels=None,
        progress_aux_mask=None,
        progress_aux_weights=None,
        progress_temporal_tokens=None,
        progress_temporal_mask=None,
        progress_causal_features=None,
        progress_aux_logits=None,
        progress_aux_query_logits=None,
    ):
        if progress_aux_labels is None or not bool(getattr(self.config, "enable_progress_aux", False)):
            return None

        target_labels = progress_aux_labels.to(device=hidden_states.device, dtype=torch.float32)
        if progress_aux_mask is None:
            mask = torch.ones_like(target_labels, dtype=torch.float32, device=hidden_states.device)
        else:
            mask = progress_aux_mask.to(device=hidden_states.device, dtype=torch.float32)
        if progress_aux_weights is None:
            weights = torch.ones(target_labels.shape[0], dtype=torch.float32, device=hidden_states.device)
        else:
            weights = progress_aux_weights.to(device=hidden_states.device, dtype=torch.float32)

        denom = (mask * weights[:, None]).sum()
        if float(denom.item()) <= 0.0:
            return None

        if progress_aux_logits is not None and progress_aux_query_logits is not None:
            final_logits = progress_aux_logits
            query_logits = progress_aux_query_logits
        else:
            memory_mask = self._build_progress_aux_memory_mask(
                hidden_states=hidden_states,
                labels=lm_labels,
                attention_mask=attention_mask,
            )
            final_logits, query_logits = self.progress_aux_head(
                hidden_states,
                attention_mask=attention_mask,
                memory_mask=memory_mask,
                temporal_tokens=progress_temporal_tokens,
                temporal_mask=progress_temporal_mask,
                causal_features=progress_causal_features,
            )
        target_mode = str(
            getattr(self.config, "progress_aux_target_mode", "legacy_multi")
        )
        if target_mode in (
            "segment_completion_success",
            "gated_boundary_completion_success",
        ):
            if target_labels.shape[-1] < 4:
                raise ValueError(
                    "boundary-aware completion requires completion, success, "
                    "segment_progress, and boundary targets"
                )
            if query_logits.shape[-1] < 4:
                raise ValueError(
                    "boundary-aware completion requires four progress queries"
                )

            final_logits = final_logits[:, :2].float()
            completion_loss = F.smooth_l1_loss(
                torch.sigmoid(final_logits[:, 0]), target_labels[:, 0], reduction="none"
            )
            success_loss = F.binary_cross_entropy_with_logits(
                final_logits[:, 1], target_labels[:, 1], reduction="none"
            )
            final_loss = torch.stack([completion_loss, success_loss], dim=1)
            completion_weight = float(
                getattr(self.config, "progress_aux_completion_loss_weight", 1.0)
            )
            success_weight = float(
                getattr(self.config, "progress_aux_success_loss_weight", 1.0)
            )
            segment_weight = float(
                getattr(self.config, "progress_aux_segment_loss_weight", 1.0)
            )
            boundary_weight = float(
                getattr(self.config, "progress_aux_boundary_loss_weight", 1.0)
            )
            final_target_weights = torch.tensor(
                [completion_weight, success_weight],
                device=hidden_states.device,
                dtype=torch.float32,
            )
            final_mask = mask[:, :2] * final_target_weights[None, :]
            final_denom = (final_mask * weights[:, None]).sum().clamp(min=1.0e-6)
            loss = (final_loss * final_mask * weights[:, None]).sum() / final_denom

            query_logits = query_logits[:, :4].float()
            query_loss = torch.stack(
                [
                    F.smooth_l1_loss(
                        torch.sigmoid(query_logits[:, 0]),
                        target_labels[:, 0],
                        reduction="none",
                    ),
                    F.smooth_l1_loss(
                        torch.sigmoid(query_logits[:, 1]),
                        target_labels[:, 2],
                        reduction="none",
                    ),
                    F.binary_cross_entropy_with_logits(
                        query_logits[:, 2], target_labels[:, 3], reduction="none"
                    ),
                    F.binary_cross_entropy_with_logits(
                        query_logits[:, 3], target_labels[:, 1], reduction="none"
                    ),
                ],
                dim=1,
            )
            query_mask = torch.stack(
                [mask[:, 0], mask[:, 2], mask[:, 3], mask[:, 1]], dim=1
            )
            query_target_weights = torch.tensor(
                [completion_weight, segment_weight, boundary_weight, success_weight],
                device=hidden_states.device,
                dtype=torch.float32,
            )
            query_weighted_mask = query_mask * query_target_weights[None, :]
            query_denom = (query_weighted_mask * weights[:, None]).sum()
            if float(query_denom.item()) > 0.0:
                query_weight = float(
                    getattr(self.config, "progress_aux_query_loss_weight", 0.3)
                )
                loss = loss + query_weight * (
                    query_loss * query_weighted_mask * weights[:, None]
                ).sum() / query_denom

            pred = torch.sigmoid(final_logits)
            rank_loss = self._compute_progress_aux_rank_loss(
                pred, target_labels[:, :2], mask[:, :2], weights
            )
            if rank_loss is not None:
                rank_weight = float(
                    getattr(self.config, "progress_aux_rank_loss_weight", 0.5)
                )
                loss = loss + rank_weight * rank_loss

            if target_mode == "gated_boundary_completion_success":
                aux_rank_losses = []
                rank_margin = float(
                    getattr(self.config, "progress_aux_rank_margin", 0.08)
                )
                for aux_pred, aux_target, aux_mask in (
                    (
                        torch.sigmoid(query_logits[:, 1]),
                        target_labels[:, 2],
                        mask[:, 2],
                    ),
                    (
                        torch.sigmoid(query_logits[:, 2]),
                        target_labels[:, 3],
                        mask[:, 3],
                    ),
                ):
                    valid = aux_mask > 0
                    if int(valid.sum().item()) < 2:
                        continue
                    current_pred = aux_pred[valid]
                    current_target = aux_target[valid]
                    current_weights = weights[valid]
                    target_diff = current_target[:, None] - current_target[None, :]
                    pred_diff = current_pred[:, None] - current_pred[None, :]
                    pair_mask = torch.triu(
                        torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
                    ) & (target_diff.abs() > rank_margin)
                    if not bool(pair_mask.any().item()):
                        continue
                    desired_diff = target_diff.sign() * pred_diff
                    pair_weights = torch.sqrt(
                        current_weights[:, None] * current_weights[None, :]
                    )
                    pair_loss = self._rank_margin_loss(
                        desired_diff[pair_mask], rank_margin
                    )
                    aux_rank_losses.append(
                        (pair_loss * pair_weights[pair_mask]).sum()
                        / pair_weights[pair_mask].sum().clamp(min=1.0e-6)
                    )
                if aux_rank_losses:
                    aux_rank_weight = float(
                        getattr(self.config, "progress_aux_aux_rank_loss_weight", 0.5)
                    )
                    loss = loss + aux_rank_weight * torch.stack(aux_rank_losses).mean()
            return loss

        if target_labels.shape[-1] == 2:
            final_logits = final_logits[:, :2].float()
            completion_pred = torch.sigmoid(final_logits[:, 0])
            completion_loss = F.smooth_l1_loss(
                completion_pred, target_labels[:, 0], reduction="none"
            )
            success_loss = F.binary_cross_entropy_with_logits(
                final_logits[:, 1], target_labels[:, 1], reduction="none"
            )
            per_target_loss = torch.stack([completion_loss, success_loss], dim=1)
            target_weights = torch.tensor(
                [
                    float(getattr(self.config, "progress_aux_completion_loss_weight", 1.0)),
                    float(getattr(self.config, "progress_aux_success_loss_weight", 1.0)),
                ],
                device=hidden_states.device,
                dtype=torch.float32,
            )
            weighted_mask = mask * target_weights[None, :]
            weighted_denom = (weighted_mask * weights[:, None]).sum().clamp(min=1.0e-6)
            loss = (per_target_loss * weighted_mask * weights[:, None]).sum() / weighted_denom

            completion_mode = str(
                getattr(self.config, "progress_aux_completion_mode", "scalar")
            )
            ordinal_completion = None
            categorical_completion = None
            if completion_mode == "categorical4":
                if query_logits.shape[-1] < 4:
                    raise ValueError("categorical4 completion requires four progress queries")
                stage_logits = query_logits[:, :4].float()
                stage_index = torch.round(target_labels[:, 0] * 3.0).clamp(0, 3).long()
                stage_loss = F.cross_entropy(stage_logits, stage_index, reduction="none")
                completion_weights = mask[:, 0] * weights
                completion_denom = completion_weights.sum()
                if float(completion_denom.item()) > 0.0:
                    query_weight = float(
                        getattr(self.config, "progress_aux_query_loss_weight", 0.3)
                    )
                    loss = loss + query_weight * (
                        stage_loss * completion_weights
                    ).sum() / completion_denom
                    stage_values = torch.arange(
                        4, device=hidden_states.device, dtype=torch.float32
                    ) / 3.0
                    categorical_completion = (
                        torch.softmax(stage_logits, dim=1) * stage_values[None, :]
                    ).sum(dim=1)
            elif completion_mode == "ordinal4":
                if query_logits.shape[-1] < 3:
                    raise ValueError("ordinal4 completion requires at least three progress queries")
                ordinal_logits = query_logits[:, :3].float()
                stage_index = torch.round(target_labels[:, 0] * 3.0).clamp(0, 3).long()
                thresholds = torch.arange(
                    1, 4, device=hidden_states.device, dtype=stage_index.dtype
                )
                ordinal_targets = (stage_index[:, None] >= thresholds[None, :]).float()
                ordinal_loss = F.binary_cross_entropy_with_logits(
                    ordinal_logits, ordinal_targets, reduction="none"
                ).mean(dim=1)
                completion_weights = mask[:, 0] * weights
                completion_denom = completion_weights.sum()
                if float(completion_denom.item()) > 0.0:
                    query_weight = float(
                        getattr(self.config, "progress_aux_query_loss_weight", 0.3)
                    )
                    loss = loss + query_weight * (
                        ordinal_loss * completion_weights
                    ).sum() / completion_denom
                    ordinal_probs = torch.sigmoid(ordinal_logits)
                    monotonic_loss = (
                        F.relu(ordinal_probs[:, 1] - ordinal_probs[:, 0])
                        + F.relu(ordinal_probs[:, 2] - ordinal_probs[:, 1])
                    )
                    loss = loss + 0.1 * (
                        monotonic_loss * completion_weights
                    ).sum() / completion_denom
                    ordinal_completion = ordinal_probs.mean(dim=1)

                if query_logits.shape[-1] >= 4:
                    success_weights = mask[:, 1] * weights * target_weights[1]
                    success_denom = success_weights.sum()
                    if float(success_denom.item()) > 0.0:
                        success_query_loss = F.binary_cross_entropy_with_logits(
                            query_logits[:, 3].float(), target_labels[:, 1], reduction="none"
                        )
                        query_weight = float(
                            getattr(self.config, "progress_aux_query_loss_weight", 0.3)
                        )
                        loss = loss + query_weight * (
                            success_query_loss * success_weights
                        ).sum() / success_denom
            else:
                query_dim = min(query_logits.shape[-1], 2)
                query_logits = query_logits[:, :query_dim].float()
                query_losses = []
                if query_dim >= 1:
                    query_losses.append(
                        F.smooth_l1_loss(
                            torch.sigmoid(query_logits[:, 0]), target_labels[:, 0], reduction="none"
                        )
                    )
                if query_dim >= 2:
                    query_losses.append(
                        F.binary_cross_entropy_with_logits(
                            query_logits[:, 1], target_labels[:, 1], reduction="none"
                        )
                    )
                query_loss = torch.stack(query_losses, dim=1)
                query_mask = mask[:, :query_dim]
                query_target_weights = target_weights[:query_dim]
                query_weighted_mask = query_mask * query_target_weights[None, :]
                query_denom = (query_weighted_mask * weights[:, None]).sum()
                if float(query_denom.item()) > 0.0:
                    query_weight = float(getattr(self.config, "progress_aux_query_loss_weight", 0.3))
                    loss = loss + query_weight * (
                        query_loss * query_weighted_mask * weights[:, None]
                    ).sum() / query_denom

            pred = torch.sigmoid(final_logits)
            if categorical_completion is not None:
                pred = pred.clone()
                pred[:, 0] = categorical_completion
            elif ordinal_completion is not None:
                pred = pred.clone()
                pred[:, 0] = ordinal_completion
            rank_loss = self._compute_progress_aux_rank_loss(
                pred, target_labels, mask, weights
            )
            if rank_loss is not None:
                rank_weight = float(getattr(self.config, "progress_aux_rank_loss_weight", 0.5))
                loss = loss + rank_weight * rank_loss
            return loss

        pred = torch.sigmoid(final_logits.float())
        label_dim = target_labels.shape[-1]
        pred = pred[:, :label_dim]
        value_loss = F.smooth_l1_loss(pred, target_labels, reduction="none")
        loss = (value_loss * mask * weights[:, None]).sum() / denom

        query_dim = min(query_logits.shape[-1], label_dim)
        if query_dim > 0:
            query_pred = torch.sigmoid(query_logits[:, :query_dim].float())
            query_loss = F.smooth_l1_loss(query_pred, target_labels[:, :query_dim], reduction="none")
            query_mask = mask[:, :query_dim]
            query_denom = (query_mask * weights[:, None]).sum()
            if float(query_denom.item()) > 0.0:
                query_weight = float(getattr(self.config, "progress_aux_query_loss_weight", 0.3))
                loss = loss + query_weight * (query_loss * query_mask * weights[:, None]).sum() / query_denom

        rank_loss = self._compute_progress_aux_rank_loss(pred, target_labels, mask, weights)
        if rank_loss is not None:
            rank_weight = float(getattr(self.config, "progress_aux_rank_loss_weight", 0.5))
            loss = loss + rank_weight * rank_loss
        return loss

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        input_ids.to(self.get_model().device)
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)

        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                # attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]

        return hidden_states

    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        if 'nextdit' in self.get_system1_type():
            scheduler = FlowMatchEulerDiscreteScheduler()
            device = traj_latents.device
            dtype = traj_latents.dtype

            traj_latents = self.get_model().cond_projector(traj_latents)
            if 'async' in self.get_system1_type():
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
                hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
            else:
                hidden_states = traj_latents
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
            batch_size = traj_latents.shape[0]
            latent_size = predict_step_nums
            latent_channels = 3

            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

            for t in scheduler.timesteps:
                latent_features = self.get_model().action_encoder(latents)
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
                latent_model_input = latent_features.repeat(2, 1, 1)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict noise model_output
                noise_pred = self.get_model().traj_dit(
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)

                # perform guidance
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous: x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        elif 'navdp' in self.get_system1_type():
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
