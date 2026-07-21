from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    system1: Optional[str] = field(default='nextdit')
    n_query: int = field(default=4)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    vln_dataset_use: str = field(default="")
    iign_dataset_use: str = field(default="")
    sample_step: int = field(default=4)
    num_history: Optional[int] = field(default=8)
    predict_step_num: Optional[int] = field(default=32)
    pixel_goal_only: Optional[bool] = field(default=False)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    resize_h: Optional[int] = field(default=384)
    resize_w: Optional[int] = field(default=384)
    num_future_steps: Optional[int] = field(default=4)
    max_dialog_turns: Optional[int] = field(default=6)
    progress_jsonl_use: str = field(
        default="",
        metadata={"help": "Comma-separated V1/V2/V3 progress jsonl files for internal DualVLN progress supervision."},
    )
    progress_jsonl_max_frames: Optional[int] = field(
        default=8,
        metadata={"help": "Maximum history/current frames loaded from each progress-jsonl row."},
    )
    progress_jsonl_rebuild_history: Optional[bool] = field(
        default=True,
        metadata={"help": "Rebuild DualVLN-style evenly sampled visual history when frame paths allow it."},
    )
    progress_jsonl_include_action_history: Optional[bool] = field(
        default=False,
        metadata={"help": "Append causal history actions to the progress-query prompt."},
    )
    progress_aux_temporal_tokens: Optional[bool] = field(
        default=True,
        metadata={"help": "Provide ordered pose/action temporal tokens to the progress auxiliary head."},
    )
    progress_aux_causal_prior: Optional[bool] = field(
        default=False,
        metadata={"help": "Add a causal step/history prior whose logits are refined by progress queries."},
    )
    progress_aux_causal_prior_only_trainable: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the loaded query head and train only its causal completion prior."},
    )
    progress_aux_target_mode: str = field(
        default="legacy_multi",
        metadata={
            "help": (
                "Auxiliary target schema: legacy_multi, completion_success, or "
                "segment_completion_success/gated_boundary_completion_success."
            )
        },
    )
    progress_aux_completion_mode: str = field(
        default="scalar",
        metadata={"help": "Completion prediction mode: scalar, ordinal4, or categorical4."},
    )
    progress_aux_num_queries: Optional[int] = field(
        default=4,
        metadata={"help": "Number of independent learnable progress query tokens."},
    )
    progress_aux_in_qwen_queries: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Insert hidden-size progress queries before the assistant action so they pass "
                "through all Qwen language layers without seeing the action label."
            )
        },
    )
    progress_aux_query_specific_readout: Optional[bool] = field(
        default=False,
        metadata={"help": "Use one scalar readout per full-width progress Query."},
    )
    progress_aux_gated_aux_fusion: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Read final completion/health directly from Q0/Q3 and inject Q1/Q2 "
                "through a zero-initialized residual gate."
            )
        },
    )
    progress_aux_qwen_backprop_fp32: Optional[bool] = field(
        default=False,
        metadata={"help": "Load Qwen in float32 for stable gradients to in-Qwen progress queries."},
    )
    progress_aux_value_stream: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Use an independent, asymmetric Value Transformer stream that reads "
                "selected frozen Qwen hidden layers. Mutually exclusive with in-Qwen queries."
            )
        },
    )
    progress_aux_value_stream_depth: Optional[int] = field(
        default=8,
        metadata={"help": "Number of evenly spaced Qwen layers read by the Value stream."},
    )
    progress_aux_value_stream_dim: Optional[int] = field(
        default=512,
        metadata={"help": "Hidden width of the independent Value stream."},
    )
    progress_aux_value_stream_ffn_dim: Optional[int] = field(
        default=2048,
        metadata={"help": "FFN width of each Value stream block."},
    )
    progress_aux_value_stream_heads: Optional[int] = field(
        default=8,
        metadata={"help": "Attention heads in each Value stream block."},
    )
    progress_jsonl_max_samples: Optional[int] = field(
        default=0,
        metadata={"help": "Optional cap for progress-jsonl samples after sampling; 0 means no cap."},
    )
    progress_jsonl_sample_rate: Optional[float] = field(
        default=1.0,
        metadata={"help": "Sampling rate applied independently to each progress-jsonl file."},
    )
    progress_jsonl_preserve_order: Optional[bool] = field(
        default=False,
        metadata={"help": "Preserve JSONL order and use sequential sampling for explicit ranking pairs."},
    )
    enable_progress_aux: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable internal progress/completion auxiliary supervision on DualVLN query tokens."},
    )
    progress_aux_loss_weight: Optional[float] = field(
        default=0.2,
        metadata={"help": "Loss weight for the internal progress/completion auxiliary head."},
    )
    progress_aux_query_loss_weight: Optional[float] = field(
        default=0.3,
        metadata={"help": "Intermediate query-score supervision weight for the progress auxiliary head."},
    )
    progress_aux_completion_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Relative loss weight for completion regression in completion_success mode."},
    )
    progress_aux_success_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Relative loss weight for success classification in completion_success mode."},
    )
    progress_aux_segment_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Relative query loss weight for local segment progress."},
    )
    progress_aux_boundary_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Relative query loss weight for segment-boundary classification."},
    )
    progress_aux_aux_rank_loss_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "Pairwise ranking weight for segment-progress and boundary Queries."},
    )
    progress_aux_rank_loss_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "Pairwise/ranking supervision weight for STOP-vs-CONTINUE and temporal progress ordering."},
    )
    progress_aux_rank_margin: Optional[float] = field(
        default=0.08,
        metadata={"help": "Margin used by progress auxiliary ranking losses."},
    )
    progress_aux_hard_negative_weight: Optional[float] = field(
        default=2.0,
        metadata={"help": "Sample weight for failure/hard-negative progress rows."},
    )
    progress_aux_stop_weight: Optional[float] = field(
        default=1.5,
        metadata={"help": "Sample weight for rows where the policy/model proposed STOP."},
    )
    progress_aux_only_trainable: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the base DualVLN model and train only the progress auxiliary head."},
    )
    progress_aux_save_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Save only the progress auxiliary head weights instead of the full base model."},
    )
    progress_aux_head_path: str = field(
        default="",
        metadata={"help": "Optional stage-1 progress_aux_head.pt path to load before stage-2 action training."},
    )
    enable_progress_action_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable the progress-conditioned action logit adapter."},
    )
    progress_action_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Loss weight for action-token CE after progress-conditioned logit adaptation."},
    )
    progress_action_logit_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "Scale applied to progress action adapter logit bias."},
    )
    progress_action_only_trainable: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the base model and train only the progress action adapter."},
    )
    progress_action_train_progress_head: Optional[bool] = field(
        default=False,
        metadata={"help": "When progress_action_only_trainable is set, also unfreeze the progress auxiliary head."},
    )
    progress_action_save_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Save only progress action adapter related weights instead of the full base model."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
