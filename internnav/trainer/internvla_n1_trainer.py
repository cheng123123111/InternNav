# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict

import torch
import transformers
from torch.utils.data import SequentialSampler
from torchvision.transforms import v2

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl_base import replace_qwen2_vl_attention_class
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Trainer,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    QWEN2_5_VL_VISION_ATTENTION_CLASSES,
)

from internnav.dataset.internvla_n1_lerobot_dataset import (
    COMPLETION_SUCCESS_LABEL_NAMES,
    PROGRESS_AUX_LABEL_NAMES,
    SEGMENT_COMPLETION_SUCCESS_LABEL_NAMES,
    make_supervised_data_module,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)


class ProgressAuxTrainer(Trainer):
    """Keep explicit preferred/rejected rows in the same local batch."""

    def _get_train_sampler(self):
        if bool(getattr(self.train_dataset, "preserve_order", False)):
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        base_model = getattr(model, "module", model)
        if bool(getattr(base_model.config, "progress_aux_in_qwen_queries", False)) or bool(
            getattr(base_model.config, "progress_aux_value_stream", False)
        ):
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    parameter.grad.clamp_(min=-100.0, max=100.0)
        return loss


def checkpoint_has_progress_aux_head(model_path: str) -> bool:
    path = Path(model_path)
    if not path.is_dir():
        return False

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.exists():
            with open(index_path, "r") as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(key.startswith("progress_aux_head.") for key in weight_map)

    try:
        from safetensors import safe_open
    except ImportError:
        return False

    for tensor_path in path.glob("*.safetensors"):
        with safe_open(tensor_path, framework="pt", device="cpu") as f:
            if any(key.startswith("progress_aux_head.") for key in f.keys()):
                return True
    return False


def load_progress_aux_head_checkpoint(model, head_path: str):
    if not head_path:
        return
    package = torch.load(head_path, map_location="cpu")
    if bool(package.get("progress_aux_value_stream", False)):
        state = package.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError(f"Missing Value-stream state_dict in {head_path}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        prefix = "progress_value_stream."
        bad_missing = [key for key in missing if key.startswith(prefix)]
        bad_unexpected = [key for key in unexpected if key.startswith(prefix)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Value-stream checkpoint mismatch: missing={bad_missing[:5]} "
                f"unexpected={bad_unexpected[:5]}"
            )
        return
    if bool(package.get("progress_aux_in_qwen_queries", False)):
        state = package.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError(f"Missing in-Qwen progress state_dict in {head_path}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_prefixes = (
            "model.progress_latent_queries",
            "progress_in_qwen_temporal_conditioner.",
            "progress_in_qwen_readout.",
        )
        bad_missing = [key for key in missing if key.startswith(expected_prefixes)]
        bad_unexpected = [key for key in unexpected if key.startswith(expected_prefixes)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"In-Qwen progress checkpoint mismatch: missing={bad_missing[:5]} "
                f"unexpected={bad_unexpected[:5]}"
            )
        return
    state = package.get("progress_aux_head", package)
    normalized_state = {}
    for key, value in state.items():
        if key.startswith("progress_aux_head."):
            normalized_state[key] = value
        else:
            normalized_state[f"progress_aux_head.{key}"] = value
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    allow_new_causal_prior = int(
        getattr(model.config, "progress_aux_causal_feature_dim", 0)
    ) > 0
    bad_missing = [
        key
        for key in missing
        if key.startswith("progress_aux_head.")
        and not (allow_new_causal_prior and key.startswith("progress_aux_head.causal_prior."))
    ]
    bad_unexpected = [key for key in unexpected if key.startswith("progress_aux_head.")]
    if bad_missing:
        raise RuntimeError(f"Missing progress_aux_head keys from {head_path}: {bad_missing[:5]} total={len(bad_missing)}")
    if bad_unexpected:
        raise RuntimeError(
            f"Unexpected progress_aux_head keys from {head_path}: {bad_unexpected[:5]} total={len(bad_unexpected)}"
        )


def progress_action_token_ids_from_tokenizer(tokenizer):
    return [int(tokenizer.encode(text, add_special_tokens=False)[0]) for text in ["STOP", "↑", "←", "→", "↓"]]


def use_bf16_flash_for_frozen_vision(model):
    """Keep frozen vision memory-efficient while Qwen language backpropagates in fp32."""
    vision_config = model.config.vision_config
    flash_attention_cls = QWEN2_5_VL_VISION_ATTENTION_CLASSES["flash_attention_2"]
    for block in model.visual.blocks:
        old_attention = block.attn
        new_attention = flash_attention_cls(
            vision_config.hidden_size,
            num_heads=vision_config.num_heads,
        )
        new_attention.load_state_dict(old_attention.state_dict(), strict=True)
        block.attn = new_attention
    model.visual.bfloat16()


def load_training_config(model_path: str, cache_dir: str = None):
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if config_dict.get("model_type") == "internvla_n1":
            return InternVLAN1ModelConfig.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

    try:
        return AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except ValueError as exc:
        if "internvla_n1" not in str(exc):
            raise
        return InternVLAN1ModelConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def save_progress_aux_head_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str,
    model_args: ModelArguments,
    data_args: DataArguments,
):
    if not trainer.args.should_save:
        return

    os.makedirs(output_dir, exist_ok=True)
    model = trainer.accelerator.unwrap_model(trainer.model)
    in_qwen_queries = bool(getattr(model.config, "progress_aux_in_qwen_queries", False))
    value_stream = bool(getattr(model.config, "progress_aux_value_stream", False))
    state_prefixes = (
        ("progress_value_stream.",)
        if value_stream
        else
        (
            "model.progress_latent_queries",
            "progress_in_qwen_temporal_conditioner.",
            "progress_in_qwen_readout.",
        )
        if in_qwen_queries
        else ("progress_aux_head.",)
    )
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(state_prefixes)
    }
    target_mode = str(getattr(model.config, "progress_aux_target_mode", "legacy_multi"))
    completion_mode = str(getattr(model.config, "progress_aux_completion_mode", "scalar"))
    if target_mode in (
        "segment_completion_success",
        "gated_boundary_completion_success",
    ):
        label_names = SEGMENT_COMPLETION_SUCCESS_LABEL_NAMES
    elif target_mode == "completion_success":
        label_names = COMPLETION_SUCCESS_LABEL_NAMES
    else:
        label_names = PROGRESS_AUX_LABEL_NAMES
    causal_feature_dim = int(getattr(model.config, "progress_aux_causal_feature_dim", 0) or 0)
    if value_stream:
        progress_aux_arch = "asymmetric_qwen_value_stream_v1"
    elif in_qwen_queries and target_mode == "gated_boundary_completion_success":
        progress_aux_arch = "gated_boundary_query_specific_full_width_in_qwen_rank_v1"
    elif in_qwen_queries and target_mode == "segment_completion_success":
        progress_aux_arch = "segment_completion_success_full_width_in_qwen_query_rank_v1"
    elif in_qwen_queries:
        progress_aux_arch = "completion_success_full_width_in_qwen_query_rank_v1"
    elif target_mode == "completion_success":
        if completion_mode == "categorical4":
            progress_aux_arch = "completion_categorical4_query_cross_attention_temporal_v1"
        elif completion_mode == "ordinal4":
            progress_aux_arch = "completion_ordinal4_query_cross_attention_temporal_v1"
        else:
            temporal_dim = int(
                getattr(model.config, "progress_aux_temporal_input_dim", 0) or 0
            )
            num_queries = int(getattr(model.config, "progress_aux_num_queries", 1) or 1)
            if causal_feature_dim > 0:
                progress_aux_arch = "completion_success_query_cross_attention_causal_prior_rank_v2"
            elif temporal_dim == 0 and num_queries == 1:
                progress_aux_arch = "completion_success_single_query_cross_attention_rank_v1"
            else:
                progress_aux_arch = "completion_success_query_cross_attention_temporal_rank_v1"
    else:
        progress_aux_arch = "query_cross_attention_temporal_rank_v2"
    checkpoint = {
            "label_names": list(label_names),
            "base_model": model_args.model_name_or_path,
            "progress_aux_initialization": (
                "warm_start" if data_args.progress_aux_head_path else "fresh_random"
            ),
            "parent_progress_aux_head": data_args.progress_aux_head_path or None,
            "progress_jsonl_use": data_args.progress_jsonl_use,
            "training_seed": trainer.args.seed,
            "training_run_name": trainer.args.run_name,
            "training_global_step": trainer.state.global_step,
            "system1": model_args.system1,
            "n_query": model_args.n_query,
            "progress_aux_arch": progress_aux_arch,
            "progress_aux_target_mode": target_mode,
            "progress_aux_completion_mode": completion_mode,
            "progress_aux_dim": getattr(model.config, "progress_aux_dim", len(label_names)),
            "progress_aux_num_queries": getattr(model.config, "progress_aux_num_queries", None),
            "progress_aux_query_loss_weight": getattr(model.config, "progress_aux_query_loss_weight", None),
            "progress_aux_completion_loss_weight": getattr(
                model.config, "progress_aux_completion_loss_weight", None
            ),
            "progress_aux_success_loss_weight": getattr(
                model.config, "progress_aux_success_loss_weight", None
            ),
            "progress_aux_segment_loss_weight": getattr(
                model.config, "progress_aux_segment_loss_weight", None
            ),
            "progress_aux_boundary_loss_weight": getattr(
                model.config, "progress_aux_boundary_loss_weight", None
            ),
            "progress_aux_aux_rank_loss_weight": getattr(
                model.config, "progress_aux_aux_rank_loss_weight", None
            ),
            "progress_aux_rank_loss_weight": getattr(model.config, "progress_aux_rank_loss_weight", None),
            "progress_aux_rank_margin": getattr(model.config, "progress_aux_rank_margin", None),
            "progress_aux_temporal_input_dim": getattr(model.config, "progress_aux_temporal_input_dim", None),
            "progress_jsonl_include_action_history": bool(
                getattr(model.config, "progress_jsonl_include_action_history", False)
            ),
            "progress_aux_temporal_max_tokens": getattr(model.config, "progress_aux_temporal_max_tokens", None),
            "progress_aux_temporal_layers": getattr(model.config, "progress_aux_temporal_layers", None),
            "progress_aux_causal_feature_dim": causal_feature_dim,
            "progress_aux_causal_prior_only_trainable": bool(
                getattr(model.config, "progress_aux_causal_prior_only_trainable", False)
            ),
            "progress_aux_in_qwen_queries": in_qwen_queries,
            "progress_aux_value_stream": value_stream,
            "progress_aux_value_stream_depth": getattr(
                model.config, "progress_aux_value_stream_depth", None
            ),
            "progress_aux_value_stream_dim": getattr(
                model.config, "progress_aux_value_stream_dim", None
            ),
            "progress_aux_value_stream_ffn_dim": getattr(
                model.config, "progress_aux_value_stream_ffn_dim", None
            ),
            "progress_aux_value_stream_heads": getattr(
                model.config, "progress_aux_value_stream_heads", None
            ),
            "progress_aux_query_specific_readout": bool(
                getattr(model.config, "progress_aux_query_specific_readout", False)
            ),
            "progress_aux_gated_aux_fusion": bool(
                getattr(model.config, "progress_aux_gated_aux_fusion", False)
            ),
            "progress_aux_query_hidden_size": (
                int(getattr(model.config, "hidden_size", 0)) if in_qwen_queries else None
            ),
            "progress_aux_qwen_backprop_fp32": bool(
                getattr(model.config, "progress_aux_qwen_backprop_fp32", False)
            ),
            "progress_aux_in_qwen_temporal_dim": getattr(
                model.config, "progress_aux_in_qwen_temporal_dim", None
            ),
            "progress_aux_in_qwen_readout_dim": getattr(
                model.config, "progress_aux_in_qwen_readout_dim", None
            ),
            "progress_aux_query_position": (
                "after_trajectory_before_assistant_action" if in_qwen_queries else "post_qwen_readout"
            ),
        }
    if in_qwen_queries or value_stream:
        checkpoint["state_dict"] = state_dict
    else:
        checkpoint["progress_aux_head"] = state_dict
    torch.save(checkpoint, os.path.join(output_dir, "progress_aux_head.pt"))
    model.config.save_pretrained(output_dir)


def save_progress_action_adapter_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str,
    model_args: ModelArguments,
):
    if not trainer.args.should_save:
        return

    os.makedirs(output_dir, exist_ok=True)
    model = trainer.accelerator.unwrap_model(trainer.model)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("progress_action_adapter.") or key.startswith("progress_aux_head.")
    }
    torch.save(
        {
            "state_dict": state_dict,
            "label_names": list(PROGRESS_AUX_LABEL_NAMES),
            "base_model": model_args.model_name_or_path,
            "system1": model_args.system1,
            "n_query": model_args.n_query,
            "progress_aux_arch": "query_cross_attention_temporal_rank_v2",
            "progress_action_arch": "progress_query_cross_attention_logit_adapter_v1",
            "progress_action_token_ids": getattr(model.config, "progress_action_token_ids", None),
            "progress_action_loss_weight": getattr(model.config, "progress_action_loss_weight", None),
            "progress_action_logit_scale": getattr(model.config, "progress_action_logit_scale", None),
            "progress_aux_temporal_input_dim": getattr(model.config, "progress_aux_temporal_input_dim", None),
            "progress_aux_temporal_max_tokens": getattr(model.config, "progress_aux_temporal_max_tokens", None),
            "progress_aux_temporal_layers": getattr(model.config, "progress_aux_temporal_layers", None),
        },
        os.path.join(output_dir, "progress_action_adapter.pt"),
    )
    model.config.save_pretrained(output_dir)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        # model.lm_head.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False

    if 'nextdit' in model_args.system1:
        modules = [
            'action_encoder',
            'action_decoder',
            'traj_dit',
            'cond_projector',
            'memory_encoder',
            'rgb_resampler',
            'rgb_model',
        ]
        for n, p in model.model.named_parameters():
            if any(k in n for k in modules):
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True
    elif 'navdp' in model_args.system1:
        for n, p in model.model.navdp.named_parameters():
            if "rgb_model" not in n:
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Dataset sampling and auxiliary-head initialization happen before Trainer is built.
    transformers.set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
                v2.Resize((data_args.resize_h, data_args.resize_w)),
            ]
        )
    else:
        data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))

    model_name_lower = model_args.model_name_or_path.lower()
    model_config = load_training_config(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    is_internvla_n1 = (
        "internvla-n1-system2" in model_name_lower
        or "internvla-n1-dualvln" in model_name_lower
        or getattr(model_config, "model_type", "") == "internvla_n1"
    )

    if is_internvla_n1:
        model_config.progress_aux_target_mode = str(data_args.progress_aux_target_mode)
        model_config.progress_aux_completion_mode = str(data_args.progress_aux_completion_mode)
        model_config.progress_aux_dim = (
            2
            if data_args.progress_aux_target_mode
            in (
                "completion_success",
                "segment_completion_success",
                "gated_boundary_completion_success",
            )
            else 10
        )
        model_config.progress_aux_num_queries = int(data_args.progress_aux_num_queries or 4)
        model_config.progress_aux_in_qwen_queries = bool(data_args.progress_aux_in_qwen_queries)
        model_config.progress_aux_query_specific_readout = bool(
            data_args.progress_aux_query_specific_readout
        )
        model_config.progress_aux_gated_aux_fusion = bool(
            data_args.progress_aux_gated_aux_fusion
        )
        model_config.progress_aux_qwen_backprop_fp32 = bool(
            data_args.progress_aux_qwen_backprop_fp32
        )
        if data_args.progress_aux_in_qwen_queries and data_args.progress_aux_value_stream:
            raise ValueError(
                "progress_aux_in_qwen_queries and progress_aux_value_stream are mutually exclusive"
            )
        model_config.progress_aux_value_stream = bool(data_args.progress_aux_value_stream)
        model_config.progress_aux_value_stream_depth = int(
            data_args.progress_aux_value_stream_depth or 8
        )
        model_config.progress_aux_value_stream_dim = int(
            data_args.progress_aux_value_stream_dim or 512
        )
        model_config.progress_aux_value_stream_ffn_dim = int(
            data_args.progress_aux_value_stream_ffn_dim or 2048
        )
        model_config.progress_aux_value_stream_heads = int(
            data_args.progress_aux_value_stream_heads or 8
        )
        model_config.progress_aux_in_qwen_temporal_dim = 256
        model_config.progress_aux_in_qwen_readout_dim = 512
        model_config.enable_progress_aux = bool(data_args.enable_progress_aux)
        model_config.progress_aux_loss_weight = float(data_args.progress_aux_loss_weight)
        model_config.progress_aux_query_loss_weight = float(data_args.progress_aux_query_loss_weight)
        model_config.progress_aux_completion_loss_weight = float(
            data_args.progress_aux_completion_loss_weight
        )
        model_config.progress_aux_success_loss_weight = float(
            data_args.progress_aux_success_loss_weight
        )
        model_config.progress_aux_segment_loss_weight = float(
            data_args.progress_aux_segment_loss_weight
        )
        model_config.progress_aux_boundary_loss_weight = float(
            data_args.progress_aux_boundary_loss_weight
        )
        model_config.progress_aux_aux_rank_loss_weight = float(
            data_args.progress_aux_aux_rank_loss_weight
        )
        model_config.progress_aux_rank_loss_weight = float(data_args.progress_aux_rank_loss_weight)
        model_config.progress_aux_rank_margin = float(data_args.progress_aux_rank_margin)
        model_config.progress_aux_temporal_input_dim = 16 if data_args.progress_aux_temporal_tokens else 0
        model_config.progress_jsonl_include_action_history = bool(
            data_args.progress_jsonl_include_action_history
        )
        model_config.progress_aux_temporal_max_tokens = int(data_args.progress_jsonl_max_frames or 8)
        model_config.progress_aux_temporal_layers = 1
        model_config.progress_aux_causal_feature_dim = (
            6 if data_args.progress_aux_causal_prior else 0
        )
        model_config.progress_aux_causal_prior_only_trainable = bool(
            data_args.progress_aux_causal_prior_only_trainable
        )
        model_config.enable_progress_action_adapter = bool(data_args.enable_progress_action_adapter)
        model_config.progress_action_loss_weight = float(data_args.progress_action_loss_weight)
        model_config.progress_action_logit_scale = float(data_args.progress_action_logit_scale)
        model_config.progress_action_num_actions = 5
        model_config.progress_action_loss_only_action_tokens = True
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=model_config,
            attn_implementation=attn_implementation,
            torch_dtype=(
                torch.float32
                if data_args.progress_aux_qwen_backprop_fp32
                else (torch.bfloat16 if training_args.bf16 else None)
            ),
        )
        if data_args.progress_aux_qwen_backprop_fp32:
            use_bf16_flash_for_frozen_vision(model)
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "internvla-n1"
    elif "qwen2.5" in model_name_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=model_config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=model_config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if is_internvla_n1 and data_args.enable_progress_action_adapter:
        model.config.progress_action_token_ids = progress_action_token_ids_from_tokenizer(tokenizer)

    if data_args.model_type == "internvla-n1":
        base_model = model.get_model()
        current_system1 = getattr(base_model.config, "system1", None)
        has_system1_modules = getattr(base_model, "latent_queries", None) is not None
        if (not has_system1_modules) or current_system1 is None or current_system1 != model_args.system1:
            base_model.initialize_vision_modules(model_args=model_args)
        model.config.progress_aux_target_mode = str(data_args.progress_aux_target_mode)
        model.config.progress_aux_completion_mode = str(data_args.progress_aux_completion_mode)
        model.config.progress_aux_dim = (
            2
            if data_args.progress_aux_target_mode
            in (
                "completion_success",
                "segment_completion_success",
                "gated_boundary_completion_success",
            )
            else 10
        )
        model.config.progress_aux_num_queries = int(data_args.progress_aux_num_queries or 4)
        model.config.progress_aux_in_qwen_queries = bool(data_args.progress_aux_in_qwen_queries)
        model.config.progress_aux_query_specific_readout = bool(
            data_args.progress_aux_query_specific_readout
        )
        model.config.progress_aux_gated_aux_fusion = bool(
            data_args.progress_aux_gated_aux_fusion
        )
        model.config.progress_aux_qwen_backprop_fp32 = bool(
            data_args.progress_aux_qwen_backprop_fp32
        )
        model.config.progress_aux_value_stream = bool(data_args.progress_aux_value_stream)
        model.config.progress_aux_value_stream_depth = int(
            data_args.progress_aux_value_stream_depth or 8
        )
        model.config.progress_aux_value_stream_dim = int(
            data_args.progress_aux_value_stream_dim or 512
        )
        model.config.progress_aux_value_stream_ffn_dim = int(
            data_args.progress_aux_value_stream_ffn_dim or 2048
        )
        model.config.progress_aux_value_stream_heads = int(
            data_args.progress_aux_value_stream_heads or 8
        )
        model.config.progress_aux_in_qwen_temporal_dim = 256
        model.config.progress_aux_in_qwen_readout_dim = 512
        model.config.enable_progress_aux = bool(data_args.enable_progress_aux)
        model.config.progress_aux_loss_weight = float(data_args.progress_aux_loss_weight)
        model.config.progress_aux_query_loss_weight = float(data_args.progress_aux_query_loss_weight)
        model.config.progress_aux_completion_loss_weight = float(
            data_args.progress_aux_completion_loss_weight
        )
        model.config.progress_aux_success_loss_weight = float(
            data_args.progress_aux_success_loss_weight
        )
        model.config.progress_aux_segment_loss_weight = float(
            data_args.progress_aux_segment_loss_weight
        )
        model.config.progress_aux_boundary_loss_weight = float(
            data_args.progress_aux_boundary_loss_weight
        )
        model.config.progress_aux_aux_rank_loss_weight = float(
            data_args.progress_aux_aux_rank_loss_weight
        )
        model.config.progress_aux_rank_loss_weight = float(data_args.progress_aux_rank_loss_weight)
        model.config.progress_aux_rank_margin = float(data_args.progress_aux_rank_margin)
        model.config.progress_aux_temporal_input_dim = 16 if data_args.progress_aux_temporal_tokens else 0
        model.config.progress_jsonl_include_action_history = bool(
            data_args.progress_jsonl_include_action_history
        )
        model.config.progress_aux_temporal_max_tokens = int(data_args.progress_jsonl_max_frames or 8)
        model.config.progress_aux_temporal_layers = 1
        model.config.progress_aux_causal_feature_dim = (
            6 if data_args.progress_aux_causal_prior else 0
        )
        model.config.progress_aux_causal_prior_only_trainable = bool(
            data_args.progress_aux_causal_prior_only_trainable
        )
        model.config.enable_progress_action_adapter = bool(data_args.enable_progress_action_adapter)
        model.config.progress_action_loss_weight = float(data_args.progress_action_loss_weight)
        model.config.progress_action_logit_scale = float(data_args.progress_action_logit_scale)
        model.config.progress_action_num_actions = 5
        model.config.progress_action_loss_only_action_tokens = True
        if (
            data_args.enable_progress_aux
            and hasattr(model, "_init_progress_aux_head")
            and not checkpoint_has_progress_aux_head(model_args.model_name_or_path)
        ):
            model._init_progress_aux_head()
        if data_args.enable_progress_aux and hasattr(model, "progress_aux_head"):
            model.progress_aux_head.float()
        if data_args.progress_aux_in_qwen_queries:
            model._init_progress_in_qwen_modules()
            if model.progress_in_qwen_temporal_conditioner is not None:
                model.progress_in_qwen_temporal_conditioner.float()
            model.progress_in_qwen_readout.float()
        if data_args.progress_aux_value_stream:
            if model.progress_value_stream is None:
                raise RuntimeError("Value stream was not initialized from the model config")
            model._init_progress_value_stream()
            model.progress_value_stream.float()
        if data_args.enable_progress_action_adapter and hasattr(model, "progress_action_adapter"):
            model.progress_action_adapter.float()
        if data_args.progress_aux_head_path:
            load_progress_aux_head_checkpoint(model, data_args.progress_aux_head_path)
    set_model(model_args, model)
    if data_args.progress_action_only_trainable:
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "progress_action_adapter") and data_args.enable_progress_action_adapter:
            for p in model.progress_action_adapter.parameters():
                p.requires_grad = True
        if (
            hasattr(model, "progress_aux_head")
            and data_args.enable_progress_aux
            and data_args.progress_action_train_progress_head
        ):
            for p in model.progress_aux_head.parameters():
                p.requires_grad = True
    elif data_args.progress_aux_only_trainable:
        for p in model.parameters():
            p.requires_grad = False
        if data_args.progress_aux_value_stream:
            if model.progress_value_stream is None:
                raise RuntimeError("Value stream was not initialized")
            for p in model.progress_value_stream.parameters():
                p.requires_grad = True
        elif data_args.progress_aux_in_qwen_queries:
            progress_queries = getattr(model.get_model(), "progress_latent_queries", None)
            if progress_queries is None or model.progress_in_qwen_readout is None:
                raise RuntimeError("In-Qwen progress query modules were not initialized")
            progress_queries.requires_grad = True
            if model.progress_in_qwen_temporal_conditioner is not None:
                for p in model.progress_in_qwen_temporal_conditioner.parameters():
                    p.requires_grad = True
            for p in model.progress_in_qwen_readout.parameters():
                p.requires_grad = True
        elif hasattr(model, "progress_aux_head") and data_args.enable_progress_aux:
            if data_args.progress_aux_causal_prior_only_trainable:
                causal_prior = getattr(model.progress_aux_head, "causal_prior", None)
                if causal_prior is None:
                    raise ValueError(
                        "progress_aux_causal_prior_only_trainable requires progress_aux_causal_prior"
                    )
                for p in causal_prior.parameters():
                    p.requires_grad = True
            else:
                for p in model.progress_aux_head.parameters():
                    p.requires_grad = True
    elif hasattr(model, "progress_aux_head"):
        for p in model.progress_aux_head.parameters():
            p.requires_grad = bool(data_args.enable_progress_aux)
        if hasattr(model, "progress_action_adapter"):
            for p in model.progress_action_adapter.parameters():
                p.requires_grad = bool(data_args.enable_progress_action_adapter)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)  # noqa: F821
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = ProgressAuxTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    try:
        from tabulate import tabulate
    except ImportError:
        def tabulate(rows, headers):
            return "\n".join(
                ["\t".join(map(str, headers))]
                + ["\t".join(map(str, row)) for row in rows]
            )

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            if p.requires_grad:
                stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
        print(f"Trainable parameter tensors: {len(stat)}")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    if data_args.progress_action_save_only:
        save_progress_action_adapter_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            model_args=model_args,
        )
    elif data_args.progress_aux_save_only:
        save_progress_aux_head_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            model_args=model_args,
            data_args=data_args,
        )
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
