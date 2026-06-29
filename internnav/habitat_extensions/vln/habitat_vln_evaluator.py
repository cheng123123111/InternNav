import argparse
import json
import math
import os
import sys
from enum import IntEnum
from typing import Optional

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict

import cv2
import habitat
import imageio
import numpy as np
import quaternion
import torch
import torch.nn.functional as F
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    HeadRGBSensorConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig
from internnav.model.basemodel.internvla_n1.internvla_n1_pes import InternVLAN1PESForCausalLM
from internnav.model.basemodel.internvla_n1.stop_residual_adapter import (
    build_online_progress_tokens,
    load_stop_residual_adapter,
    logit as stop_residual_logit,
)
from internnav.model.basemodel.LongCLIP.model import longclip
from internnav.model.utils.attention import load_pretrained_with_attention_fallback
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.trainer.repo_lora import apply_repo_lora
from scripts.data_collect.stop_alignment_utils import (
    extract_stop_target_elements,
    extract_stop_object_phrase,
    extract_stop_phrase,
    is_usable_stop_object_phrase,
)
from scripts.data_collect.train_stop_element_longclip_history import (
    CrossAttentionStopHead,
    GroundingQueryStopHead,
    HistoryElementLongCLIPHead,
    encode_image_with_patch_tokens,
    forward_stop_head,
)

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4

VIVA_CANDIDATE_TYPE_IDS = {
    "stop_token": 1,
    "history_continue": 2,
    "primitive_action": 3,
    "rule_interrupt": 4,
    "synthetic_continue": 5,
    "legacy_qwen_candidate": 6,
    "legacy_continue_proxy": 7,
    "diffusion_trajectory": 8,
    "expert_trajectory": 9,
    "perturbed_trajectory": 10,
    "stop_trajectory": 11,
}


def _iter_pes_head_tensors(checkpoint_dir: str):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    model_path = os.path.join(checkpoint_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_to_keys = {}
        for name, shard in index_data["weight_map"].items():
            if name.startswith("pes_head."):
                shard_to_keys.setdefault(shard, []).append(name)
        for shard, keys in shard_to_keys.items():
            shard_path = os.path.join(checkpoint_dir, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    yield key, f.get_tensor(key)
        return

    if os.path.exists(model_path):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("pes_head."):
                    yield key, f.get_tensor(key)
        return

    raise FileNotFoundError(f"Unable to find safetensors weights under {checkpoint_dir}")


def _load_pes_head_override(model, checkpoint_dir: str):
    pes_state_dict = {name: tensor for name, tensor in _iter_pes_head_tensors(checkpoint_dir)}
    if not pes_state_dict:
        raise ValueError(f"No pes_head tensors found under {checkpoint_dir}")
    incompatible = model.load_state_dict(pes_state_dict, strict=False)
    missing_pes = [name for name in incompatible.missing_keys if name.startswith("pes_head.")]
    unexpected_pes = [name for name in incompatible.unexpected_keys if name.startswith("pes_head.")]
    if missing_pes:
        raise RuntimeError(f"Missing pes_head keys while loading override: {missing_pes}")
    if unexpected_pes:
        raise RuntimeError(f"Unexpected pes_head keys while loading override: {unexpected_pes}")
    return len(pes_state_dict)


def _enable_progress_input_lora_config(config, model_args):
    config.enable_progress_latent_adapter = False
    config.enable_progress_input_encoder = True
    config.enable_viva_progress_conditioner = False
    config.progress_adapter_num_heads = int(getattr(model_args, "progress_adapter_num_heads", 8))
    config.progress_adapter_dropout = float(getattr(model_args, "progress_adapter_dropout", 0.1))
    config.progress_adapter_ff_mult = int(getattr(model_args, "progress_adapter_ff_mult", 2))
    config.progress_adapter_num_layers = int(getattr(model_args, "progress_adapter_num_layers", 2))
    config.progress_adapter_num_queries = int(getattr(model_args, "progress_adapter_num_queries", 6))
    config.progress_adapter_num_stages = int(getattr(model_args, "progress_adapter_num_stages", 8))
    config.progress_adapter_residual_scale = float(getattr(model_args, "progress_adapter_residual_scale", 0.2))
    config.progress_max_image_tokens = int(getattr(model_args, "progress_max_image_tokens", 64))
    config.progress_max_text_tokens = int(getattr(model_args, "progress_max_text_tokens", 96))


def _enable_viva_progress_conditioner_config(config, model_args):
    config.enable_progress_latent_adapter = False
    config.enable_progress_input_encoder = False
    config.enable_viva_progress_conditioner = True
    config.progress_adapter_num_heads = int(getattr(model_args, "progress_adapter_num_heads", 8))
    config.progress_adapter_dropout = float(getattr(model_args, "progress_adapter_dropout", 0.1))
    config.progress_adapter_ff_mult = int(getattr(model_args, "progress_adapter_ff_mult", 2))
    config.progress_adapter_num_layers = int(getattr(model_args, "progress_adapter_num_layers", 2))
    config.progress_adapter_num_queries = int(getattr(model_args, "progress_adapter_num_queries", 6))
    config.progress_adapter_num_stages = int(getattr(model_args, "progress_adapter_num_stages", 8))
    config.progress_adapter_residual_scale = float(getattr(model_args, "progress_adapter_residual_scale", 0.2))
    config.progress_max_image_tokens = int(getattr(model_args, "progress_max_image_tokens", 64))
    config.progress_max_text_tokens = int(getattr(model_args, "progress_max_text_tokens", 96))
    config.progress_stop_token_loss_weight = float(getattr(model_args, "progress_stop_token_loss_weight", 1.0))
    config.progress_kl_loss_weight = float(getattr(model_args, "progress_kl_loss_weight", 0.0))
    config.viva_action_vocab_size = int(getattr(model_args, "viva_action_vocab_size", 8))
    config.viva_candidate_vocab_size = int(getattr(model_args, "viva_candidate_vocab_size", 8))


def _load_progress_trainable(model, model_args, device):
    checkpoint_path = getattr(model_args, "progress_trainable_path", "")
    if not checkpoint_path:
        return 0, 0
    target_modules = str(
        getattr(model_args, "repo_lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    )
    replaced = apply_repo_lora(
        model.model,
        target_modules=target_modules,
        r=int(getattr(model_args, "repo_lora_r", 8)),
        alpha=float(getattr(model_args, "repo_lora_alpha", 16.0)),
        dropout=float(getattr(model_args, "repo_lora_dropout", 0.05)),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected progress trainable keys while loading {checkpoint_path}: {unexpected[:20]}")
    return len(state_dict), len(replaced)


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


ACTION_NAMES = {
    int(action_code.STOP): "stop",
    int(action_code.FORWARD): "forward",
    int(action_code.LEFT): "left",
    int(action_code.RIGHT): "right",
    int(action_code.LOOKUP): "look_up",
    int(action_code.LOOKDOWN): "look_down",
}

ACTION_TEXT = {
    int(action_code.STOP): "STOP",
    int(action_code.FORWARD): "MOVE_FORWARD",
    int(action_code.LEFT): "TURN_LEFT",
    int(action_code.RIGHT): "TURN_RIGHT",
    int(action_code.LOOKUP): "LOOK_UP",
    int(action_code.LOOKDOWN): "LOOK_DOWN",
}


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.save_front_video_only = bool(getattr(args, "save_front_video_only", False))
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path
        self.viva_export_dir = os.environ.get("INTERNNAV_VIVA_EXPORT_DIR", "").strip()
        self.viva_export_video = os.environ.get("INTERNNAV_VIVA_EXPORT_VIDEO", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.viva_export_only_failed = os.environ.get("INTERNNAV_VIVA_EXPORT_ONLY_FAILED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.viva_export_jpeg_quality = int(os.environ.get("INTERNNAV_VIVA_EXPORT_JPEG_QUALITY", "90"))
        self.viva_export_cameras = OrderedDict(
            [
                ("125cm_0deg", ("rgb_125cm_0deg", [0.0, 1.25, 0.0], [0.0, 0.0, 0.0])),
                ("125cm_30deg", ("rgb_125cm_30deg", [0.0, 1.25, 0.0], [math.radians(-30.0), 0.0, 0.0])),
                ("125cm_45deg", ("rgb_125cm_45deg", [0.0, 1.25, 0.0], [math.radians(-45.0), 0.0, 0.0])),
                ("60cm_15deg", ("rgb_60cm_15deg", [0.0, 0.60, 0.0], [math.radians(-15.0), 0.0, 0.0])),
                ("60cm_30deg", ("rgb_60cm_30deg", [0.0, 0.60, 0.0], [math.radians(-30.0), 0.0, 0.0])),
            ]
        )

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        requested_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpu_count = max(torch.cuda.device_count(), 1)
        requested_device_id = requested_local_rank % gpu_count

        with habitat.config.read_write(self.config):
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = requested_device_id
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            if self.viva_export_dir:
                sim_sensors = self.config.habitat.simulator.agents.main_agent.sim_sensors
                base_rgb = sim_sensors.rgb_sensor
                for _, (sensor_name, position, orientation) in self.viva_export_cameras.items():
                    sensor_cfg = HeadRGBSensorConfig(
                        height=int(base_rgb.height),
                        width=int(base_rgb.width),
                        hfov=int(base_rgb.hfov),
                        position=list(position),
                        orientation=list(orientation),
                        uuid=sensor_name,
                    )
                    sim_sensors[sensor_name] = sensor_cfg
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        # ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))
        self.disable_pixel_goal = bool(getattr(self.model_args, "disable_pixel_goal", False))
        self.pixel_goal_strict_parse = bool(getattr(self.model_args, "pixel_goal_strict_parse", False))
        self.pixel_goal_max_jump = float(getattr(self.model_args, "pixel_goal_max_jump", 0.0))
        self.enable_qwen_stop_verify = bool(getattr(self.model_args, "enable_qwen_stop_verify", False))
        self.qwen_stop_verify_max_new_tokens = int(getattr(self.model_args, "qwen_stop_verify_max_new_tokens", 8))
        self.qwen_stop_reject_action = str(getattr(self.model_args, "qwen_stop_reject_action", "lookdown")).lower()
        self.enable_longclip_stop_verify = bool(getattr(self.model_args, "enable_longclip_stop_verify", False))
        self.longclip_stop_model_path = getattr(self.model_args, "longclip_stop_model_path", "")
        self.longclip_stop_weight_path = getattr(self.model_args, "longclip_stop_weight_path", "")
        self.longclip_stop_threshold = float(getattr(self.model_args, "longclip_stop_threshold", 0.1))
        self.log_pes_outputs = bool(getattr(self.model_args, "log_pes_outputs", False))
        self.enable_pes_stop_verify = bool(getattr(self.model_args, "enable_pes_stop_verify", False))
        self.pes_stop_event_threshold = float(getattr(self.model_args, "pes_stop_event_threshold", 0.8))
        self.pes_stop_progress_threshold = float(getattr(self.model_args, "pes_stop_progress_threshold", 0.4))
        self.enable_stop_residual_verify = bool(getattr(self.model_args, "enable_stop_residual_verify", False))
        self.stop_residual_adapter_path = str(getattr(self.model_args, "stop_residual_adapter_path", ""))
        self.stop_residual_accept_threshold = float(getattr(self.model_args, "stop_residual_accept_threshold", 0.5))
        self.stop_residual_delta_scale = float(getattr(self.model_args, "stop_residual_delta_scale", 1.0))
        self.stop_residual_reject_on_error = bool(getattr(self.model_args, "stop_residual_reject_on_error", False))
        self.stop_residual_use_distance_progress = bool(
            getattr(self.model_args, "stop_residual_use_distance_progress", True)
        )
        self.enable_viva_online_structured_inputs = bool(
            getattr(
                self.model_args,
                "enable_viva_online_structured_inputs",
                bool(getattr(self.model_args, "enable_viva_progress_conditioner", False)),
            )
        )
        self.viva_online_max_action_block = int(getattr(self.model_args, "viva_online_max_action_block", 12))
        self.viva_online_max_candidates = int(getattr(self.model_args, "viva_online_max_candidates", 8))
        self.log_viva_online_inputs = bool(getattr(self.model_args, "log_viva_online_inputs", False))

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'
        if getattr(processor, "chat_template", None) is None and getattr(processor.tokenizer, "chat_template", None):
            processor.chat_template = processor.tokenizer.chat_template

        device_id = self.local_rank % gpu_count
        device = torch.device(f"cuda:{device_id}")
        if self.model_args.mode == 'dual_system':
            config_path = os.path.join(self.model_args.model_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            architectures = set(model_config.get("architectures") or [])
            use_pes_model = bool(model_config.get("enable_pes_head", False)) or "InternVLAN1PESForCausalLM" in architectures
            model_cls = InternVLAN1PESForCausalLM if use_pes_model else InternVLAN1ForCausalLM
            progress_trainable_path = getattr(self.model_args, "progress_trainable_path", "")
            model_load_kwargs = {}
            if progress_trainable_path:
                progress_config = InternVLAN1ModelConfig.from_pretrained(self.model_args.model_path)
                if bool(getattr(self.model_args, "enable_viva_progress_conditioner", False)):
                    _enable_viva_progress_conditioner_config(progress_config, self.model_args)
                else:
                    _enable_progress_input_lora_config(progress_config, self.model_args)
                model_load_kwargs["config"] = progress_config
            model = load_pretrained_with_attention_fallback(
                model_cls,
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                **model_load_kwargs,
            )
        elif self.model_args.mode == 'system2':
            model = load_pretrained_with_attention_fallback(
                Qwen2_5_VLForConditionalGeneration,
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        progress_trainable_path = getattr(self.model_args, "progress_trainable_path", "")
        if progress_trainable_path:
            loaded_tensors, lora_modules = _load_progress_trainable(model, self.model_args, device)
            if self.rank == 0:
                print(
                    f"Loaded progress trainable from {progress_trainable_path} "
                    f"({loaded_tensors} tensors, {lora_modules} LoRA modules)",
                    flush=True,
                )

        pes_head_override_path = getattr(self.model_args, "pes_head_override_path", "")
        if pes_head_override_path:
            if not hasattr(model, "pes_head"):
                raise ValueError("pes_head_override_path was set, but the loaded model has no pes_head")
            loaded_tensors = _load_pes_head_override(model, pes_head_override_path)
            if self.rank == 0:
                print(
                    f"Loaded PES head override from {pes_head_override_path} ({loaded_tensors} tensors)",
                    flush=True,
                )

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor
        self._pes_log_path = os.path.join(self.output_path, f"pes_outputs_rank{self.rank}.jsonl")
        self._pes_stop_verify_log_path = os.path.join(self.output_path, f"pes_stop_verify_rank{self.rank}.jsonl")
        self._stop_residual_verify_log_path = os.path.join(self.output_path, f"stop_residual_verify_rank{self.rank}.jsonl")
        self._viva_online_input_log_path = os.path.join(self.output_path, f"viva_online_inputs_rank{self.rank}.jsonl")
        self.stop_residual_adapter = None
        self.stop_residual_meta = {}
        self.stop_residual_stop_token_ids = self._build_stop_residual_stop_token_ids()
        self._stop_residual_progress_history = []
        if self.enable_stop_residual_verify:
            if not self.stop_residual_adapter_path:
                raise ValueError("enable_stop_residual_verify=True requires stop_residual_adapter_path")
            self.stop_residual_adapter, self.stop_residual_meta = load_stop_residual_adapter(
                self.stop_residual_adapter_path,
                device,
            )
            if self.rank == 0:
                print(
                    "Loaded STOP residual adapter "
                    f"from {self.stop_residual_adapter_path} "
                    f"meta={self.stop_residual_meta}",
                    flush=True,
                )
        if self.viva_export_dir:
            os.makedirs(self.viva_export_dir, exist_ok=True)
            self._viva_episode_trace_path = os.path.join(self.viva_export_dir, f"episode_traces_rank{self.rank}.jsonl")
            self._viva_stop_intervention_path = os.path.join(
                self.viva_export_dir,
                f"stop_interventions_rank{self.rank}.jsonl",
            )
        else:
            self._viva_episode_trace_path = None
            self._viva_stop_intervention_path = None
        self.longclip_model = None
        self.longclip_preprocess = None
        self.longclip_projector = None
        self.longclip_head = None
        self.longclip_mode = "projector"
        self.longclip_head_type = "projector"

        if self.enable_longclip_stop_verify:
            self.longclip_model, self.longclip_preprocess = longclip.load(self.longclip_stop_model_path, device=device)
            ckpt = torch.load(self.longclip_stop_weight_path, map_location="cpu")
            feature_dim = int(self.longclip_model.text_projection.shape[-1])
            if "head" in ckpt:
                self.longclip_mode = "elements"
                self.longclip_head_type = str(ckpt.get("head_type", "mlp"))
                head_state = ckpt["head"]
                obj_threshold = float(ckpt.get("obj_threshold", 0.25))
                coverage_temperature = float(ckpt.get("coverage_temperature", 12.0))
                max_target_elements = int(ckpt.get("max_target_elements", 4))
                history_frames = int(ckpt.get("history_frames", 5))
                if self.longclip_head_type == "grounding_v4":
                    self.longclip_head = GroundingQueryStopHead(
                        feature_dim,
                        int(ckpt["proj_dim"]),
                        hidden_dim=int(ckpt["hidden_dim"]),
                        num_heads=int(ckpt.get("attn_heads", 4)),
                        dropout=float(ckpt.get("attn_dropout", 0.0)),
                    ).to(device)
                    self.longclip_head.load_state_dict(head_state, strict=True)
                    self.longclip_head.eval()
                    self.longclip_head_meta = {
                        "obj_threshold": obj_threshold,
                        "coverage_temperature": coverage_temperature,
                        "max_target_elements": max_target_elements,
                        "history_frames": history_frames,
                    }
                elif self.longclip_head_type == "cross_attn":
                    self.longclip_head = CrossAttentionStopHead(
                        feature_dim,
                        int(ckpt["proj_dim"]),
                        hidden_dim=int(ckpt["hidden_dim"]),
                        num_heads=int(ckpt.get("attn_heads", 4)),
                        dropout=float(ckpt.get("attn_dropout", 0.0)),
                    ).to(device)
                    self.longclip_head.load_state_dict(head_state, strict=True)
                    self.longclip_head.eval()
                    self.longclip_head_meta = {
                        "obj_threshold": obj_threshold,
                        "coverage_temperature": coverage_temperature,
                        "max_target_elements": max_target_elements,
                        "history_frames": history_frames,
                    }
                else:
                    score_head_state = {
                        key: value.to(device)
                        for key, value in head_state.items()
                        if key.startswith("score_head.")
                    }
                    self.longclip_head = {
                        "image_proj.weight": head_state["image_proj.weight"].to(device),
                        "image_proj.bias": head_state["image_proj.bias"].to(device),
                        "text_proj.weight": head_state["text_proj.weight"].to(device),
                        "text_proj.bias": head_state["text_proj.bias"].to(device),
                        "score_head_state": score_head_state,
                        "obj_threshold": obj_threshold,
                        "coverage_temperature": coverage_temperature,
                        "max_target_elements": max_target_elements,
                        "history_frames": history_frames,
                    }
            else:
                proj_dim = int(ckpt["proj_dim"])
                self.longclip_projector = {
                    "image_proj.weight": ckpt["projector"]["image_proj.weight"].to(device),
                    "image_proj.bias": ckpt["projector"]["image_proj.bias"].to(device),
                    "text_proj.weight": ckpt["projector"]["text_proj.weight"].to(device),
                    "text_proj.bias": ckpt["projector"]["text_proj.bias"].to(device),
                    "logit_scale": ckpt["projector"]["logit_scale"].to(device),
                    "feature_dim": feature_dim,
                    "proj_dim": proj_dim,
                }
            self.longclip_model.eval()

        # refactor: this part used in three places
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.stop_verify_prompt = (
            "You are a binary stop verifier, not a navigation planner.\n"
            "Instruction: <instruction>\n"
            "Use only the current view to decide whether the final stopping condition in the instruction is already satisfied.\n"
            "Focus only on the final landmark/object and the required relative position, such as stop by, stop near, stop at, stop in front of, stop beside, stop under, or stop at the doorway.\n"
            "If the final stopping condition is clearly satisfied, answer YES.\n"
            "If the final stopping condition is not satisfied or you are uncertain, answer NO.\n"
            "Do not output arrows, coordinates, actions, explanations, or any extra words.\n"
            "The only valid answers are exactly YES or NO."
        )
        self.pixel_recover_prompt = (
            "You are recovering navigation after a rejected STOP.\n"
            "Instruction: <instruction>\n"
            "Use only the current look-down view and output the next waypoint coordinates in the image.\n"
            "Output exactly two integers for the waypoint coordinates and nothing else.\n"
            "Do not output STOP, arrows, actions, explanations, or any extra words."
        )

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.num_history = self.model_args.num_history

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def _run_vlm_text(self, prompt: str, images: list, max_new_tokens: int = 16) -> str:
        messages = [{"role": "user", "content": []}]
        if images and DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        parts = split_and_clean(prompt)
        input_img_id = 0
        for part in parts:
            if part == DEFAULT_IMAGE_TOKEN:
                messages[0]["content"].append({"type": "image", "image": images[input_img_id]})
                input_img_id += 1
            else:
                messages[0]["content"].append({"type": "text", "text": part})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        return self.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

    def _score_vlm_choices(self, prompt: str, images: list, choices: list[str]) -> list[float]:
        messages = [{"role": "user", "content": []}]
        if images and DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        parts = split_and_clean(prompt)
        input_img_id = 0
        for part in parts:
            if part == DEFAULT_IMAGE_TOKEN:
                messages[0]["content"].append({"type": "image", "image": images[input_img_id]})
                input_img_id += 1
            else:
                messages[0]["content"].append({"type": "text", "text": part})

        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt_text], images=images, return_tensors="pt").to(self.model.device)
        prompt_len = int(prompt_inputs.input_ids.shape[1])

        scores = []
        with torch.no_grad():
            for choice in choices:
                choice_text = prompt_text + choice
                choice_inputs = self.processor(text=[choice_text], images=images, return_tensors="pt").to(self.model.device)
                logits = self.model(**choice_inputs).logits[:, :-1, :]
                labels = choice_inputs.input_ids[:, 1:]
                choice_token_count = int(choice_inputs.input_ids.shape[1] - prompt_len)
                if choice_token_count <= 0:
                    scores.append(float("-inf"))
                    continue
                choice_logits = logits[:, prompt_len - 1 :, :]
                choice_labels = labels[:, prompt_len - 1 :]
                token_log_probs = F.log_softmax(choice_logits, dim=-1).gather(-1, choice_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_log_probs.sum().item()))
        return scores

    def _log_pes_outputs(
        self,
        inputs,
        scene_id,
        episode_id,
        step_id,
        current_metrics=None,
        initial_distance_to_goal=None,
    ):
        if not self.log_pes_outputs or not hasattr(self.model, "pes_head"):
            return

        try:
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, use_cache=False)
        except Exception as exc:
            if self.rank == 0:
                print(f"pes_output_log_failed step={step_id}: {exc}", flush=True)
            return

        stop_logits = getattr(outputs, "stop_logits", None)
        progress_scores = getattr(outputs, "progress_scores", None)
        event_logits = getattr(outputs, "event_logits", None)
        pes_latent = getattr(outputs, "pes_latent", None)
        pes_query_features = getattr(outputs, "pes_query_features", None)
        if stop_logits is None or progress_scores is None or event_logits is None:
            return

        stop_logit = float(stop_logits.detach().float().flatten()[0].cpu())
        progress = float(progress_scores.detach().float().flatten()[0].cpu())
        event_probs = torch.softmax(event_logits.detach().float(), dim=-1).flatten().cpu().tolist()
        pes_norm = None
        if pes_latent is not None:
            pes_norm = float(pes_latent.detach().float().norm(dim=-1).flatten()[0].cpu())
        pes_query_norm = None
        if pes_query_features is not None:
            pes_query_norm = float(pes_query_features.detach().float().norm(dim=-1).mean().cpu())

        distance_to_goal = None
        oracle_stop_label = None
        distance_progress = None
        if current_metrics is not None and "distance_to_goal" in current_metrics:
            distance_to_goal = float(current_metrics["distance_to_goal"])
            oracle_stop_label = int(distance_to_goal < 3.0)
            if initial_distance_to_goal is not None and initial_distance_to_goal > 1e-6:
                distance_progress = 1.0 - min(distance_to_goal / initial_distance_to_goal, 1.0)

        row = {
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "stop_logit": stop_logit,
            "stop_prob": float(torch.sigmoid(torch.tensor(stop_logit)).item()),
            "progress": progress,
            "event_pred": int(np.argmax(event_probs)) if event_probs else -1,
            "event_probs": event_probs,
            "pes_latent_norm": pes_norm,
            "pes_query_norm": pes_query_norm,
            "distance_to_goal": distance_to_goal,
            "oracle_stop_label": oracle_stop_label,
            "distance_progress": distance_progress,
        }
        os.makedirs(self.output_path, exist_ok=True)
        with open(self._pes_log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _parse_pixel_goal(
        self, llm_outputs: str, last_pixel_goal: Optional[list] = None
    ) -> tuple[Optional[list], Optional[str]]:
        coords = [int(c) for c in re.findall(r"\d+", llm_outputs)]
        if self.pixel_goal_strict_parse and len(coords) != 2:
            return None, f"strict_parse_reject coords={coords}"
        if len(coords) < 2:
            return None, f"coord_count_reject coords={coords}"

        pixel_goal = [int(coords[1]), int(coords[0])]
        pixel_goal[0] = int(np.clip(pixel_goal[0], 0, self.model_args.resize_w - 1))
        pixel_goal[1] = int(np.clip(pixel_goal[1], 0, self.model_args.resize_h - 1))

        if self.pixel_goal_max_jump > 0 and last_pixel_goal is not None:
            jump = float(np.linalg.norm(np.array(pixel_goal, dtype=np.float32) - np.array(last_pixel_goal, dtype=np.float32)))
            if jump > self.pixel_goal_max_jump:
                return None, f"jump_reject jump={jump:.1f} last={last_pixel_goal} now={pixel_goal}"

        return pixel_goal, None

    def _stop_reject_action_code(self) -> int:
        mapping = {
            "lookdown": action_code.LOOKDOWN,
            "forward": action_code.FORWARD,
            "left": action_code.LEFT,
            "right": action_code.RIGHT,
        }
        return int(mapping.get(self.qwen_stop_reject_action, action_code.LOOKDOWN))

    def _build_stop_residual_stop_token_ids(self) -> list[int]:
        tokenizer = self.processor.tokenizer
        token_ids = set()
        for text in ("STOP", " STOP", "\nSTOP"):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if ids:
                token_ids.add(int(ids[0]))
        return sorted(token_ids)

    def _stop_residual_current_progress(
        self,
        current_metrics=None,
        initial_distance_to_goal=None,
    ) -> Optional[float]:
        if not self.stop_residual_use_distance_progress:
            return None
        if current_metrics is None or "distance_to_goal" not in current_metrics:
            return None
        if initial_distance_to_goal is None or initial_distance_to_goal <= 1e-6:
            return None
        try:
            distance_to_goal = float(current_metrics["distance_to_goal"])
            progress = 1.0 - min(distance_to_goal / float(initial_distance_to_goal), 1.0)
            return float(np.clip(progress, 0.0, 1.0))
        except Exception:
            return None

    def _append_stop_residual_progress_history(
        self,
        *,
        step_id: int,
        current_metrics=None,
        initial_distance_to_goal=None,
    ) -> None:
        progress = self._stop_residual_current_progress(current_metrics, initial_distance_to_goal)
        if progress is None:
            progress = 0.0
        if self._stop_residual_progress_history and int(self._stop_residual_progress_history[-1]["step_id"]) == int(step_id):
            self._stop_residual_progress_history[-1]["progress"] = float(progress)
            return
        self._stop_residual_progress_history.append({"step_id": int(step_id), "progress": float(progress)})

    def _stop_residual_prompt_features(self, inputs, forward_kwargs: Optional[dict] = None) -> dict:
        forward_kwargs = forward_kwargs or {}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
                **forward_kwargs,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("model forward returned no hidden_states")
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("model forward returned no logits")

        last_hidden = hidden_states[-1]
        token_dim = int(self.stop_residual_meta.get("token_dim", last_hidden.shape[-1]))
        num_qwen_tokens = int(self.stop_residual_meta.get("num_qwen_tokens", 5))
        hidden_dim = int(self.stop_residual_meta.get("hidden_dim", num_qwen_tokens * token_dim))
        if last_hidden.shape[-1] != token_dim:
            raise RuntimeError(f"hidden width mismatch: got {last_hidden.shape[-1]}, expected {token_dim}")
        if last_hidden.shape[1] < num_qwen_tokens:
            raise RuntimeError(f"prompt too short for {num_qwen_tokens} hidden tokens")
        hidden = last_hidden[:, -num_qwen_tokens:, :].detach().float().contiguous().view(1, -1)
        if hidden.shape[-1] != hidden_dim:
            raise RuntimeError(f"hidden dim mismatch: got {hidden.shape[-1]}, expected {hidden_dim}")

        next_logits = logits[:, -1, :].detach().float()
        probs = torch.softmax(next_logits, dim=-1)
        stop_ids = [idx for idx in self.stop_residual_stop_token_ids if idx < probs.shape[-1]]
        if not stop_ids:
            raise RuntimeError("no valid STOP token ids for tokenizer")
        stop_prob = float(probs[:, stop_ids].max().item())
        first_token_prob = float(probs.max(dim=-1).values.item())
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).item())
        return {
            "hidden": hidden,
            "stop_prob": stop_prob,
            "base_binary_logit": stop_residual_logit(stop_prob),
            "first_token_prob": first_token_prob,
            "first_token_entropy": entropy,
            "stop_token_ids": stop_ids,
        }

    def _log_stop_residual_verify(self, row: dict) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        with open(self._stop_residual_verify_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _verify_stop_with_residual_adapter(
        self,
        inputs,
        scene_id=None,
        episode_id=None,
        step_id=None,
        current_metrics=None,
        initial_distance_to_goal=None,
        llm_outputs=None,
        forward_kwargs: Optional[dict] = None,
    ) -> tuple[bool, str]:
        if self.stop_residual_adapter is None:
            answer = "stop_residual_adapter_missing_accept"
            return True, answer

        distance_to_goal = None
        if current_metrics is not None and "distance_to_goal" in current_metrics:
            try:
                distance_to_goal = float(current_metrics["distance_to_goal"])
            except Exception:
                distance_to_goal = None
        progress = self._stop_residual_current_progress(current_metrics, initial_distance_to_goal)
        if progress is None:
            progress = 0.0

        try:
            prompt_features = self._stop_residual_prompt_features(inputs, forward_kwargs=forward_kwargs)
            progress_tokens, progress_valid = build_online_progress_tokens(
                self._stop_residual_progress_history,
                step_id=int(step_id or 0),
                current_progress=float(progress),
                qstop_prob=float(prompt_features["stop_prob"]),
                first_token_prob=float(prompt_features["first_token_prob"]),
                first_token_entropy=float(prompt_features["first_token_entropy"]),
                qwen_says_stop=True,
                windows=list(self.stop_residual_meta.get("progress_windows", [10, 20, 30, 50])),
                device=self.model.device,
            )
            hidden = prompt_features["hidden"].to(self.model.device)
            with torch.no_grad():
                delta = float(self.stop_residual_adapter(hidden, progress_tokens, progress_valid).float().flatten()[0].item())
            scaled_delta = float(delta * self.stop_residual_delta_scale)
            final_logit = float(prompt_features["base_binary_logit"] + scaled_delta)
            final_prob = float(torch.sigmoid(torch.tensor(final_logit)).item())
            accept = final_prob >= self.stop_residual_accept_threshold
            answer = (
                f"final_prob={final_prob:.4f}>={self.stop_residual_accept_threshold:.4f} "
                f"base_stop_prob={prompt_features['stop_prob']:.4f} "
                f"delta={scaled_delta:.4f} progress={progress:.4f}"
            )
            row = {
                "rank": int(self.rank),
                "scene_id": scene_id,
                "episode_id": int(episode_id) if episode_id is not None else None,
                "step_id": int(step_id) if step_id is not None else None,
                "llm_outputs": llm_outputs,
                "accept": bool(accept),
                "answer": answer,
                "base_stop_prob": float(prompt_features["stop_prob"]),
                "base_binary_logit": float(prompt_features["base_binary_logit"]),
                "adapter_delta": scaled_delta,
                "adapter_delta_raw": delta,
                "final_prob": final_prob,
                "threshold": float(self.stop_residual_accept_threshold),
                "progress": float(progress),
                "history_len": len(self._stop_residual_progress_history),
                "distance_to_goal": distance_to_goal,
                "initial_distance_to_goal": initial_distance_to_goal,
                "stop_token_ids": prompt_features["stop_token_ids"],
                "error": None,
            }
            self._log_stop_residual_verify(row)
            return accept, answer
        except Exception as exc:
            accept = not self.stop_residual_reject_on_error
            answer = f"stop_residual_failed={exc!r} accept_on_error={accept}"
            self._log_stop_residual_verify(
                {
                    "rank": int(self.rank),
                    "scene_id": scene_id,
                    "episode_id": int(episode_id) if episode_id is not None else None,
                    "step_id": int(step_id) if step_id is not None else None,
                    "llm_outputs": llm_outputs,
                    "accept": bool(accept),
                    "answer": answer,
                    "distance_to_goal": distance_to_goal,
                    "initial_distance_to_goal": initial_distance_to_goal,
                    "progress": float(progress),
                    "history_len": len(self._stop_residual_progress_history),
                    "error": repr(exc),
                }
            )
            return accept, answer

    def _log_pes_stop_verify(
        self,
        *,
        scene_id,
        episode_id,
        step_id,
        accept,
        answer,
        stop_logit=None,
        progress=None,
        event_probs=None,
        current_metrics=None,
        initial_distance_to_goal=None,
        llm_outputs=None,
        error=None,
    ) -> None:
        event4 = float(event_probs[4]) if event_probs and len(event_probs) > 4 else None
        distance_to_goal = None
        oracle_stop_label = None
        distance_progress = None
        if current_metrics is not None and "distance_to_goal" in current_metrics:
            distance_to_goal = float(current_metrics["distance_to_goal"])
            oracle_stop_label = int(distance_to_goal < 3.0)
            if initial_distance_to_goal is not None and initial_distance_to_goal > 1e-6:
                distance_progress = 1.0 - min(distance_to_goal / initial_distance_to_goal, 1.0)

        row = {
            "rank": int(self.rank),
            "scene_id": scene_id,
            "episode_id": int(episode_id) if episode_id is not None else None,
            "step_id": int(step_id) if step_id is not None else None,
            "llm_outputs": llm_outputs,
            "accept": bool(accept),
            "answer": answer,
            "stop_logit": stop_logit,
            "stop_prob": float(torch.sigmoid(torch.tensor(stop_logit)).item()) if stop_logit is not None else None,
            "progress": progress,
            "event_pred": int(np.argmax(event_probs)) if event_probs else None,
            "event4": event4,
            "event_probs": event_probs,
            "pes_stop_event_threshold": float(self.pes_stop_event_threshold),
            "pes_stop_progress_threshold": float(self.pes_stop_progress_threshold),
            "distance_to_goal": distance_to_goal,
            "oracle_stop_label": oracle_stop_label,
            "distance_progress": distance_progress,
            "error": error,
        }
        os.makedirs(self.output_path, exist_ok=True)
        with open(self._pes_stop_verify_log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _metric_distance_to_goal(self) -> Optional[float]:
        try:
            metrics = self.env.get_metrics()
        except Exception:
            return None
        value = metrics.get("distance_to_goal")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _viva_export_image_path(self, scene_id: str, episode_id: int, frame_idx: int, camera: str) -> str:
        safe_scene = str(scene_id).replace("/", "_")
        return os.path.join(
            self.viva_export_dir,
            "images",
            safe_scene,
            f"episode_{int(episode_id):06d}",
            camera,
            f"frame_{int(frame_idx):06d}.jpg",
        )

    def _viva_agent_state(self) -> dict:
        try:
            state = self.env._env.sim.get_agent_state()
            rot = state.rotation
            return {
                "position": [float(x) for x in np.asarray(state.position, dtype=np.float32).tolist()],
                "rotation_wxyz": [float(rot.w), float(rot.x), float(rot.y), float(rot.z)],
            }
        except Exception as exc:
            return {"position": None, "rotation_wxyz": None, "error": repr(exc)}

    def _viva_reference_path(self, episode) -> list:
        ref = getattr(episode, "reference_path", None) or []
        out = []
        for point in ref:
            try:
                out.append([float(x) for x in point])
            except Exception:
                continue
        return out

    def _viva_goal_positions(self, episode) -> list:
        goals = getattr(episode, "goals", None) or []
        out = []
        for goal in goals:
            pos = getattr(goal, "position", None)
            if pos is None:
                continue
            try:
                out.append([float(x) for x in pos])
            except Exception:
                continue
        return out

    def _viva_capture_frame(
        self,
        *,
        episode_frames: list,
        observations: dict,
        scene_id: str,
        episode_id: int,
        policy_step_id: int,
        instruction: str,
        llm_outputs: str,
        initial_distance_to_goal: Optional[float],
    ) -> None:
        if not self.viva_export_dir:
            return
        frame_idx = len(episode_frames)
        camera_paths = {}
        missing_cameras = []
        for camera, (sensor_name, _, _) in self.viva_export_cameras.items():
            image_arr = observations.get(sensor_name)
            if image_arr is None and camera == "125cm_0deg":
                image_arr = observations.get("rgb")
            if image_arr is None:
                missing_cameras.append(camera)
                continue
            path = self._viva_export_image_path(scene_id, episode_id, frame_idx, camera)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            arr = np.asarray(image_arr)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                arr = arr[..., :3]
            Image.fromarray(arr.astype(np.uint8)).convert("RGB").save(
                path,
                quality=int(self.viva_export_jpeg_quality),
            )
            camera_paths[camera] = path

        metrics = self.env.get_metrics()
        distance_to_goal = metrics.get("distance_to_goal")
        try:
            distance_to_goal = float(distance_to_goal)
        except (TypeError, ValueError):
            distance_to_goal = None
        distance_progress = None
        if distance_to_goal is not None and initial_distance_to_goal is not None and initial_distance_to_goal > 1e-6:
            distance_progress = 1.0 - min(distance_to_goal / initial_distance_to_goal, 1.0)

        episode_frames.append(
            {
                "frame_idx": int(frame_idx),
                "policy_step_id": int(policy_step_id),
                "instruction": instruction,
                "llm_outputs_before_action": llm_outputs,
                "camera_paths": camera_paths,
                "missing_cameras": missing_cameras,
                "agent_state": self._viva_agent_state(),
                "distance_to_goal": distance_to_goal,
                "distance_progress": distance_progress,
                "action": None,
                "action_name": None,
                "action_text": None,
                "distance_to_goal_before": None,
                "distance_to_goal_after": None,
                "distance_delta": None,
                "qwen_says_stop": False,
                "stop_interrupted": False,
                "stop_verify_tag": None,
                "stop_verify_answer": None,
                "stop_reject_action": None,
                "done_after_action": False,
            }
        )

    def _viva_update_last_frame(
        self,
        *,
        episode_frames: list,
        action,
        before_distance: Optional[float],
        after_distance: Optional[float],
        done: bool,
        llm_outputs: str,
        stop_event: Optional[dict] = None,
    ) -> None:
        if not self.viva_export_dir or not episode_frames:
            return
        frame = episode_frames[-1]
        action_int = int(action)
        frame["action"] = action_int
        frame["action_name"] = ACTION_NAMES.get(action_int, str(action_int))
        frame["action_text"] = ACTION_TEXT.get(action_int, "MOVE")
        if action_int == int(action_code.STOP):
            frame["qwen_says_stop"] = True
        frame["llm_outputs_after_action"] = llm_outputs
        frame["distance_to_goal_before"] = before_distance
        frame["distance_to_goal_after"] = after_distance
        if before_distance is not None and after_distance is not None:
            frame["distance_delta"] = float(before_distance) - float(after_distance)
        frame["done_after_action"] = bool(done)
        if stop_event:
            frame.update(
                {
                    "qwen_says_stop": True,
                    "stop_interrupted": True,
                    "stop_verify_tag": stop_event.get("stop_verify_tag"),
                    "stop_verify_answer": stop_event.get("stop_verify_answer"),
                    "stop_reject_action": stop_event.get("stop_reject_action"),
                }
            )

    def _viva_log_stop_intervention(self, row: dict) -> None:
        if not self._viva_stop_intervention_path:
            return
        with open(self._viva_stop_intervention_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _viva_write_episode_trace(
        self,
        *,
        episode,
        scene_id: str,
        episode_id: int,
        instruction: str,
        initial_distance_to_goal: Optional[float],
        frames: list,
        interventions: list,
        metrics: dict,
    ) -> None:
        if not self._viva_episode_trace_path:
            return
        success = float(metrics.get("success", 0.0))
        if self.viva_export_only_failed and success >= 0.5:
            return
        if success >= 0.5:
            failure_type = "success"
        elif interventions:
            failure_type = "interrupted_stop_failure"
        else:
            failure_type = "native_failure"
        row = {
            "rank": int(self.rank),
            "scene_id": scene_id,
            "episode_id": int(episode_id),
            "instruction": instruction,
            "source_split": str(getattr(self.config.habitat.dataset, "split", "")),
            "failure_type": failure_type,
            "initial_distance_to_goal": initial_distance_to_goal,
            "metrics": {
                "success": success,
                "spl": float(metrics.get("spl", 0.0)),
                "oracle_success": float(metrics.get("oracle_success", 0.0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
                "ndtw": float(metrics["ndtw"]) if "ndtw" in metrics else None,
            },
            "reference_path": self._viva_reference_path(episode),
            "goal_positions": self._viva_goal_positions(episode),
            "interventions": interventions,
            "frames": frames,
        }
        with open(self._viva_episode_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _append_viva_online_action(
        self,
        action_block: list,
        action,
        before_distance: Optional[float],
        after_distance: Optional[float],
        origin: str,
    ) -> None:
        if not self.enable_viva_online_structured_inputs:
            return
        action_int = int(action)
        distance_delta = 0.0
        if before_distance is not None and after_distance is not None:
            # Match training convention: positive means distance-to-goal decreased.
            distance_delta = float(before_distance) - float(after_distance)
        action_block.append(
            {
                "action": action_int,
                "action_origin": origin,
                "distance_to_goal_before": before_distance,
                "distance_to_goal_after": after_distance,
                "distance_delta": distance_delta,
            }
        )
        keep = max(1, int(self.viva_online_max_action_block))
        del action_block[:-keep]

    def _viva_online_action_tensors(self, action_block: list) -> tuple[torch.Tensor, torch.Tensor]:
        max_actions = max(1, int(self.viva_online_max_action_block))
        ids = torch.zeros(max_actions, dtype=torch.long)
        deltas = torch.zeros(max_actions, dtype=torch.float32)
        for idx, item in enumerate((action_block or [])[-max_actions:]):
            action_id = int(item.get("action", 6))
            ids[idx] = max(1, min(action_id + 1, 7))
            delta = item.get("distance_delta")
            try:
                deltas[idx] = max(-1.0, min(1.0, float(delta) / 2.0))
            except (TypeError, ValueError):
                deltas[idx] = 0.0
        return ids, deltas

    def _actions_to_viva_traj_features(self, actions: list) -> list[float]:
        actions = [int(a) for a in (actions or [])]
        is_stop = bool(actions and actions[0] == action_code.STOP)
        if not actions or is_stop:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if is_stop else 0.0]

        x, z, yaw = 0.0, 0.0, 0.0
        pts = []
        turn = math.radians(15.0)
        for action in actions[:MAX_STEPS]:
            if action == action_code.LEFT:
                yaw += turn
            elif action == action_code.RIGHT:
                yaw -= turn
            elif action == action_code.FORWARD:
                x += math.sin(yaw) * 0.25
                z += math.cos(yaw) * 0.25
            elif action == action_code.STOP:
                is_stop = True
                pts.append((x, z, yaw))
                break
            pts.append((x, z, yaw))

        if not pts:
            return [0.0] * 8
        end_x, end_z, end_yaw = pts[-1]
        prev = (0.0, 0.0)
        path_len = 0.0
        max_step = 0.0
        yaw_abs = 0.0
        for px, pz, pyaw in pts:
            step = ((px - prev[0]) ** 2 + (pz - prev[1]) ** 2) ** 0.5
            path_len += step
            max_step = max(max_step, step)
            yaw_abs += abs(pyaw)
            prev = (px, pz)
        direct = (end_x * end_x + end_z * end_z) ** 0.5
        n = max(1, len(pts))

        def clip(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        return [
            clip(end_x / 3.0),
            clip(end_z / 3.0),
            clip(end_yaw / math.pi),
            clip(path_len / 6.0),
            clip(direct / 6.0),
            clip(max_step / 1.5),
            clip((yaw_abs / n) / math.pi),
            1.0 if is_stop else 0.0,
        ]

    def _trajectory_points_to_viva_features(self, points, is_stop: bool = False) -> list[float]:
        if points is None:
            return [0.0] * 7 + [1.0 if is_stop else 0.0]
        pts = []
        for item in points:
            if len(item) < 2:
                continue
            x = float(item[0])
            z = float(item[1])
            yaw = float(item[2]) if len(item) > 2 else 0.0
            pts.append((x, z, yaw))
        if not pts:
            return [0.0] * 7 + [1.0 if is_stop else 0.0]
        end_x, end_z, end_yaw = pts[-1]
        prev = (0.0, 0.0)
        path_len = 0.0
        max_step = 0.0
        yaw_abs = 0.0
        for x, z, yaw in pts:
            step = ((x - prev[0]) ** 2 + (z - prev[1]) ** 2) ** 0.5
            path_len += step
            max_step = max(max_step, step)
            yaw_abs += abs(yaw)
            prev = (x, z)
        direct = (end_x * end_x + end_z * end_z) ** 0.5
        n = max(1, len(pts))

        def clip(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        return [
            clip(end_x / 3.0),
            clip(end_z / 3.0),
            clip(end_yaw / math.pi),
            clip(path_len / 6.0),
            clip(direct / 6.0),
            clip(max_step / 1.5),
            clip((yaw_abs / n) / math.pi),
            1.0 if is_stop else 0.0,
        ]

    def _nextdit_raw_to_viva_candidate_features(self, dp_actions) -> torch.Tensor:
        max_diffusion = max(0, int(self.viva_online_max_candidates) - 1)
        out = torch.zeros(max_diffusion, 8, dtype=torch.float32)
        if max_diffusion == 0 or dp_actions is None:
            return out
        samples = dp_actions.detach().float().cpu()
        if samples.dim() == 2:
            samples = samples.unsqueeze(0)
        if samples.dim() != 3 or samples.size(-1) < 2:
            return out
        samples = samples[:max_diffusion].clone()
        # Match traj_to_actions: NextDiT x/y deltas are unnormalized by /4 before cumulative rollout.
        samples[:, :, :2] = samples[:, :, :2] / 4.0
        xy = samples[:, :, :2].cumsum(dim=1)
        if samples.size(-1) >= 3:
            yaw = samples[:, :, 2].cumsum(dim=1).clamp(min=-math.pi, max=math.pi)
        else:
            yaw = torch.zeros(samples.size(0), samples.size(1))
        for idx in range(samples.size(0)):
            points = torch.cat([xy[idx], yaw[idx].unsqueeze(-1)], dim=-1).tolist()
            out[idx] = torch.tensor(self._trajectory_points_to_viva_features(points, is_stop=False), dtype=torch.float32)
        return out

    def _viva_online_candidate_tensors(
        self,
        local_actions: list,
        action_seq: list,
        nextdit_candidate_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_candidates = max(1, int(self.viva_online_max_candidates))
        ids = torch.zeros(max_candidates, dtype=torch.long)
        values = torch.zeros(max_candidates, dtype=torch.float32)
        traj_features = torch.zeros(max_candidates, 8, dtype=torch.float32)
        ids[0] = int(VIVA_CANDIDATE_TYPE_IDS["stop_trajectory"])
        traj_features[0] = torch.tensor(self._actions_to_viva_traj_features([action_code.STOP]), dtype=torch.float32)
        if nextdit_candidate_features is not None:
            feats = nextdit_candidate_features.detach().float().cpu()
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            count = min(max_candidates - 1, feats.size(0))
            if count > 0:
                ids[1 : 1 + count] = int(VIVA_CANDIDATE_TYPE_IDS["diffusion_trajectory"])
                traj_features[1 : 1 + count] = feats[:count]
        return ids, values, traj_features

    def _build_viva_online_generation_kwargs(
        self,
        action_block: list,
        local_actions: list,
        action_seq: list,
        nextdit_candidate_features: Optional[torch.Tensor] = None,
        *,
        scene_id,
        episode_id,
        step_id,
    ) -> dict:
        if not self.enable_viva_online_structured_inputs or not bool(
            getattr(self.model_args, "enable_viva_progress_conditioner", False)
        ):
            return {}
        action_ids, action_deltas = self._viva_online_action_tensors(action_block)
        candidate_type_ids, candidate_values, candidate_traj_features = self._viva_online_candidate_tensors(
            local_actions,
            action_seq,
            nextdit_candidate_features,
        )
        kwargs = {
            "progress_action_ids": action_ids.unsqueeze(0).to(self.model.device),
            "progress_action_deltas": action_deltas.unsqueeze(0).to(self.model.device),
            "progress_candidate_type_ids": candidate_type_ids.unsqueeze(0).to(self.model.device),
            "progress_candidate_values": candidate_values.unsqueeze(0).to(self.model.device),
            "progress_candidate_traj_features": candidate_traj_features.unsqueeze(0).to(self.model.device),
        }
        if self.log_viva_online_inputs:
            os.makedirs(self.output_path, exist_ok=True)
            row = {
                "rank": int(self.rank),
                "scene_id": scene_id,
                "episode_id": int(episode_id),
                "step_id": int(step_id),
                "action_ids": action_ids.tolist(),
                "action_deltas": [float(x) for x in action_deltas.tolist()],
                "candidate_type_ids": candidate_type_ids.tolist(),
                "candidate_values": [float(x) for x in candidate_values.tolist()],
                "candidate_traj_features": candidate_traj_features.tolist(),
                "action_block_size": len(action_block or []),
                "nextdit_candidate_count": int(candidate_type_ids.eq(VIVA_CANDIDATE_TYPE_IDS["diffusion_trajectory"]).sum().item()),
            }
            with open(self._viva_online_input_log_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return kwargs

    def _verify_stop_with_pes(
        self,
        inputs,
        scene_id=None,
        episode_id=None,
        step_id=None,
        current_metrics=None,
        initial_distance_to_goal=None,
        llm_outputs=None,
    ) -> tuple[bool, str]:
        if not hasattr(self.model, "pes_head"):
            answer = "pes_head_missing"
            self._log_pes_stop_verify(
                scene_id=scene_id,
                episode_id=episode_id,
                step_id=step_id,
                accept=False,
                answer=answer,
                current_metrics=current_metrics,
                initial_distance_to_goal=initial_distance_to_goal,
                llm_outputs=llm_outputs,
                error=answer,
            )
            return False, answer

        try:
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, use_cache=False)
        except Exception as exc:
            answer = f"pes_forward_failed={exc!r}"
            self._log_pes_stop_verify(
                scene_id=scene_id,
                episode_id=episode_id,
                step_id=step_id,
                accept=False,
                answer=answer,
                current_metrics=current_metrics,
                initial_distance_to_goal=initial_distance_to_goal,
                llm_outputs=llm_outputs,
                error=answer,
            )
            return False, answer

        stop_logits = getattr(outputs, "stop_logits", None)
        progress_scores = getattr(outputs, "progress_scores", None)
        event_logits = getattr(outputs, "event_logits", None)
        if progress_scores is None or event_logits is None:
            answer = "pes_outputs_missing"
            self._log_pes_stop_verify(
                scene_id=scene_id,
                episode_id=episode_id,
                step_id=step_id,
                accept=False,
                answer=answer,
                current_metrics=current_metrics,
                initial_distance_to_goal=initial_distance_to_goal,
                llm_outputs=llm_outputs,
                error=answer,
            )
            return False, answer

        stop_logit = float(stop_logits.detach().float().flatten()[0].cpu()) if stop_logits is not None else None
        progress = float(progress_scores.detach().float().flatten()[0].cpu())
        event_probs = torch.softmax(event_logits.detach().float(), dim=-1).flatten().cpu().tolist()
        event4 = float(event_probs[4]) if len(event_probs) > 4 else 0.0
        accept = event4 >= self.pes_stop_event_threshold and progress >= self.pes_stop_progress_threshold
        answer = (
            f"event4={event4:.4f}>={self.pes_stop_event_threshold:.4f} "
            f"progress={progress:.4f}>={self.pes_stop_progress_threshold:.4f} "
            f"event_pred={int(np.argmax(event_probs)) if event_probs else -1}"
        )
        self._log_pes_stop_verify(
            scene_id=scene_id,
            episode_id=episode_id,
            step_id=step_id,
            accept=accept,
            answer=answer,
            stop_logit=stop_logit,
            progress=progress,
            event_probs=event_probs,
            current_metrics=current_metrics,
            initial_distance_to_goal=initial_distance_to_goal,
            llm_outputs=llm_outputs,
        )
        return accept, answer

    def _verify_stop_with_qwen(self, instruction: str, image: Image.Image) -> tuple[bool, str]:
        prompt = self.stop_verify_prompt.replace("<instruction>", instruction)
        choices = [" YES", " NO"]
        scores = self._score_vlm_choices(prompt, [image], choices)
        best_idx = int(np.argmax(scores))
        answer = choices[best_idx].strip()
        return best_idx == 0, f"{answer} scores={scores}"

    def _longclip_project(self, feats: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(feats, weight, bias)

    def _longclip_score_head(self, stats: torch.Tensor) -> torch.Tensor:
        x = stats
        score_head_state = self.longclip_head["score_head_state"]
        linear_ids = sorted(
            {
                int(key.split(".")[1])
                for key in score_head_state
                if key.endswith(".weight") and key.startswith("score_head.")
            }
        )
        last_linear_id = linear_ids[-1]
        for layer_id in linear_ids:
            x = F.linear(
                x,
                score_head_state[f"score_head.{layer_id}.weight"],
                score_head_state[f"score_head.{layer_id}.bias"],
            )
            if layer_id != last_linear_id:
                x = F.gelu(x)
        return x.squeeze(-1)

    def _verify_stop_with_longclip(self, instruction: str, images) -> tuple[bool, str]:
        if isinstance(images, Image.Image):
            images = [images]
        images = list(images)
        if not images:
            return False, "no_image"

        history_frames = int(
            (self.longclip_head_meta["history_frames"] if hasattr(self, "longclip_head_meta") else self.longclip_head.get("history_frames", 1))
            if self.longclip_mode == "elements"
            else 1
        )
        images = images[-history_frames:]
        image_tensor = torch.stack([self.longclip_preprocess(image) for image in images], dim=0).to(self.device)
        with torch.no_grad():
            image_features = self.longclip_model.encode_image(image_tensor).float()
            if self.longclip_mode == "elements":
                elements = extract_stop_target_elements(
                    instruction,
                    max_elements=(
                        self.longclip_head_meta["max_target_elements"]
                        if hasattr(self, "longclip_head_meta")
                        else self.longclip_head["max_target_elements"]
                    ),
                )
                if not elements:
                    stop_object_phrase = extract_stop_object_phrase(instruction)
                    stop_phrase = extract_stop_phrase(instruction)
                    text = stop_object_phrase if is_usable_stop_object_phrase(stop_object_phrase) else stop_phrase
                    elements = [text]
                if self.longclip_head_type in {"grounding_v4", "cross_attn"}:
                    frame_cls, frame_patch = encode_image_with_patch_tokens(self.longclip_model, image_tensor)
                    text_tokens = longclip.tokenize(elements, truncate=True).to(self.device)
                    stop_phrase = extract_stop_phrase(instruction)
                    relation_text = stop_phrase or instruction
                    stop_object_phrase = extract_stop_object_phrase(instruction)
                    action_text = stop_phrase or stop_object_phrase or instruction
                    text_features = self.longclip_model.encode_text(text_tokens).float()
                    action_features = self.longclip_model.encode_text(
                        longclip.tokenize([action_text], truncate=True).to(self.device)
                    ).float()
                    relation_features = self.longclip_model.encode_text(
                        longclip.tokenize([relation_text], truncate=True).to(self.device)
                    ).float()
                    logits, raw_scores, uncertainties, embeddings = forward_stop_head(
                        self.longclip_head,
                        frame_cls.float(),
                        frame_patch.float() if self.longclip_head_type == "grounding_v4" else None,
                        [0] * len(images),
                        text_features,
                        [0] * len(elements),
                        action_features,
                        relation_features,
                        1,
                        self.longclip_head_meta["max_target_elements"],
                        self.longclip_head_meta["obj_threshold"],
                        self.longclip_head_meta["coverage_temperature"],
                    )
                    score = float(torch.sigmoid(logits)[0].item())
                    answer = (
                        f"score={score:.4f} threshold={self.longclip_stop_threshold:.4f} "
                        f"elements={elements!r} action={action_text!r}"
                    )
                else:
                    text_tokens = longclip.tokenize(elements, truncate=True).to(self.device)
                    text_features = self.longclip_model.encode_text(text_tokens).float()
                    image_proj = self._longclip_project(
                        image_features[-1:],
                        self.longclip_head["image_proj.weight"],
                        self.longclip_head["image_proj.bias"],
                    )
                    text_proj = self._longclip_project(
                        text_features,
                        self.longclip_head["text_proj.weight"],
                        self.longclip_head["text_proj.bias"],
                    )
                    image_proj = F.normalize(image_proj, dim=-1)
                    text_proj = F.normalize(text_proj, dim=-1)
                    sims = (image_proj * text_proj).sum(dim=-1)
                    soft_coverage = torch.sigmoid(
                        (sims - self.longclip_head["obj_threshold"]) * self.longclip_head["coverage_temperature"]
                    ).mean()
                    count_ratio = torch.tensor(
                        len(elements) / max(self.longclip_head["max_target_elements"], 1),
                        device=self.device,
                        dtype=sims.dtype,
                    )
                    stats = torch.stack([sims.mean(), sims.max(), soft_coverage, count_ratio], dim=0).unsqueeze(0)
                    score = float(self._longclip_score_head(stats).item())
                    answer = (
                        f"score={score:.4f} threshold={self.longclip_stop_threshold:.4f} "
                        f"elements={elements!r} sims={[round(float(x), 4) for x in sims.tolist()]}"
                    )
            else:
                stop_object_phrase = extract_stop_object_phrase(instruction)
                stop_phrase = extract_stop_phrase(instruction)
                text = stop_object_phrase if is_usable_stop_object_phrase(stop_object_phrase) else stop_phrase
                text_tokens = longclip.tokenize([text], truncate=True).to(self.device)
                text_features = self.longclip_model.encode_text(text_tokens).float()
                image_features = self._longclip_project(
                    image_features,
                    self.longclip_projector["image_proj.weight"],
                    self.longclip_projector["image_proj.bias"],
                )
                text_features = self._longclip_project(
                    text_features,
                    self.longclip_projector["text_proj.weight"],
                    self.longclip_projector["text_proj.bias"],
                )
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                score = float((image_features[-1:] * text_features).sum(dim=-1).item())
                answer = f"score={score:.4f} threshold={self.longclip_stop_threshold:.4f} text={text!r}"
        return score >= self.longclip_stop_threshold, answer

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)
            initial_metrics = self.env.get_metrics()
            initial_distance_to_goal = initial_metrics.get("distance_to_goal", None)
            if initial_distance_to_goal is not None:
                initial_distance_to_goal = float(initial_distance_to_goal)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            force_fresh_lookdown_prompt = False
            force_pixel_only_once = False
            block_action_branch_once = False
            local_actions = []
            progress_action_block = []
            latest_nextdit_candidate_features = None
            self._stop_residual_progress_history = []
            viva_episode_frames = []
            viva_stop_interventions = []

            done = False
            flag = False
            pixel_goal = None
            last_valid_pixel_goal = None

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                viva_stop_event = None
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()
                self._viva_capture_frame(
                    episode_frames=viva_episode_frames,
                    observations=observations,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    policy_step_id=step_id,
                    instruction=episode_instruction,
                    llm_outputs=llm_outputs,
                    initial_distance_to_goal=initial_distance_to_goal,
                )

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)

                if len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        if force_fresh_lookdown_prompt:
                            sources = [{"from": "human", "value": self.pixel_recover_prompt}]
                            sources[0]["value"] = sources[0]["value"].replace(
                                "<instruction>", episode.instruction.instruction_text
                            )
                            input_images = [look_down_image]
                            messages = []
                            input_img_id = 0
                            force_fresh_lookdown_prompt = False
                            force_pixel_only_once = True
                            block_action_branch_once = True
                        else:
                            # last action is look down
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            input_images += [look_down_image]
                            messages.append(
                                {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                            )
                            input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)
                    current_metrics = self.env.get_metrics()
                    self._append_stop_residual_progress_history(
                        step_id=step_id,
                        current_metrics=current_metrics,
                        initial_distance_to_goal=initial_distance_to_goal,
                    )
                    viva_generation_kwargs = self._build_viva_online_generation_kwargs(
                        progress_action_block,
                        local_actions,
                        action_seq,
                        latest_nextdit_candidate_features,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        step_id=step_id,
                    )
                    self._log_pes_outputs(
                        inputs,
                        scene_id,
                        episode_id,
                        step_id,
                        current_metrics=current_metrics,
                        initial_distance_to_goal=initial_distance_to_goal,
                    )

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                            **viva_generation_kwargs,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    if force_pixel_only_once and not bool(re.search(r'\d', llm_outputs)):
                        retry_prompt = self.pixel_recover_prompt.replace("<instruction>", episode.instruction.instruction_text)
                        llm_outputs = self._run_vlm_text(retry_prompt, [look_down_image], max_new_tokens=16)
                        force_pixel_only_once = False
                        print('step_id:', step_id, 'pixel_recover_retry output text:', llm_outputs)
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        if self.disable_pixel_goal:
                            action_seq = [action_code.FORWARD]
                            print('step_id:', step_id, 'disable_pixel_goal -> fallback actions', action_seq)
                            llm_outputs = "↑"
                            print('actions', action_seq, flush=True)
                            pixel_goal = None
                            draw_pixel_goal = False
                            output_ids = None
                        else:
                            forward_action = 0
                            pixel_goal, reject_reason = self._parse_pixel_goal(llm_outputs, last_valid_pixel_goal)
                            if pixel_goal is None:
                                if block_action_branch_once and last_valid_pixel_goal is not None:
                                    pixel_goal = list(last_valid_pixel_goal)
                                    draw_pixel_goal = True
                                    print(
                                        'step_id:',
                                        step_id,
                                        'pixel_goal_reject -> reuse_last_valid_pixel_goal',
                                        pixel_goal,
                                        reject_reason,
                                    )
                                else:
                                    action_seq = [action_code.LOOKDOWN] if block_action_branch_once else [action_code.FORWARD]
                                    print('step_id:', step_id, 'pixel_goal_reject -> fallback actions', action_seq, reject_reason)
                                    llm_outputs = "↓" if action_seq[0] == action_code.LOOKDOWN else "↑"
                                    print('actions', action_seq, flush=True)
                                    draw_pixel_goal = False
                                    output_ids = None
                            else:
                                draw_pixel_goal = True
                                last_valid_pixel_goal = list(pixel_goal)
                            block_action_branch_once = False

                            if pixel_goal is None:
                                pass
                            else:
                                # look down --> horizontal
                                self.env.step(action_code.LOOKUP)
                                self.env.step(action_code.LOOKUP)

                                local_actions = []
                                pixel_values = inputs.pixel_values
                                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                                with torch.no_grad():
                                    traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                                # prepocess align with navdp
                                image_dp = (
                                    torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                )
                                pix_goal_image = copy.copy(image_dp)
                                images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                                pix_goal_depth = copy.copy(depth_dp)
                                depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                                latest_nextdit_candidate_features = self._nextdit_raw_to_viva_candidate_features(dp_actions)

                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < MAX_STEPS:
                                    action_list += [0] * (MAX_STEPS - len(action_list))

                                local_actions = action_list
                                if len(local_actions) >= MAX_LOCAL_STEPS:
                                    local_actions = local_actions[:MAX_LOCAL_STEPS]

                                action = local_actions[0]
                                if action == action_code.STOP:
                                    pixel_goal = None
                                    output_ids = None
                                    action = action_code.LEFT
                                    before_distance = self._metric_distance_to_goal()
                                    observations, _, done, _ = self.env.step(action)
                                    after_distance = self._metric_distance_to_goal()
                                    self._viva_update_last_frame(
                                        episode_frames=viva_episode_frames,
                                        action=action,
                                        before_distance=before_distance,
                                        after_distance=after_distance,
                                        done=done,
                                        llm_outputs=llm_outputs,
                                        stop_event=None,
                                    )
                                    self._append_viva_online_action(
                                        progress_action_block,
                                        action,
                                        before_distance,
                                        after_distance,
                                        "local_stop_replaced_left",
                                    )
                                    step_id += 1
                                    messages = []
                                    continue
                                print('predicted goal', pixel_goal, flush=True)

                    else:
                        if block_action_branch_once:
                            if last_valid_pixel_goal is not None:
                                pixel_goal = list(last_valid_pixel_goal)
                                draw_pixel_goal = True
                                block_action_branch_once = False
                                print(
                                    'step_id:',
                                    step_id,
                                    'recover_non_digit -> reuse_last_valid_pixel_goal',
                                    pixel_goal,
                                    flush=True,
                                )
                                # look down --> horizontal
                                self.env.step(action_code.LOOKUP)
                                self.env.step(action_code.LOOKUP)

                                local_actions = []
                                pixel_values = inputs.pixel_values
                                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                                with torch.no_grad():
                                    traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                                image_dp = (
                                    torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                                )
                                pix_goal_image = copy.copy(image_dp)
                                images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                                depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                                pix_goal_depth = copy.copy(depth_dp)
                                depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                                latest_nextdit_candidate_features = self._nextdit_raw_to_viva_candidate_features(dp_actions)

                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < MAX_STEPS:
                                    action_list += [0] * (MAX_STEPS - len(action_list))

                                local_actions = action_list
                                if len(local_actions) >= MAX_LOCAL_STEPS:
                                    local_actions = local_actions[:MAX_LOCAL_STEPS]

                                action = local_actions[0]
                                if action == action_code.STOP:
                                    pixel_goal = None
                                    output_ids = None
                                    action = action_code.LEFT
                                    before_distance = self._metric_distance_to_goal()
                                    observations, _, done, _ = self.env.step(action)
                                    after_distance = self._metric_distance_to_goal()
                                    self._viva_update_last_frame(
                                        episode_frames=viva_episode_frames,
                                        action=action,
                                        before_distance=before_distance,
                                        after_distance=after_distance,
                                        done=done,
                                        llm_outputs=llm_outputs,
                                        stop_event=None,
                                    )
                                    self._append_viva_online_action(
                                        progress_action_block,
                                        action,
                                        before_distance,
                                        after_distance,
                                        "local_stop_replaced_left",
                                    )
                                    step_id += 1
                                    messages = []
                                    continue
                                print('predicted goal', pixel_goal, flush=True)
                            else:
                                action_seq = [action_code.LOOKDOWN]
                                block_action_branch_once = False
                                print('step_id:', step_id, 'recover_non_digit -> force lookdown retry', flush=True)
                                print('actions', action_seq, flush=True)
                                continue
                        action_seq = self.parse_actions(llm_outputs)
                        if len(action_seq) != 0 and action_seq[0] == action_code.STOP and (
                            self.enable_stop_residual_verify
                            or self.enable_pes_stop_verify
                            or self.enable_qwen_stop_verify
                            or self.enable_longclip_stop_verify
                        ):
                            stop_verify_images = rgb_list[-max(int(getattr(self, "num_history", 1)), 1):] or [save_raw_image]
                            stop_accept = True
                            stop_answer = ""
                            stop_tag = "stop_verify"
                            if self.enable_stop_residual_verify:
                                stop_accept, stop_answer = self._verify_stop_with_residual_adapter(
                                    inputs,
                                    scene_id=scene_id,
                                    episode_id=episode_id,
                                    step_id=step_id,
                                    current_metrics=current_metrics,
                                    initial_distance_to_goal=initial_distance_to_goal,
                                    llm_outputs=llm_outputs,
                                    forward_kwargs=viva_generation_kwargs,
                                )
                                stop_tag = "stop_residual_verify"
                            if stop_accept and self.enable_pes_stop_verify:
                                stop_accept, stop_answer = self._verify_stop_with_pes(
                                    inputs,
                                    scene_id=scene_id,
                                    episode_id=episode_id,
                                    step_id=step_id,
                                    current_metrics=current_metrics,
                                    initial_distance_to_goal=initial_distance_to_goal,
                                    llm_outputs=llm_outputs,
                                )
                                stop_tag = "pes_stop_verify"
                            if stop_accept and self.enable_longclip_stop_verify:
                                stop_accept, stop_answer = self._verify_stop_with_longclip(
                                    episode_instruction,
                                    stop_verify_images,
                                )
                                stop_tag = "longclip_stop_verify"
                            elif stop_accept and self.enable_qwen_stop_verify:
                                stop_accept, stop_answer = self._verify_stop_with_qwen(
                                    episode_instruction,
                                    save_raw_image,
                                )
                                stop_tag = "stop_verify"
                            print(
                                f"{stop_tag} step={step_id} answer={stop_answer!r} accept={stop_accept}",
                                flush=True,
                            )
                            if not stop_accept:
                                action_seq = [self._stop_reject_action_code()]
                                viva_stop_event = {
                                    "rank": int(self.rank),
                                    "scene_id": scene_id,
                                    "episode_id": int(episode_id),
                                    "frame_idx": int(viva_episode_frames[-1]["frame_idx"]) if viva_episode_frames else None,
                                    "policy_step_id": int(step_id),
                                    "llm_outputs": llm_outputs,
                                    "stop_verify_tag": stop_tag,
                                    "stop_verify_answer": stop_answer,
                                    "stop_accept": False,
                                    "stop_reject_action": int(action_seq[0]),
                                    "stop_reject_action_name": ACTION_NAMES.get(int(action_seq[0]), str(int(action_seq[0]))),
                                    "distance_to_goal": float(current_metrics["distance_to_goal"])
                                    if current_metrics and "distance_to_goal" in current_metrics
                                    else None,
                                }
                                viva_stop_interventions.append(viva_stop_event)
                                self._viva_log_stop_intervention(viva_stop_event)
                                llm_outputs = "↓" if action_seq[0] == action_code.LOOKDOWN else llm_outputs
                                pixel_goal = None
                                output_ids = None
                                input_images = []
                                messages = []
                                force_fresh_lookdown_prompt = action_seq[0] == action_code.LOOKDOWN
                                force_pixel_only_once = force_fresh_lookdown_prompt
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # navdp
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255

                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                        latest_nextdit_candidate_features = self._nextdit_raw_to_viva_candidate_features(dp_actions)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if self.save_video:
                    if info['top_down_map'] is not None and not self.save_front_video_only:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        frame = np.asarray(save_raw_image).copy()
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    if info['top_down_map'] is not None and not self.save_front_video_only:
                        vis = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if pixel_goal is not None and draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                before_distance = self._metric_distance_to_goal()
                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False
                after_distance = self._metric_distance_to_goal()
                self._viva_update_last_frame(
                    episode_frames=viva_episode_frames,
                    action=action,
                    before_distance=before_distance,
                    after_distance=after_distance,
                    done=done,
                    llm_outputs=llm_outputs,
                    stop_event=viva_stop_event,
                )
                self._append_viva_online_action(
                    progress_action_block,
                    action,
                    before_distance,
                    after_distance,
                    "policy_execute",
                )

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics.get('oracle_success', 0.0))  # Use 0.0 if not available
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics.get('oracle_success', 0.0)}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics.get('oracle_success', 0.0),
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            self._viva_write_episode_trace(
                episode=episode,
                scene_id=scene_id,
                episode_id=episode_id,
                instruction=episode_instruction,
                initial_distance_to_goal=initial_distance_to_goal,
                frames=viva_episode_frames,
                interventions=viva_stop_interventions,
                metrics=metrics,
            )

            # save video for both successful and failed episodes in separate directories
            if self.save_video:
                video_bucket = 'success' if metrics['success'] == 1.0 else 'failed'
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', video_bucket, f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []
            last_valid_pixel_goal = None

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        if self.disable_pixel_goal:
                            action_seq = [action_code.FORWARD]
                            print('step_id:', step_id, 'disable_pixel_goal -> fallback actions', action_seq)
                            llm_outputs = "↑"
                            print('actions', action_seq, flush=True)
                            goal = None
                            draw_pixel_goal = False
                            output_ids = None
                        else:
                            forward_action = 0
                            pixel_goal, reject_reason = self._parse_pixel_goal(llm_outputs, last_valid_pixel_goal)
                            if pixel_goal is None:
                                action_seq = [action_code.FORWARD]
                                print('step_id:', step_id, 'pixel_goal_reject -> fallback actions', action_seq, reject_reason)
                                llm_outputs = "↑"
                                print('actions', action_seq, flush=True)
                                goal = None
                                draw_pixel_goal = False
                                output_ids = None
                            else:
                                draw_pixel_goal = True
                                last_valid_pixel_goal = list(pixel_goal)

                            if pixel_goal is None:
                                pass
                            else:
                                # look down --> horizontal
                                self.env.step(action_code.LOOKUP)
                                self.env.step(action_code.LOOKUP)

                                goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                                goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                                if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                                    goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                                action = agent.get_next_action(goal)
                                if action == action_code.STOP:
                                    goal = None
                                    output_ids = None
                                    action = action_code.LEFT  # random action to avoid deadlock
                                    observations, _, done, _ = self.env.step(action)
                                    step_id += 1
                                    messages = []
                                    continue
                                print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if self.save_video:
                    if info['top_down_map'] is not None and not self.save_front_video_only:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        frame = np.asarray(save_raw_image).copy()
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    if info['top_down_map'] is not None and not self.save_front_video_only:
                        vis = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    else:
                        vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics.get('oracle_success', 0.0))  # Use 0.0 if not available
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics.get('oracle_success', 0.0)}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics.get('oracle_success', 0.0),
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            # save video for both successful and failed episodes in separate directories
            if self.save_video:
                video_bucket = 'success' if metrics['success'] == 1.0 else 'failed'
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', video_bucket, f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
