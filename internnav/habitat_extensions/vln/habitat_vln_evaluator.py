import argparse
import json
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
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
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
from internnav.model.basemodel.LongCLIP.model import longclip
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from scripts.data_collect.stop_alignment_utils import (
    extract_stop_object_phrase,
    extract_stop_phrase,
    is_usable_stop_object_phrase,
)

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln')
class HabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
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

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:{self.local_rank}")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Changed from flash_attention_2
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # Changed from flash_attention_2
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor
        self.longclip_model = None
        self.longclip_preprocess = None
        self.longclip_projector = None

        if self.enable_longclip_stop_verify:
            self.longclip_model, self.longclip_preprocess = longclip.load(self.longclip_stop_model_path, device=device)
            ckpt = torch.load(self.longclip_stop_weight_path, map_location="cpu")
            proj_dim = int(ckpt["proj_dim"])
            feature_dim = int(self.longclip_model.text_projection.shape[-1])
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

    def _verify_stop_with_qwen(self, instruction: str, image: Image.Image) -> tuple[bool, str]:
        prompt = self.stop_verify_prompt.replace("<instruction>", instruction)
        choices = [" YES", " NO"]
        scores = self._score_vlm_choices(prompt, [image], choices)
        best_idx = int(np.argmax(scores))
        answer = choices[best_idx].strip()
        return best_idx == 0, f"{answer} scores={scores}"

    def _longclip_project(self, feats: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(feats, weight, bias)

    def _verify_stop_with_longclip(self, instruction: str, image: Image.Image) -> tuple[bool, str]:
        stop_object_phrase = extract_stop_object_phrase(instruction)
        stop_phrase = extract_stop_phrase(instruction)
        text = stop_object_phrase if is_usable_stop_object_phrase(stop_object_phrase) else stop_phrase
        image_tensor = self.longclip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = longclip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            image_features = self.longclip_model.encode_image(image_tensor).float()
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
            score = float((image_features * text_features).sum(dim=-1).item())
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

            done = False
            flag = False
            pixel_goal = None
            last_valid_pixel_goal = None

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

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
                                    observations, _, done, _ = self.env.step(action)
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
                                    observations, _, done, _ = self.env.step(action)
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
                            self.enable_qwen_stop_verify or self.enable_longclip_stop_verify
                        ):
                            if self.enable_longclip_stop_verify:
                                stop_accept, stop_answer = self._verify_stop_with_longclip(
                                    episode_instruction,
                                    save_raw_image,
                                )
                                stop_tag = "longclip_stop_verify"
                            else:
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

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    if info['top_down_map'] is not None:
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

            # save video
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
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

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    if info['top_down_map'] is not None:
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
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
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
