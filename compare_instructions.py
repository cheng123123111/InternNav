#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比通用指令 vs 真实指令的推理结果
"""

import os, sys, torch, json
from PIL import Image
from pathlib import Path

print("=" * 70)
print("对比推理: 通用指令 vs 真实指令")
print("=" * 70)
print()

# 加载模型
print("加载模型...")
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from transformers import AutoTokenizer, AutoProcessor

model_path = "checkpoints/InternVLA-N1-DualVLN"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer = tokenizer
processor.tokenizer.padding_side = 'left'

model = InternVLAN1ForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="auto")
model.eval()
print("✅ 完成\n")

# 读取真实指令
trajectory_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample/office_3/trajectory_68"
with open(f"{trajectory_path}/meta/episodes.jsonl") as f:
    episode_data = json.loads(f.readline())
task_data = json.loads(episode_data['tasks'][0])
real_instruction = task_data['sub_instruction']

print("真实指令:")
print(f"  '{real_instruction}'\n")

# 读取图像
image_dir = Path(trajectory_path) / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))

# 测试3帧
DEFAULT_IMAGE_TOKEN = "<image>"
test_frames = [0, 90, 181]

for idx in test_frames:
    if idx >= len(image_files):
        idx = len(image_files) - 1

    print(f"\n{'='*70}")
    print(f"帧 {idx}/{len(image_files)-1}")
    print(f"{'='*70}")

    image = Image.open(image_files[idx]).convert('RGB')

    # 1. 通用指令
    generic_prompt = "You are an autonomous navigation assistant. Navigate through the office. Where should you go next?"
    generic_prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': generic_prompt.replace(DEFAULT_IMAGE_TOKEN, "")},
            {'type': 'image', 'image': image}
        ]
    }]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                use_cache=True, return_dict_in_generate=True)

    generic_result = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # 2. 真实指令
    real_prompt = f"You are an autonomous navigation assistant. Your task: {real_instruction}. Where should you go next?"
    real_prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': real_prompt.replace(DEFAULT_IMAGE_TOKEN, "")},
            {'type': 'image', 'image': image}
        ]
    }]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                use_cache=True, return_dict_in_generate=True)

    real_result = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # 显示对比
    print(f"\n通用指令预测: {generic_result}")
    print(f"真实指令预测: {real_result}")

    if generic_result != real_result:
        print("  ⚠️  预测不同！")
    else:
        print("  ℹ️  预测相同")

print(f"\n{'='*70}")
print("✅ 对比完成")
print(f"{'='*70}\n")
