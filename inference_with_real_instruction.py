#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实导航指令进行推理
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path
import json

print("=" * 70)
print("使用真实导航指令进行推理")
print("=" * 70)
print()

# 配置
model_path = "checkpoints/InternVLA-N1-DualVLN"
trajectory_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample/office_3/trajectory_68"

# 1. 读取真实导航指令
print("📖 读取真实导航指令...")
with open(f"{trajectory_path}/meta/episodes.jsonl", 'r') as f:
    episode_data = json.loads(f.readline())

# 解析任务
task_data = json.loads(episode_data['tasks'][0])
real_instruction = task_data['sub_instruction']
revised_instruction = task_data['revised_sub_instruction']

print(f"原始指令:\n  {real_instruction}")
print(f"\n改写指令:\n  {revised_instruction}")
print()

# 2. 加载模型
print("🚀 加载模型...")
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
processor = AutoProcessor.from_pretrained(model_path)
processor.tokenizer = tokenizer
processor.tokenizer.padding_side = 'left'

model = InternVLAN1ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
model.eval()
print(f"✅ 模型加载成功")
print()

# 3. 读取图像
print("📂 读取图像...")
image_dir = Path(trajectory_path) / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"找到 {len(image_files)} 帧")
print()

# 4. 在关键帧上进行推理
print("🎯 在关键帧上进行推理...")
print("=" * 70)

DEFAULT_IMAGE_TOKEN = "<image>"

# 选择开始、中间、结束的几帧
test_frames = [0, 30, 60, 90, 120, 150, 181]

for idx in test_frames:
    if idx >= len(image_files):
        idx = len(image_files) - 1

    print(f"\n--- 帧 {idx}/{len(image_files)-1} ---")

    img_path = image_files[idx]
    image = Image.open(img_path).convert('RGB')

    # 使用真实指令
    prompt = f"You are an autonomous navigation assistant. Your task is: {real_instruction}. Where should you go next to follow this instruction?"
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt.replace(DEFAULT_IMAGE_TOKEN, "")},
                {'type': 'image', 'image': image}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    generated_text = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"预测: {generated_text}")

print()
print("=" * 70)
print("✅ 推理完成")
print()
print("对比说明:")
print("  - 使用通用指令时: 决策一直是 ←←←← (向左)")
print("  - 使用真实指令后: 应该看到决策随着任务进度变化")
print()
