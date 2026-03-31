#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速推理测试 - 对几帧进行推理
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path

print("=" * 70)
print("InternVLA-N1 快速推理测试")
print("=" * 70)
print()

# 配置
model_path = "checkpoints/InternVLA-N1-DualVLN"
trajectory_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample/office_3/trajectory_68"
navigation_instruction = "Navigate through the office and reach the desk area."

# 1. 加载模型
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

# 2. 读取几帧图像进行测试
print("📂 读取测试图像...")
image_dir = Path(trajectory_path) / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"找到 {len(image_files)} 帧，选择前3帧进行测试")
print()

# 3. 推理测试
DEFAULT_IMAGE_TOKEN = "<image>"

for idx in [0, 50, 100]:
    if idx >= len(image_files):
        continue

    print(f"--- 帧 {idx} ---")
    img_path = image_files[idx]
    image = Image.open(img_path).convert('RGB')
    print(f"图像大小: {image.size}")

    # 构建prompt
    prompt = f"You are an autonomous navigation assistant. Your task is to {navigation_instruction}. Where should you go next?"
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    # 准备输入
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

    # 推理
    print("推理中...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    # 解码
    generated_text = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"预测: {generated_text}")
    print()

print("✅ 测试完成")
