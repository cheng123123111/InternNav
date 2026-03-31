#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成推理视频 - 优化版本
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("InternVLA-N1 推理视频生成")
print("=" * 70)
print()

# 配置
model_path = "checkpoints/InternVLA-N1-DualVLN"
trajectory_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample/office_3/trajectory_68"
output_video_path = "inference_output_video.mp4"
navigation_instruction = "Navigate through the office"

# 1. 加载模型
print("🚀 加载InternVLA-N1模型...")
try:
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
    print(f"  ✅ 模型加载成功")
except Exception as e:
    print(f"  ❌ 模型加载失败: {e}")
    sys.exit(1)
print()

# 2. 读取图像序列
print("📂 读取轨迹图像...")
image_dir = Path(trajectory_path) / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"  找到 {len(image_files)} 帧图像")
print()

# 3. 对采样帧进行推理
print("🎯 开始推理...")
DEFAULT_IMAGE_TOKEN = "<image>"
results = {}

# 每10帧采样一次
sample_interval = 10
sampled_indices = list(range(0, len(image_files), sample_interval))
sampled_indices.append(len(image_files) - 1)  # 添加最后一帧

print(f"  采样 {len(sampled_indices)} 帧进行推理")

for idx in tqdm(sampled_indices, desc="推理进度"):
    img_path = image_files[idx]
    image = Image.open(img_path).convert('RGB')

    prompt = f"You are an autonomous navigation assistant. Your task is to {navigation_instruction}. Where should you go next?"
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

    results[idx] = generated_text

print(f"\n✅ 推理完成！")
print()

# 4. 生成视频
print("🎬 生成推理视频...")

# 获取图像尺寸
first_img = Image.open(str(image_files[0]))
width, height = first_img.size

# 创建视频
text_height = 100
new_height = height + text_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, new_height))

# 尝试加载字体（如果失败使用默认字体）
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

for idx in tqdm(range(len(image_files)), desc="生成视频"):
    # 读取图像
    img = Image.open(str(image_files[idx]))

    # 创建扩展画布
    canvas = Image.new('RGB', (width, new_height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    # 找到最近的推理结果
    current_prediction = "Processing..."
    for sample_idx in reversed(sorted(results.keys())):
        if sample_idx <= idx:
            current_prediction = results[sample_idx]
            break

    # 在画布上绘制文本
    draw = ImageDraw.Draw(canvas)

    # 帧信息
    draw.text((10, height + 10), f"Frame: {idx}/{len(image_files)}",
              fill=(0, 0, 0), font=font_small)

    # 指令
    draw.text((10, height + 35), f"Task: {navigation_instruction}",
              fill=(0, 100, 0), font=font_small)

    # 推理结果
    draw.text((10, height + 60), f"Prediction: {current_prediction}",
              fill=(200, 0, 0), font=font)

    # 转换为OpenCV格式并写入
    frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()

print(f"\n✅ 视频生成完成！")
print(f"   输出: {output_video_path}")
print(f"   分辨率: {width}x{new_height}")
print(f"   帧率: {fps} fps")
print(f"   时长: {len(image_files)/fps:.1f} 秒")
print()

# 5. 显示推理结果样例
print("📝 推理结果样例:")
for idx in sorted(results.keys())[:5]:
    print(f"  帧 {idx:3d}: {results[idx]}")
if len(results) > 5:
    print(f"  ... (共 {len(results)} 个结果)")
print()

# GPU显存
if torch.cuda.is_available():
    print("💾 GPU显存:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print()

print("=" * 70)
print("🎉 完成！")
print("=" * 70)
print()
print(f"视频已保存: {output_video_path}")
print("可以使用 vlc、mpv 或其他播放器查看")
print()
