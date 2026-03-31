#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实导航指令生成推理视频
"""

import os, sys, torch, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("使用真实指令生成推理视频")
print("=" * 70)
print()

model_path = "checkpoints/InternVLA-N1-DualVLN"
trajectory_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample/office_3/trajectory_68"
output_video_path = "inference_real_instruction_video.mp4"

# 1. 读取真实指令
print("📖 读取真实导航指令...")
with open(f"{trajectory_path}/meta/episodes.jsonl") as f:
    episode_data = json.loads(f.readline())
task_data = json.loads(episode_data['tasks'][0])
real_instruction = task_data['sub_instruction']

print(f"指令: {real_instruction}")
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
    model_path, torch_dtype=torch.bfloat16,
    attn_implementation="eager", device_map="auto")
model.eval()
print("✅ 完成")
print()

# 3. 读取图像
print("📂 读取图像...")
image_dir = Path(trajectory_path) / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"  {len(image_files)} 帧")
print()

# 4. 推理
print("🎯 推理...")
DEFAULT_IMAGE_TOKEN = "<image>"
results = {}

sample_interval = 10
sampled_indices = list(range(0, len(image_files), sample_interval))
sampled_indices.append(len(image_files) - 1)

for idx in tqdm(sampled_indices, desc="推理进度"):
    image = Image.open(image_files[idx]).convert('RGB')

    prompt = f"You are an autonomous navigation assistant. Your task: {real_instruction}. Where should you go next?"
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."

    conversation = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt.replace(DEFAULT_IMAGE_TOKEN, "")},
            {'type': 'image', 'image': image}
        ]
    }]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            use_cache=True, return_dict_in_generate=True)

    generated_text = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True).strip()

    results[idx] = generated_text

print(f"\n✅ 推理完成")
print()

# 5. 生成视频
print("🎬 生成视频...")

first_img = Image.open(str(image_files[0]))
width, height = first_img.size
text_height = 120
new_height = height + text_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, new_height))

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

for idx in tqdm(range(len(image_files)), desc="生成视频"):
    img = Image.open(str(image_files[idx]))
    canvas = Image.new('RGB', (width, new_height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    # 找到最近的推理结果
    current_prediction = "Processing..."
    for sample_idx in reversed(sorted(results.keys())):
        if sample_idx <= idx:
            current_prediction = results[sample_idx]
            break

    draw = ImageDraw.Draw(canvas)

    # 帧信息
    draw.text((10, height + 10), f"Frame: {idx}/{len(image_files)}",
              fill=(0, 0, 0), font=font_small)

    # 指令（截断显示）
    instruction_display = real_instruction[:60] + "..." if len(real_instruction) > 60 else real_instruction
    draw.text((10, height + 35), f"Task: {instruction_display}",
              fill=(0, 100, 0), font=font_small)

    # 推理结果 - 更大更醒目
    draw.text((10, height + 65), f"Action: {current_prediction}",
              fill=(200, 0, 0), font=font)

    # 进度条
    progress = idx / len(image_files)
    bar_width = width - 20
    bar_height = 15
    draw.rectangle([(10, height + 95), (10 + bar_width, height + 95 + bar_height)],
                   outline=(100, 100, 100), width=1)
    draw.rectangle([(10, height + 95), (10 + int(bar_width * progress), height + 95 + bar_height)],
                   fill=(0, 150, 0))

    frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()

print(f"\n✅ 视频生成完成")
print(f"   输出: {output_video_path}")
print(f"   分辨率: {width}x{new_height}")
print(f"   时长: {len(image_files)/fps:.1f}秒")
print()

# 显示推理结果统计
print("📝 推理结果统计:")
unique_predictions = {}
for idx, pred in results.items():
    unique_predictions[pred] = unique_predictions.get(pred, 0) + 1

for pred, count in sorted(unique_predictions.items()):
    print(f"  {pred}: {count}次")
print()

print("📋 关键帧推理:")
for idx in [0, 60, 120, 181]:
    if idx in results:
        print(f"  帧{idx:3d}: {results[idx]}")
print()

print("=" * 70)
print("🎉 完成！")
print("=" * 70)
print()
print(f"现在可以看到模型的决策是如何随着任务进度变化的！")
print(f"  开始: 调整方向")
print(f"  中期: 直行/沿着长凳")
print(f"  结束: 转向面对白墙")
print()
