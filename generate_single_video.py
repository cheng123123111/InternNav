#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为单个轨迹生成推理视频
"""

import os, sys, torch, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

if len(sys.argv) < 4:
    print("用法: python generate_single_video.py <scene> <trajectory> <scene_name>")
    print("例如: python generate_single_video.py apartment_2 trajectory_0 'Apartment'")
    sys.exit(1)

scene = sys.argv[1]
trajectory = sys.argv[2]
scene_name = sys.argv[3]

print("=" * 70)
print(f"生成推理视频: {scene_name}")
print("=" * 70)
print()

model_path = "checkpoints/InternVLA-N1-DualVLN"
base_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample"
trajectory_path = Path(base_path) / scene / trajectory
output_video = f"inference_{scene}_{trajectory}.mp4"

# 1. 加载模型
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

# 2. 读取指令
meta_file = trajectory_path / "meta/episodes.jsonl"
if meta_file.exists():
    with open(meta_file) as f:
        episode_data = json.loads(f.readline())
    task_data = json.loads(episode_data['tasks'][0])
    # 处理两种格式
    if 'sub_instruction' in task_data:
        instruction = task_data['sub_instruction']
    elif 'sum_instruction' in task_data:
        instruction = task_data['sum_instruction']
    else:
        instruction = "Navigate through the environment"
else:
    instruction = "Navigate through the environment"

print(f"📖 指令: {instruction}")
print()

# 3. 读取图像
image_dir = trajectory_path / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"📂 {len(image_files)} 帧图像")
print()

# 4. 推理
print("🎯 推理中...")
DEFAULT_IMAGE_TOKEN = "<image>"
results = {}

sample_interval = max(10, len(image_files) // 15)
sampled_indices = list(range(0, len(image_files), sample_interval))
if len(image_files) - 1 not in sampled_indices:
    sampled_indices.append(len(image_files) - 1)

for idx in tqdm(sampled_indices, desc="推理进度"):
    image = Image.open(image_files[idx]).convert('RGB')

    prompt = f"You are an autonomous navigation assistant. Your task: {instruction}. Where should you go next?"
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

print("✅ 推理完成")
print()

# 5. 生成视频
print("🎬 生成视频...")

first_img = Image.open(str(image_files[0]))
width, height = first_img.size
text_height = 120
new_height = height + text_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(output_video, fourcc, fps, (width, new_height))

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

    current_prediction = "Processing..."
    for sample_idx in reversed(sorted(results.keys())):
        if sample_idx <= idx:
            current_prediction = results[sample_idx]
            break

    draw = ImageDraw.Draw(canvas)
    draw.text((10, height + 10), f"{scene_name} - Frame: {idx}/{len(image_files)}",
              fill=(0, 0, 0), font=font_small)

    inst_display = instruction[:55] + "..." if len(instruction) > 55 else instruction
    draw.text((10, height + 35), f"Task: {inst_display}",
              fill=(0, 100, 0), font=font_small)

    draw.text((10, height + 65), f"Action: {current_prediction}",
              fill=(200, 0, 0), font=font)

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

# 统计
unique_preds = {}
for pred in results.values():
    unique_preds[pred] = unique_preds.get(pred, 0) + 1

print(f"\n✅ 视频生成完成!")
print(f"   文件: {output_video}")
print(f"   时长: {len(image_files)/fps:.1f}秒")
print(f"   推理结果统计:")
for pred, count in sorted(unique_preds.items(), key=lambda x: -x[1]):
    print(f"     {pred}: {count}次")
print()
