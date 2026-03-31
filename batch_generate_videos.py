#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成多个轨迹的推理视频
"""

import os, sys, torch, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("批量生成推理视频")
print("=" * 70)
print()

model_path = "checkpoints/InternVLA-N1-DualVLN"
base_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample"

# 定义要生成的轨迹列表
trajectories = [
    ("office_3", "trajectory_68", "Office Scene 1"),
    ("office_3", "trajectory_61", "Office Scene 2"),
    ("apartment_2", "trajectory_0", "Apartment Scene"),
    ("room_0", "trajectory_0", "Room Scene"),
]

# 1. 加载模型（只加载一次）
print("🚀 加载InternVLA-N1模型...")
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

DEFAULT_IMAGE_TOKEN = "<image>"

# 处理每个轨迹
for scene, trajectory, scene_name in trajectories:
    print("=" * 70)
    print(f"处理: {scene_name} ({scene}/{trajectory})")
    print("=" * 70)

    trajectory_path = Path(base_path) / scene / trajectory

    # 检查路径是否存在
    if not trajectory_path.exists():
        print(f"  ⚠️  路径不存在，跳过")
        print()
        continue

    output_video = f"inference_{scene}_{trajectory}.mp4"

    # 读取真实指令
    meta_file = trajectory_path / "meta/episodes.jsonl"
    if meta_file.exists():
        with open(meta_file) as f:
            episode_data = json.loads(f.readline())
        task_data = json.loads(episode_data['tasks'][0])
        instruction = task_data['sub_instruction']
    else:
        instruction = "Navigate through the environment"

    print(f"  指令: {instruction[:80]}...")
    print()

    # 读取图像
    image_dir = trajectory_path / "videos/chunk-000/observation.images.rgb"
    if not image_dir.exists():
        print(f"  ⚠️  图像目录不存在，跳过")
        print()
        continue

    image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
    print(f"  📂 {len(image_files)} 帧图像")

    # 推理（采样）
    print("  🎯 推理中...")
    results = {}
    sample_interval = max(10, len(image_files) // 15)  # 最多15个采样点
    sampled_indices = list(range(0, len(image_files), sample_interval))
    if len(image_files) - 1 not in sampled_indices:
        sampled_indices.append(len(image_files) - 1)

    for idx in tqdm(sampled_indices, desc="  推理进度", leave=False):
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

    print(f"  ✅ 推理完成")

    # 生成视频
    print("  🎬 生成视频...")

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

    for idx in tqdm(range(len(image_files)), desc="  生成视频", leave=False):
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

        # 场景名称
        draw.text((10, height + 10), f"{scene_name} - Frame: {idx}/{len(image_files)}",
                  fill=(0, 0, 0), font=font_small)

        # 指令（截断）
        inst_display = instruction[:55] + "..." if len(instruction) > 55 else instruction
        draw.text((10, height + 35), f"Task: {inst_display}",
                  fill=(0, 100, 0), font=font_small)

        # 推理结果
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

    # 统计
    unique_preds = {}
    for pred in results.values():
        unique_preds[pred] = unique_preds.get(pred, 0) + 1

    print(f"  ✅ 视频生成: {output_video}")
    print(f"     时长: {len(image_files)/fps:.1f}秒")
    print(f"     推理结果: {', '.join([f'{p}({c})' for p, c in sorted(unique_preds.items())])}")
    print()

print("=" * 70)
print("🎉 批量生成完成！")
print("=" * 70)
print()
print("生成的视频文件:")
for scene, trajectory, scene_name in trajectories:
    video_file = f"inference_{scene}_{trajectory}.mp4"
    if os.path.exists(video_file):
        size = os.path.getsize(video_file) / 1024
        print(f"  ✅ {video_file} ({size:.0f} KB)")
print()
