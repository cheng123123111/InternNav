#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已保存的推理结果快速生成视频（不需要重新推理）
"""

import json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("从保存的推理结果生成视频")
print("=" * 70)
print()

# 加载推理结果
print("📂 加载推理结果...")
with open('official_inference_results.json', 'r') as f:
    data = json.load(f)

frame_results = {int(k): v for k, v in data['frame_results'].items()}
statistics = data['statistics']

print(f"✅ 加载完成: {statistics['total_inferences']} 次推理结果")
print()

# 轨迹路径
scene = "office_3"
trajectory = "trajectory_68"
base_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample"
trajectory_path = Path(base_path) / scene / trajectory
output_video = f"official_inference_{scene}_{trajectory}.mp4"

# 读取指令
meta_file = trajectory_path / "meta/episodes.jsonl"
if meta_file.exists():
    with open(meta_file) as f:
        episode_data = json.loads(f.readline())
    task_data = json.loads(episode_data['tasks'][0])
    instruction = task_data.get('sub_instruction') or task_data.get('sum_instruction', "Navigate")
else:
    instruction = "Navigate through the environment"

print(f"📖 指令: {instruction}")
print()

# 读取图像
image_dir = trajectory_path / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"📂 {len(image_files)} 帧图像")
print()

# 生成视频
print("🎬 生成视频...")

first_img = Image.open(str(image_files[0]))
width, height = first_img.size
text_height = 150
new_height = height + text_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
out = cv2.VideoWriter(output_video, fourcc, fps, (width, new_height))

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

for idx in tqdm(range(len(image_files)), desc="生成视频"):
    # 读取图像
    img = Image.open(str(image_files[idx]))
    canvas = Image.new('RGB', (width, new_height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    # 找到当前帧对应的推理结果
    current_action = None
    for inference_frame in reversed(sorted(frame_results.keys())):
        if inference_frame <= idx:
            current_action = frame_results[inference_frame]
            last_inference_frame = inference_frame
            break

    # 绘制文本信息
    draw = ImageDraw.Draw(canvas)
    y_offset = height + 5

    # 基本信息
    draw.text((10, y_offset), f"Frame: {idx}/{len(image_files)} | Scene: Office 3 | Method: Official Agent",
              fill=(0, 0, 0), font=font_small)
    y_offset += 20

    # 指令
    inst_display = instruction[:55] + "..." if len(instruction) > 55 else instruction
    draw.text((10, y_offset), f"Task: {inst_display}",
              fill=(0, 100, 0), font=font_small)
    y_offset += 25

    # 推理结果
    if current_action:
        draw.text((10, y_offset), f"System 2 (VLN): {current_action}",
                  fill=(200, 0, 0), font=font)
        y_offset += 20

        # 显示上次推理的帧号
        draw.text((10, y_offset), f"  (Last inference at frame {last_inference_frame})",
                  fill=(100, 100, 100), font=font_small)
        y_offset += 20
    else:
        draw.text((10, y_offset), "Processing...",
                  fill=(150, 150, 150), font=font)
        y_offset += 20

    # 统计信息
    draw.text((10, y_offset), f"Inference frequency: every 4 frames (plan_step_gap=4)",
              fill=(100, 100, 100), font=font_small)

    # 进度条
    y_offset = height + 130
    progress = idx / len(image_files)
    bar_width = width - 20
    bar_height = 12
    draw.rectangle([(10, y_offset), (10 + bar_width, y_offset + bar_height)],
                   outline=(100, 100, 100), width=1)
    draw.rectangle([(10, y_offset), (10 + int(bar_width * progress), y_offset + bar_height)],
                   fill=(0, 150, 0))

    # 写入帧
    frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()

print(f"\n✅ 视频生成完成!")
print(f"   文件: {output_video}")
print(f"   时长: {len(image_files)/fps:.1f}秒")
print(f"   分辨率: {width}x{new_height}")

# 检查文件大小
import os
file_size = os.path.getsize(output_video)
print(f"   大小: {file_size/1024:.0f}KB")
print()
