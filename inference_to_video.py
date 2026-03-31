#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InternVLA-N1 轨迹推理视频生成脚本
对轨迹中的每一帧进行推理，并生成带有推理结果的视频
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
navigation_instruction = "Navigate through the office and reach the desk area."  # 示例指令

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

    print(f"  ✅ 模型加载成功 ({next(model.parameters()).device})")
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

# 3. 对每帧进行推理
print("🎯 开始推理...")
DEFAULT_IMAGE_TOKEN = "<image>"
results = []

# 为了加速，我们每5帧采样一次进行推理
sample_interval = 5
sampled_indices = list(range(0, len(image_files), sample_interval))

for idx in tqdm(sampled_indices, desc="推理进度"):
    img_path = image_files[idx]
    image = Image.open(img_path).convert('RGB')

    # 构建prompt
    prompt = f"You are an autonomous navigation assistant. Your task is to {navigation_instruction}. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image."
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    # 解码结果
    generated_text = processor.tokenizer.decode(
        outputs.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    results.append({
        'frame_idx': idx,
        'prediction': generated_text.strip()
    })

print()
print(f"✅ 推理完成！共处理 {len(results)} 帧")
print()

# 4. 生成视频
print("🎬 生成推理视频...")

# 获取第一帧的尺寸
first_frame = cv2.imread(str(image_files[0]))
height, width = first_frame.shape[:2]

# 扩展画布高度以容纳文本
text_height = 120
new_height = height + text_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10  # 10 fps
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, new_height))

# 创建一个结果字典，用于快速查找
result_dict = {r['frame_idx']: r['prediction'] for r in results}

for idx, img_path in enumerate(tqdm(image_files, desc="生成视频")):
    # 读取图像
    frame = cv2.imread(str(img_path))

    # 创建扩展画布
    canvas = np.ones((new_height, width, 3), dtype=np.uint8) * 255
    canvas[:height, :] = frame

    # 获取当前帧的推理结果（如果有）
    # 如果当前帧没有推理结果，使用最近的推理结果
    current_prediction = None
    for sample_idx in reversed(sampled_indices):
        if sample_idx <= idx:
            current_prediction = result_dict.get(sample_idx, "Processing...")
            break

    if current_prediction is None:
        current_prediction = "Initializing..."

    # 添加文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 帧信息
    cv2.putText(canvas, f"Frame: {idx}/{len(image_files)}",
                (10, height + 30), font, 0.6, (0, 0, 0), 2)

    # 指令
    cv2.putText(canvas, f"Instruction: {navigation_instruction[:40]}...",
                (10, height + 60), font, 0.5, (0, 100, 0), 1)

    # 推理结果
    prediction_text = f"Prediction: {current_prediction[:50]}"
    cv2.putText(canvas, prediction_text,
                (10, height + 90), font, 0.5, (255, 0, 0), 2)

    out.write(canvas)

out.release()

print(f"✅ 视频生成完成！")
print(f"   输出文件: {output_video_path}")
print()

# 5. 显示一些推理结果样例
print("📝 推理结果样例:")
for i, result in enumerate(results[:5]):
    print(f"  帧 {result['frame_idx']:3d}: {result['prediction']}")
if len(results) > 5:
    print(f"  ... (共 {len(results)} 个推理结果)")
print()

# 6. 显存使用
if torch.cuda.is_available():
    print("💾 GPU显存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print()

print("=" * 70)
print("🎉 完成！")
print("=" * 70)
print()
print(f"现在可以使用视频播放器打开: {output_video_path}")
print()
