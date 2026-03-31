#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用官方 InternVLAN1AsyncAgent 生成推理视频
展示双系统输出：System 1 (轨迹航点) + System 2 (动作符号)
"""

import os, sys, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import torch

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# ============ 配置 ============
class Args:
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "checkpoints/InternVLA-N1-DualVLN"
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        # D435i 相机内参
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 4  # 官方推荐：每4帧推理一次

args = Args()

# 轨迹路径
scene = "office_3"
trajectory = "trajectory_68"
base_path = "/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample"
trajectory_path = Path(base_path) / scene / trajectory
output_video = f"official_inference_{scene}_{trajectory}.mp4"

print("=" * 70)
print("官方推理方法 - 双系统输出演示")
print("=" * 70)
print()

# ============ 1. 加载模型 ============
print("🚀 加载官方 InternVLAN1AsyncAgent...")

# 临时修改以避免 flash_attention 依赖
import internnav.agent.internvla_n1_agent_realworld as agent_module
original_init = agent_module.InternVLAN1AsyncAgent.__init__

def patched_init(self, args):
    # 调用原始 __init__，但先修改模型加载
    from transformers import AutoProcessor
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
    from datetime import datetime

    self.device = torch.device(args.device)
    self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"args.model_path{args.model_path}")

    # 使用 eager attention 代替 flash_attention_2
    self.model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 修改这里
        device_map={"": self.device},
    )
    self.model.eval()
    self.model.to(self.device)

    self.processor = AutoProcessor.from_pretrained(args.model_path)
    self.processor.tokenizer.padding_side = 'left'

    self.resize_w = args.resize_w
    self.resize_h = args.resize_h
    self.num_history = args.num_history
    self.PLAN_STEP_GAP = args.plan_step_gap

    prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
    answer = ""
    self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
    self.conjunctions = [
        'you can see ',
        'in front of you is ',
        'there is ',
        'you can spot ',
        'you are toward the ',
        'ahead of you is ',
        'in your sight is ',
    ]

    # 继续初始化其他属性（完整版）
    self.actions2idx = agent_module.OrderedDict({
        'STOP': [0],
        "↑": [1],
        "←": [2],
        "→": [3],
        "↓": [5],
    })

    self.rgb_list = []
    self.depth_list = []
    self.pose_list = []
    self.episode_idx = 0
    self.conversation_history = []
    self.llm_output = ""
    self.past_key_values = None
    self.last_s2_idx = -100  # 重要：初始化为 -100

    # output
    self.output_action = None
    self.output_latent = None
    self.output_pixel = None
    self.pixel_goal_rgb = None
    self.pixel_goal_depth = None

agent_module.InternVLAN1AsyncAgent.__init__ = patched_init

agent = InternVLAN1AsyncAgent(args)

# Warm up
print("⚡ 预热模型...")
dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_depth = np.zeros((480, 640), dtype=np.float32)
dummy_pose = np.eye(4)
agent.reset()
agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
print("✅ 模型加载完成")
print()

# ============ 2. 读取指令 ============
meta_file = trajectory_path / "meta/episodes.jsonl"
if meta_file.exists():
    with open(meta_file) as f:
        episode_data = json.loads(f.readline())
    task_data = json.loads(episode_data['tasks'][0])
    instruction = task_data.get('sub_instruction') or task_data.get('sum_instruction', "Navigate through the environment")
else:
    instruction = "Navigate through the environment"

print(f"📖 指令: {instruction}")
print()

# ============ 3. 读取图像 ============
image_dir = trajectory_path / "videos/chunk-000/observation.images.rgb"
image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: int(x.stem))
print(f"📂 {len(image_files)} 帧图像")
print(f"⚙️  plan_step_gap={args.plan_step_gap} (每{args.plan_step_gap}帧推理一次)")
print()

# ============ 4. 推理 ============
print("🎯 推理中...")
agent.reset()
results = {}  # idx -> (output_action, output_trajectory, output_pixel)

# 只在 plan_step_gap 的倍数帧进行推理
inference_indices = list(range(0, len(image_files), args.plan_step_gap))
if len(image_files) - 1 not in inference_indices:
    inference_indices.append(len(image_files) - 1)

for idx in tqdm(inference_indices, desc="推理进度"):
    # 读取图像
    rgb = np.asarray(Image.open(image_files[idx]).convert('RGB'))

    # 创建深度图（实际数据中没有，用常量填充）
    depth = 10.0 * np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    # 相机位姿（单位矩阵）
    camera_pose = np.eye(4)

    # 推理
    with torch.no_grad():
        dual_sys_output = agent.step(
            rgb,
            depth,
            camera_pose,
            instruction,
            intrinsic=args.camera_intrinsic
        )

    # 保存结果
    output_action = dual_sys_output.output_action if dual_sys_output.output_action else None
    output_trajectory = dual_sys_output.output_trajectory.tolist() if dual_sys_output.output_trajectory is not None else None
    output_pixel = dual_sys_output.output_pixel if dual_sys_output.output_pixel is not None else None

    results[idx] = {
        'action': output_action,
        'trajectory': output_trajectory,
        'pixel': output_pixel
    }

print("✅ 推理完成")
print()

# ============ 5. 生成视频 ============
print("🎬 生成视频...")

first_img = Image.open(str(image_files[0]))
width, height = first_img.size
text_height = 150  # 增加高度以显示更多信息
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

    # 找到当前帧对应的推理结果（使用最近的推理结果）
    current_result = None
    for inference_idx in reversed(sorted(results.keys())):
        if inference_idx <= idx:
            current_result = results[inference_idx]
            break

    # 绘制文本信息
    draw = ImageDraw.Draw(canvas)
    y_offset = height + 5

    # 基本信息
    draw.text((10, y_offset), f"Frame: {idx}/{len(image_files)} | Scene: Office 3",
              fill=(0, 0, 0), font=font_small)
    y_offset += 20

    # 指令
    inst_display = instruction[:60] + "..." if len(instruction) > 60 else instruction
    draw.text((10, y_offset), f"Task: {inst_display}",
              fill=(0, 100, 0), font=font_small)
    y_offset += 25

    # 推理结果
    if current_result:
        # System 2: 动作符号
        if current_result['action']:
            # 处理 action 可能是字符串列表或整数列表
            if isinstance(current_result['action'], list):
                action_text = " ".join([str(a) for a in current_result['action']])
            else:
                action_text = str(current_result['action'])
            draw.text((10, y_offset), f"System 2 (VLN): {action_text}",
                      fill=(200, 0, 0), font=font)
            y_offset += 20

        # System 1: 轨迹航点
        if current_result['trajectory']:
            traj_str = f"{len(current_result['trajectory'])} waypoints"
            draw.text((10, y_offset), f"System 1 (VN): {traj_str}",
                      fill=(0, 0, 200), font=font)
            y_offset += 20

            # 显示前3个航点
            traj_preview = str(current_result['trajectory'][:3])[:50] + "..."
            draw.text((10, y_offset), f"  Waypoints: {traj_preview}",
                      fill=(100, 100, 100), font=font_small)
    else:
        draw.text((10, y_offset), "Processing...",
                  fill=(150, 150, 150), font=font)

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

# ============ 6. 统计结果 ============
print(f"\n✅ 视频生成完成!")
print(f"   文件: {output_video}")
print(f"   时长: {len(image_files)/fps:.1f}秒")
print(f"\n📊 推理统计:")
print(f"   总帧数: {len(image_files)}")
print(f"   推理次数: {len(results)} (每{args.plan_step_gap}帧)")

# 统计 System 2 动作
action_count = {}
trajectory_count = 0
for result in results.values():
    if result['action']:
        action = " ".join(result['action']) if isinstance(result['action'], list) else str(result['action'])
        action_count[action] = action_count.get(action, 0) + 1
    if result['trajectory']:
        trajectory_count += 1

print(f"\nSystem 2 (VLN) 动作统计:")
for action, count in sorted(action_count.items(), key=lambda x: -x[1]):
    print(f"  {action}: {count}次")

print(f"\nSystem 1 (VN) 轨迹输出: {trajectory_count}次")
print()
