#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从日志中解析推理结果并生成分析报告
"""

import re
from collections import Counter

log_file = "/home/dell/tmp/claude/-mnt-data-0923-Interndata-vln-n1-traj-data-replica-d435i/tasks/b8c3a59.output"

# 解析日志
results = {}
with open(log_file, 'r') as f:
    # 跳过预热的 output 1
    lines = f.readlines()
    output_started = False
    for line in lines:
        if "🎯 推理中..." in line:
            output_started = True
            continue
        if output_started:
            match = re.match(r'output (\d+)\s+(\S+)\s+cost: ([\d.]+)s', line)
            if match:
                idx = int(match.group(1))
                action = match.group(2)
                cost = float(match.group(3))
                results[idx] = {'action': action, 'cost': cost}

print("=" * 70)
print("官方推理结果分析")
print("=" * 70)
print()

# 计算帧索引（每4帧推理一次）
frame_results = {}
for idx, data in results.items():
    frame_idx = (idx - 1) * 4  # idx从1开始，转换为帧索引
    frame_results[frame_idx] = data['action']

print(f"📊 总推理次数: {len(results)}")
print(f"📂 对应帧数: {max(frame_results.keys())} / 182 帧")
print(f"⚙️  推理频率: plan_step_gap=4 (每4帧)")
print()

# 统计动作分布
action_counter = Counter([d['action'] for d in results.values()])
print("📈 动作分布统计:")
for action, count in sorted(action_counter.items(), key=lambda x: -x[1]):
    percentage = count / len(results) * 100
    print(f"  {action:<10} : {count:2d}次 ({percentage:5.1f}%)")
print()

# 按阶段分析
print("🔍 决策阶段分析:")
stages = []
current_action = None
stage_start = None
stage_frames = []

for idx in sorted(results.keys()):
    action = results[idx]['action']
    frame_idx = (idx - 1) * 4

    if action != current_action:
        # 结束上一个阶段
        if current_action is not None:
            stages.append({
                'action': current_action,
                'start_frame': stage_start,
                'end_frame': frame_idx - 4,
                'count': len(stage_frames)
            })
        # 开始新阶段
        current_action = action
        stage_start = frame_idx
        stage_frames = [idx]
    else:
        stage_frames.append(idx)

# 添加最后一个阶段
if current_action is not None:
    stages.append({
        'action': current_action,
        'start_frame': stage_start,
        'end_frame': max(frame_results.keys()),
        'count': len(stage_frames)
    })

for i, stage in enumerate(stages, 1):
    print(f"  阶段{i}: 帧 {stage['start_frame']:3d}-{stage['end_frame']:3d} | "
          f"{stage['action']:<10} | {stage['count']:2d}次推理")
print()

# 推理时间分析
costs = [d['cost'] for d in results.values()]
print("⏱️  推理时间统计:")
print(f"  最快: {min(costs):.2f}秒")
print(f"  最慢: {max(costs):.2f}秒")
print(f"  平均: {sum(costs)/len(costs):.2f}秒")
print(f"  总计: {sum(costs)/60:.1f}分钟")
print()

# 保存结果到文件
import json
with open('official_inference_results.json', 'w') as f:
    json.dump({
        'frame_results': {str(k): v for k, v in frame_results.items()},
        'statistics': {
            'total_inferences': len(results),
            'action_distribution': dict(action_counter),
            'stages': stages,
            'time_stats': {
                'min': min(costs),
                'max': max(costs),
                'avg': sum(costs)/len(costs),
                'total': sum(costs)
            }
        }
    }, f, indent=2, ensure_ascii=False)

print("✅ 结果已保存到: official_inference_results.json")
