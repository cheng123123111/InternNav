# 📹 InternVLA-N1 推理视频库

生成时间: 2026-03-10

## 🎬 视频列表

### 1. inference_real_instruction_video.mp4 (718KB) ⭐ 推荐
**场景**: Office 3 - trajectory 68
**指令**: "Walk straight ahead, passing the beige sofa on your left and the blue bench on your right. Stop at the end of the blue bench, facing the white wall."
**时长**: 18.2秒 (182帧)
**特点**: 使用真实导航指令，决策动态变化
- ←←←← (60%) - 调整方向
- ↓ (25%) - 直行
- →→→→ (15%) - 转向目标

**效果**: ✅ 最佳示例 - 展示了模型如何根据具体指令动态调整决策

---

### 2. inference_office_3_trajectory_68.mp4 (729KB)
**场景**: Office 3 - trajectory 68
**指令**: "Walk straight ahead, passing the beige sofa on your left and the blue bench on your right..."
**时长**: 18.2秒 (182帧)
**推理结果**:
- ←←←← (9次)
- ↓ (6次)
- →→→→ (2次)

**特点**: 同一轨迹的另一版本生成

---

### 3. inference_office_3_trajectory_61.mp4 (434KB)
**场景**: Office 3 - trajectory 61
**指令**: "Stroll forward, with the ivory couch drifting by to your..."
**时长**: 12.3秒 (123帧)
**分辨率**: 480x390

**特点**: 办公室场景的另一个轨迹

---

### 4. inference_apartment_2_trajectory_0.mp4 (244KB)
**场景**: Apartment 2 - trajectory 0
**指令**: "Continuing straight past the ivory-upholstered armchair..."
**时长**: 7.8秒 (78帧)
**分辨率**: 480x390

**特点**: 公寓场景 - 展示不同环境下的导航

---

### 5. inference_output_video.mp4 (651KB)
**场景**: Office 3 - trajectory 68
**指令**: "Navigate through the office" (通用指令)
**时长**: 18.2秒 (182帧)
**问题**: 所有帧都是 `←←←←` - 决策单一

**用途**: ⚠️ 对比示例 - 展示通用指令的局限性

---

## 📊 对比分析

### 通用指令 vs 真实指令

| 指令类型 | 决策变化 | 示例 |
|---------|---------|------|
| **通用指令** | ❌ 单一 | 所有帧: ←←←← |
| **真实指令** | ✅ 动态 | 开始:←←←← → 中期:↓ → 结束:→→→→ |

### 不同场景对比

| 场景类型 | 特点 | 视频数量 |
|---------|------|---------|
| **Office** | 办公室环境，有家具参考 | 3个 |
| **Apartment** | 公寓环境，更狭窄 | 1个 |

---

## 🎯 关键发现

### 1. 指令的重要性
- ✅ **具体指令** (包含参考物、方向) → 决策准确且动态
- ❌ **模糊指令** ("Navigate through...") → 决策单一

### 2. 模型能力
- ✅ 能理解复杂的自然语言指令
- ✅ 能根据视觉输入和指令做出合理决策
- ✅ 决策随任务进度变化

### 3. 决策符号含义
- `←←←←` : 向左转
- `→→→→` : 向右转
- `↓` : 直行/准备停止
- `STOP` : 停止（任务完成）

---

## 📂 数据来源

- **环境**: Habitat-Sim 仿真器
- **场景**: Replica 数据集
  - office_3: 办公室场景
  - apartment_2: 公寓场景
  - room_0: 房间场景
- **摄像头**: 仿真 D435i RGB-D相机
- **数据集**: InternData-N1 (370k+ 轨迹)

---

## 🎬 如何查看

```bash
cd /mnt/data/0923_Interndata/InternNav

# 查看推荐视频（真实指令版本）
vlc inference_real_instruction_video.mp4

# 对比通用指令 vs 真实指令
vlc inference_output_video.mp4 inference_real_instruction_video.mp4

# 查看不同场景
vlc inference_office_3_*.mp4 inference_apartment_2_*.mp4

# 或使用mpv
mpv inference_*.mp4
```

---

## 📝 技术细节

### 模型信息
- **模型**: InternVLA-N1-DualVLN (8.38B 参数)
- **精度**: bfloat16
- **推理**: 每10帧采样1次
- **GPU**: Quadro GV100 (15.64GB 显存)

### 视频生成
- **分辨率**: 480x390 (原图270p + 120px文本区)
- **帧率**: 10 fps
- **格式**: MP4 (H.264)
- **内容**:
  - 导航场景（第一人称视角）
  - 任务描述
  - 实时推理结果
  - 进度条

---

## 🔍 下一步

1. **测试更多场景**: room场景、其他apartment轨迹
2. **测试不同指令**: 尝试更复杂的导航任务
3. **完整评估**: 在VLN-CE benchmark上运行完整评估
4. **对比分析**: 与其他VLN模型对比

---

生成工具:
- `generate_inference_video.py` - 基础视频生成
- `generate_video_with_real_instruction.py` - 真实指令版本
- `batch_generate_videos.py` - 批量生成
- `generate_single_video.py` - 单个轨迹生成
