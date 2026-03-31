# 🎬 视频查看指南

## 📺 快速观看

### 方式1: VLC播放器
```bash
cd /mnt/data/0923_Interndata/InternNav

# 官方完整版（最推荐）⭐⭐⭐⭐⭐
vlc official_inference_office_3_trajectory_68.mp4

# 简化版（真实指令）⭐⭐⭐⭐
vlc inference_real_instruction_video.mp4

# 对比通用指令 vs 真实指令
vlc inference_output_video.mp4 inference_real_instruction_video.mp4
```

### 方式2: mpv播放器
```bash
mpv official_inference_office_3_trajectory_68.mp4
```

---

## 📊 视频对比一览

| 视频 | 大小 | 方法 | 动作种类 | 推理次数 | 推荐度 |
|------|------|------|---------|---------|--------|
| **official_inference_office_3_trajectory_68.mp4** | 810KB | 官方Agent | **6种** | 47次 | ⭐⭐⭐⭐⭐ |
| inference_real_instruction_video.mp4 | 718KB | 简化版 | 3种 | 19次 | ⭐⭐⭐⭐ |
| inference_output_video.mp4 | 651KB | 简化版 | 1种 | 19次 | ⭐⭐ |

---

## 🎯 官方视频亮点

**文件**: `official_inference_office_3_trajectory_68.mp4`

### 决策时间线
```
0-4秒:   ←←←← (调整方向)
1-4秒:   ↓    (开始直行)
4-7秒:   ←/←← (左转避障)
8-15秒:  ↓    (长距离直行)
16秒:    STOP (到达目标)
16.4秒:  →→   (右转面墙)
17-18秒: STOP (任务完成)
```

### 动作统计
- ↓ (直行): 28次 (59.6%)
- STOP (停止): 6次 (12.8%)
- ← (左转): 5次 (10.6%)
- ←← (大幅左转): 5次 (10.6%)
- ←←←← (初始调整): 2次 (4.3%)
- →→ (右转): 1次 (2.1%)

---

## 📝 相关文档

- **VIDEO_FINAL_SUMMARY.md** - 完整对比总结
- **INFERENCE_COMPARISON.md** - 方法详细对比
- **VIDEO_GALLERY.md** - 视频库文档
- **official_inference_results.json** - 推理数据

---

生成时间: 2026-03-10
