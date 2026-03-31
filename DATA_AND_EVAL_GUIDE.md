# 数据集与VLN-CE评估完全指南

生成时间: 2026-03-10

---

## 🎯 回答你的问题

### Q1: 有没有在VLN-CE里面的？

**✅ 有的！** InternNav完整支持VLN-CE评估。

VLN-CE = **Vision-Language Navigation in Continuous Environments** (连续环境中的视觉-语言导航)

---

### Q2: 这是在什么数据上跑的？

**我们刚才用的数据**:
- **来源**: InternData-N1 数据集的 Replica 场景样本
- **路径**: `/mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i/extracted_sample`
- **场景**: office_3, apartment_2, room_0
- **用途**: 快速推理演示（不是完整评估）

---

## 📊 数据集详解

### 1. InternData-N1 数据集（我们用的）

**规模**:
- 3000+ 场景
- 830,000+ VLN轨迹数据
- 6个不同的场景数据集
- 2种相机型号（D435i, ZED）

**场景类型** (在 `/mnt/data/0923_Interndata/vln_n1/traj_data/`):
```
├── 3dfront_d435i/          # 3D-Front 室内场景 (70k条)
├── 3dfront_zed/
├── gibson_d435i/           # Gibson 场景 (20k条)
├── gibson_zed/
├── hm3d_d435i/             # Habitat-Matterport3D (37k条)
├── hm3d_zed/
├── hssd_d435i/             # Habitat合成场景 (12k条)
├── hssd_zed/
├── matterport3d_d435i/     # Matterport3D (MP3D)
├── matterport3d_zed/
├── replica_d435i/          # ← 我们用的这个
└── replica_zed/
```

**特点**:
- ✅ 高质量轨迹数据（专家演示）
- ✅ 自然语言指令（详细的导航描述）
- ✅ RGB-D图像序列
- ✅ 相机位姿信息
- ✅ 动作序列

**我们用的样本**:
```
replica_d435i/extracted_sample/
├── office_3/trajectory_68     ← 主要测试场景
├── office_3/trajectory_61
├── apartment_2/trajectory_0
└── room_0/trajectory_13
```

---

### 2. VLN-CE Benchmark（完整评估用）

**VLN-CE = Vision-Language Navigation in Continuous Environments**

**数据集**:
- **R2R (Room-to-Room)**: 基础VLN数据集，Matterport3D场景
  - val_seen: 1020 episodes
  - val_unseen: 2349 episodes
  - test: 4173 episodes

- **RxR (Room-across-Room)**: 多语言VLN数据集
  - 支持英语、印地语、泰卢固语

**评估指标**:
- SR (Success Rate): 成功率
- SPL (Success weighted by Path Length): 路径长度加权成功率
- nDTW (normalized Dynamic Time Warping): 归一化动态时间规整
- SDTW: SPL和nDTW的调和平均

**配置文件**:
```bash
scripts/eval/configs/vln_r2r.yaml         # R2R评估配置
scripts/eval/configs/vln_rxr.yaml         # RxR评估配置
scripts/eval/configs/h1_internvla_n1_async_cfg.py  # InternVLA-N1配置
```

---

### 3. VLN-PE Benchmark（物理具身导航）

**VLN-PE = Vision-Language Navigation for Physical Embodiment**

**特点**:
- 连续动作空间（速度控制 v, w）
- 物理仿真（考虑碰撞、惯性）
- 具身机器人（H1人形机器人）

**数据集**:
- 基于R2R扩展
- 支持Isaac Sim仿真

---

## 🔧 如何在VLN-CE上运行完整评估

### 方式1: 使用Habitat仿真（推荐）

#### 步骤1: 准备VLN-CE数据

```bash
# 下载场景数据（Matterport3D）
# 需要申请权限：https://niessner.github.io/Matterport/

# 数据结构
data/
├── scene_data/
│   └── mp3d_ce/              # Matterport3D场景文件
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
└── vln_ce/
    └── raw_data/
        └── r2r/
            ├── val_seen/
            │   └── val_seen.json.gz
            ├── val_unseen/
            │   └── val_unseen.json.gz
            └── test/
                └── test.json.gz
```

#### 步骤2: 修改评估配置

编辑 `scripts/eval/configs/h1_internvla_n1_async_cfg.py`:

```python
dataset=EvalDatasetCfg(
    dataset_type="mp3d",
    dataset_settings={
        'base_data_dir': 'data/vln_ce/raw_data/r2r',
        'split_data_types': ['val_unseen'],  # 或 'val_seen', 'test'
        'filter_stairs': True,
    },
)
```

#### 步骤3: 运行评估

```bash
cd /mnt/data/0923_Interndata/InternNav

# 在VLN-CE val_unseen上评估
python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

**评估过程**:
1. 加载InternVLA-N1模型
2. 初始化Habitat仿真环境
3. 加载VLN-CE episodes
4. 对每个episode运行导航
5. 计算SR, SPL, nDTW等指标
6. 保存结果到JSON

**预计时间**: 取决于数据集大小
- val_unseen (2349 episodes): ~10-20小时（单GPU）
- test (4173 episodes): ~20-40小时

---

### 方式2: 使用Isaac Sim（VLN-PE）

#### 步骤1: 准备环境

```bash
# 安装Isaac Sim
# 参考: https://developer.nvidia.com/isaac-sim

# 配置VLN-PE环境
export ISAACSIM_PATH=/path/to/isaac-sim
```

#### 步骤2: 运行评估

```bash
python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

---

## 📊 对比：我们的演示 vs 完整VLN-CE评估

| 维度 | 我们的演示 | VLN-CE完整评估 |
|------|----------|---------------|
| **数据来源** | InternData-N1 样本 | VLN-CE R2R/RxR |
| **场景数** | 3个 (office_3, apartment_2, room_0) | 90个 (Matterport3D) |
| **轨迹数** | 4条 | 2349+ (val_unseen) |
| **用途** | 快速推理演示 | 标准化基准测试 |
| **评估指标** | 无（仅可视化） | SR, SPL, nDTW, SDTW |
| **耗时** | ~6分钟（1条轨迹） | 10-20小时（全数据集） |
| **硬件需求** | 1x GPU | 多GPU（分布式） |

---

## 🎯 我们演示用的数据详情

### Replica场景

**Replica** = Facebook Research的高质量3D场景数据集

**特点**:
- 真实室内场景的高精度重建
- 18个场景（办公室、公寓、房间等）
- 干净、无杂物（易于测试）
- 用于Habitat仿真

**我们用的具体场景**:
```
office_3/trajectory_68:
  场景: Replica办公室3号
  指令: "Walk straight ahead, passing the beige sofa on your left..."
  帧数: 182帧 (18.2秒)
  分辨率: 480x270 (D435i相机)

apartment_2/trajectory_0:
  场景: Replica公寓2号
  帧数: 78帧 (7.8秒)

room_0/trajectory_13:
  场景: Replica房间0号
  帧数: 若干帧
```

**数据格式**: LeRobot v2.1 格式
```
trajectory_68/
├── meta/
│   └── episodes.jsonl         # 元数据（指令、场景信息）
├── videos/
│   └── chunk-000/
│       ├── observation.images.rgb/  # RGB图像序列
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
│       └── observation.images.depth/  # 深度图序列
└── data/
    └── chunk-000/
        └── episode_000000.parquet  # 轨迹数据（位姿、动作等）
```

---

## 📚 相关资源

### 论文
- [VLN-CE论文](https://arxiv.org/abs/2004.02857) - VLN-CE benchmark
- [DualVLN论文](https://arxiv.org/abs/2512.08186) - InternVLA-N1模型架构
- [InternVLA-N1技术报告](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)

### 数据集下载
- **InternData-N1**: [HuggingFace](https://huggingface.co/datasets/InternRobotics/InternData-N1)
- **VLN-CE (R2R)**: [GitHub](https://github.com/jacobkrantz/VLN-CE)
- **Matterport3D**: [官网](https://niessner.github.io/Matterport/) (需申请)
- **Replica**: [GitHub](https://github.com/facebookresearch/Replica-Dataset)

### 官方文档
- [InternNav文档](https://internrobotics.github.io/user_guide/internnav/index.html)
- [快速开始](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html)
- [评估指南](https://internrobotics.github.io/user_guide/internnav/quick_start/evaluation.html)

---

## 🚀 下一步建议

### 如果你想快速测试（当前方式）
✅ 继续使用InternData-N1样本
- 优点：快速、易用、不需要下载大型数据集
- 适合：模型开发、快速验证、演示

### 如果你想标准化评估
✅ 在VLN-CE上运行完整评估
- 准备VLN-CE数据（R2R）
- 运行官方评估脚本
- 获得可比较的指标（SR, SPL等）
- 适合：论文发表、模型对比

### 如果你想训练模型
✅ 使用InternData-N1完整数据集
- 下载完整的830k轨迹数据
- 使用多场景数据（3dfront, gibson, hm3d等）
- 参考训练文档

---

## 📝 总结

**我们用的数据**:
- **数据集**: InternData-N1 (Replica场景样本)
- **规模**: 4条轨迹，3个场景
- **用途**: 快速推理演示和视频生成

**完整VLN-CE评估**:
- **数据集**: VLN-CE (R2R/RxR)
- **规模**: 2349+ episodes (val_unseen)
- **用途**: 标准化基准测试

**两者关系**:
- InternData-N1是**训练数据**（大规模、多样化）
- VLN-CE是**评估基准**（标准化、可比较）
- 模型在InternData-N1上训练，在VLN-CE上评估

---

生成时间: 2026-03-10
作者: Claude Code
