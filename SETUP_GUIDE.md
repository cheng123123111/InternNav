# InternVLA-N1 完整设置指南

## 📊 当前环境状态

### ✅ 已完成安装
- **Conda环境**: `internvln` (Python 3.10.19)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **InternNav**: 0.3.1 (已安装)
- **核心依赖**: 60+个Python包已安装
- **模型依赖**: transformers==4.51.0, diffusers==0.33.1, accelerate==1.4.0

### ⚠️ 待解决问题

#### 1. Habitat-Sim 版本兼容性
- **问题**: Habitat-Sim 0.3.2 需要 Python 3.9
- **当前**: Python 3.10
- **影响**: 无法运行VLN-CE完整评估
- **不影响**: 模型推理、可视化功能

#### 2. 模型下载网络问题
- **问题**: Hugging Face连接不稳定
- **需要**: 手动下载或使用镜像

---

## 🚀 解决方案

### 方案A: 快速推理测试（推荐！⚡）

**适合**: 快速体验模型功能，无需完整仿真环境

#### 步骤1: 下载模型（3种方法）

**方法1: 使用浏览器下载**
1. 访问: https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN
2. 下载所有文件到本地
3. 放置到: `/mnt/data/0923_Interndata/InternNav/checkpoints/InternVLA-N1-DualVLN/`

**方法2: 使用Python huggingface_hub**
```bash
conda activate internvln
python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="InternRobotics/InternVLA-N1-DualVLN",
    local_dir="/mnt/data/0923_Interndata/InternNav/checkpoints/InternVLA-N1-DualVLN",
    local_dir_use_symlinks=False
)
EOF
```

**方法3: 使用镜像站（国内推荐）**
```bash
export HF_ENDPOINT=https://hf-mirror.com
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/InternRobotics/InternVLA-N1-DualVLN checkpoints/InternVLA-N1-DualVLN
cd checkpoints/InternVLA-N1-DualVLN
git lfs pull
```

#### 步骤2: 安装剩余依赖
```bash
conda activate internvln
cd /mnt/data/0923_Interndata/InternNav

# 初始化git submodules（重要！）
git submodule update --init --recursive

# 安装InternVLA-N1特定依赖
pip install ftfy==6.3.1
pip install scipy matplotlib
```

#### 步骤3: 运行推理Demo
```bash
cd scripts/notebooks
jupyter notebook inference_only_demo.ipynb
```

---

### 方案B: 完整VLN-CE评估环境

**适合**: 需要在Habitat仿真环境中进行完整benchmark评估

#### 步骤1: 创建Python 3.9环境
```bash
conda create -n internvln39 python=3.9 -y
conda activate internvln39
```

#### 步骤2: 安装所有依赖
```bash
cd /mnt/data/0923_Interndata/InternNav

# 安装PyTorch
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 安装InternNav
pip install -e .

# 安装核心依赖
pip install -r requirements/core_requirements.txt

# 安装Habitat-Sim (重要！需要Python 3.9)
conda install -y habitat-sim=0.3.2 withbullet headless -c conda-forge -c aihabitat

# 安装Habitat-Lab
pip install habitat-lab==0.3.2

# 安装Habitat和InternVLA-N1依赖
pip install -r requirements/habitat_requirements.txt
pip install -r requirements/internvla_n1.txt

# 初始化submodules
git submodule update --init --recursive
```

#### 步骤3: 准备数据
```bash
cd /mnt/data/0923_Interndata/InternNav

# 创建数据目录
mkdir -p data/scene_data
mkdir -p data/vln_ce/raw_data/r2r

# 链接或复制场景数据（示例）
# 你需要准备mp3d_ce场景数据
ln -s /path/to/mp3d_ce data/scene_data/mp3d_ce

# 下载VLN-CE任务数据
# 从 https://github.com/jacobkrantz/VLN-CE 获取
```

#### 步骤4: 下载模型（同方案A）

#### 步骤5: 运行评估
```bash
# 测试R2R任务
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py

# 或使用脚本
bash scripts/eval/bash/start_eval.sh --config scripts/eval/configs/habitat_dual_system_cfg.py
```

---

## 📁 数据目录结构

```
/mnt/data/0923_Interndata/
├── InternNav/                    # 代码仓库
│   ├── checkpoints/
│   │   └── InternVLA-N1-DualVLN/    # 模型权重 (~8GB)
│   ├── data/
│   │   ├── scene_data/
│   │   │   ├── mp3d_ce/             # Matterport3D for VLN-CE
│   │   │   ├── mp3d_pe/             # Matterport3D for VLN-PE
│   │   │   └── mp3d_n1/             # Matterport3D for VLN-N1
│   │   └── vln_ce/
│   │       └── raw_data/
│   │           └── r2r/
│   │               ├── train/
│   │               ├── val_seen/
│   │               └── val_unseen/
│   │                   └── val_unseen.json.gz
│   └── logs/                     # 评估结果
└── vln_n1/                       # 你的数据集
    └── traj_data/
```

---

## 🎯 快速启动脚本

### 激活环境（推理）
```bash
eval "$(conda shell.bash hook)"
conda activate internvln
cd /mnt/data/0923_Interndata/InternNav
```

### 激活环境（完整评估）
```bash
eval "$(conda shell.bash hook)"
conda activate internvln39
cd /mnt/data/0923_Interndata/InternNav
```

---

## 🔍 故障排查

### 模型下载失败
- 尝试使用HuggingFace镜像站
- 或通过Python的huggingface_hub库下载
- 检查网络代理设置

### Habitat-Sim安装失败
- 确认Python版本是3.9
- 检查CUDA版本兼容性
- 参考官方文档: https://github.com/facebookresearch/habitat-sim

### Git submodule错误
```bash
cd /mnt/data/0923_Interndata/InternNav
git submodule update --init --recursive
```

### 缺少依赖
```bash
# 检查已安装的包
pip list | grep -E "transformers|diffusers|accelerate|habitat"

# 重新安装特定包
pip install --force-reinstall transformers==4.51.0
```

---

## 📚 参考资源

- **InternNav GitHub**: https://github.com/InternRobotics/InternNav
- **InternVLA-N1 模型**: https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN
- **技术报告**: https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf
- **文档**: https://internrobotics.github.io/user_guide/internnav/index.html

---

## ✅ 检查清单

在运行评估前，确认：
- [ ] 环境已正确激活（internvln 或 internvln39）
- [ ] 模型已下载到 checkpoints/InternVLA-N1-DualVLN/
- [ ] Git submodules已初始化
- [ ] 所有依赖包已安装
- [ ] 场景数据已准备（如需评估）
- [ ] GPU驱动和CUDA正常工作

测试GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

好运！🚀
