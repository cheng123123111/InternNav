# InternVLA-N1 完整安装进度报告

**开始时间**: 2026-03-09
**环境**: Python 3.9 conda环境 `internvln39`

---

## ✅ 已完成步骤

### 1. 环境创建 ✅
- **环境名**: `internvln39`
- **Python版本**: 3.9.25
- **状态**: 创建成功

### 2. PyTorch安装 ✅
- **版本**: PyTorch 2.6.0 + torchvision 0.21.0
- **CUDA**: 12.4
- **大小**: ~768MB + 依赖
- **状态**: 安装成功
- **验证命令**:
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

### 3. InternNav核心包 ✅
- **版本**: 0.3.1
- **状态**: 安装成功（editable mode）

### 4. 核心依赖包 ✅
- **总数**: 60+个Python包
- **关键包**:
  - numpy==1.26.4
  - pillow==9.5.0
  - opencv-python-headless==4.11.0.86
  - gymnasium==0.29.1
  - gym==0.26.2
  - pydantic==2.11.10
  - fastapi==0.110.3
  - rich==14.3.3
  - 等等...
- **状态**: 全部安装成功

---

## ✅ 已完成步骤 (续)

### 5. Habitat-Sim ✅
- **版本**: 0.3.2
- **特性**: with bullet physics, headless
- **大小**: 304.5MB + 依赖包
- **状态**: 安装成功

### 6. Habitat-Lab ✅
- **版本**: 0.3.20231024
- **状态**: 安装成功

### 7. InternVLA-N1模型依赖 ✅
- transformers==4.51.0
- diffusers==0.33.1
- accelerate==1.4.0
- ftfy==6.3.1
- dtw==1.4.0
- **状态**: 全部安装成功
- **注**: flash_attn 跳过（可选优化）

### 8. Git Submodules ✅
- diffusion_policy
- Long-CLIP
- **状态**: 初始化成功

---

## ⚠️ 需要手动完成

### 9. 模型权重下载 (网络问题)
- **模型**: InternVLA-N1-DualVLN
- **大小**: ~8GB
- **问题**: HuggingFace连接超时
- **状态**: 需要手动下载

---

## 🎯 安装完成后的后续步骤

### 数据准备
```
data/
├── scene_data/
│   ├── mp3d_ce/        # Matterport3D场景（VLN-CE）
│   ├── mp3d_pe/        # Matterport3D场景（VLN-PE）
│   └── mp3d_n1/        # Matterport3D场景（VLN-N1）
└── vln_ce/
    └── raw_data/
        └── r2r/
            └── val_unseen/
                └── val_unseen.json.gz
```

### 验证安装
```bash
# 激活环境
conda activate internvln39

# 测试Habitat-Sim
python -c "import habitat_sim; print(habitat_sim.__version__)"

# 测试Habitat-Lab
python -c "import habitat; print(habitat.__version__)"

# 测试InternNav
python -c "import internnav; print('InternNav OK')"

# 测试PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 运行评估
```bash
# 进入项目目录
cd /mnt/data/0923_Interndata/InternNav

# 运行VLN-CE R2R评估（Dual System）
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py

# 或使用启动脚本
bash scripts/eval/bash/start_eval.sh --config scripts/eval/configs/habitat_dual_system_cfg.py
```

---

## 📝 安装命令速查

### 激活环境
```bash
eval "$(conda shell.bash hook)"
conda activate internvln39
```

### 手动安装步骤（如果需要）
```bash
# Habitat-Lab
pip install habitat-lab==0.3.2

# Habitat依赖
pip install -r requirements/habitat_requirements.txt

# InternVLA-N1依赖
pip install -r requirements/internvla_n1.txt

# 初始化submodules
git submodule update --init --recursive

# 下载模型
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='InternRobotics/InternVLA-N1-DualVLN',
    local_dir='checkpoints/InternVLA-N1-DualVLN',
    local_dir_use_symlinks=False
)
"
```

---

## 🔧 故障排查

### Habitat-Sim安装失败
- 确认Python版本为3.9
- 检查conda channels是否正确
- 尝试: `conda clean --all && conda install ...`

### 模型下载失败
- 使用HF镜像: `export HF_ENDPOINT=https://hf-mirror.com`
- 或使用浏览器手动下载

### GPU不可用
- 检查CUDA驱动: `nvidia-smi`
- 检查PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

---

最后更新: 正在安装Habitat-Sim...
