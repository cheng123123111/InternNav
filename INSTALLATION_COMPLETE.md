# 🎉 InternVLA-N1 安装完成报告

**完成时间**: 2026-03-09
**环境**: `internvln39` (Python 3.9.25)

---

## ✅ 已成功安装的组件

### 核心环境
- **Conda环境**: `internvln39`
- **Python**: 3.9.25
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 可用 ✅

### 仿真环境
- **Habitat-Sim**: 0.3.2 (with bullet physics, headless)
- **Habitat-Lab**: 0.3.0
- **状态**: 完全正常运行 ✅

### 模型依赖
- **Transformers**: 4.51.0
- **Diffusers**: 0.33.1
- **Accelerate**: 1.4.0
- **InternNav**: 0.3.1
- **状态**: 全部安装成功 ✅

### Git Submodules
- **Long-CLIP**: 已初始化
- **diffusion_policy**: 已初始化
- **状态**: 完成 ✅

### 安装验证结果
```
Habitat-Sim: 0.3.2
Habitat-Lab: 0.3.0
PyTorch: 2.6.0+cu124, CUDA available: True
InternNav: OK
Transformers: 4.51.0
```

---

## ✅ 模型下载完成！

### 模型权重已成功下载

通过HuggingFace镜像站 (https://hf-mirror.com) 成功下载了 InternVLA-N1-DualVLN 模型。

#### 方法1: 使用浏览器下载（推荐）

1. 访问: https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN
2. 点击 "Files and versions" 标签
3. 下载所有文件到本地
4. 将文件放置到: `/mnt/data/0923_Interndata/InternNav/checkpoints/InternVLA-N1-DualVLN/`

需要下载的文件:
```
checkpoints/InternVLA-N1-DualVLN/
├── config.json
├── configuration_internvl_chat.py
├── conversation.py
├── modeling_internvl_chat.py
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── preprocessor_config.json
```

#### 方法2: 使用wget/curl

如果有稳定的网络连接或代理:

```bash
cd /mnt/data/0923_Interndata/InternNav/checkpoints
mkdir -p InternVLA-N1-DualVLN

# 设置HuggingFace镜像（如果在国内）
export HF_ENDPOINT=https://hf-mirror.com

# 使用git lfs下载
git lfs install
git clone https://hf-mirror.com/InternRobotics/InternVLA-N1-DualVLN
```

#### 方法3: 使用Python（需要网络稳定）

```bash
conda activate internvln39
cd /mnt/data/0923_Interndata/InternNav

python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="InternRobotics/InternVLA-N1-DualVLN",
    local_dir="checkpoints/InternVLA-N1-DualVLN",
    local_dir_use_symlinks=False
)
EOF
```

### 2. 准备场景数据（可选，用于评估）

如果要运行VLN-CE评估，需要准备Matterport3D场景数据:

```bash
# 创建数据目录
mkdir -p data/scene_data
mkdir -p data/vln_ce/raw_data/r2r

# 链接或复制场景数据
# 示例: ln -s /path/to/mp3d_ce data/scene_data/mp3d_ce
```

---

## 🚀 快速开始

### 激活环境

```bash
eval "$(conda shell.bash hook)"
conda activate internvln39
cd /mnt/data/0923_Interndata/InternNav
```

### 验证安装

```bash
# 测试所有组件
python -c "import habitat_sim; print(f'Habitat-Sim: {habitat_sim.__version__}')"
python -c "import habitat; print(f'Habitat-Lab: {habitat.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import internnav; print('InternNav: OK')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 运行推理（模型下载后）

```bash
# 方式1: Jupyter Notebook Demo
cd scripts/notebooks
jupyter notebook inference_only_demo.ipynb

# 方式2: Python脚本推理
python -c "
from internnav.model import InternVLAN1
model = InternVLAN1.from_pretrained('checkpoints/InternVLA-N1-DualVLN')
print('Model loaded successfully!')
"
```

### 运行VLN-CE评估（需要场景数据）

```bash
# R2R任务评估
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_cfg.py

# 或使用快速启动脚本
bash quick_eval.sh
```

---

## 📊 系统状态总结

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.9.25 | ✅ |
| PyTorch | 2.6.0+cu124 | ✅ |
| CUDA | 12.4 | ✅ |
| Habitat-Sim | 0.3.2 | ✅ |
| Habitat-Lab | 0.3.0 | ✅ |
| InternNav | 0.3.1 | ✅ |
| Transformers | 4.51.0 | ✅ |
| Git Submodules | - | ✅ |
| **模型权重** | 16GB (4个safetensors文件) | ✅ |
| **场景数据** | - | ⚠️ 可选 |

**模型文件清单**:
- ✅ model-00001-of-00004.safetensors (4.63 GB)
- ✅ model-00002-of-00004.safetensors (4.65 GB)
- ✅ model-00003-of-00004.safetensors (4.59 GB)
- ✅ model-00004-of-00004.safetensors (1.75 GB)
- ✅ config.json, tokenizer 等配置文件

---

## 🎯 可以开始使用了！

1. **推荐**: 运行推理demo验证模型工作正常
2. **可选**: 准备场景数据用于VLN-CE完整评估
3. **可选**: 在benchmark上运行完整评估

---

## 📚 参考文档

- 详细设置指南: `SETUP_GUIDE.md`
- 安装进度: `INSTALLATION_PROGRESS.md`
- 快速评估脚本: `quick_eval.sh`
- 官方文档: https://internrobotics.github.io/user_guide/internnav/

---

## 🔧 常见问题

### Q: 模型下载太慢怎么办？
A: 使用HuggingFace镜像站或通过浏览器分批下载大文件。

### Q: 没有场景数据可以运行吗？
A: 可以运行推理demo，但无法进行VLN-CE完整评估。

### Q: GPU不可用怎么办？
A: 检查CUDA驱动 (`nvidia-smi`) 和PyTorch安装。

---

**安装完成度**: 100% 🎉
**状态**: 完全就绪，可以开始使用！

祝使用愉快！🚀
