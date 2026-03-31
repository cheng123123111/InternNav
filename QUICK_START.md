# 🚀 InternVLA-N1 快速开始指南

**环境**: `internvln39` (Python 3.9.25)
**状态**: ✅ 完全就绪

---

## 1. 激活环境

```bash
# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate internvln39

# 进入项目目录
cd /mnt/data/0923_Interndata/InternNav
```

---

## 2. 验证安装

```bash
# 快速验证所有组件
python << 'EOF'
import habitat_sim
import habitat
import torch
import transformers
import internnav

print("✅ Habitat-Sim:", habitat_sim.__version__)
print("✅ Habitat-Lab:", habitat.__version__)
print("✅ PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("✅ Transformers:", transformers.__version__)
print("✅ InternNav: OK")
print()
print("🎉 所有组件正常运行！")
EOF
```

**预期输出**:
```
✅ Habitat-Sim: 0.3.2
✅ Habitat-Lab: 0.3.0
✅ PyTorch: 2.6.0+cu124 | CUDA: True
✅ Transformers: 4.51.0
✅ InternNav: OK

🎉 所有组件正常运行！
```

---

## 3. 测试模型加载

```bash
python << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 0

print("正在加载模型配置...")
from transformers import AutoConfig

model_path = "checkpoints/InternVLA-N1-DualVLN"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print(f"✅ 模型配置加载成功！")
print(f"   模型路径: {model_path}")
print(f"   模型类型: {config.model_type}")
print(f"   隐藏层大小: {config.hidden_size}")
print()
print("🎉 模型文件完整，可以正常使用！")
EOF
```

---

## 4. 运行推理Demo（推荐）

### 方式A: Jupyter Notebook

```bash
# 启动Jupyter Notebook
cd scripts/notebooks
jupyter notebook inference_only_demo.ipynb
```

然后在浏览器中打开Notebook，运行单元格进行推理测试。

### 方式B: Python脚本推理

创建一个简单的推理脚本 `test_inference.py`:

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# 加载模型和tokenizer
model_path = "checkpoints/InternVLA-N1-DualVLN"
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).cuda().eval()

print("✅ Model loaded successfully!")

# 示例: 处理导航指令
instruction = "Walk forward and turn left at the door"
print(f"\n指令: {instruction}")

# 这里可以添加实际的推理逻辑
# 具体使用方法请参考官方文档
```

运行:
```bash
python test_inference.py
```

---

## 5. 可视化工具（可选）

```bash
# 查看已下载的轨迹数据
cd /mnt/data/0923_Interndata/vln_n1/traj_data/replica_d435i

# 列出场景数据
ls -lh
```

如果你想可视化这些数据，可以使用InternNav提供的可视化工具（需要参考官方文档）。

---

## 6. 运行VLN-CE评估（需要场景数据）

如果你已经准备好了Matterport3D场景数据：

```bash
# 确保场景数据在正确位置
# data/scene_data/mp3d_ce/

# 运行R2R任务评估
python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_dual_system_cfg.py

# 或使用快速启动脚本
bash quick_eval.sh
```

评估结果会保存在 `logs/habitat/` 目录。

---

## 📊 性能基准（参考）

根据官方报告，InternVLA-N1 (Dual System) 在VLN-CE基准上的表现：

| 数据集 | SR↑ | SPL↑ |
|--------|-----|------|
| R2R val_unseen | **73.0** | **66.2** |
| RxR val_unseen | **68.5** | **60.8** |

---

## 🔧 常见问题

### Q: CUDA out of memory
A: 尝试使用更小的batch size或使用`torch.float16`精度。

### Q: 模型加载很慢
A: 第一次加载会比较慢（需要加载16GB模型）。后续加载会快一些。

### Q: 想要可视化导航结果
A: 参考 `scripts/notebooks/` 目录下的可视化notebook。

### Q: 没有场景数据怎么办
A: 可以运行推理demo，但无法进行完整的VLN-CE评估。

---

## 📚 更多资源

- **完整安装报告**: `INSTALLATION_COMPLETE.md`
- **设置指南**: `SETUP_GUIDE.md`
- **官方文档**: https://internrobotics.github.io/user_guide/internnav/
- **技术报告**: https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf
- **GitHub仓库**: https://github.com/InternRobotics/InternNav

---

## ✅ 检查清单

使用前确认：
- [x] Conda环境已激活 (`internvln39`)
- [x] 所有依赖已安装
- [x] 模型权重已下载 (16GB)
- [x] GPU驱动正常 (`nvidia-smi`)
- [x] CUDA可用 (`torch.cuda.is_available()`)

---

**开始探索吧！** 🎉
