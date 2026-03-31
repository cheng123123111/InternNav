#!/bin/bash
# InternVLA-N1 快速评估脚本

# 设置环境
eval "$(conda shell.bash hook)"
conda activate internvln

# 设置工作目录
cd /mnt/data/0923_Interndata/InternNav

# 检查模型是否存在
if [ ! -d "checkpoints/InternVLA-N1-DualVLN" ]; then
    echo "❌ 模型未找到！请先下载模型："
    echo "   mkdir -p checkpoints && cd checkpoints"
    echo "   git lfs install"
    echo "   git clone https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN"
    exit 1
fi

# 检查数据是否存在
if [ ! -d "data/scene_data/mp3d_ce" ]; then
    echo "❌ 场景数据未找到！请确保数据在 data/scene_data/mp3d_ce"
    exit 1
fi

echo "✅ 环境检查通过，开始评估..."

# 运行评估 (可选配置文件)
CONFIG=${1:-"scripts/eval/configs/habitat_dual_system_cfg.py"}

python scripts/eval/eval.py --config $CONFIG

echo "✅ 评估完成！结果保存在 logs/habitat/ 目录"
