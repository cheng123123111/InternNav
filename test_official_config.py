# 测试官方配置能否加载
import sys
sys.path.insert(0, '.')
from habitat_baselines.config.default import get_config as get_habitat_config

config = get_habitat_config('scripts/eval/configs/vln_r2r.yaml')
print("✅ 官方配置加载成功！")
print(f"数据路径: {config.habitat.dataset.data_path}")
print(f"场景路径: {config.habitat.dataset.scenes_dir}")
print(f"split: {config.habitat.dataset.split}")
