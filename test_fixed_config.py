import sys
sys.path.insert(0, '.')
from habitat_baselines.config.default import get_config as get_habitat_config

config = get_habitat_config('scripts/eval/configs/vln_r2r_fixed.yaml')
print("✅ 修复后的配置加载成功！")
print(f"数据路径: {config.habitat.dataset.data_path}")
print(f"场景路径: {config.habitat.dataset.scenes_dir}")
print(f"Success距离: {config.habitat.task.measurements.success.success_distance}")
