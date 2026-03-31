#!/usr/bin/env python3
import sys
sys.path.append('.')

from habitat.datasets.vln.r2r_vln_dataset import R2RVLNDatasetV1
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default import get_agent_config
import habitat
import gzip
import json

config_path = 'scripts/eval/configs/vln_ce_simple.yaml'
print(f"Loading config from: {config_path}")

config = get_habitat_config(config_path)
print(f"\nDataset config:")
print(f"  Type: {config.habitat.dataset.type}")
print(f"  Split: {config.habitat.dataset.split}")
print(f"  Data path: {config.habitat.dataset.data_path}")
print(f"  Scenes dir: {config.habitat.dataset.scenes_dir}")
print(f"  Content scenes: {config.habitat.dataset.content_scenes}")

# Manually check the data file
data_path_template = config.habitat.dataset.data_path
split = config.habitat.dataset.split
data_path = data_path_template.replace('{split}', split)
print(f"\n  Resolved data path: {data_path}")

try:
    with gzip.open(data_path, 'rt') as f:
        data = json.load(f)
        all_episodes = data['episodes']
        print(f"\n  Total episodes in file: {len(all_episodes)}")

        # Check filtering by content_scenes
        content_scenes = config.habitat.dataset.content_scenes
        filtered = [ep for ep in all_episodes if ep['scene_id'] in content_scenes]
        print(f"  Episodes matching content_scenes {content_scenes}: {len(filtered)}")

        if len(filtered) > 0:
            print(f"\n  First matched episode:")
            print(f"    Episode ID: {filtered[0]['episode_id']}")
            print(f"    Scene ID: {filtered[0]['scene_id']}")
            print(f"    Instruction: {filtered[0]['instruction']['instruction_text'][:100]}...")
except Exception as e:
    print(f"\n  Error reading data file: {e}")
    import traceback
    traceback.print_exc()

# Now try creating dataset object
print(f"\n--- Creating dataset object ---")
try:
    dataset = R2RVLNDatasetV1(config.habitat.dataset)
    print(f"Dataset created!")
    print(f"Number of episodes: {len(dataset.episodes)}")
    if len(dataset.episodes) > 0:
        print(f"First episode: {dataset.episodes[0]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
