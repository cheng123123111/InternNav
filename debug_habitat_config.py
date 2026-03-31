#!/usr/bin/env python3
import sys
sys.path.append('.')

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default import get_agent_config
import habitat

config_path = 'scripts/eval/configs/vln_ce_simple.yaml'
print(f"Loading config from: {config_path}")

config = get_habitat_config(config_path)
print(f"\nConfig loaded successfully!")
print(f"Dataset type: {config.habitat.dataset.type}")
print(f"Dataset split: {config.habitat.dataset.split}")
print(f"Data path: {config.habitat.dataset.data_path}")
print(f"Scenes dir: {config.habitat.dataset.scenes_dir}")
print(f"Content scenes: {config.habitat.dataset.content_scenes}")

# Try to load the dataset
print(f"\n--- Attempting to create environment ---")
try:
    env = habitat.Env(config=config)
    print(f"Environment created successfully!")
    print(f"Number of episodes: {len(env.episodes)}")
    if len(env.episodes) > 0:
        print(f"First episode: {env.episodes[0]}")
except Exception as e:
    print(f"Error creating environment: {e}")
    import traceback
    traceback.print_exc()
