# VLN-CE Simple Test - Minimal config to test if basic setup works
# This is a simplified version to diagnose performance issues

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            # Try WITHOUT these parameters first to see if they cause the issue:
            # "infer_mode": "partial_async",
            # "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            # "num_frames": 32,
            # "num_future_steps": 4,
            # "predict_step_nums": 32,
            # "continuous_traj": True,
            "vis_debug": True,
            "vis_debug_path": "./logs/vlnce_simple_test/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_ce_simple_single_scene.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/vlnce_simple_test",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 1000,
        "port": "2333",
        "dist_url": "env://",
    },
)
