# VLN-CE Quick Test with Habitat - Single Scene Validation
# Tests scene: 8194nk5LbLH (office scene with 39 episodes in val_unseen)

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",  # dual_system mode for VLN
            "model_path": "checkpoints/InternVLA-N1-DualVLN",

            # Critical parameters from official config
            "infer_mode": "partial_async",  # KEY: async mode for better performance
            "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            "width": 640,
            "height": 480,
            "hfov": 79,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "num_frames": 32,
            "num_history": 8,
            "num_future_steps": 4,
            "predict_step_nums": 32,
            "continuous_traj": True,
            "device": "cuda:0",

            # Visualization
            "vis_debug": True,
            "vis_debug_path": "./logs/vlnce_medium100_async/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',  # Use Habitat instead of InterNUtopia
        env_settings={
            # Use complete config with all measurements
            'config_path': 'scripts/eval/configs/vln_ce_complete.yaml',
        },
    ),
    eval_type='habitat_vln',  # Use habitat_vln evaluation type
    eval_settings={
        "output_path": "./logs/vlnce_medium100_async",
        "save_video": True,  # Save videos for verification
        "epoch": 0,
        "max_steps_per_episode": 1000,
        # Distributed settings (not used for single scene test)
        "port": "2333",
        "dist_url": "env://",
    },
)
