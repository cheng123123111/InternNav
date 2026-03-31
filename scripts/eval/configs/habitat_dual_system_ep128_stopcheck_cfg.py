from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 320,
            "resize_h": 320,
            "max_new_tokens": 256,
            "vis_debug": True,
            "vis_debug_path": "./logs/habitat/ep128_stopcheck/vis_debug",
            "stop_check": True,
            "stop_reject_action": "forward",
            "stop_check_max_new_tokens": 8,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_ep128.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/ep128_stopcheck",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
    },
)
