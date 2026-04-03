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
            "vis_debug": True,
            "vis_debug_path": "./logs/habitat/fail4_longclipelement/vis_debug",
            "enable_qwen_stop_verify": False,
            "enable_longclip_stop_verify": True,
            "longclip_stop_model_path": "checkpoints/clip-long/longclip-B.pt",
            "longclip_stop_weight_path": "/root/code/InternNav/tmp_stop_phrase/run_longclip_element_v1/best_longclip_stop_elements.pt",
            "longclip_stop_threshold": 0.0,
            "qwen_stop_reject_action": "lookdown",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_fail4_local.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/fail4_longclipelement",
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
    },
)
