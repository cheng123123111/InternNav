from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name="internvla_n1",
        model_settings={
            "mode": "dual_system",
            "model_path": "/vepfs-B区/vlnce/InternNav/checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "vis_debug": False,
            "vis_debug_path": "/vepfs-C/qiancheng/stop_residual_online_unseen100_same_as_viva_20260629/vis_debug",
            "enable_stop_residual_verify": True,
            "stop_residual_adapter_path": "/vepfs-C/qiancheng/stop_residual_adapter_20260629/first2000_scores_multitoken_smoke/stop_residual_adapter.pt",
            "stop_residual_accept_threshold": 0.5,
            "stop_residual_delta_scale": 1.0,
            "stop_residual_reject_on_error": False,
            "stop_residual_use_distance_progress": True,
            "qwen_stop_reject_action": "lookdown",
        },
    ),
    env=EnvCfg(
        env_type="habitat",
        env_settings={
            "config_path": "scripts/eval/configs/vln_r2r_progress_unseen100_same_as_viva.yaml",
        },
    ),
    eval_type="habitat_vln",
    eval_settings={
        "output_path": "/vepfs-C/qiancheng/stop_residual_online_unseen100_same_as_viva_20260629",
        "save_video": False,
        "save_front_video_only": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2349",
        "dist_url": "env://",
    },
)
