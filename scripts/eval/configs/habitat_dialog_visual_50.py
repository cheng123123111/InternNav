from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, TaskCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name="dialog",
        ckpt_path="",
        model_settings={
            "mode": "system2",
            "dialog_enabled": True,
            "model_path": "/mnt/data/0923_Interndata/VL-LN-Bench/base_model/iign",
            "append_look_down": False,
            "num_history": 8,
            "resize_w": 320,
            "resize_h": 320,
            "max_new_tokens": 64,
        },
    ),
    env=EnvCfg(
        env_type="habitat_vlln",
        env_settings={
            "habitat_config_path": "scripts/eval/configs/instance_dialog_50.yaml",
        },
    ),
    task=TaskCfg(task_name="instance_dialog"),
    eval_type="habitat_dialog",
    eval_settings={
        "output_path": "./logs/habitat/dialog_visual_50",
        "epoch": 0,
        "max_steps_per_episode": 500,
        "eval_split": "unseen_mini",
        "turn": 5,
        "save_video": True,
        "base_url": "https://api.chatanywhere.tech/v1",
        "model_name": "gpt-4o",
        "openai_api_key": "internnav/habitat_extensions/vlln/simple_npc/api_key.txt",
        "scene_summary": "/mnt/data/0923_Interndata/VL-LN-Bench/raw_data/mp3d/scene_summary",
        "port": "2333",
        "dist_url": "env://",
    },
)
