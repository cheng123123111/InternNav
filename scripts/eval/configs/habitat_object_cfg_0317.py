from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg, TaskCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name="dialog",
        ckpt_path="",
        model_settings={
            "mode": "system2",
            "dialog_enabled": False,
            "model_path": "/mnt/data/0923_Interndata/VL-LN-Bench/base_model/iign",
            "append_look_down": True,
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 128,
        },
    ),
    env=EnvCfg(
        env_type="habitat_vlln",
        env_settings={
            "habitat_config_path": "scripts/eval/configs/objectnav_hm3d_0317.yaml",
        },
    ),
    task=TaskCfg(
        task_name="objectnav",
    ),
    eval_type="habitat_dialog",
    eval_settings={
        "output_path": "./logs/habitat/object_0317",
        "epoch": 0,
        "max_steps_per_episode": 500,
        "eval_split": "val",
        "turn": 5,
        "save_video": True,
        "base_url": "http://35.220.164.252:3888/v1",
        "model_name": "gpt-4o",
        "openai_api_key": "internnav/habitat_extensions/vlln/simple_npc/api_key.txt",
        "scene_summary": "/mnt/data/0923_Interndata/VL-LN-Bench/raw_data/mp3d/scene_summary",
        "port": "2333",
        "dist_url": "env://",
    },
)
