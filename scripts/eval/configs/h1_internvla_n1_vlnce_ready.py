# VLN-CE评估配置 - 数据已准备好
# VLN-CE数据: /mnt/data/0923_Interndata/vln_ce/raw_data/r2r
# 场景文件: /mnt/data/VL-LN-Bench/scene_datasets/mp3d

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8023,
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1-DualVLN",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'infer_mode': 'partial_async',
            # debug
            'vis_debug': True,
            'vis_debug_path': './logs/vlnce_eval/vis_debug',
        },
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='vlnce_eval',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
            'max_step': 1000,
        },
        scene=SceneCfg(
            scene_type='mp3d',
            # 指向实际的场景文件路径
            scene_data_dir='/mnt/data/VL-LN-Bench/scene_datasets/mp3d',
        ),
        robot_name='h1',
        robot_flash=True,
        flash_collision=False,
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            # 指向实际的VLN-CE数据路径
            'base_data_dir': '/mnt/data/0923_Interndata/vln_ce/raw_data/r2r',

            # 选择要评估的split（可改为 'val_seen', 'train'）
            'split_data_types': ['val_unseen'],

            # 是否过滤楼梯场景
            'filter_stairs': True,

            # 可选：只评估特定场景（注释掉则评估所有）
            # 'selected_scans': ['8194nk5LbLH', 'pLe4wQe7qrG'],
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True,
        'use_agent_server': False,
    },
)
