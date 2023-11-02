# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



# from Pi_robot.tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
# from Pi_robot.tasks.base.vec_task_rlgames import RLgamesVecTaskPython

from Pi_robot.utils.config import warn_task_name
from Pi_robot.tasks.Pi_terrain import PiTerrain

import json


def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "Python":
        print("Python")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env=task
        return task, env

    elif args.task_type == "RLgames":
        print("Task type: RLgames")
        try:
            task = eval(args.task)(
                cfg=cfg,
                rl_device=cfg["rl_device"],
                sim_device=cfg["sim_device"],
                graphics_device_id=cfg["graphics_device_id"],
                virtual_screen_capture=cfg["capture_video"],
                force_render=cfg["force_render"],
                # sim_params=sim_params,
                # physics_engine=args.physics_engine,
                # device_type=args.device,
                # device_id=device_id,
                headless=cfg['headless'])
                # is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        # env = RLgamesVecTaskPython(task, rl_device)
        env=task
    return task, env


