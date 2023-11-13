
from vehicle.utils.config import warn_task_name




def parse_task(cfg, cfg_train, sim_param):
    device_id=cfg.device_id
    rl_device=cfg.rl_device
    try:
        task=eval(cfg.task)(
            cfg=cfg,
            sim_param=sim_param,
            physics_engine=cfg.physics_engine,
            device_type=cfg.device,
            device_id=device_id,
            headless=cfg.headless,
            is_multi_agent=False)
    except NameError as e:
        print(e)
        warn_task_name()

    env=VecTaskPython(task, rl_device)