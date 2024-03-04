import gym
from datetime import datetime

# from vehicle.utils.reformat import omegaconf_to_dict
from vehicle_Isaacgym.vehicle.tasks import task_map
from vehicle_Isaacgym.vehicle.utils.rlgames_utils import get_rlgames_env_creator

time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make(seed,
         task,
         num_envs,
         sim_device,
         rl_device,
         graphics_device_id,
         headless,
         cfg,
         cfg_task,
         force_render=True,
         capture_video=False
         ):

    create_env=get_rlgames_env_creator(seed=seed,
                                       task_config=cfg_task,
                                       task_name=cfg_task["name"],
                                       sim_device=sim_device,
                                       rl_device=rl_device,
                                       graphics_device_id=graphics_device_id,
                                       headless=headless,
                                       force_render=force_render
                                       )

    # create_env=task_map[cfg_task["name"]](
    #         cfg=cfg_task,
    #         rl_device=rl_device,
    #         sim_device=sim_device,
    #         graphics_device_id=graphics_device_id,
    #         headless=headless,
    #         virtual_screen_capture=virtual_screen_capture,
    #         force_render=force_render,
    #     )


    if capture_video:
        run_name = f"{cfg.wandb_name}_{time_str}"
        create_env.is_vector_env= True
        create_env=gym.Wrapper.RecordVideo(
            create_env,
            f"videos/{run_name}",
            step_trigger=lambda step: step%cfg.capture_video_freq==0,
            video_length=cfg.capture_video_len
        )

    return create_env