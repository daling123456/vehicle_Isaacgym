import sys
sys.path.append("/home/mutong/RobotDoc/vehicle_Isaacgym")
import os
import hydra


from omegaconf import DictConfig, OmegaConf

import vehicle
from vehicle.utils.config import set_np_formatting
from vehicle.algorithms.rl.ppo import PPO
from vehicle.algorithms.rl.dpg import DDPG


def process_sarl(cfg, env, cfg_train, cfg_task):
    logdir = cfg.logdir + "_seed{}".format(cfg["seed"])
    algorithm = eval(cfg.algo_name.upper())(
        vec_env=env,
        cfg_train=cfg_train,
        device=cfg.rl_device,
        log_dir=logdir,
        sampler=cfg_train["learn"].get("sampler",'sequential'),
        is_testing=False,
        print_log=cfg_train["learn"]["print_log"],
        apply_reset=False,
        asymmetric=(env.num_states>0)
    )

    return algorithm

@hydra.main(config_name="config", config_path="./cfg")
def train(cfg: DictConfig):

    cfg_train=cfg.train
    cfg_task=cfg.task
    logdir=cfg.logdir
    env=vehicle.make(seed=cfg.seed,
                     task=cfg.task_name,
                     num_envs=cfg.task.env.numEnvs,
                     sim_device=cfg.sim_device,
                     rl_device=cfg.rl_device,
                     graphics_device_id=cfg.graphics_device_id,
                     headless=cfg.headless,
                     cfg_task=cfg_task,
                     cfg=cfg,
                     force_render=cfg.force_render
                     )
    # env.stand_by()
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    algo=process_sarl(cfg, env, cfg_train, logdir,)
    iterations=cfg_train["learn"]["max_iterations"]
    algo.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])



if __name__=="__main__":
    set_np_formatting()

    train()