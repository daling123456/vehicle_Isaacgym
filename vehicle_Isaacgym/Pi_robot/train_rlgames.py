import yaml
import sys
sys.path.append("/home/mutong/RobotDoc/vehicle_Isaacgym")

import Pi_robot
from Pi_robot.utils.parse_task import parse_task
from Pi_robot.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg

from rl_games.common.algo_observer import AlgoObserver, IsaacAlgoObserver
from rl_games.torch_runner import Runner


if __name__=="__main__":
    set_np_formatting()
    args=get_args(use_rlg_config=True)
    if args.checkpoint=="Base":
        args.checkpoint=""

    #配置算法config
    if args.algo=="ppo":
        config_name="cfg/{}/ppo_continuous.yaml".format(args.algo)
    elif args.algo=="ppo_lstm":
        config_name="cfg/{}/ppo_continuous_lstm.yaml".format(args.algo)
    else:
        print("We don't support this config in RL-games now")

    args.task_type= "RLgames"
    print('Loading config:', config_name)

    args.cfg_train=config_name
    cfg, cfg_train, logdir= load_cfg(args, use_rlg_config=True)
    sim_params=parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed",-1), cfg_train.get("torch_deterministic", False))

    task, env= parse_task(args, cfg, cfg_train, sim_params)

    with open(config_name, 'r') as stream:
        rlgames_cfg=yaml.safe_load(stream)
        rlgames_cfg['params']['config']['name']=args.task
        rlgames_cfg['params']['config']['num_actors']=env.num_environments
        rlgames_cfg['params']['seed']=42
        rlgames_cfg['params']['config']['env_config']['seed']=42
        rlgames_cfg['params']['config']['vec_env']=env
        rlgames_cfg['params']['config']['env_info']= env.get_env_info()
        rlgames_cfg['params']['config']['minibatch_size']= rlgames_cfg['params']['config']['minibatch_size']* env.num_environments

    vargs=vars(args)
    algo_observer= IsaacAlgoObserver()
    runner= Runner(algo_observer=algo_observer)
    runner.load(rlgames_cfg)
    runner.reset()
    runner.run(vargs)