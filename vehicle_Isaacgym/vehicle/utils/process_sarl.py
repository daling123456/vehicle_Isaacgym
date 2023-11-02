


def process_sarl(cfg, env, cfg_train, logdir):

    logdir=logdir+"_seed{}".format(env.task.cfg["seed"])
    algorithm=eval(cfg.train.algo.name)(
        cfg_train,
        device=env.rl_device,
        log_dir=logdir,

    )

    return algorithm