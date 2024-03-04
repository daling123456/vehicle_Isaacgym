import os
import statistics
import numpy as np
import time
from collections import deque
from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
# from torch import Tensor
from gym.spaces import Space
from vehicle_Isaacgym.vehicle.algorithms.rl.dpg import MLPActorCritic
from vehicle_Isaacgym.vehicle.algorithms.rl.dpg import ReplayBuffer
from vehicle_Isaacgym.vehicle.algorithms.low_level_control.cpg import CPG
from vehicle_Isaacgym.vehicle.algorithms.low_level_control.inverse_locomotion import conventional_control
from torch.utils.tensorboard import SummaryWriter

class DDPG:
    def __init__(self, vec_env, cfg_train, is_testing, sampler, apply_reset,print_log, asymmetric, device='cpu', log_dir='run',):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.device = device
        learn_cfg = cfg_train["learn"]
        self.learning_rate = learn_cfg["learning_rate"]
        self.act_limit = vec_env.action_space.high[0]
        self.action_shape=vec_env.action_space
        self.observation_shape=vec_env.observation_space
        ###########DDPG parameters############
        self.act_noise = learn_cfg["act_noise"]
        self.num_transitions_per_env = learn_cfg["nsteps"]
        self.replay_size = learn_cfg["replay_size"]
        self.batch_size=learn_cfg["batch_size"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.num_learning_epoches = learn_cfg["noptepochs"]
        self.polyak = learn_cfg["polyak"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.gamma = learn_cfg["gamma"]
        self.target_noise = learn_cfg["target_noise"]
        self.act_noise = learn_cfg["act_noise"]
        self.noise_clip = learn_cfg["noise_clip"]
        self.warm_up=True

        self.print_log = print_log
        ###########DDPG components############
        self.vec_env = vec_env
        net_hidden_layer = dict(hidden_sizes=[learn_cfg["hidden_nodes"]] * learn_cfg["hidden_layer"])
        self.actor_critic = MLPActorCritic(vec_env.observation_space, vec_env.action_space, self.act_noise, self.device,
                                           **net_hidden_layer).to(self.device)
        self.actor_cricti_target = deepcopy(self.actor_critic)

        self.storage=ReplayBuffer(num_envs=vec_env.num_envs, replay_size=self.replay_size, batch_size=self.batch_size,
                                   obs_shape=self.observation_shape.shape, actions_shape=self.action_shape.shape, device=self.device)
        for p in self.actor_cricti_target.parameters():
            p.requires_grad = False

        self.critic_params=self.actor_critic.critic.parameters()
        self.actor_optimizer=Adam(self.actor_critic.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer=Adam(self.critic_params, lr=self.learning_rate)

        self.cpg_data=self.cpg_hopf()

        self.log_dir = log_dir
        self.is_testing = is_testing

        self.current_learning_iteration = 0
        self.asset_reply = False

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()['obs']
        if self.is_testing:
            while True:
                with torch.no_grad():
                    # if self.apply_reset:

                    # Compute the action
                    actions = self.actor_critic.act(current_obs)
                    actions += self.CPG_actions()
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs['obs'])

        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos=[]

                for _ in range(self.num_transitions_per_env):
                    # if self.apply_reset:
                    #     current_obs = self.vec_env.reset()
                    actions=self.actor_critic.act(current_obs, deterministic=False)
                    actions += self.CPG_actions()

                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    self.storage.add_transitions(current_obs, actions, rews, next_obs['obs'], dones)
                    current_obs.copy_(next_obs['obs'])
                    ep_infos.append(infos)


                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    if self.storage.step>self.batch_size:
                        self.warm_up=False
                    if not self.warm_up:
                        self.update()
                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                stop=time.time()
                collection_time=stop-start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()
                print("colletion_time", collection_time)

                start=stop
                if self.warm_up == False:
                    stop=time.time()
                    learn_time=stop-start
                    # if self.print_log:
                    #     self.log(locals())
                    if it % log_interval ==0:
                        self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))

                    ep_infos.clear()
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))



    def update(self):
        mean_value_loss=0
        mean_surrogate_loss=0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epoches):
            for indices in batch:
                obs_batch=self.storage.observations[indices]
                rewards_batch=self.storage.rewards[indices]
                nextobs_batch=self.storage.next_observations[indices]
                actions_batch=self.storage.actions[indices]
                dones_batch=self.storage.dones[indices]

            data={'obs': obs_batch,
                  'act': actions_batch,
                  'rew': rewards_batch,
                  'n_obs': nextobs_batch,
                  'dones': dones_batch}



            self.critic_optimizer.zero_grad()
            loss_critic=self.compute_loss_critic(data)
            loss_critic.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
            self.critic_optimizer.step()

            # Record things
            mean_value_loss+=loss_critic.item()

            ####Freeze Q net so you don't waste computational effort########
            for param in self.critic_params:
                param.requires_grad=False
            self.actor_optimizer.zero_grad()
            loss_actor=self.compute_loss_actor(data)
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
            self.actor_optimizer.step()

            mean_surrogate_loss += loss_actor.item()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for param in self.critic_params:
                param.requires_grad=True

            #########target 网络参数软更新############
            with torch.no_grad():
                for param,param_target in zip(self.actor_critic.parameters(), self.actor_cricti_target.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    param_target.data.mul_(self.polyak)
                    param_target.data.add_((1-self.polyak)*param.data)
        num_updates=self.num_learning_epoches*self.num_mini_batches
        mean_value_loss/=num_updates
        mean_surrogate_loss/=num_updates

        return mean_surrogate_loss,mean_value_loss


    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()


    def save(self, path):
        torch.save(self.actor_critic.state_dict(),path)


    def compute_loss_actor(self, data):
        obs = data['obs']
        q_actor=self.actor_critic.critic(obs, self.actor_critic.actor(obs))
        loss_actor=-q_actor.mean()
        return loss_actor


    def compute_loss_critic(self, data):
        obs, action, reward, next_obs, done = data['obs'], data['act'], data['rew'], data['n_obs'], data['dones']
        q_critic=self.actor_critic.critic(obs, action)

        with torch.no_grad():
            act_target=self.actor_cricti_target.actor(next_obs)

            epsilon = torch.randn_like(act_target)* self.target_noise
            epsilon=torch.clamp(epsilon, -self.noise_clip, self.noise_clip)

            next_action=act_target+epsilon
            next_action=torch.clamp(next_action, -self.act_limit, self.act_limit)

            #####target Q value##########
            q_actor_target=self.actor_cricti_target.critic(next_obs, next_action)

            backup= reward +self.gamma * (1-done)*q_actor_target

        loss_critic=((q_critic-backup)**2).mean()
        return loss_critic


    def cpg_hopf(self):
        self.cpg_gait_mode = {
            "walk": 1,
            "trot": 2,
            "pace": 3,
            "gallop": 4
        }
        ###############步态模式############
        gait_mode = self.cpg_gait_mode["trot"]
        cpg = CPG()
        timestep = np.arange(0, 200, 0.02)
        ############计算一次时间长达0.6s
        data = cpg.hopf(gait_mode, 0.02, timestep)
        return data



    def CPG_actions(self):

        norm_control = conventional_control()

        # data[:, 0:4] = 0.1 * data[:, 0:4]
        # data[:, 8:12] = 0.3 - data[:, 8:12]
        data=self.cpg_data
        i = 2000

        motor_leg1 = norm_control.inverse_locomotion(data[i, 0], data[i, 8])
        motor_leg2 = norm_control.inverse_locomotion(data[i, 1], data[i, 9])
        motor_leg3 = norm_control.inverse_locomotion(data[i, 2], data[i, 10])
        motor_leg4 = norm_control.inverse_locomotion(data[i, 3], data[i, 11])
        motor_leg2[1] = -motor_leg2[1]
        motor_leg2[2] = -motor_leg2[2]
        motor_leg3[2] = -motor_leg3[2]
        motor_leg4[1] = -motor_leg4[1]
        action = (motor_leg1 + motor_leg2 + motor_leg3 + motor_leg4)
        action = [3 * value / 4 for value in action]

        actions = torch.zeros(self.vec_env.num_envs, self.vec_env.action_space.shape[0], dtype=torch.float32)
        actions[:, :] = torch.tensor(action, dtype=torch.float32)
        ########需要赋值才能到device#########
        actions = actions.to(self.device)

        return actions


    def log(self, locs, width=80, pad=35):
        """
        print training info
        :param locs:
        :param width:
        :param pad:
        :return:
        """

        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)