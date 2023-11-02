import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from .model import (Actor, Critic)
from .memory import SequentialMemory
from .random_process import OrnsteinUhlenbeckProcess
from .util import *
import glob
from vehicle.algorithms.low_level_control.inverse_locomotion import conventional_control
from vehicle.algorithms.low_level_control.cpg import CPG
# from ipdb import set_trace as debug
from copy import deepcopy
import xlwt
import matplotlib.pyplot as plt

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, vec_env, device, cfg_train, log_dir, sampler, is_testing, print_log, apply_reset, asymmetric):


        self.seed(cfg_train.seed)

        self.cfg_train=cfg_train
        self.env=vec_env

        self.nb_states = vec_env.obs_space.shape[0]
        self.nb_actions= vec_env.act_space.shape[0]

        # Create Actor and Critic Network TODO Make this a proper model
        # net_cfg = {
        #     'hidden1':126,
        #     'hidden2':126,
        #     'init_w':0.003
        # }

        net_cfg = {
            'hidden1':400,
            'hidden2':300,
            'init_w':0.003
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=0.0001)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=6000000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, theta=0.15, mu=0.0, sigma=0.2)

        # Hyper-parameters
        self.batch_size = 64
        self.tau = 0.001
        self.discount = 0.99
        self.depsilon = 1.0 / 150000

        #
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.output="vehicle/weights/Pi/cpg"

        self.cpg_gait_mode ={
                                "walk": 1,
                                "trot": 2,
                                "pace": 3,
                                "gallop": 4
                            }
        self.episode_reward_list = []



        #
        if USE_CUDA: self.cuda()

    def run(self,num_learning_iterations, log_interval):
        step=episode=episode_steps=0
        output = self.output + "_*"
        output = glob.glob(output)
        if len(output):
            output = max(output, key=os.path.getctime)
        else:
            output = str(output)

        if os.path.exists(output):
            print("############### is something here! ###############")
            self.load_weights(output)
            step = int(output.split("_")[-1].split(".")[0])
        episode_reward = np.zeros(64)
        last_step = 0
        observation = None

        ###############步态模式############
        gait_mode = self.cpg_gait_mode["trot"]

        cpg = CPG()
        norm_control = conventional_control()
        time = np.arange(0, 200, 0.02)
        data = cpg.hopf(gait_mode, 0.02, time)
        # data[:, 0:4] = 0.1 * data[:, 0:4]
        # data[:, 8:12] = 0.3 - data[:, 8:12]

        i = 2000
        while step < num_learning_iterations:
            motor_leg1 = norm_control.inverse_locomotion(data[i, 0], data[i, 8])
            motor_leg2 = norm_control.inverse_locomotion(data[i, 1], data[i, 9])
            motor_leg3 = norm_control.inverse_locomotion(data[i, 2], data[i, 10])
            motor_leg4 = norm_control.inverse_locomotion(data[i, 3], data[i, 11])
            action = motor_leg1 + motor_leg2 + motor_leg3 + motor_leg4

            actions=torch.zeros(self.env.num_envs, self.nb_actions, dtype=torch.float32)
            actions[:, :]= torch.tensor(action, dtype=torch.float32)

            # reset if it is the start of episode
            if observation is None:
                observation = deepcopy(self.env.reset())
                observation = observation["obs"]
                self.reset(observation)

            # agent pick action ...
            if step <= 100:  # Warmup
                actions += self.random_action() * 0.2
            else:
                print("select action")
                actions += self.select_action(observation) * 0.2

            # env response with next_observation, reward, terminate_info
            actions=torch.tensor(actions,dtype=torch.float32)

            observation2, reward, done, info = self.env.step(actions)
            observation2 = observation2['obs']
            observation2 = deepcopy(observation2)

            if self.cfg_train.learn.max_episode_length and episode_steps >= self.cfg_train.learn.max_episode_length - 1:
                done = True

            # agent observe and update policy
            self.observe(reward, observation2, done)
            if step > 100:  # Warmup
                self.update_policy()

            # update
            i += 1
            step += 1
            episode_steps += 1
            reward=reward.cpu().numpy()

            episode_reward += reward


            observation = deepcopy(observation2)
            # print(type(observation))
            obs=observation.cpu().numpy()

            for j in range(self.nb_actions):
                if done[j]:  # end of episode
                    self.memory.append(
                        obs[j],
                        self.select_action(observation)[j],
                        0.,
                        False
                    )

                    """把回合奖励值加入奖励值列表中"""
                    self.episode_reward_list.append(episode_reward[j])
                    if True: prGreen('#{}: episode_reward:{} steps:{}\n'
                                      '     average_reward:{}'.format(episode, episode_reward[j], step - last_step,
                                                                      np.mean(self.episode_reward_list[-128:])))

                    if np.mean(self.episode_reward_list[-64:]) > 6.0:
                        output = output + f"_{episode}"
                        self.save_model(output)
                        break

                    # reset
                    make_tensor=torch.ones_like(observation)
                    make_tensor[j,:]=0.0
                    observation=observation* make_tensor
                    episode_steps = 0
                    episode_reward = np.zeros(64)
                    episode += 1
                    last_step = step
                    i = 0

        ###############保存奖励值写入excel表格################
        write_to_excel(self.episode_reward_list)
        plot(self.episode_reward_list)



    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions,)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # print("observation",s_t)

        action = to_numpy(
            self.actor(s_t)
        )
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

###########写入excel表格#############
def write_to_excel(episode_reward_list, excel_name="episode_reward.xls", ):
    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('episode_reward', cell_overwrite_ok=True)  # 创建sheet工作表
    sheet1.write(0, 0, "episode")
    sheet1.write(0, 1, "reward")
    sheet1.write(0, 2, "average_reward")
    for i in range(len(episode_reward_list)):
        sheet1.write(i+1, 0, i+1)    # 写入数据参数对应 行, 列, 值
        sheet1.write(i+1, 1, episode_reward_list[i])
        sheet1.write(i+1, 2, np.mean(episode_reward_list[i-128:i]))
    f.save(excel_name)  # 保存.xls到当前工作目录

###########绘制回合奖励值图###########
def plot(episode_reward_list):
    # 开始画图
    plt.title('episode reward')
    plt.plot(range(len(episode_reward_list)), episode_reward_list, color='red', label='episode reward')
    plt.legend()  # 显示图例

    plt.xlabel('episode')
    plt.ylabel('episode reward')
    plt.show()