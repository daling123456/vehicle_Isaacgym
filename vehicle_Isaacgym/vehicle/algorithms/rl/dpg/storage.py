import random

import torch


class ReplayBuffer:
    def __init__(self, num_envs, replay_size, batch_size, obs_shape, actions_shape, device='cpu'):
        self.device=device

        self.observations=torch.zeros(replay_size, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(replay_size, num_envs, 1, device=self.device)
        self.next_observations=torch.zeros(replay_size,num_envs, *obs_shape, device=self.device)
        self.actions=torch.zeros(replay_size,num_envs, *actions_shape, device=device)
        self.dones=torch.zeros(replay_size, num_envs, 1, device=self.device).byte()

        self.replay_size=replay_size
        self.batch_size=batch_size
        self.num_envs=num_envs
        self.fullfill=False

        self.step=0


    #######将得到的transition 放入buffer中#########
    def add_transitions(self, observations, actions, rewards, next_obs, dones):
        if self.step >= self.replay_size:
            self.step = (self.step + 1) % self.replay_size
            self.fullfill = True

        self.observations[self.step].copy_(observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.next_observations[self.step].copy_(next_obs)
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.step += 1


    #############得到每段轨迹的平均长度和平均奖励 ##############
    def get_statistics(self):
        done = self.dones.cpu()
        done[-1]=1
        flat_done=done.permute(1,0,2).reshape(-1,1)
        done_indices=torch.cat((flat_done.new_tensor([-1], dtype=torch.int64), flat_done.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:]-done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards[:self.step].mean()



    #############通过给定的选取次数选出batch_size大小的索引值##############
    def mini_batch_generator(self, num_mini_batches):
        batch_size=self.batch_size
        mini_batch_size=batch_size//num_mini_batches
        batch=[]

        for _ in range(num_mini_batches):
            if self.fullfill:
                subset=random.sample(range(self.replay_size),mini_batch_size)
            else:
                subset=random.sample(range(self.step),mini_batch_size)

            batch.append(subset)

        return batch
