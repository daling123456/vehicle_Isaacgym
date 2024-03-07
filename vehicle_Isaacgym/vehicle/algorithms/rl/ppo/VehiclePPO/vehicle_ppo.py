import copy
import os.path
import time

import torch
from gym.spaces import Space
import torch.optim as optim
import torch.nn as nn

from vehicle_Isaacgym.vehicle.algorithms.rl.ppo.VehiclePPO.module import ActorCritic
from vehicle_Isaacgym.vehicle.algorithms.rl.ppo.VehiclePPO.storage import RolloutStorage


class VEHICLEPPO:
    def __init__(self,vec_env, cfg_train, device='cpu', sampler='sequential',log_dir='run', print_log=True, is_testing=False, apply_reset=False, asymmetric=False):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

        self.device=device
        self.is_testing=is_testing
        self.cfg_train=copy.deepcopy(cfg_train)
        learn_cfg=self.cfg_train["learn"]

        self.observation_space=vec_env.observation_space
        self.action_space=vec_env.action_space


        #PPO params
        self.clip_param = learn_cfg["cliprange"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env=learn_cfg["nsteps"]
        self.learning_rate=learn_cfg["optim_stepsize"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]

        self.history_length=self.cfg_train["policy"]["actor"]["tsteps"]


        #PPO compoment
        self.vec_env=vec_env
        self.actor_critic=ActorCritic(self.observation_space.shape, self.action_space.shape, self.init_noise_std, self.model_cfg).to(self.device)
        self.storage=RolloutStorage(self.vec_env.num_envs, self.num_transitions_per_env, self.observation_space.shape, self.action_space.shape, self.history_length, self.device, sampler=sampler)
        self.optimizer=optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        #log
        self.log_dir = log_dir
        self.current_learning_iterations = 0


    def run(self, num_learning_iterations, log_interval=1):
        current_obs=self.vec_env.reset()['obs']
        if self.is_testing:
            while True:

                with torch.no_grad():

                    actions=self.actor_critic.act_inference(current_obs)
                    # print(actions)
                    next_obs, rewards, dones, infos=self.vec_env.step(actions)
                    # print(torch.sum(rewards))
                    # print(actions.detach().cpu().numpy())
                    current_obs.copy_(next_obs['obs'])

        else:

            for it in range(self.current_learning_iterations, num_learning_iterations):
                start=time.time()
                ep_infos=[]

                for _ in range(self.num_transitions_per_env):

                    actions, actions_log_prob, values, mu, sigma=self.actor_critic.act(current_obs)
                    # actions=torch.clamp(actions, )
                    next_obs, rewards, dones, infos=self.vec_env.step(actions)
                    # print(torch.sum(rewards))
                    # print(actions.detach().cpu().numpy())
                    self.storage.add_transitions(current_obs, actions, rewards, dones, values, actions_log_prob, mu, sigma)
                    current_obs.copy_(next_obs['obs'])

                    ep_infos.append(infos)

                _, _, last_values, _, _=self.actor_critic.act(current_obs)
                stop=time.time()
                colletion_time=stop-start
                # print("colletion_time",colletion_time)

                mean_trajectory_length, mean_reward= self.storage.get_statistics()

                start=stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss=self.update()
                self.storage.clear()
                # print("the iterations now is:", it)
                if it%log_interval==0:
                    if not os.path.exists(self.log_dir):
                        os.makedirs(self.log_dir)
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    print("save_once")
                    print(self.log_dir)
                stop=time.time()
                learn_time=stop-start
                # print("learn_time",learn_time
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))



    def update(self, ):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch=self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch=self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]        #去除reset的观测？
                act_batch=self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch=self.storage.values.view(-1, 1)[indices]
                return_batch=self.storage.returns.view(-1,1)[indices]
                old_actions_log_prob_batch=self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch= self.storage.advantages.view(-1,1)[indices]
                old_mu_batch=self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch=self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch=self.actor_critic.evaluate(obs_batch, act_batch)


                if self.desired_kl !=None and self.schedule == 'adaptive':
                    kl=torch.sum(
                        sigma_batch-old_sigma_batch+(torch.square(old_mu_batch.exp())+
                                                     torch.square(old_mu_batch-mu_batch))/(2.0*torch.square(sigma_batch.exp()))-0.5,axis=-1)
                    kl_mean=torch.mean(kl)

                    if kl_mean>self.desired_kl * 2.0:
                        self.step_size=max(1e-5,self.step_size/1.5)
                    elif kl_mean < self.desired_kl/2.0 and  kl_mean>0.0:
                        self.step_size=min(1e-2, self.step_size*1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr']=self.step_size

                    #surrogate loss
                    ratio=torch.exp(actions_log_prob_batch-torch.squeeze(old_actions_log_prob_batch))
                    surrogate=-torch.squeeze(advantages_batch)*ratio
                    surrogate_clipped=-torch.squeeze(advantages_batch)*torch.clamp(ratio,1.0-self.clip_param, 1.0+self.clip_param)
                    surrogate_loss=torch.max(surrogate_clipped,surrogate).mean()

                    #value loss
                    if self.use_clipped_value_loss:
                        value_clipped=target_values_batch+(value_batch-target_values_batch).clamp(-self.clip_param,self.clip_param)
                        value_losses=(value_batch-return_batch).pow(2)
                        value_losses_clipped=(value_clipped-return_batch).pow(2)
                        value_loss=torch.max(value_losses,value_losses_clipped).mean()
                    else:
                        value_loss=(return_batch-value_batch).pow(2).mean()
                    loss=surrogate_loss+self.value_loss_coef*value_loss-self.entropy_coef*entropy_batch.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),self.max_grad_norm)
                    self.optimizer.step()

                    mean_value_loss+=value_loss.item()
                    mean_surrogate_loss+=surrogate_loss.item()
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates

            return mean_value_loss, mean_surrogate_loss



    def save(self, path):
        torch.save(self.actor_critic.state_dict(),path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iterations= int(path.split('_')[-1].split('.')[0])

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()