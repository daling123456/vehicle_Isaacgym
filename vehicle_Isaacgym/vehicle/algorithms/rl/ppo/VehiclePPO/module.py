import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg):
        super(ActorCritic, self).__init__()
        # self.actor_hidden_shape=model_cfg.actor_hidden_shape
        self.activation_fn = model_cfg.activation_fn
        self.output_activation_fn = model_cfg.output_activation_fn
        self.small_init = model_cfg.small_init
        critic_shape=model_cfg.critic_shape

        self.device="cuda:0"

        self.actor=MLPEncode(self.device, model_cfg, self.activation_fn, self.output_activation_fn, self.small_init)
        # self.critic=MLPEncode(self.device, model_cfg, *obs_shape, 1, self.actor_hidden_shape, self.activation_fn)
        self.critic=MLP(self.device, critic_shape, *obs_shape, 1, self.activation_fn, self.output_activation_fn)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, obs_history):
        action_latent= self.actor(observations)
        history_latent=self.actor.get_history_encoding(obs_history)

        suspend_latent=self.actor.get_suspend_latent(action_latent[:,:8], history_latent)
        # print(suspend_latent)

        steer_latent=self.actor.get_steer_latent(action_latent[:,8:])
        actions_mean=torch.cat((suspend_latent, steer_latent), dim=1)
        # actions_mean = self.actor(obs)
        # print(actions_mean.detach().cpu().numpy()[0])

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)


        value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations, obs_history):
        action_latent = self.actor(observations)
        history_latent = self.actor.get_history_encoding(obs_history)

        suspend_latent = self.actor.get_suspend_latent(action_latent[:, :8], history_latent)

        steer_latent = self.actor.get_steer_latent(action_latent[:, 8:])
        actions_mean = torch.cat((suspend_latent, steer_latent), dim=1)
        return actions_mean

    def evaluate(self, observations, actions, obs_history):
        # print(observations.shape)
        action_latent = self.actor(observations)
        history_latent = self.actor.get_history_encoding(obs_history).float()
        mean_history_latent=history_latent.mean(dim=0)
        history_latent=mean_history_latent.unsqueeze(0)
        history_latent=history_latent.repeat(observations.shape[0],1)

        suspend_latent = self.actor.get_suspend_latent(action_latent[:, :8], history_latent)

        steer_latent = self.actor.get_steer_latent(action_latent[:, 8:])
        actions_mean = torch.cat((suspend_latent, steer_latent), dim=1)
        # actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()


        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class MLPEncode(nn.Module):
    def __init__(self, device, model_cfg, activation_fn, output_activation_fn = None, small_init= False):
        super(MLPEncode,self).__init__()

        # self.n_futures=model_cfg.n_futures
        actor_hidden_shape=model_cfg.actor.actor_hidden_shape       #actor 网络结构
        suspensin_hidden_shape= model_cfg.actor.suspension_hidden_shape        #suspension net
        steer_hidden_shape= model_cfg.actor.steer_hidden_shape      #steer_net

        # self.state_history_dim = model_cfg.actor.state_history_dim      #statehistory输入维度
        self.vision_dim=model_cfg.actor.vision_dim       #视觉输入维度
        self.obs_dim=model_cfg.actor.obs_dim       #本体感知观测空间大小


        history_latent_dim=model_cfg.actor.history_latent_dim   #状态历史编码向量大小
        vision_latent_dim=model_cfg.actor.vision_latent_dim     #视觉向量大小
        suspension_latten_dim=model_cfg.actor.suspension_latten_dim     #主动悬架模块输入参数大小
        steer_latten_dim=model_cfg.actor.steer_latten_dim       #转向模块输入参数大小
        suspension_input_size=suspension_latten_dim+history_latent_dim
        self.actor_output_size=suspension_latten_dim+steer_latten_dim

        suspension_output_dim=model_cfg.actor.suspension_output_dim     #12
        steer_output_dim=model_cfg.actor.steer_output_dim   #13
        self.actor_input_size=self.obs_dim+self.vision_dim

        self.device=device
        self.small_init=small_init
        self.tsteps=model_cfg.actor.tsteps

        self.base_obs_size=self.obs_dim+1
        self.history_length=self.tsteps
        self.state_history_dim=self.base_obs_size * self.history_length
        self.statehistory_encoder=model_cfg.actor.statehistory_encoder

        self.activation_fn=activation_fn
        self.output_activation_fn=output_activation_fn

        scale_encoder = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        if self.vision_dim >0:
            self.vision_encoder = VisionEncoder()
            # self.init_weights(self.vision_encoder, scale_encoder)
        else:
            # raise IOError("Not implemented self.geom_dim")
            print("Not implemented self.vision_dim")
        if self.state_history_dim > 0:
            self.state_history_encoder = StateHistoryEncoder(activation_fn, self.tsteps, self.state_history_dim, history_latent_dim, self.statehistory_encoder)
            # self.init_weights(self.s_encoder, scale_encoder)

        else:
            # raise IOError("Not implemented self.geom_dim")
            print("Not implemented self.geom_dim")

        # creating the action encoder
        self.action_mlp=MLP(device, actor_hidden_shape, self.actor_input_size, self.actor_output_size, self.activation_fn, self.output_activation_fn, self.small_init)
        # modules = [nn.Linear(self.obs_dim+vision_latent_dim, actor_hidden_shape[0]), self.activation_fn()]
        # scale = [np.sqrt(2)]
        # for l in range(len(actor_hidden_shape)):
        #     if l == len(actor_hidden_shape) - 1:
        #         modules.append(nn.Linear(actor_hidden_shape[l], suspension_latten_dim+steer_latten_dim))
        #     else:
        #         modules.append(nn.Linear(actor_hidden_shape[l], actor_hidden_shape[l + 1]))
        #         modules.append(self.activation_fn())
        #     scale.append(np.sqrt(2))
        # action_output_layer = modules[-1]
        # if self.output_activation_fn is not None:
        #     modules.append(self.output_activation_fn())
        # self.action_mlp = nn.Sequential(*modules)
        # scale.append(np.sqrt(2))
        # self.init_weights(self.action_mlp, scale)
        #
        # if self.small_init: action_output_layer.weight.data *= 1e-6

        #creating the suspension mlp
        self.suspension_mlp=MLP(device, suspensin_hidden_shape, suspension_input_size, suspension_output_dim, self.activation_fn, self.output_activation_fn, self.small_init)
        self.steer_mlp=MLP(device, steer_hidden_shape, steer_latten_dim, steer_output_dim, self.activation_fn, self.output_activation_fn, self.small_init)
    # @staticmethod
    # def init_weights(sequential, scales):
    #     [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
    #      enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def forward(self, x):
        if self.vision_dim>0:
            vision_latent=self.get_vision_latent(x[:,-self.vision_dim:])
        act_output=self.action_mlp(x[:,:self.obs_dim])
        return act_output

    # def forward(self,x):
    #     prop_latent=torch.tensor([], device=self.device)
    #     if self.prop_dim > 0:
    #         prop_latent = self.prop_encoder(x[:, self.regular_obs_dim:-self.geom_dim * (self.n_futures + 1) - 1])
    #     geom_latents = torch.tensor([], device=self.device)
    #     if self.geom_dim > 0:
    #         for i in reversed(range(self.n_futures + 1)):
    #             start = -(i + 1) * self.geom_dim - 1
    #             end = -i * self.geom_dim - 1
    #             if (end == 0):
    #                 end = None
    #             geom_latent = self.geom_encoder(x[:, start:end])
    #             geom_latents.append(geom_latent)
    #         geom_latents = torch.hstack(geom_latents)
    #     return self.action_mlp(torch.cat((x[:, :self.regular_obs_dim], prop_latent, geom_latents), dim=1))

    def get_history_encoding(self,obs_history):
        # hlen = self.base_obs_size * self.history_length
        # raw_obs = obs[:, : hlen]
        obs_history=torch.transpose(obs_history, 0, 1)
        # obs_history=torch.reshape(obs_history, [obs_history.size(0), -1])
        history_latent=self.state_history_encoder(obs_history)
        return history_latent

    def get_vision_latent(self, obs):
        # hlen=self.
        pass

    def get_steer_latent(self, x):
        steer_latent=self.steer_mlp(x)
        return steer_latent

    def get_suspend_latent(self, x, y):
        # y=self.get_history_encoding(y)
        # print(x.shape,y.shape)
        input=torch.cat((x, y), dim=1)
        suspend_latent=self.suspension_mlp(input)
        return suspend_latent

class MLP(nn.Module):
    def  __init__(self, device, shape, input_size, output_size, activation_fn, output_activation_fn=None, small_init=False, ):
        super(MLP,self).__init__()
        self.activation_fn = get_activation(activation_fn)()
        self.output_activation_fn = get_activation(output_activation_fn)()
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn)
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn)
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        if small_init: action_output_layer.weight.data *= 1e-6



    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self,x):
        output=self.architecture(x)
        return output


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, tsteps, input_size, output_size, statehistory_encoder):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn=get_activation(activation_fn)
        self.tsteps=tsteps
        # self.input_shape= input_size* tsteps
        self.output_shape=output_size

        if statehistory_encoder=="MPL":
            self.encoder=nn.Sequential(nn.Linear(input_size, 256), self.activation_fn(),
                                       nn.Linear(256,512), self.activation_fn(),
                                       nn.Linear(512, output_size), self.activation_fn())

        if statehistory_encoder=="CNN":
            if tsteps == 50:
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, 32), self.activation_fn()
                )
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels=50, out_channels=32, kernel_size=8, stride=4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1), nn.LeakyReLU(),
                    nn.Flatten())
                self.linear_output = nn.Sequential(
                    nn.Linear(1728, output_size), self.activation_fn()
                )
            else:
                raise NotImplementedError()

    def forward(self, obs):
        bs=obs.shape[0]
        T=self.tsteps
        # projection=self.encoder(obs.reshape(bs*T, -1))
        # projection = self.encoder(obs)
        output=self.conv_layers(obs)
        output=self.linear_output(output)
        return output


class VisionEncoder(nn.Module):         #TODO:
    def __init__(self):
        super(VisionEncoder).__init__()

        # self.encoder=nn.Sequential(
        #     nn.Conv2d()
        # )


class SteerEncoder():
    def __init__(self):
        pass


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU
    elif act_name == "selu":
        return nn.SELU
    elif act_name == "relu":
        return nn.ReLU
    elif act_name == "crelu":
        return nn.ReLU
    elif act_name == "lrelu":
        return nn.LeakyReLU
    elif act_name == "tanh":
        return nn.Tanh
    elif act_name == "sigmoid":
        return nn.Sigmoid
    else:
        print("invalid activation function!")
        return None
