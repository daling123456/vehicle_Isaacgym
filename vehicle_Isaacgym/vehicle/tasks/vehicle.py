# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch

import torch
import cv2

from vehicle_Isaacgym.vehicle.tasks.base.vec_task import VecTask
from vehicle_Isaacgym.vehicle.utils.torch_jit_utils import *

class Vehicle(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.height_samples=None
        self.init_done=False
        self.break_=False


        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.camera_width=self.cfg["env"]["viewer"]["camera_width"]
        self.camera_height = self.cfg["env"]["viewer"]["camera_height"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.base_height=self.cfg['env']['baseInitState']['pos'][2]

        # reward scales
        self.print_reward=self.cfg['env']['learn']['printreward']
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["wheelStumbleRewardScale"]
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["wheelAirTimeRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales['base_contact']= self.cfg['env']['learn']['basecontactRewardScale']


        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        super().__init__(cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.camera_buffer=torch.zeros((self.num_envs, self.camera_width, self.camera_height, 4), device=self.device, dtype=torch.int8)

        # get gym GPU state tensors
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)      #position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        #create wrapper tensors
        self.dof_state=gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos=self.dof_state.view(self.num_envs, self.num_dof, 2)[...,0]     #省略多个冒号
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_states=gymtorch.wrap_tensor(_actor_root_state)    #position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # init some data used later
        self.action_lower_limit = torch.tensor([-0.5, -0.5, -0.7, -100, -0.5, -0.5, -0.7, -100, -0.5, -0.5, -0.7, -100,
                -0.5, -0.5, -0.7, -100, -0.7, -0.5, -0.5, -0.7, -100, -0.5, -0.5, -0.7, -100, ], device=self.device)
        self.action_upper_limit = torch.tensor([0.5, 0.5, 0.7, 100, 0.5, 0.5, 0.7, 100, 0.5, 0.5, 0.7, 100,
                0.5, 0.5, 0.7, 100, 0.7, 0.5, 0.5, 0.7, 100, 0.5, 0.5, 0.7, 100, ], device=self.device)
        self.common_step_counter = 0
        self.commands=torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.wheel_air_time = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)


        self.height_points=self.init_height_points()
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name=self.dof_names[i]
            angle=self.named_default_joint_angles[name]
            self.default_dof_pos[:, i]=angle
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_x": torch_zeros(), "lin_vel_yz": torch_zeros(), "ang_vel_z": torch_zeros(),       #TODO:  修改
                             "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "hip": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_done=True


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim=super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type=self.cfg['env']['terrain']['terrainType']
        if terrain_type=='plane':
            self._create_ground_plane()
            self.wheel_air_tips = 0.5
        elif terrain_type=="trimesh":
            self.wheel_air_tips=2
            self._create_trimesh()
            self.custom_origins=True

        self._create_envs(self.num_envs, self.cfg['env']['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction=self.cfg['env']['terrain']['staticFriction']
        plane_params.dynamic_friction=self.cfg['env']['terrain']['dynamicFriction']
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain=Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples=torch.tensor(self.terrain.heightsamples).view(self.terrain.total_rows, self.terrain.total_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]

        asset_options=gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode=gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints=self.cfg.env.urdfAsset.collapseFixedJoints
        asset_options.flip_visual_attachments=True
        asset_options.armature=0.0
        asset_options.disable_gravity=False
        # asset_options.replace_cylinder_with_capsule=True  #罪魁祸首
        asset_options.fix_base_link=self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        vehicle_asset=self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof=self.gym.get_asset_dof_count(vehicle_asset)
        self.num_bodies=self.gym.get_asset_rigid_body_count(vehicle_asset)

        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose=gymapi.Transform()
        start_pose.p=gymapi.Vec3(*self.base_init_state[:3])

        #prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(vehicle_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)

        dof_props = self.gym.get_asset_dof_properties(vehicle_asset)
        # dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        # dof_props["stiffness"].fill(self.cfg["env"]["control"]["stiffness"])
        # dof_props["damping"].fill(self.cfg['env']['control']['damping'])
        body_names = self.gym.get_asset_rigid_body_names(vehicle_asset)

        slider_index=self.gym.find_asset_dof_index(vehicle_asset, "slider_joint")
        dof_props['driveMode'][slider_index] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][slider_index] = 1000000
        dof_props['damping'][slider_index] = 200000
        wheel_names=[s for s in body_names if "wheel" in s]
        steer_names=[s for s in body_names if "steer" in s]
        dof_names=self.gym.get_asset_dof_names(vehicle_asset)
        wheel_dof_names=[s for s in dof_names if "wheel" in s]
        for s in wheel_dof_names:
            wheel_index=self.gym.find_asset_dof_index(vehicle_asset, s)
            dof_props['driveMode'][wheel_index] = gymapi.DOF_MODE_VEL
            dof_props["stiffness"][wheel_index] = 0
            dof_props['damping'][wheel_index] = 2000

        self.wheel_index=torch.zeros(len(wheel_names),dtype=torch.long, device=self.device, requires_grad=False)
        self.wheel_dof_index = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.steer_dof_index = torch.zeros(len(steer_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.fl_lift1_index = torch.tensor([1, 5, 9, 13, 18, 22], dtype=torch.long, device=self.device)
        self.dof_names = self.gym.get_asset_dof_names(vehicle_asset)
        self.base_index = 0
        self.slider_index=0
        #env origins
        self.env_origins=torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg['env']['terrain']['maxInitMapLevel']=self.cfg['env']['terrain']['numLevels']-1     #出生点差异由此导致
        self.terrain_levels=torch.randint(0, self.cfg['env']['terrain']["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)

        if self.custom_origins:
            self.terrain_origins=torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing=0.
        env_lower=gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper=gymapi.Vec3(spacing, spacing, spacing)
        self.vehicle_handles = []
        self.envs = []
        self.cameras=[]
        for i in range(self.num_envs):
            env_handle=self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]       #从这个地方导致的出生点差异
                pos=self.env_origins[i].clone()
                pos[:2]+=torch_rand_float(-1, 1., (2, 1), device=self.device).squeeze(1)
                pos[2]=0.1
                start_pose.p=gymapi.Vec3(*pos)
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction=friction_buckets[i%num_buckets]        #随机化摩擦力
            self.gym.set_asset_rigid_shape_properties(vehicle_asset, rigid_shape_prop)
            vehicle_handle=self.gym.create_actor(env_handle, vehicle_asset, start_pose, 'vehicle', i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, vehicle_handle, dof_props)

            camera_handle=self.set_camera_sensors(env_handle, vehicle_handle)
            self.cameras.append(camera_handle)
            self.envs.append(env_handle)
            self.vehicle_handles.append(vehicle_handle)
        # print(dof_props['driveMode'])
        # print(dof_props["stiffness"])
        # print(dof_props["damping"])
        # print(self.cameras)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.camera_video = cv2.VideoWriter('video1.mp4', fourcc, 15.0, (self.camera_width, self.camera_height))

        dof_names=self.gym.get_actor_dof_names(self.envs[0], self.vehicle_handles[0])
        wheel_dof_names=[s for s in dof_names if "wheel" in s]
        steer_dof_names=[s for s in dof_names if "steer" in s]
        for i in range(len(wheel_names)):
            self.wheel_index[i]=self.gym.find_actor_rigid_body_handle(self.envs[0], self.vehicle_handles[0], wheel_names[i])
            self.wheel_dof_index[i]=self.gym.find_actor_dof_index(self.envs[0], self.vehicle_handles[0], wheel_dof_names[i], gymapi.DOMAIN_ENV)
        for i in range(len(steer_names)):
            self.steer_dof_index[i]=self.gym.find_actor_dof_index(self.envs[0], self.vehicle_handles[0], steer_dof_names[i], gymapi.DOMAIN_ENV)
        self.base_index=self.gym.find_actor_rigid_body_handle(self.envs[0], self.vehicle_handles[0], "base_link")
        self.slider_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.vehicle_handles[0], "slider")

        ########计算bodies、joints、dofs的数量###########
        env = self.envs[0]
        actor_handle = self.vehicle_handles[0]
        num_bodies = self.gym.get_actor_rigid_body_count(env, actor_handle)
        num_joints = self.gym.get_actor_joint_count(env, actor_handle)
        num_dofs = self.gym.get_actor_dof_count(env, actor_handle)
        print(f"-----num_bodies:{num_bodies},------num_joints:{num_joints},------num_dofs:{num_dofs}--------")

    def reset_idx(self, env_ids):
        position_offset=torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities= torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids]=self.default_dof_pos[env_ids]
        self.dof_vel[env_ids]=velocities

        env_ids_int32=env_ids.to(dtype=torch.int32)
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
        #重置指定索引的无人车的初始位置为 self.root_states
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        #重置指定索引的无人车的初始自由度位姿为 self.dof_state
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids]*= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1)

        # 清除一些buffer    #TODO： 增加
        self.last_actions[env_ids]= 0.
        self.last_dof_vel[env_ids]= 0.
        self.progress_buf[env_ids]=0
        self.reset_buf[env_ids]= 1.         #TODO:why?
        self.wheel_air_time.zero_()

        #填写extras 及计算清理env_ids 对应的episode_sums
        self.extras['episode']={}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_'+key]=torch.mean(self.episode_sums[key][env_ids])/self.max_episode_length_s
            self.episode_sums[key][env_ids]= 0.
        self.extras['episode']['terrain_level']=    torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):    #TODO：
        if not self.init_done or not self.curriculum:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25)
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.     #base 的力
        # print(self.contact_forces[0])
        # print(self.contact_forces.shape)
        # print(torch.sum(self.reset_buf.cpu()))
        # print(self.contact_forces[:, self.base_index, :].cpu().numpy())
        self.reset_buf|=torch.norm(self.contact_forces[:,self.slider_index,:],dim=1)>1
        # print(torch.sum(self.reset_buf.cpu()))

        #six wheels left the plane
        contact=self.contact_forces[:, self.wheel_index, 2]>1.
        # print(~torch.any(contact, dim=1))
        self.reset_buf |=~torch.any(contact, dim=1)
        # print(torch.sum(self.reset_buf.cpu()))

        # print(torch.any(self.wheel_air_time>0.5, dim=1))
        self.reset_buf |= torch.any(self.wheel_air_time>self.wheel_air_tips, dim=1)     #车轮长时间滞空则重置
        # print(torch.sum(self.reset_buf.cpu()))

        #orientation >45度
        # print("self.reset_buf2", self.reset_buf)
        # torch.unique((self.wheel_air_time > 0.5).nonzero()[:, 0])

        self.reset_buf |= torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # print(torch.sum(self.reset_buf.cpu()))
        # print("---------------------------------------------")

    def compute_observations(self):
        self.measured_heights=self.get_heights()
        heights=torch.clip(self.root_states[:,2].unsqueeze(1)-self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf=torch.cat((self.dof_pos*self.dof_pos_scale,
                                self.dof_vel*self.dof_vel_scale,
                                self.actions,
                                self.commands[:, :3] * self.commands_scale,
                                self.base_lin_vel * self.lin_vel_scale,
                                self.base_ang_vel * self.ang_vel_scale,
                                self.projected_gravity,
                                # self.torques,
                                # self.contact_forces,
                                heights,), dim=-1)
        print(self.dof_pos.shape)
        print(self.dof_vel.shape)

    def compute_reward(self):
        # velocity tracking reward
        # lin_vel_error = torch.sum(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), dim=1)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_x=torch.exp(-lin_vel_error/0.25)*self.rew_scales['lin_vel_xy']
        rew_ang_vel_z=torch.exp(-ang_vel_error/0.25)*self.rew_scales['ang_vel_z']
        # print(self.base_lin_vel[:, 0])

        # other base velocity penalties
        # rew_lin_vel_yz = torch.square(self.base_lin_vel[:, 1:3]) * self.rew_scales["lin_vel_z"]
        rew_lin_vel_yz = torch.sum(torch.square(self.base_lin_vel[:, 1:3]), dim=1) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # 5 orientation penalty（pitch roll）
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]
        # rew_orient = torch.square(self.projected_gravity[:, 2]) * self.rew_scales["orient"]

        # 6 wheel air time penalty 对轮子不贴地的惩罚
        contact = self.contact_forces[:, self.wheel_index, 2] > 1
        self.wheel_air_time += self.dt
        rew_airTime = torch.sum(self.wheel_air_time - 0.01, dim=1) * self.rew_scales["air_time"]  # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.wheel_air_time *= ~contact
        # print(self.wheel_air_time)


        # 7 joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]
        # 8 action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        # 9 stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.wheel_index, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.wheel_index, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # 10 torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # 11 base height penalty
        rew_base_height=torch.square(self.root_states[:,2]-self.base_height)*self.rew_scales['base_height']

        # 12 base contact penalty
        rew_base_contact=torch.norm(self.contact_forces[:, self.base_index, :], dim=1) * self.rew_scales['base_contact']

        # 13 stability margin
        rew_stab_margin=0

        self.rew_buf= rew_lin_vel_x + rew_lin_vel_yz + rew_ang_vel_xy + rew_ang_vel_z  + rew_orient + rew_joint_acc + rew_action_rate \
                      + rew_airTime + rew_base_contact
        self.rew_buf = torch.clip(self.rew_buf, min=0, max=None)
        self.rew_buf+= self.rew_scales['termination'] * self.reset_buf * ~self.timeout_buf
        if self.print_reward:
            print(f"rew_lin_vel_x:{torch.sum(rew_lin_vel_x)},rew_lin_vel_yz:{torch.sum(rew_lin_vel_yz)},rew_ang_vel_xy:{torch.sum(rew_ang_vel_xy)},"
                  f"rew_ang_vel_z:{torch.sum(rew_ang_vel_z)}，\nrew_orient:{torch.sum(rew_orient)},rew_joint_acc:{torch.sum(rew_joint_acc)},"
                  f"rew_action_rate:{torch.sum(rew_action_rate)}, rew_base_height:{torch.sum(rew_base_height)},\nrew_airTime:{torch.sum(rew_airTime)}, "
                  f"rew_base_contact:{torch.sum(rew_base_contact)}, rew_torque:{torch.sum(rew_torque)}")
            print(f"        total_rewards:{torch.sum(self.rew_buf)}")
            # print(self.rew_scales['termination'] * self.reset_buf * ~self.timeout_buf)

        # log episode rewar d sums
        self.episode_sums["lin_vel_x"] += rew_lin_vel_x
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_yz"] += rew_lin_vel_yz
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        # self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        # self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate

    def pre_physics_step(self, actions):
        self.actions=actions.clone().to(self.device)
        actions=torch.max(torch.min(actions, self.action_upper_limit), self.action_lower_limit)     # 将actions限制
        self.actions[:,self.slider_index-1]=0.0     #将中间轴的运动设置为0
        for i in self.fl_lift1_index:
            self.actions[:, i]= -self.actions[:, i-1]

        # self.actions=torch.zeros(32, 25, dtype=torch.float32, device="cuda:0")
        # actions=torch.tensor([0.2, -0.2, 0, 0, 0.2, -0.2, 0, 0, 0.2, -0.2, 0, 0, 0.2, -0.2, 0, 0, 0, 0.2, -0.2, 0, 0, 0.2, -0.2, 0, 0], dtype=torch.float32, device=self.device)
        # self.actions=actions.repeat(32, 1)
        # for i in self.wheel_dof_index:
        #     self.actions[:, i]=-2

        for i in self.steer_dof_index:
            self.actions[:, i]=0
        # print(self.steer_dof_index)
        # print(self.actions.detach().cpu().numpy()[0])
        # 一次控制多步运动
        for i in range(self.decimation):
            torques=torch.clip(self.Kp*(self.action_scale*self.actions+self.default_dof_pos-self.dof_pos)-self.Kd*self.dof_vel, -800, 800)
            self.torques = torques.view(self.torques.shape)
            # print(self.torques[0])
            # print(self.dof_pos[0])
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.actions))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))          # 神经网络输出的直接是torque控制wheel
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.actions))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        # if self.common_step_counter % self.push_interval==0:
        #     self.push_robots()
        #prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel=quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2]= torch.clip(0.5 * wrap_to_pi(self.commands[:, 3]- heading), -1., 1)


        self.check_termination()
        self.compute_reward()
        if self.cfg.env.enableCameraSensors:
            self.play_video()


        env_ids=self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids)>0:
            self.reset_idx(env_ids)

        self.compute_observations()     #设置obs_buf
        if self.add_noise:
            self.obs_buf+=(2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec

        self.last_actions[:]=self.actions[:]
        self.last_dof_vel[:]=self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            print("-----------------------")
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom=gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _get_noise_scale_vec(self, cfg):
        noise_vec=torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:37] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[37:62] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale

        return noise_vec

    def push_robots(self):  #刷新无人车的车身线速度
        self.root_states[:, 7:9]=torch_rand_float(-1, 1, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def init_height_points(self):
        y = 0.1*torch.tensor([-5,-4,-3,-2,-1,1,2,3,4,5], device=self.device, requires_grad=False)# 10-50cm on each side
        x = 0.1*torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False) # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.num_height_points=grid_x.numel()
        points=torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:,:,0]=grid_x.flatten()
        points[:,:,1]=grid_y.flatten()
        return points

    def set_camera_sensors(self, env, actor_handle):
        camera_props=gymapi.CameraProperties()
        camera_props.width=self.camera_width
        camera_props.height=self.camera_height
        camera_props.enable_tensors=self.cfg.env.viewer.enable_tensors
        camera_handle=self.gym.create_camera_sensor(env, camera_props)

        # body_handle=self.gym.find_actor_rigid_body_handle(env, actor_handle, "camera_link")
        body_handle=0
        transform=gymapi.Transform()
        transform.p = gymapi.Vec3(*self.cfg.env.viewer.camera_pos)
        transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*self.cfg.env.viewer.camera_axis), np.radians(self.cfg.env.viewer.camera_angle))
        self.gym.attach_camera_to_body(camera_handle, env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)
        return camera_handle

    def play_video(self):
        # _camera_tensor=self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
        # camera_tensor=gymtorch.wrap_tensor(_camera_tensor)
        # print(camera_tensor.shape)
        for i in range(len(self.envs)):
            _camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            camera_tensor = gymtorch.wrap_tensor(_camera_tensor)
            self.camera_buffer[i]=camera_tensor

        self.gym.render_all_camera_sensors(self.sim)
        # color_image=self.gym.get_camera_image(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)

        image_data=np.frombuffer(self.camera_buffer[2].cpu().numpy(), dtype=np.uint8)
        image=image_data.reshape((self.camera_height, self.camera_width, 4))
        image = image.astype(np.uint8)  # 确保图像数据是uint8类型
        if image.ndim == 3 and image.shape[2] == 4:  # 如果图像是RGBA格式的，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if self.cfg.env.viewer.create_video:
            self.camera_video.write(image)
            rgba_for_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            cv2.imshow('Real-time Video', rgba_for_display)
            # print(rgba_for_display[0][0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.break_=True

    def get_heights(self, env_ids=None):
        if self.cfg['env']['terrain']['terrainType'] == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points=quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, 3]).unsqueeze(1)
        else:
            points=quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.border_size
        points=(points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale



def stand_by(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

            self.check_termination()
            self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


from isaacgym.terrain_utils import *
class Terrain():
    def __init__(self, cfg, num_robots):
        self.horizontal_scale=0.1
        self.border_size = 20
        self.vertical_scale=0.005
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions=[np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg['terrainProportions']))]   #[0.1, 0.2, 0.55, 0.8, 1.0]

        self.env_rows=cfg["numLevels"]
        self.env_cols=cfg["numTerrains"]
        self.num_maps= self.env_rows * self.env_cols
        self.env_origins=np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels= int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels= int(self.env_length/ self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.total_cols=int(self.env_cols*self.width_per_env_pixels)+2*self.border
        self.total_rows=int(self.env_rows*self.width_per_env_pixels)+2*self.border

        self.height_field_raw=np.zeros((self.total_rows, self.total_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()

        self.heightsamples=self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])


    def randomized_terrain(self):
        for k in range(self.num_maps):
            (i, j)= np.unravel_index(k, (self.env_rows, self.env_cols))

            start_x=self.border+i * self.length_per_env_pixels
            end_x=self.border+(i+1) * self.length_per_env_pixels
            start_y=self.border+j * self.length_per_env_pixels
            end_y=self.border+(j+1) * self.length_per_env_pixels

            terrain=SubTerrain(width=self.width_per_env_pixels,
                               length=self.width_per_env_pixels,
                               vertical_scale=self.vertical_scale,
                               horizontal_scale=self.horizontal_scale)

            choice=np.random.uniform(0,1)

            if choice<0.1:
                if np.random.choice([0,1]):
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice<0.6:
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice<1:
                discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x=(i+0.5)*self.env_length
            env_origin_y=(j+0.5)*self.env_width
            x1=int((self.env_length/2.-1)/self.horizontal_scale)
            x2=int((self.env_length/2+1)/self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z=np.max(terrain.height_field_raw[x1:x2,y1:y2])*self.vertical_scale
            self.env_origins[i,j]=[env_origin_x, env_origin_y, env_origin_z]


    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots/ num_terrains)
        left_over=num_robots % num_terrains
        for j in range(num_terrains):
            for i in range(num_levels):
                terrain=SubTerrain("terrain", width=self.width_per_env_pixels, length=self.width_per_env_pixels,
                                   vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
                difficulty=i/num_levels
                choice=j/num_terrains
                slope=difficulty*0.4
                step_height=0.05+0.175*difficulty
                discrete_obstacles_height=0.025+difficulty*0.15
                amplitude=0.1+difficulty*0.5
                stepping_stones_size=2-1.8*difficulty

                if choice<self.proportions[0]:
                    if choice<0.05:
                        slope*=-1
                    pyramid_sloped_terrain(terrain, slope, platform_size=3.)
                elif choice<self.proportions[1]:
                    if choice<0.15:
                        slope*=-1
                    pyramid_sloped_terrain(terrain, slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step= 0.025, downsampled_scale=0.2)
                elif choice<self.proportions[3]:
                    if choice<self.proportions[2]:
                        step_height*=-1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    wave_terrain(terrain, num_waves=1, amplitude=amplitude)
                    # stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                    #                         platform_size=3.)

                start_x=self.border+i * self.length_per_env_pixels
                end_x=self.border+(i+1) * self.length_per_env_pixels
                start_y=self.border+j * self.length_per_env_pixels
                end_y=self.border+(j+1) * self.length_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y]=terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map+=1

                env_origin_x=(i-0.5) * self.env_length
                env_origin_y=(j+0.5) * self.env_width

                x1= int((self.env_length/2-1)/self.horizontal_scale)
                x2= int((self.env_length/2+1)/self.horizontal_scale)
                y1= int((self.env_width/2-1)/self.horizontal_scale)
                y2= int((self.env_width/2+1)/self.horizontal_scale)
                # env_origin_z=np.max(terrain.height_field_raw[x1:x2, y1:y2])* self.vertical_scale
                env_origin_z=0.
                self.env_origins[i,j] = [env_origin_x, env_origin_y, env_origin_z]



@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles
