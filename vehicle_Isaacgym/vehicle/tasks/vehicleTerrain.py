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

from isaacgym import gymapi
from isaacgym import gymtorch

import torch

from vehicle.tasks.base.vec_task import VecTask
from vehicle.utils.torch_jit_utils import *

class VehicleTerrain(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]


        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["wheelStumbleRewardScale"]



        #command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]


        super().__init__(cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        # get gym GPU state tensors
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)      #position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)

        #create wrapper tensors
        self.dof_state=gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos=self.dof_state.view(self.num_envs, self.num_dof, 2)[...,0]     #省略多个冒号
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_states=gymtorch.wrap_tensor(_actor_root_state)    #position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        # print(self.contact_forces.shape)

        # init some data used later
        self.common_step_counter = 0
        self.commands=torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))


        self.height_points=self.init_height_points()
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(),       #TODO:  修改
                             "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "hip": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim=super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type=self.cfg['env']['terrain']['terrainType']
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=="trimesh":
            self._create_trimesh()

        self._create_envs(self.num_envs, self.cfg['env']['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction=self.cfg['env']['terrain']['staticFriction']
        plane_params.dynamic_friction=self.cfg['env']['terrain']['dynamicFriction']
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)



    def _create_trimesh(self):
        self.terrain=Terrain()
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices=self.terrain.vertices.shape[0]
        tm_params.nb_triangles=self.terrain.triangles.shape[0]

        # self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C', tm_params))



    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]

        asset_options=gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode=gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints=True
        asset_options.flip_visual_attachments=True
        asset_options.armature=0.0
        asset_options.disable_gravity=False
        asset_options.fix_base_link=False
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
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(100000.0)
        dof_props["damping"].fill(20000.0)
        body_names = self.gym.get_asset_rigid_body_names(vehicle_asset)
        wheel_names=[s for s in body_names if "wheel" in s]
        self.wheel_index=torch.zeros(len(wheel_names),dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0
        #env origins
        self.env_origins=torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg['env']['terrain']['maxInitMapLevel']=self.cfg['env']['terrain']['numLevels']-1
        self.terrain_levels=torch.randint(0, self.cfg['env']['terrain']["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins=torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing=0.
        env_lower=gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper=gymapi.Vec3(spacing, spacing, spacing)
        self.vehicle_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_handle=self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos=self.env_origins[i].clone()
                pos[:2]+=torch_rand_float(-1, 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p=gymapi.Vec3(*pos)

            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction=friction_buckets[i%num_buckets]        #随机化摩擦力
            self.gym.set_asset_rigid_shape_properties(vehicle_asset, rigid_shape_prop)
            vehicle_handle=self.gym.create_actor(env_handle, vehicle_asset, start_pose, 'vehicle', i, 0, 0)
            self.envs.append(env_handle)
            self.vehicle_handles.append(vehicle_handle)

            actor_index = self.gym.get_actor_dof_names(env_handle, vehicle_handle)
            j=0
            for s in wheel_names:
                wheel_indices=self.gym.find_actor_dof_index(env_handle, vehicle_handle, s, gymapi.DOMAIN_ENV)
                dof_props['driveMode'][wheel_indices]=gymapi.DOF_MODE_EFFORT
                dof_props['damping'][wheel_indices]=1500.0
            self.gym.set_actor_dof_properties(env_handle, vehicle_handle, dof_props)
        for i in range(len(wheel_names)):
            self.wheel_index[i]=self.gym.find_actor_rigid_body_handle(self.envs[0], self.vehicle_handles[0], wheel_names[i])


        self.base_index=self.gym.find_actor_rigid_body_handle(self.envs[0], self.vehicle_handles[0], "base")


    def reset_idx(self, env_ids):
        position_offset=torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities= torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        # print(self.default_dof_pos[env_ids].shape)
        self.dof_pos[env_ids]=self.default_dof_pos[env_ids]
        self.dof_vel[env_ids]=velocities
        # print(position_offset)
        # print(self.dof_pos[env_ids])

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
        # print(self.dof_state[env_ids])
        # print(self.dof_pos[env_ids])

        #重置指定索引的无人车的初始自由度位姿为 self.dof_state
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # print(self.dof_pos[env_ids])

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids]*= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(1)

        # 清除一些buffer    #TODO： 增加
        self.last_actions[env_ids]= 0.
        self.last_dof_vel[env_ids]= 0.
        self.reset_buf[env_ids]= 1.         #TODO:why?

        #填写extras 及计算清理env_ids 对应的episode_sums
        self.extras['episode']={}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_'+key]=torch.mean(self.episode_sums[key][env_ids])/self.max_episode_length_s
            self.episode_sums[key][env_ids]= 0.
        self.extras['episode']['terrain_level']=    torch.mean(self.terrain_levels.float())



    def update_terrain_level(self, env_ids):    #TODO：
        pass


    def check_termination(self):    #TODO: 确定一些终止条件
        # self.reset_buf=torch.norm(self.)
        self.reset_buf = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 10.     #base 的力
        print(torch.norm(self.contact_forces[:, self.base_index, :], dim=1))
        # print(self.contact_forces[:,self.base_index, :])
        # print("self.reset_buf1",self.reset_buf)
        #six wheels left the plane
        contact=self.contact_forces[:, self.wheel_index, 2]>1.
        print(~torch.any(contact, dim=1))
        self.reset_buf |=~torch.any(contact, dim=1)
        #orientation >45度
        # print("self.reset_buf2", self.reset_buf)


        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                     torch.ones_like(self.reset_buf), self.reset_buf)


    def compute_observations(self):
        self.measured_heights=self.get_heights()
        heights=torch.clip(self.root_states[:,2].unsqueeze(-1)-0.5-self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf=torch.cat((self.base_lin_vel*self.lin_vel_scale,
                                self.base_ang_vel*self.ang_vel_scale,
                                self.projected_gravity,
                                self.commands[:, :3]*self.commands_scale,
                                self.dof_pos*self.dof_pos_scale,
                                self.dof_vel*self.dof_vel_scale,
                                heights,
                                self.actions), dim=-1)


    def compute_reward(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy=torch.exp(-lin_vel_error/0.25)*self.rew_scales['lin_vel_xy']
        rew_ang_vel_z=torch.exp(-ang_vel_error/0.25)*self.rew_scales['ang_vel_z']

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # torque penalty
        # rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]
        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.wheel_index, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.wheel_index, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        self.rew_buf= rew_lin_vel_xy + rew_lin_vel_z + rew_ang_vel_xy + rew_ang_vel_z  + rew_joint_acc + rew_action_rate
        self.rew_buf = torch.clip(self.rew_buf, min=0, max=None)
        self.rew_buf+= self.rew_scales['termination'] * self.reset_buf * ~self.timeout_buf


        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        # self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        # self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate


    def pre_physics_step(self, actions):
        # print(actions.cpu().numpy())
        self.actions=actions.clone().to(self.device)
        # 一次控制多步运动
        for i in range(self.decimation):
            # torques=torch.clip(self.Kp*(self.action_scale*self.actions+self.default_dof_pos-self.dof_pos)-self.Kd*self.dof_vel, -80,80)     #TODO:急需修改
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(actions))          # 神经网络输出的直接是torque控制wheel
            # self.torques=torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)



    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim) # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)     #TODO:why?

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval==0:
            self.push_robots()
        #prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel=quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)


        self.check_termination()
        self.compute_reward()

        env_ids=self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids)>0:
            self.reset_idx(env_ids)
        print(env_ids)
        # print(env_ids)
        self.compute_observations()     #设置obs_buf
        if self.add_noise:
            self.obs_buf+=(2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec

        self.last_actions[:]=self.actions[:]
        self.last_dof_vel[:]=self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # sphere_geom=gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


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
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points=grid_x.numel()
        points=torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:,:,0]=grid_x.flatten()
        points[:,:,1]=grid_y.flatten()
        return points



    def get_heights(self, env_ids=None): #TODO: not finish
        if self.cfg['env']['terrain']['terrainType'] == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points=quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, 3]).unsqueeze(1)
        else:
            points=quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.border_size
        points=(points)

    def stand_by(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class Terrain():
    pass



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
