import math
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch

import torch

def create_sim():
    compute_device_id=0
    graphics_device_id=0
    gym=gymapi.acquire_gym()
    ### set common parameters ###
    sim_params=gymapi.SimParams()
    sim_params.dt=1/100.
    sim_params.substeps=2
    sim_params.up_axis=gymapi.UP_AXIS_Z
    sim_params.gravity=gymapi.Vec3(0.,0.,-9.8)
    # sim_params.use_gpu_pipeline=True                    #使返回张量留在gpu上，容易导致报错
    ### set PhysX-specific parameters ###
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type=1
    sim_params.physx.num_position_iterations=6
    sim_params.physx.num_velocity_iterations=1
    sim_params.physx.contact_offset=0.01
    sim_params.physx.rest_offset=0.0
    # sim_params.physx.bounce_threshold_velocity=2*9.8*1/240./2
    sim= gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

    return sim
def create_plane(sim):
    plane_params=gymapi.PlaneParams()
    plane_params.normal=gymapi.Vec3(0,0,1)  #Z - up
    plane_params.distance=0
    plane_params.static_friction=5
    plane_params.dynamic_friction=5
    plane_params.restitution=0          #控制与地平面碰撞的弹性
    gym.add_ground(sim, plane_params)
def load_assets(sim):
    asset_root = "/home/mutong/RobotDoc/vehicle_Isaacgym/vehicle/assets"
    asset_file = "urdf/vehicle/cwego.urdf"
    # asset_file="urdf/anymal_c/urdf/anymal.urdf"
    asset_options=gymapi.AssetOptions()
    # asset_options.armature=0.01         #
    asset_options.flip_visual_attachments = True
    # asset_options.fix_base_link = True
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity=False
    # asset_options.replace_cylinder_with_capsule=True        #用胶囊替代圆柱体
    asset=gym.load_asset(sim,asset_root,asset_file,asset_options)
    return asset
def env_actor_create(sim,asset):
    spacing=6.0
    envs_per_row = 3
    env_lower=gymapi.Vec3(-spacing, -spacing, -spacing)
    env_upper=gymapi.Vec3(spacing,spacing,spacing)
    env=gym.create_env(sim, env_lower, env_upper, envs_per_row)            #create_env
    pose=gymapi.Transform()
    pose.p=gymapi.Vec3(0,0,0)
    # pose.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), -0.5*math.pi)
    actor_handle=gym.create_actor(env, asset, pose, "MyActor", 0, 1)        #create_actor
    return env, actor_handle

def create_viewer(sim):
    cam_props=gymapi.CameraProperties()
    cam_props.use_collision_geometry=True       #渲染碰撞网格
    viewer=gym.create_viewer(sim, cam_props)
    return viewer

def mouse_input():
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")


def get_dof_props(env, actor_hand):
    props = gym.get_actor_dof_properties(env, actor_handle)
    # print("dof_props:\n")
    # print(props)
    return props

def get_body_states():
    ########获取body_states###########
    body_actor_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
    body_env_states = gym.get_env_rigid_body_states(env, gymapi.STATE_ALL)
    body_sim_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)
    print(f"-----body_actor_states:\n{body_actor_states.shape},\n------body_env_states:\n{body_env_states.shape},\n------body_sim_states:\n{body_sim_states.shape}--------")
    print(f"-----body_actor_states:\n{body_actor_states},\n------body_env_states:\n{body_env_states},\n------body_sim_states:\n{body_sim_states}--------")
    ########获取特定body_states###########
    i1 = gym.find_actor_rigid_body_index(env, actor_handle, "body_actor_states", gymapi.DOMAIN_ACTOR)
    i2 = gym.find_actor_rigid_body_index(env, actor_handle, "body_env_states", gymapi.DOMAIN_ENV)
    i3 = gym.find_actor_rigid_body_index(env, actor_handle, "body_sim_states", gymapi.DOMAIN_SIM)
    return body_actor_states, body_env_states, body_sim_states

def set_dof_props(env, ActorHandle):
    props=gym.get_actor_dof_properties(env, ActorHandle,)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(100000.0)
    props["damping"].fill(20000.0)
    wheel_index, num_wheel = find_wheel_index()
    for i in wheel_index:
        props["driveMode"][i]=gymapi.DOF_MODE_VEL
        props["damping"][i]=150.0

    gym.set_actor_dof_properties(env, actor_handle, props)

#
def apply_actor_effort(props,actorhandle):
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, actorhandle, props)
    efforts=np.full(num_dofs, 100.0).astype(np.float32)
    gym.apply_actor_dof_efforts(env, actorhandle, efforts)
#
def apply_actor_position(props,actorhandle):
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(100000.0)
    props["damping"].fill(2000.0)

    lower_limits = props['lower']
    upper_limits = props['upper']
    ranges = upper_limits - lower_limits
    pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')

    gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)
    gym.set_actor_dof_properties(env, actorhandle, props)
    target=np.zeros(num_dofs).astype('f')
    gym.set_actor_dof_position_targets(env, actorhandle, target)
#
def apply_actor_velocity( actorhandle,):
    # props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    # props["stiffness"].fill(0.0)
    # props["damping"].fill(60000.0)
    # gym.set_actor_dof_properties(env, actorhandle, props)

    # vel_targets=np.random.uniform(-math.pi, math.pi, num_dofs,).astype('f')
    # print(vel_targets)

    wheel_index, num_wheel = find_wheel_index()
    vel_targets = np.zeros(25).astype('f')
    for i in wheel_index:
        vel_targets[i] = -0.2
    print(vel_targets)
    gym.set_actor_dof_velocity_targets(env, actorhandle, vel_targets)


def set_body_states(body_states):
    gym.set_actor_rigid_body_states(env, actor_handle, body_states, gymapi.STATE_ALL)
    gym.set_env_rigid_body_states(env, body_states, gymapi.STATE_ALL)
    gym.set_sim_rigid_body_states(sim, body_states, gymapi.STATE_ALL)

def set_dof_states():
    dof_states=gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)



######### Tensor 操作 ###########
def acquire_root_tensor():
    _root_tensor=gym.acquire_actor_root_state_tensor(sim)     #只用调用一次
    root_tensor=gymtorch.wrap_tensor(_root_tensor)      #shape(num_actors, 13)
    root_positions = root_tensor[:, 0:3]
    root_orientations = root_tensor[:, 3:7]
    root_linvels = root_tensor[:, 7:10]
    root_angvels = root_tensor[:, 10:13]

    return _root_tensor,root_positions, root_orientations, root_linvels, root_angvels

def acquire_dof_state():
    _dof_states=gym.acquire_dof_state_tensor(sim)
    dof_states=gymtorch.wrap_tensor(_dof_states)
    # print(dof_states)
    return _dof_states

def acquire_mass_tensor():
    name_of_actor=gym.get_actor_name(env, actor_handle)
    _mass_matrix=gym.acquire_mass_matrix_tensor(sim, name_of_actor)
    mass_matrix=gymtorch.wrap_tensor(_mass_matrix)
    return mass_matrix


def set_root_tensor(num_actors, _root_tensor, root_positions, offsets):
    ######失败########
    root_positions += offsets
    gym.set_actor_root_state_tensor(sim, _root_tensor)

    #####
    # actor_indices=torch.tensor([0, 17, 22], dtype=torch.int32, device="cuda:0")
    # gym.set_actor_root_state_tensor_indexed(sim, _root_tensor, gymtorch.unwrap_tensor(actor_indices), 3)

def set_dof_state(actorhandle, dof_states_stand_targets):
    props=gym.get_actor_dof_properties(env, actorhandle)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(10000.0)
    props["damping"].fill(2000.0)
    gym.set_actor_dof_properties(env, actorhandle, props)
    print(dof_states_stand_targets)
    _dof_states_targets=gymtorch.unwrap_tensor(dof_states_stand_targets)
    gym.set_dof_state_tensor(sim, _dof_states_targets)

def set_dof_state_indexed(props,actorhandle,actor_indices, num_indices):
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(100000.0)
    props["damping"].fill(2000.0)
    gym.set_actor_dof_properties(env, actorhandle, props)
    _dof_states=acquire_dof_state()
    dof_states=gymtorch.wrap_tensor(_dof_states)
    _actor_indices = torch.tensor([0, 17, 33, 42], dtype=torch.int32, device="cuda:0")
    gym.set_dof_state_tensor_indexed(sim, _dof_states, gymtorch.unwrap_tensor(_actor_indices), 4)       #指定的是actor而不是dof

def set_dof_pos_target(actorhandle,positions_tensor):
    # props["driveMode"].fill(gymapi.DOF_MODE_POS)
    # props["stiffness"].fill(1000000.0)
    # props["damping"].fill(20000.0)
    # gym.set_actor_dof_properties(env, actorhandle, props)
    _positions_tensor=gymtorch.unwrap_tensor(positions_tensor)
    # print(positions_tensor)
    gym.set_dof_position_target_tensor(sim, _positions_tensor)

def set_dof_target_velocity(actorhandle, dof_vel_targets):
    # props["driveMode"].fill(gymapi.DOF_MODE_POS)
    # props["stiffness"].fill(11000.0)
    # props["damping"].fill(60000.0)
    # gym.set_actor_dof_properties(env, actorhandle, props)
    wheel_index, num_wheel = find_wheel_index()
    for i in wheel_index:
        dof_vel_targets[i] = -2
    _dof_vel_targets=gymtorch.unwrap_tensor(dof_vel_targets)
    # print(dof_vel_targets.dtype)
    # print(dof_vel_targets)
    gym.set_dof_velocity_target_tensor(sim, _dof_vel_targets)
    # gym.set_dof_target_velocity(env, )

def set_dof_force_tensor(actorhandle, effort_target):
    # props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    # props["stiffness"].fill(0.0)
    # props["damping"].fill(0)
    # gym.set_actor_dof_properties(env, actorhandle, props)
    # num_dofs = gym.get_sim_dof_count(sim)
    # actions = 100.0 - 200.0 * torch.rand(num_dofs, dtype=torch.float32, device="cuda:0")
    # print(actions)

    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_target))


def apply_body_force():

    # gym.apply_rigid_body_force_tensors(sim, force_tensor, torque_tensor, gymapi.ENV_PLACE)
    pass



############控制操作###########
def vehicle_standing(env, actorhandle):
    # props = get_dof_props(env, actorhandle)
    # apply_actor_position(actorhandle=actorhandle, props=props)
    ######tensor
    dof_states_stand_targets = torch.zeros(25, 2, dtype=torch.float32, device="cuda:0")
    set_dof_state(actorhandle=actorhandle, dof_states_stand_targets=dof_states_stand_targets)

def wheel_moving(actorhandle, dof_states_moving_targets, target_vel):
    wheel_index, num_wheel=find_wheel_index()
    for i in wheel_index:
        dof_states_moving_targets[i]+=target_vel
    # print(dof_states_moving_targets)
    # set_dof_velocity(actorhandle=actorhandle, dof_vel_targets=dof_states_moving_targets)
    # set_dof_state(props=props, actorhandle=actorhandle, dof_states_stand_targets=dof_states_moving_targets)
    set_dof_props(env, actorhandle)
    # print(dof_states_moving_targets)
    set_dof_pos_target(actorhandle, dof_states_moving_targets)
    # set_dof_force_tensor(actorhandle, dof_states_moving_targets)
    set_dof_target_velocity(actorhandle, dof_states_moving_targets)



def middle_wheel_cycle(props, actorhandle, dof_states_moving_targets, target_vel1,target_vel2, to_limit1, to_limit2):
    if to_limit2:
        to_limit1=slider_translate(props, actorhandle, dof_states_moving_targets, target_vel1,)
        if to_limit1:
            target_vel2=-target_vel2
            to_limit2=not to_limit2
    if to_limit1:
        to_limit21 = wheel_rise("ml", actorhandle, dof_states_moving_targets, target_vel2)
        to_limit22 = wheel_rise("mr", actorhandle, dof_states_moving_targets, target_vel2)
        to_limit2 = to_limit21 and to_limit22
        if to_limit2:
            target_vel1=-target_vel1
            to_limit1=not to_limit1
    return to_limit1, to_limit2, target_vel1, target_vel2

def turn_around(actorhandle, dof_states_moving_targets, target_vel):
    wheel_index, num_wheel = find_wheel_index()
    j=0
    for i in wheel_index:
        if j%2==0:
            dof_states_moving_targets[i] += target_vel
        else:
            dof_states_moving_targets[i]-=target_vel
        # j+=1
    # print(dof_states_moving_targets)
    # set_dof_velocity(actorhandle=actorhandle, dof_vel_targets=dof_states_moving_targets)
    # set_dof_state(props=props, actorhandle=actorhandle, dof_states_stand_targets=dof_states_moving_targets)
    set_dof_pos_target(actorhandle, dof_states_moving_targets)


def front_wheel_rise(actorhandle, dof_states_moving_targets,target_vel):
    to_limit1=wheel_rise("fl", actorhandle, dof_states_moving_targets, target_vel)
    to_limit2=wheel_rise("fr", actorhandle, dof_states_moving_targets, target_vel)
    to_limit=to_limit1 and to_limit2
    return to_limit


#########单一功能#########
def slider_translate(props, actorhandle, dof_states_moving_targets, target_vel,):
    lower_limit = torch.tensor(-0.7, dtype=torch.float32)
    upper_limit = torch.tensor(0.7, dtype=torch.float32)
    dof_states_moving_targets[16, 0] += target_vel  # 需要clamp
    dof_states_moving_targets[16, 0] = torch.clamp(dof_states_moving_targets[16, 0], lower_limit.to("cuda:0"), upper_limit.to("cuda:0"), )
    set_dof_state(props=props, actorhandle=actorhandle, dof_states_stand_targets=dof_states_moving_targets)
    if dof_states_moving_targets[16, 0] <= lower_limit or dof_states_moving_targets[16, 0] >= upper_limit:
        to_limit = True
    else:
        to_limit = False
    # print(dof_states_moving_targets[16, 0])
    return to_limit

def wheel_rise(wheel_name:str, actorhandle, dof_states_moving_targets, target_vel):
    lower_limit = torch.tensor(-0.5, dtype=torch.float32)
    upper_limit = torch.tensor(0.5, dtype=torch.float32)
    lift1_indice, lift2_indice = find_lift_index(wheel_name)
    dof_states_moving_targets[lift1_indice, 0] += target_vel
    dof_states_moving_targets[lift2_indice, 0] -= target_vel
    dof_states_moving_targets[lift1_indice, 0] = torch.clamp(dof_states_moving_targets[lift1_indice, 0], lower_limit, upper_limit, )
    dof_states_moving_targets[lift2_indice, 0] = torch.clamp(dof_states_moving_targets[lift2_indice, 0], lower_limit, upper_limit, )
    set_dof_state( actorhandle=actorhandle, dof_states_stand_targets=dof_states_moving_targets)
    if lower_limit>=dof_states_moving_targets[lift1_indice, 0] or dof_states_moving_targets[lift1_indice, 0]>=upper_limit or \
            dof_states_moving_targets[lift2_indice, 0] <=lower_limit or dof_states_moving_targets[lift2_indice, 0]>=upper_limit:
        to_limit = True
    else:
        to_limit = False
    return to_limit



def find_wheel_index():
    wheel_index=[]
    actor_index=gym.get_actor_dof_names(env, actor_handle)
    # wheel_index.append(s for s in actor_index if "wheel" in s)
    for s in actor_index:
        if "wheel" in s:
            # wheel_name.append(s)
            _index=gym.find_actor_dof_index(env,actor_handle, s, gymapi.DOMAIN_ENV)
            wheel_index.append(_index)
    return wheel_index, len(wheel_index)

def find_lift_index(lift_name):
    lift1_index,lift2_index=-1,-1
    actor_index = gym.get_actor_dof_names(env, actor_handle)
    for name in actor_index:
        if lift_name+"_lift1" in name:
            lift1_index = gym.find_actor_dof_index(env, actor_handle, name, gymapi.DOMAIN_ENV)
        elif lift_name+"_lift2" in name:
            lift2_index = gym.find_actor_dof_index(env, actor_handle, name, gymapi.DOMAIN_ENV)
    if (lift1_index or lift2_index) ==-1:
        print("The lift which you find is not exist!")
        return
    return lift1_index, lift2_index


if __name__=="__main__":
    gym=gymapi.acquire_gym()
    sim=create_sim()
    create_plane(sim)
    asset=load_assets(sim)
    env, actor_handle=env_actor_create(sim,asset)
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    viewer=create_viewer(sim)
    gym.prepare_sim(sim)
    ########计算bodies、joints、dofs的数量###########
    num_bodies=gym.get_actor_rigid_body_count(env, actor_handle)
    num_joints=gym.get_actor_joint_count(env, actor_handle)
    num_dofs=gym.get_actor_dof_count(env, actor_handle)
    print(f"-----num_bodies:{num_bodies},------num_joints:{num_joints},------num_dofs:{num_dofs}--------")

    # body_actor_states, body_env_states, body_sim_states=get_body_states()
    # props=get_dof_props(env, actor_handle)

    #力控制
    # apply_actor_effort(props,actor_handle)
    #位置控制
    # apply_actor_position(actorhandle=actor_handle, props=props)
    #速度控制
    # apply_actor_velocity(actorhandle=actor_handle)

    _root_tensor, root_positions, root_orientations, root_linvels, root_angvels=acquire_root_tensor()
    root_tensor=gymtorch.wrap_tensor(_root_tensor)
    save_root_tensor=root_tensor.clone()
    _dof_state=acquire_dof_state()


    dof_states_moving_targets = torch.zeros(25, 2, dtype=torch.float32, )
    positions_tensor=torch.zeros(25, dtype=torch.float32, )
    velocity_tensor=torch.zeros(25, dtype=torch.float32)
    # offsets = torch.tensor([0, 0, 0.1]).repeat(1).to("cuda:0")
    # gym.refresh_actor_root_state_tensor(sim)  # 在重置时使用，不要每帧都刷新       #不刷新机器人会消失？
    step=0
    to_limit=False
    to_limit1=False
    to_limit2=True
    target_vel=-0.005
    target_vel1=0.001
    target_vel2=0.001


    # set_dof_pos_target(actor_handle, positions_tensor)
    # set_dof_target_velocity(actor_handle, velocity_tensor)          #不能用use_gpu_pipeline  #why?
    # set_dof_force_tensor(actorhandle=actor_handle)
    while not gym.query_viewer_has_closed(viewer):
        ######物理仿真########
        gym.simulate(sim)
        gym.fetch_results(sim,True)

        ######viewer仿真########
        gym.step_graphics(sim)      #使模拟的视觉表示与物理状态同步
        gym.draw_viewer(viewer, sim, True)      #查看器中渲染最新的快照
        gym.sync_frame_time(sim)        #视觉更新频率与实时同步

        # 刷新root——tensor
        step+=1
        if step%100==0:
            # gym.refresh_dof_state_tensor(sim)            #使用模型中的最新数据填充张量
            # set_root_tensor(1, _root_tensor=_root_tensor, root_positions=root_positions, offsets=offsets)
            # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(save_root_tensor))
            # set_dof_state(_dof_state)
            # set_dof_force()
            pass

        # gym.refresh_mass_matrix_tensors(sim)
        # gym.refresh_dof_state_tensor(sim)
        # print(acquire_mass_tensor().shape)
        # print(acquire_mass_tensor())
        # print(gymtorch.wrap_tensor(acquire_dof_state()).shape)
        # print(gymtorch.wrap_tensor(acquire_dof_state()))

        # 运动控制
        # vehicle_standing(env, actorhandle=actor_handle)
        # turn_around(actor_handle,positions_tensor,target_vel)
        # to_limit=slider_translate(props=props,actorhandle=actor_handle,dof_states_moving_targets=dof_states_moving_targets,target_vel=target_vel,)
        # to_limit=wheel_rise("fr",actorhandle=actor_handle, dof_states_moving_targets=dof_states_moving_targets, target_vel=target_vel,)
        to_limit=front_wheel_rise(actor_handle, dof_states_moving_targets, target_vel)
        # wheel_moving(actorhandle=actor_handle, dof_states_moving_targets=velocity_tensor, target_vel=target_vel)

        # to_limit1,to_limit2, target_vel1, target_vel2=middle_wheel_cycle(props=props,actorhandle=actor_handle,dof_states_moving_targets=dof_states_moving_targets,target_vel1=target_vel1, target_vel2=target_vel2,to_limit1=to_limit1,to_limit2=to_limit2)

        if to_limit:
            target_vel = -target_vel
        # print(to_limit1, to_limit2)
        if to_limit1:
            if to_limit2:
                target_vel1=-target_vel1
        if to_limit2:
            if to_limit1:
                print("+++++++++++++++++++")
                target_vel2=-target_vel2

        ####键盘控制#####
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action=="reset" and evt.value>0:
                gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)