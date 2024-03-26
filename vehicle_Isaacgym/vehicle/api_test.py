import math
import numpy as np

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.terrain_utils import *
import cv2
import matplotlib.pyplot as plt
import torch
Device="cuda:0"

def create_sim():
    compute_device_id=0
    graphics_device_id=0
    gym=gymapi.acquire_gym()
    ### set common parameters ###
    sim_params=gymapi.SimParams()
    sim_params.dt=1/100.
    sim_params.substeps=1
    sim_params.up_axis=gymapi.UP_AXIS_Z
    sim_params.gravity=gymapi.Vec3(0.,0.,-9.8)
    sim_params.use_gpu_pipeline=True                    #使返回张量留在gpu上，容易导致报错
    ### set PhysX-specific parameters ###
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type=1
    sim_params.physx.num_position_iterations=6
    sim_params.physx.num_velocity_iterations=1
    sim_params.physx.contact_offset=0.02
    sim_params.physx.rest_offset=0.0
    sim_params.physx.num_threads=4
    sim_params.physx.bounce_threshold_velocity=0.2
    sim_params.physx.max_depenetration_velocity=100.0
    sim_params.physx.default_buffer_size_multiplier=5.0
    sim_params.physx.num_subscenes=4
    sim_params.physx.contact_collection=gymapi.ContactCollection(1)
    # sim_params.physx.bounce_threshold_velocity=2*9.8*1/240./2
    sim= gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    return sim
def create_plane(sim):
    plane_params=gymapi.PlaneParams()
    plane_params.normal=gymapi.Vec3(0,0,1)  #Z - up
    plane_params.distance=0
    plane_params.static_friction=5
    plane_params.dynamic_friction=5
    plane_params.restitution=0          #控制与地平面碰撞的弹性
    gym.add_ground(sim, plane_params)

def create_terrain():
    # create all available terrain types
    num_terains = 8
    terrain_width = 12.
    terrain_length = 12.
    horizontal_scale = 0.25  # [m]
    vertical_scale = 0.005  # [m]
    num_rows = int(terrain_width / horizontal_scale)
    num_cols = int(terrain_length / horizontal_scale)
    heightfield = np.zeros((num_terains * num_rows, num_cols), dtype=np.int16)

    def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                                             horizontal_scale=horizontal_scale)

    heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2,
                                                        downsampled_scale=0.5).height_field_raw
    heightfield[num_rows:2 * num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
    heightfield[2 * num_rows:3 * num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
    heightfield[3 * num_rows:4 * num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5,
                                                                           min_size=1., max_size=5.,
                                                                           num_rects=20).height_field_raw
    heightfield[4 * num_rows:5 * num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2.,
                                                             amplitude=1.).height_field_raw
    heightfield[5 * num_rows:6 * num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75,
                                                               step_height=-0.5).height_field_raw
    heightfield[6 * num_rows:7 * num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75,
                                                                       step_height=-0.5).height_field_raw
    heightfield[7 * num_rows:8 * num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
                                                                        stone_distance=1., max_height=0.5,
                                                                        platform_size=0.).height_field_raw

    # add the terrain as a triangle mesh
    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                         vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = -1.
    tm_params.transform.p.y = -1.
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

def load_assets(sim):
    asset_root = "/home/mutong/RobotDoc/vehicle_Isaacgym/vehicle_Isaacgym/vehicle/assets"
    asset_file = "urdf/vehicle/cwego.urdf"
    # asset_file="urdf/anymal_c/urdf/anymal.urdf"
    # asset_file="urdf/anymal_c/urdf/anymal_minimal.urdf"
    asset_options=gymapi.AssetOptions()
    asset_options.armature = 0.0
    asset_options.flip_visual_attachments = True
    asset_options.collapse_fixed_joints=True
    # asset_options.fix_base_link = True
    # asset_options.use_mesh_materials = True
    asset_options.disable_gravity=False
    # asset_options.replace_cylinder_with_capsule=True        #用胶囊替代圆柱体
    asset=gym.load_asset(sim,asset_root,asset_file,asset_options)
    return asset
def env_actor_create(sim,asset, num_envs):
    spacing=6.0
    envs_per_row = 8
    env_lower=gymapi.Vec3(-spacing, -spacing, -spacing)
    env_upper=gymapi.Vec3(spacing,spacing,spacing)
    envs=[]
    ActorHandles=[]
    cameras=[]
    for i in range(num_envs):
        env=gym.create_env(sim, env_lower, env_upper, envs_per_row)            #create_env
        envs.append(env)
        pose=gymapi.Transform()
        pose.p=gymapi.Vec3(0,0,0)
        # pose.r=gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), -math.pi/2)
        actor_handle=gym.create_actor(env, asset, pose, "MyActor", 0, 0)        #create_actor
        ActorHandles.append(actor_handle)
        gym.enable_actor_dof_force_sensors(env, actor_handle)
        camera_actor_handle = gym.find_actor_rigid_body_handle(env, actor_handle, "camera_link")
        # print("camera_handle:", camera_actor_handle)
        loc = [2.1, 0, 1.]
        target_loc = [1, 0, 0]
        angle = torch.tensor(10)
        image_width = 640
        image_height = 640

        camera_handle = camera_sensors(env, 0, loc, target_locs=target_loc, angle=angle,
                                       image_width=image_width, image_height=image_height)
        cameras.append(camera_handle)
        # print(camera_handle)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output5.mp4', fourcc, 30.0, (image_width, image_height))
    return envs, ActorHandles, out, cameras

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

def set_dof_props(envs, ActorHandles):
    for j in range(len(envs)):
        props=gym.get_actor_dof_properties(envs[j], ActorHandles[j],)
        # print(props['driveMode'])
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(40000.0)
        props["damping"].fill(2000.0)
        wheel_index, num_wheel = find_wheel_index()
        for i in wheel_index:
            props["driveMode"][i]=gymapi.DOF_MODE_VEL
            props["damping"][i]=1500.0
        # print(props["driveMode"])
        gym.set_actor_dof_properties(envs[j], ActorHandles[j], props)
        # print(props['driveMode'])

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
    # print(vel_targets)
    gym.set_actor_dof_velocity_targets(env, actorhandle, vel_targets)


def set_body_states(body_states):
    gym.set_actor_rigid_body_states(env, actor_handle, body_states, gymapi.STATE_ALL)
    gym.set_env_rigid_body_states(env, body_states, gymapi.STATE_ALL)
    gym.set_sim_rigid_body_states(sim, body_states, gymapi.STATE_ALL)

def set_dof_states():
    dof_states=gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


def camera_sensors(env, body_handle, locs, target_locs, angle, image_width, image_height):
    camera_props=gymapi.CameraProperties()
    camera_props.width=image_width
    camera_props.height=image_height
    camera_handle=gym.create_camera_sensor(env, camera_props)

    # gym.set_camera_location(camera_handle, env, gymapi.Vec3(locs[0], locs[1], locs[2]), gymapi.Vec3(target_locs[0], target_locs[1], target_locs[2]))
    transform= gymapi.Transform()
    transform.p = gymapi.Vec3(locs[0], locs[1], locs[2])
    # transform.p = gymapi.Vec3(0, 0, 0)
    transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(angle))
    # gym.set_camera_transform(camera_handle, env, transform)
    gym.attach_camera_to_body(camera_handle, env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)

    # gym.render_all_camera_sensors(sim)
    return  camera_handle

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

def acquire_contact_force():
    _net_contact_force=gym.acquire_net_contact_force_tensor(sim)
    net_contact_force=gymtorch.wrap_tensor(_net_contact_force)
    return net_contact_force

def acquire_rigid_body_state():
    _rb_states=gym.acquire_rigid_body_state_tensor(sim)
    rb_states=gymtorch.wrap_tensor(_rb_states)
    # print(rb_states.cpu().numpy())

def set_root_tensor(root_positions, ):
    ######失败########
    # root_positions += offsets
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_positions))

    #####
    # actor_indices=torch.tensor([0, 17, 22], dtype=torch.int32, device="cuda:0")
    # gym.set_actor_root_state_tensor_indexed(sim, _root_tensor, gymtorch.unwrap_tensor(actor_indices), 3)

def set_dof_state(actorhandle, dof_states_stand_targets):
    # props=gym.get_actor_dof_properties(env, actorhandle)
    # props["driveMode"].fill(gymapi.DOF_MODE_POS)
    # props["stiffness"].fill(10000.0)
    # props["damping"].fill(2000.0)
    # gym.set_actor_dof_properties(env, actorhandle, props)
    # print(dof_states_stand_targets)
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
    positions_tensors=positions_tensor.unsqueeze(1).expand(0,4)
    _positions_tensors=gymtorch.unwrap_tensor(positions_tensors)
    # print(positions_tensor)
    gym.set_dof_position_target_tensor(sim, _positions_tensors)

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
        dof_states_moving_targets[i]+=-20
    # print(dof_states_moving_targets)
    # set_dof_velocity(actorhandle=actorhandle, dof_vel_targets=dof_states_moving_targets)
    # set_dof_state( actorhandle=actorhandle, dof_states_stand_targets=dof_states_moving_targets)

    # print(dof_states_moving_targets)
    # set_dof_pos_target(actorhandle, dof_states_moving_targets)
    # set_dof_force_tensor(actorhandle, dof_states_moving_targets)
    # print(dof_states_moving_targets)
    # set_dof_target_velocity(actorhandle, dof_states_moving_targets)



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
    to_limit1 = wheel_rise("ml", actorhandle, dof_states_moving_targets, target_vel)
    to_limit2 = wheel_rise("mr", actorhandle, dof_states_moving_targets, target_vel)
    to_limit1 = wheel_rise("rl", actorhandle, dof_states_moving_targets, target_vel)
    to_limit2 = wheel_rise("rr", actorhandle, dof_states_moving_targets, target_vel)
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
    lower_limit = torch.tensor(-0.5, dtype=torch.float32, device="cuda:0")
    upper_limit = torch.tensor(0.5, dtype=torch.float32, device="cuda:0")
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


def pre_physics_step(actions):
    # print(actions.cpu().numpy())
    # actions=actions.clone().to("cuda:0")
    actions_pos=actions.clone()
    actions_force=torch.zeros_like(actions_pos).to("cuda:0")
    wheel_index,_=find_wheel_index()
    # for i in wheel_index:
    #     actions_pos[:,i]=-1

    # print(actions_pos)
    # 一次控制多步运动
    for i in range(1):
        # torques=torch.clip(self.Kp*(self.action_scale*self.actions+self.default_dof_pos-self.dof_pos)-self.Kd*self.dof_vel, -80,80)     #TODO:急需修改
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(actions_pos))
        # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions_force))          # 神经网络输出的直接是torque控制wheel
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions_pos))
        # print("step once")
        # self.torques=torques.view(self.torques.shape)
        gym.simulate(sim)
        gym.refresh_dof_state_tensor(sim)


if __name__=="__main__":
    gym=gymapi.acquire_gym()
    sim=create_sim()
    create_plane(sim)
    # create_terrain()
    asset=load_assets(sim)
    envs, actor_handles, out, cameras=env_actor_create(sim,asset, num_envs=1)
    # initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    viewer=create_viewer(sim)
    gym.prepare_sim(sim)
    ########计算bodies、joints、dofs的数量###########
    env=envs[0]
    actor_handle=actor_handles[0]
    num_bodies=gym.get_actor_rigid_body_count(env, actor_handle)
    actor_props=gym.get_actor_rigid_body_properties(env, actor_handle)
    for properties in actor_props:
        mass = properties.mass
        print(f"刚体质量: {mass}")
    # print(actor_props[:].mass)
    # print(gymtorch.wrap_tensor(actor_props))

    # print(gym.get_actor_joint_transforms(env, actor_handle))
    image_width = 640
    image_height = 640



    num_joints=gym.get_actor_joint_count(env, actor_handle)
    num_dofs=gym.get_actor_dof_count(env, actor_handle)
    print(f"-----num_bodies:{num_bodies},------num_joints:{num_joints},------num_dofs:{num_dofs}--------")
    from vehicle_Isaacgym.vehicle.utils.torch_jit_utils import *
    # default_root=torch.FloatTensor([0,0,0,0,0,0,1,0,0,0,0,0,0])
    # default_dof=torch_rand_float(0.5, 1.5,( 25, 2),device="cpu")
    # body_actor_states, body_env_states, body_sim_states=get_body_states()
    # props=get_dof_props(env, actor_handle)

    #力控制
    # apply_actor_effort(props,actor_handle)
    #位置控制
    # apply_actor_position(actorhandle=actor_handle, props=props)
    #速度控制
    # apply_actor_velocity(actorhandle=actor_handle)
    #
    _root_tensor, root_positions, root_orientations, root_linvels, root_angvels=acquire_root_tensor()
    gravity_vector=torch.tensor([0,0,-1],dtype=torch.float32, device=Device).unsqueeze(0)

    # root_tensor=gymtorch.wrap_tensor(_root_tensor)
    # save_root_tensor=root_tensor.clone()
    # _dof_state=acquire_dof_state()

    # dof=torch.FloatTensor([-3.0416, -0.1903,  0.1302, 0, -3.0416, -0.1903,  0.1302, 0, -3.0416, -0.1903,  0.1302, 0, -3.0416, -0.1903,  0.1302, 0,  0, -3.0416, -0.1903,  0.1302, 0, -3.0416, -0.1903,  0.1302, 0, ]).to("cuda:0")
    dof_states_moving_targets = torch.zeros(25, 2, dtype=torch.float32, device="cuda:0", requires_grad=False)
    positions_tensor=torch.ones(num_dofs, dtype=torch.float32, device="cuda:0", requires_grad=False)
    # velocity_tensor=torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0", requires_grad=False)
    key=-0.5
    velocity_tensor=torch.FloatTensor([-key, key, 0, 0, -key, key, 0, 0, -0, 0, 0, 0, -0, 0, 0, 0, 0, -0, 0, 0, 0, -0, 0, 0, 0]).to(Device)
    efforts = np.full(num_dofs, 100.0).astype(np.float32)
    efforts=torch.from_numpy(efforts)
    # offsets = torch.tensor([0, 0, 0.1]).repeat(1).to("cuda:0")
    # gym.refresh_actor_root_state_tensor(sim)  # 在重置时使用，不要每帧都刷新       #不刷新机器人会消失？
    # step=0
    # to_limit=False
    # to_limit1=False
    # to_limit2=True
    target_vel=0.005
    # target_vel1=0.001
    # target_vel2=0.001
    contact_force = acquire_contact_force().view(len(envs),-1,3)
    # dof_force=gym.get_actor_dof_forces(env, actor_handle)
    _force=gym.acquire_dof_force_tensor(sim)
    dof_forces=gymtorch.wrap_tensor(_force)

    set_dof_props(envs, actor_handles)
    # acquire_rigid_body_state()
    # set_dof_pos_target(actor_handle, positions_tensor)
    # set_dof_target_velocity(actor_handle, efforts)          #不能用use_gpu_pipeline  #why?
    while not gym.query_viewer_has_closed(viewer):
        ######物理仿真########
        gym.simulate(sim)
        gym.fetch_results(sim,True)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        ######viewer仿真########
        gym.step_graphics(sim)      #使模拟的视觉表示与物理状态同步
        gym.draw_viewer(viewer, sim, True)      #查看器中渲染最新的快照
        gym.sync_frame_time(sim)        #视觉更新频率与实时同步

        gym.render_all_camera_sensors(sim)

        # print(viewer)
        # print(contact_force[:, 0, :])

        color_image=gym.get_camera_image(sim, env, cameras[0], gymapi.IMAGE_COLOR)
        # print("----------------------------")
        # print(color_image.shape)
        image_data=np.frombuffer(color_image, dtype=np.uint8)
        image=image_data.reshape((image_height, image_width, 4))
        image = image.astype(np.uint8)  # 确保图像数据是uint8类型
        if image.ndim == 3 and image.shape[2] == 4:  # 如果图像是RGBA格式的，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # out.write(image)
        rgba_for_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        cv2.imshow('Real-time Video', rgba_for_display)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #
        #     # 显示图像
        # plt.imshow(image)
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        # 刷新root——tensor
        # step+=1
        # if step%100==0:
        #     # gym.refresh_dof_state_tensor(sim)            #使用模型中的最新数据填充张量
        #     # set_root_tensor(1, _root_tensor=_root_tensor, root_positions=root_positions, offsets=offsets)
        #     # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(save_root_tensor))
        #     # set_dof_state(_dof_state)
        #     # set_dof_force()
        #     pass
        # set_dof_force_tensor(actorhandle=actor_handle,effort_target=positions_tensor)
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
        #
        # wheel_moving(actorhandle=actor_handle, dof_states_moving_targets=velocity_tensor, target_vel=0.01)

        # to_limit1,to_limit2, target_vel1, target_vel2=middle_wheel_cycle(props=props,actorhandle=actor_handle,dof_states_moving_targets=dof_states_moving_targets,target_vel1=target_vel1, target_vel2=target_vel2,to_limit1=to_limit1,to_limit2=to_limit2)

        # if to_limit:
        #     target_vel = -target_vel
        # # print(to_limit1, to_limit2)
        # if to_limit1:
        #     if to_limit2:
        #         target_vel1=-target_vel1
        # if to_limit2:
        #     if to_limit1:
        #         print("+++++++++++++++++++")
        #         target_vel2=-target_vel2

        # set_root_tensor(default_root, )
        # set_dof_state(actor_handle,default_dof)
        # set_dof_pos_target(actor_handle, positions_tensor)
        # set_dof_state(actor_handle, dof_states_moving_targets)
        actions_pos = velocity_tensor.unsqueeze(0).expand(len(envs), -1)
        if not actions_pos.is_contiguous():
            # 如果不连续，使用 contiguous() 函数来创建一个连续的张量
            actions_pos = actions_pos.contiguous()
        # pre_physics_step(actions_pos)
        # print(contact_force)

        # print(root_orientations)
        # print(gravity_vector)
        # orient = quat_rotate_inverse(root_orientations, gravity_vector)
        # print(orient)

        ####键盘控制#####
        # for evt in gym.query_viewer_action_events(viewer):
        #     if evt.action=="reset" and evt.value>0:
        #         gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
    out.release()
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)