import math
import random
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil
# from isaacgym.terrain_utils import *
from vehicle_Isaacgym.vehicle import assets
Kp=200
def standing():

    for i in range(num_envs):
        current_dof = gym.get_actor_dof_states(envs[i], actor_handles[i], gymapi.STATE_POS)
        print(current_dof['pos'])
        print("+++++++++++++++++++")
        # position=((dof_target_position-current_dof['pos'])+current_dof['pos']).copy()
        pos_gap=torch.from_numpy(dof_target_position-current_dof['pos']).to(torch.float32)
        # position=torch.from_numpy(position,).to(torch.float32)
        torques = torch.clip(
            Kp *pos_gap,
            -180., 180.)
        print(torques)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))
        # gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_ALL)
        # gym.set_actor_dof_position_targets(envs[i], actor_handles[i],position)
        # dof_hand=gym.find_actor_dof_handle(envs[i],)
        # gym.set_dof_target_position(envs[i],dof_handle, position)

def moving():
    dof_vel=dof_states['vel']
    for i in range(num_envs):
        wheel_index=[]
        actor_index=gym.get_actor_dof_names(envs[i], actor_handles[i])
        # wheel_index.append(s for s in actor_index if "wheel" in s)
        for s in actor_index:
            if "wheel" in s:
                # wheel_name.append(s)
                _index=gym.find_actor_dof_index(envs[i],actor_handles[i], s, gymapi.DOMAIN_ENV)
                wheel_index.append(_index)
    # wheel_index_tensor=torch.zeros(num_envs,len(wheel_index))
    # wheel_index_tensor[:,:]=torch.tensor(wheel_index,dtype=torch.float32)
    # wheel_vel=torch.ones(num_envs, len(wheel_index))
    # gym.set_dof_velocity_target_tensor_indexed(sim, gymtorch.unwrap_tensor(wheel_vel), gymtorch.unwrap_tensor(wheel_index_tensor), len(wheel_index))
    for i in wheel_index:
        dof_vel[i]=1
        dof_position[i] += 0.15 * dt
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_ALL)
        print(dof_states)


def front_wheel_rise():
    pass

def middle_wheel_translate():
    dof_vel=dof_states['vel']
    for i in range(num_envs):
        wheel_index=[]
        actor_index=gym.get_actor_dof_names(envs[i], actor_handles[i])
        # wheel_index.append(s for s in actor_index if "wheel" in s)
        for s in actor_index:
            if "wheel" in s:
                # wheel_name.append(s)
                _index=gym.find_actor_dof_index(envs[i],actor_handles[i], s, gymapi.DOMAIN_ENV)
                wheel_index.append(_index)
    for i in wheel_index:
        dof_vel[i]=1
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_ALL)

def barries1():
    pass



import torch
def clamp(x,min_value,max_value):
    return max(min(x,max_value),min_value)

default_joint_angle={

   "LF_HAA": 0.03,    # [rad]
   "LH_HAA": 0.03,    # [rad]
   "RF_HAA": -0.03,   # [rad]
   "RH_HAA": -0.03,   # [rad]

   "LF_HFE": 0.4,     # [rad]
   "LH_HFE": -0.4,    # [rad]
   "RF_HFE": 0.4,     # [rad]
   "RH_HFE": -0.4,    # [rad]

   "LF_KFE": -0.8,    # [rad]
   "LH_KFE": 0.8,     # [rad]
   "RF_KFE": -0.8,    # [rad]
   "RH_KFE": 0.8     # [rad]
    }
Pi_default_dof={
    "FL_crotch_joint": -0.03,
    "HL_crotch_joint": -0.03,
    "FR_crotch_joint": 0.03,
    "HR_crotch_joint": 0.03,

    "FL_thigh_joint": -0.4,
    "HL_thigh_joint": 0.4,
    "FR_thigh_joint": 0.4,
    "HR_thigh_joint": -0.4,

    "FL_calf_joint": 0.8,
    "HL_calf_joint": 0.8,
    "FR_calf_joint": -0.8,
    "HR_calf_joint": -0.8
}
vehicle_default_dof={
# "slider_joint": +-0.7,
# "lift1_joint": +-0.5
# "lift2_joint": +-0.5
# "steer_joint": +-0.7

    "slider_joint": 0.,

    "fl_lift1_joint": 0.,
    'fr_lift1_joint': 0,
    'rl_lift1_joint': 0,
    'rr_lift1_joint': 0,
    'ml_lift1_joint': 0,
    'mr_lift1_joint': 0,

    'fl_lift2_joint': 0,
    'fr_lift2_joint': 0,
    'rl_lift2_joint': 0,
    'rr_lift2_joint': 0,
    'ml_lift2_joint': 0,
    'mr_lift2_joint': 0,

    'fl_steer_joint': 0,
    'fr_steer_joint': 0,
    'rl_steer_joint': -0,
    'rr_steer_joint': 0,
    'ml_steer_joint': 0,
    'mr_steer_joint': 0,

    'fl_wheel_joint': 0,
    'fr_wheel_joint': 0,
    'rl_wheel_joint': 0,
    'rr_wheel_joint': 0,
    'ml_wheel_joint': 0,
    'mr_wheel_joint': 0
}


computer_device_id=0
graphics_device_id=0

gym=gymapi.acquire_gym()
sim_params=gymapi.SimParams()

sim_params.dt=1/100
sim_params.substeps=2
sim_params.up_axis=gymapi.UP_AXIS_Z
sim_params.gravity=gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.gravity=gymapi.Vec3(0.0, 0.0, 0)

# sim_params.use_gpu_pipeline = True
sim_params.use_gpu_pipeline=False

sim_params.physx.use_gpu=True
sim_params.physx.solver_type=1
sim_params.physx.num_position_iterations=6
sim_params.physx.num_velocity_iterations=1
sim_params.physx.contact_offset=0.01
sim_params.physx.rest_offset=0.0
# sim_params.physx.bounce_threshold_velocity=2*9.8

#
sim=gym.create_sim(computer_device_id,graphics_device_id,gymapi.SIM_PHYSX,sim_params)
# gym.prepare_sim(sim)
# # print("+++++++++++++++++++++++++++++++")
plane_params=gymapi.PlaneParams()
plane_params.normal=gymapi.Vec3(0, 0, 1)
plane_params.distance=0
plane_params.static_friction=1
plane_params.dynamic_friction=1
plane_params.restitution=0.5

gym.add_ground(sim,plane_params)


####################地形设置#########################
# num_terrain=9
# terrain_width=12
# terrain_length=12
# horizontal_scale=0.25
# vertical_scale=0.005
# num_rows=int(terrain_width/horizontal_scale)
# num_cols=int(terrain_length/horizontal_scale)
# height_field=np.zeros((int(np.sqrt(num_terrain))*num_rows, int(np.sqrt(num_terrain))*num_cols),dtype=np.int16)
#
# def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
#
# height_field[0:num_rows, 0:num_cols]=random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
# height_field[num_rows:2*num_rows,0:num_cols]=sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
# height_field[2*num_rows:3*num_rows,0:num_cols]=pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
#
# height_field[0:num_rows, num_cols:2*num_cols]=discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5, min_size=1, max_size=5, num_rects=20).height_field_raw
# height_field[num_rows:2*num_rows,num_cols:2*num_cols]=wave_terrain(new_sub_terrain(), num_waves=2, amplitude=1.).height_field_raw
# height_field[2*num_rows:3*num_rows,num_cols:2*num_cols]=stairs_terrain(new_sub_terrain(), step_width=0.75,step_height=-0.5).height_field_raw
#
# height_field[0:num_rows, 2*num_cols:3*num_cols]=stepping_stones_terrain(new_sub_terrain(), stone_size=1., stone_distance=2., max_height=0.5, platform_size=0.).height_field_raw
# height_field[num_rows:2*num_rows, 2*num_cols:3*num_cols]=pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
# height_field[2*num_rows:3*num_rows, 2*num_cols:3*num_cols]=random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
#
# verticals, triangles= convert_heightfield_to_trimesh(height_field, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
# tm_params=gymapi.TriangleMeshParams()
# tm_params.nb_vertices=verticals.shape[0]
# tm_params.nb_triangles=triangles.shape[0]
# tm_params.transform.p.x=-1.
# tm_params.transform.p.y=-1.
# gym.add_triangle_mesh(sim, verticals.flatten(), triangles.flatten(), tm_params)


#
# ###############机器人载入#####################
asset_root="/home/mutong/RobotDoc/vehicle_Isaacgym/vehicle/assets"
asset_file="urdf/vehicle/cwego.urdf"
# asset_file="urdf/anymal_c/urdf/anymal.urdf"
# asset_file="urdf/Pi/urdf/pi.urdf"
# asset_file="urdf/a1/urdf/a1.urdf"
#

asset_options = gymapi.AssetOptions()
# asset_options.density = 100.
asset_options.flip_visual_attachments = True
# asset_options.fix_base_link = True
asset_options.use_mesh_materials = True
asset=gym.load_asset(sim,asset_root,asset_file,asset_options)



# for i in range(num_dof):
#     if has_limits[i]:
#         if dof_types[i]==gymapi.DOF_ROTATION:
#             lower_limits[i]=clamp(lower_limits[i],-math.pi,math.pi)
#             upper_limits[i]=clamp(upper_limits[i],-math.pi,math.pi)
#
#         if lower_limits[i]>0.0:
#             defaults[i]=lower_limits[i]
#         elif upper_limits[i]<0.0:
#             defaults[i]=upper_limits[i]
#     else:
#         if dof_types[i]==gymapi.DOF_ROTATION:
#             lower_limits[i]=-math.pi
#             upper_limits[i]=math.pi
#         elif dof_types[i]==gymapi.DOF_TRANSLATION:
#             lower_limits[i]=-1.0
#             upper_limits[i]=1.0
#
#     dof_position[i]=defaults[i]
#     if dof_types[i]==gymapi.DOF_ROTATION:
#         speeds[i]= clamp(2*(upper_limits[i]-lower_limits[i]), 0.25*math.pi, 3.0*math.pi)
#     else:
#         speeds[i]=clamp(2*(upper_limits[i]-lower_limits[i]), 0.1, 7.0)
#
# for i in range(num_dof):
#     print("DOF %d" %i)
#     print("  Name:      '%s'" % dof_names[i])
#     print("  Type:      %s"  %gym.get_dof_type_string(dof_types[i]))
#     print("  Stiffness:     %r"  %stiffnesses[i])
#     print("  Damping:     %r"  %damping[i])
#     print("  Armature:     %r"  %armatures[i])
#     print("  Limited?     %r"  %has_limits[i])
#     if has_limits[i]:
#         print("     Lower   %f " % lower_limits[i])
#         print("     Upper   %f " % upper_limits[i])

# #########机器人的create##########
num_envs=1
envs_per_row=3
env_spaceing=6.
env_lower=gymapi.Vec3(-env_spaceing, -env_spaceing, -env_spaceing)
env_upper=gymapi.Vec3(env_spaceing, env_spaceing, env_spaceing)
envs=[]
actor_handles=[]
scale=[1.0, 2.0, 3.0, 4.0]

for i in range(num_envs):
    env=gym.create_env(sim,env_lower,env_upper,envs_per_row)
    envs.append(env)
    height=random.uniform(1.0,2.5)
    pose=gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)


    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)
    # gym.set_actor_scale(env, actor_handle, scale[3])

    # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
    # props=gym.get_actor_dof_properties(envs[i],actor_handles[i])
    # props['driveMode'].fill(gymapi.DOF_MODE_POS)
    # props['stiffness'].fill(1000)
    # props['damping'].fill(200.0)
    # gym.set_actor_dof_properties(envs[i], actor_handles[i],props)


#############机器人关节驱动###################
dof_names=gym.get_asset_dof_names(asset)
dof_props=gym.get_asset_dof_properties(asset)
num_dof=gym.get_asset_dof_count(asset)
dof_states=np.zeros(num_dof, dtype=gymapi.DofState.dtype)
dof_types=[gym.get_asset_dof_type(asset,i)for i in range(num_dof)]
dof_position=dof_states['pos']
#
dof_props["driveMode"]=gymapi.DOF_MODE_POS
dof_props['stiffness'].fill(2)
dof_props['damping'].fill(2)

stiffnesses= dof_props['stiffness']     #刚度
damping=dof_props['damping']    #阻尼
armatures=dof_props['armature']     #电
has_limits=dof_props['hasLimits']
lower_limits=dof_props['lower']
upper_limits=dof_props['upper']

defaults=np.zeros(num_dof)
speeds=np.zeros(num_dof)


# #############查看器设置##################
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

gym.subscribe_viewer_keyboard_event(viewer,gymapi.KEY_SPACE,"space_shoot")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

current_dof=0
dt=1/60.
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4
dof_target_position=np.zeros(num_dof)

anim_state=ANIM_SEEK_LOWER
print("Animating DOF %d('%s')"% (current_dof, dof_names[current_dof]))


names=gym.get_asset_dof_names(asset)
from isaacgym import gymtorch
# net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
# contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(-1, 3)
# last_contact_forces=torch.zeros_like(contact_forces)
# base_index = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], "base")
# mass_matrix =gym.acquire_mass_matrix_tensor(sim, "MyActor")
# mass=gymtorch.wrap_tensor(mass_matrix)
# print(mass_matrix.shape)
# print(mass.shape)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
print(dof_state_tensor.dtype)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)

# dof_pos=dof_state.view(12, 2)[...,0]

# default_dof_pos=torch.zeros_like(dof_pos,dtype=torch.float,requires_grad=False)
for i in range(num_dof):
    name=names[i]
    angle=vehicle_default_dof[name]
    dof_target_position[i]=angle
# print(dof_position)
# # default_dof_pos=[0.03, 0.03, -0.03, -0.03, 0.4, -0.4, 0.4, -0.4, -0.8, 0.8, -0.8, 0.8, ]
# # # default_dof_pos=[0.03, 0.03, -0.03, -0.03, 0.4, -0.4, 0.4, -0.4, 0, 0, 0, 0, ]
# dof_pos= default_dof_pos


while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # if torch.all(contact_forces==last_contact_forces):
    #     print("?????????????????????????????????")
    #     print("it\'s change")
    # last_contact_forces=contact_forces
    # print("""""""")
    # print(contact_forces)
    # print(names)


    # print("++++++++++++++++++++++++++")
    # print(mass_matrix.shape)
    # print(mass)

    # speed=speeds[current_dof]
    # #
    # if anim_state==ANIM_SEEK_LOWER:
    #     dof_position[current_dof]-=speed* dt
    #     if dof_position[current_dof]<=lower_limits[current_dof]:
    #         dof_position[current_dof]=lower_limits[current_dof]
    #         anim_state=ANIM_SEEK_UPPER
    # elif anim_state==ANIM_SEEK_UPPER:
    #     dof_position[current_dof]+=speed* dt
    #     if dof_position[current_dof]>=upper_limits[current_dof]:
    #         dof_position[current_dof]=upper_limits[current_dof]
    #         anim_state=ANIM_SEEK_DEFAULT
    # elif anim_state==ANIM_SEEK_DEFAULT:
    #     dof_position[current_dof]-=speed*dt
    #     if dof_position[current_dof] <= defaults[current_dof]:
    #         dof_position[current_dof] = defaults[current_dof]
    #         anim_state = ANIM_FINISHED
    # elif anim_state == ANIM_FINISHED:
    #     dof_position[current_dof] = defaults[current_dof]
    #     current_dof = (current_dof + 1) % num_dof
    #     anim_state = ANIM_SEEK_LOWER


    # standing()
    # moving()
    # middle_wheel_translate()



    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    # gym.refresh_dof_state_tensor(sim)
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

