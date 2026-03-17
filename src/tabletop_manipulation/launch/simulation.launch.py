"""
simulation.launch.py  -- v11
================================
Key fixes:
  - rgbd_camera sensor: correct Fortress topic paths remapped via bridge
  - static_transform_publisher world→base so RViz can anchor robot
  - shoulder_pan initial_value=3.1416 in URDF makes arm face table (+X)
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'

# Startup delay constants (seconds)
DELAY_SPAWN_ROBOT  = 3.0    # wait for Gazebo world to fully load before spawning robot
DELAY_START_BRIDGE = 6.0    # wait for robot URDF to be spawned before starting ROS bridge
DELAY_START_CTRL   = 15.0   # wait for bridge topics to be available before starting controllers

PKG        = get_package_share_directory('tabletop_manipulation')
ROBOTIQ_SHARE = get_package_share_directory('robotiq_description')
# parent of robotiq_description share dir so model://robotiq_description/... resolves
ROBOTIQ_SHARE_PARENT = os.path.dirname(ROBOTIQ_SHARE)
CTRL_YAML  = os.path.join(PKG, 'config', 'ur5_controllers.yaml')
WORLD_FILE = os.path.join(PKG, 'worlds', 'tabletop.sdf')
URDF_XACRO = os.path.join(PKG, 'urdf', 'ur5_with_camera.urdf.xacro')
TMP_URDF   = '/tmp/ur5_with_camera.urdf'
# models/ dir: Ignition resolves model://mug, model://bottle etc. from here
PKG_MODELS = os.path.join(PKG, 'models')

ROS_LIB      = '/opt/ros/humble/lib'
IGN_BUILTIN  = '/usr/lib/x86_64-linux-gnu/ign-gazebo-6/plugins'
PLUGIN_PATH  = ':'.join(filter(None, [ROS_LIB, IGN_BUILTIN,
                os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')]))
RESOURCE_PATH = ':'.join(filter(None, ['/opt/ros/humble/share',
                PKG_MODELS,                          # ← YCB model packages
                ROBOTIQ_SHARE_PARENT,               # ← model://robotiq_description/...
                os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')]))
# model:// URIs in <include> tags are resolved by libsdf, which checks:
#   SDF_PATH  and  GAZEBO_MODEL_PATH  (not IGN_GAZEBO_RESOURCE_PATH)
SDF_MODEL_PATH = ':'.join(filter(None, [PKG_MODELS,
                  os.environ.get('GAZEBO_MODEL_PATH', '')]))

os.environ['IGN_GAZEBO_SYSTEM_PLUGIN_PATH'] = PLUGIN_PATH
os.environ['IGN_GAZEBO_RESOURCE_PATH']      = RESOURCE_PATH
os.environ['SDF_PATH']                      = SDF_MODEL_PATH
os.environ['GAZEBO_MODEL_PATH']             = SDF_MODEL_PATH

print(f'[sim.launch] RMW              = rmw_cyclonedds_cpp')
print(f'[sim.launch] PLUGIN_PATH      = {PLUGIN_PATH}')
print(f'[sim.launch] RESOURCE_PATH    = {RESOURCE_PATH}')

print(f'[sim.launch] xacro -> {URDF_XACRO}')
try:
    ROBOT_DESCRIPTION = subprocess.check_output(
        ['xacro', URDF_XACRO], stderr=subprocess.STDOUT
    ).decode('utf-8')
except subprocess.CalledProcessError as e:
    raise RuntimeError(f'xacro failed:\n{e.output.decode()}')

with open(TMP_URDF, 'w') as fh:
    fh.write(ROBOT_DESCRIPTION)
print(f'[sim.launch] URDF written -> {TMP_URDF}  ({len(ROBOT_DESCRIPTION)} bytes)')

CHILD_ENV = {
    'RMW_IMPLEMENTATION':            'rmw_cyclonedds_cpp',
    'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': PLUGIN_PATH,
    'IGN_GAZEBO_RESOURCE_PATH':      RESOURCE_PATH,
    'SDF_PATH':                      SDF_MODEL_PATH,
    'GAZEBO_MODEL_PATH':             SDF_MODEL_PATH,
}


def generate_launch_description():

    set_rmw         = SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    set_plugin_path = SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', PLUGIN_PATH)
    set_res_path    = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', RESOURCE_PATH)

    robot_desc_provider = Node(
        package='tabletop_manipulation',
        executable='robot_description_server.py',
        name='robot_desc_provider',
        output='screen',
        parameters=[{'robot_description': ROBOT_DESCRIPTION}],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
    )

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', WORLD_FILE, '-r', '-v', '1'],
        output='screen',
        additional_env=CHILD_ENV,
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ROBOT_DESCRIPTION,
            'use_sim_time': True,
        }],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
    )

    # NOTE: world→base_link offset (1.35, 0, 0.765, yaw=π) is baked into the
    # URDF's xacro:ur_robot <origin> so robot_state_publisher publishes it.
    # A separate static_transform_publisher for world→base would conflict with
    # the URDF's base_link→base joint, so it is intentionally absent here.

    # Static TF: camera_mount -> ur5/wrist_3_link/rgbd_camera
    # Ignition collapses fixed-joint links into their nearest moveable parent.
    # camera_mount → tool0 → wrist_3_link are all fixed joints, so the sensor ends up
    # stamped as 'ur5/wrist_3_link/rgbd_camera' but physically lives at camera_mount.
    # Identity TF: camera_mount IS the sensor origin in the SDF.
    # (camera_color_optical_frame is a FURTHER rotation from camera_mount via the URDF
    # joint, so it is NOT the same frame and must NOT be used as the parent here.)
    static_tf_sensor = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_sensor_frame',
        arguments=[
            '0', '0', '0',
            '0', '0', '0', '1',   # identity — sensor origin = camera_mount
            'camera_mount', 'ur5/wrist_3_link/rgbd_camera',
        ],
        parameters=[{'use_sim_time': True}],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )

    # Static TF: lidar_mount -> ur5/wrist_3_link/lidar
    # Ignition collapses all fixed joints onto the nearest moveable link
    # (wrist_3_link), so the LiDAR sensor gets stamped as
    # 'ur5/wrist_3_link/lidar' even though it physically lives at lidar_mount.
    # Identity TF bridges the two names so RViz + lidar_fusion_node can both
    # resolve the transform chain:
    #   world → ... → wrist_3_link → camera_mount → lidar_mount
    #                                                     ↕ identity
    #                                         ur5/wrist_3_link/lidar
    static_tf_lidar_sensor = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar_sensor_frame',
        arguments=[
            '0', '0', '0',
            '0', '0', '0', '1',   # identity — sensor origin = lidar_mount
            'lidar_mount', 'ur5/wrist_3_link/lidar',
        ],
        parameters=[{'use_sim_time': True}],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_ur5',
        arguments=[
            '-name', 'ur5', '-file', TMP_URDF,
            # Spawn the URDF root link ('world') at the world origin.
            # The actual robot base position (1.35, 0, 0.765, yaw=π) is encoded in the
            # URDF's world→base_link fixed joint, so Gazebo + RSP both get it from there.
            # Spawning at any non-zero offset here would double-stack the offset.
            '-x', '0', '-y', '0', '-z', '0',
            '-R', '0',  '-P', '0', '-Y', '0',
        ],
        output='screen',
        additional_env=CHILD_ENV,
    )

    # Bridge: rgbd_camera in Fortress publishes at <sensor_topic>/camera_info etc.
    # Our sensor_topic = /camera/color/image_raw, so Fortress publishes:
    #   /camera/color/image_raw/camera_info
    #   /camera/color/image_raw/depth_image
    #   /camera/color/image_raw/points
    # We remap these to the names perception_node expects.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            # rgbd_camera in Fortress publishes RGB at <topic>/image (NOT <topic>)
            '/camera/color/image_raw/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/color/image_raw/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/camera/color/image_raw/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/color/image_raw/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
            # NOTE: Do NOT bridge /tf from Ignition — robot_state_publisher owns the
            # full TF tree (world→base→...→camera_color_optical_frame) and the
            # static_transform_publisher covers world→base. Bridging Ignition Pose_V
            # injects conflicting/NaN transforms for the same frames during startup.
            # Suction gripper: ROS 2 Empty → Ignition Empty (] = ROS2→IGN direction)
            '/gripper/suction_on@std_msgs/msg/Empty]ignition.msgs.Empty',
            '/gripper/suction_off@std_msgs/msg/Empty]ignition.msgs.Empty',
            # Scene overview camera (static stand, not on the robot)
            '/scene_camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            # GPU LiDAR stand — Fortress appends /points for PointCloudPacked output
            '/lidar/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
        ],
        remappings=[
            ('/camera/color/image_raw/image',       '/camera/color/image_raw'),
            ('/camera/color/image_raw/camera_info', '/camera/color/camera_info'),
            ('/camera/color/image_raw/depth_image', '/camera/depth/image_rect_raw'),
            ('/camera/color/image_raw/points',      '/camera/depth/points'),
        ],
        parameters=[{'use_sim_time': True}],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )

    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', CTRL_YAML,
        ],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )
    jtc_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', CTRL_YAML,
        ],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )
    gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'gripper_action_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', CTRL_YAML,
        ],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        output='screen',
    )
    jtc_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=jsb_spawner, on_exit=[jtc_spawner]))
    gripper_after_jtc = RegisterEventHandler(
        OnProcessExit(target_action=jtc_spawner, on_exit=[gripper_spawner]))

    return LaunchDescription([
        set_rmw,
        set_plugin_path,
        set_res_path,
        robot_desc_provider,
        gazebo,
        rsp,
        static_tf_sensor,
        static_tf_lidar_sensor,
        TimerAction(period=DELAY_SPAWN_ROBOT,  actions=[spawn]),
        TimerAction(period=DELAY_START_BRIDGE, actions=[bridge]),
        TimerAction(period=DELAY_START_CTRL,   actions=[jsb_spawner]),
        jtc_after_jsb,
        gripper_after_jtc,
    ])
