import os
import subprocess
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    tabletop_pkg  = get_package_share_directory('tabletop_manipulation')
    moveit_pkg    = get_package_share_directory('tabletop_moveit_config')
    ur_moveit_pkg = get_package_share_directory('ur_moveit_config')

    # ── URDF (xacro → string) ───────────────────────────────────────────────
    urdf_xacro = os.path.join(tabletop_pkg, 'urdf', 'ur5_with_camera.urdf.xacro')
    robot_description = subprocess.check_output(
        ['xacro', urdf_xacro], stderr=subprocess.STDOUT
    ).decode('utf-8')

    # ── SRDF ────────────────────────────────────────────────────────────────
    srdf_file = os.path.join(moveit_pkg, 'config', 'ur5_with_camera.srdf')
    with open(srdf_file) as f:
        robot_description_semantic = f.read()

    # ── Kinematics (KDL for ur_manipulator) ─────────────────────────────────
    robot_description_kinematics = {
        'ur_manipulator': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.5,
            'kinematics_solver_attempts': 20,
        }
    }

    # ── OMPL planner configs (load from ur_moveit_config) ───────────────────
    ompl_file = os.path.join(ur_moveit_pkg, 'config', 'ompl_planning.yaml')
    with open(ompl_file) as f:
        ompl_cfg = yaml.safe_load(f)
    # ompl_planning.yaml contains planner_configs + per-group planners.
    # Inject planning_plugin at the top of the ompl pipeline namespace.
    ompl_cfg['planning_plugin'] = 'ompl_interface/OMPLPlanner'
    ompl_cfg['request_adapters'] = (
        'default_planner_request_adapters/AddTimeOptimalParameterization '
        'default_planner_request_adapters/FixWorkspaceBounds '
        'default_planner_request_adapters/FixStartStateBounds '
        'default_planner_request_adapters/FixStartStateCollision '
        'default_planner_request_adapters/FixStartStatePathConstraints'
    )

    # ── Controllers (nested under moveit_simple_controller_manager) ──────────
    moveit_controllers = {
        'moveit_simple_controller_manager': {
            'controller_names': ['joint_trajectory_controller'],
            'joint_trajectory_controller': {
                'type': 'FollowJointTrajectory',
                'action_ns': 'follow_joint_trajectory',
                'default': True,
                'joints': [
                    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
                ],
            },
        },
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    # ── move_group parameters ────────────────────────────────────────────────
    move_group_params = {
        'robot_description':             robot_description,
        'robot_description_semantic':    robot_description_semantic,
        'robot_description_kinematics':  robot_description_kinematics,
        'planning_pipelines':            ['ompl'],
        'default_planning_pipeline':     'ompl',
        'ompl':                          ompl_cfg,
        'use_sim_time':                  True,
        # Grasping forces cause joints to drift a few mrad from the planned
        # position. The default 0.01 rad tolerance rejects retreat trajectories
        # planned before the gripper closed. 0.05 rad covers normal drift.
        'trajectory_execution': {
            'allowed_start_tolerance': 0.05,
        },
    }
    move_group_params.update(moveit_controllers)

    return LaunchDescription([
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[move_group_params],
        )
    ])
