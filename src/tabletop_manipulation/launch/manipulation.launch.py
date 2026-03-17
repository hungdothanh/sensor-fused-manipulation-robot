"""
manipulation.launch.py
=======================
Second-stage launch: run AFTER simulation.launch.py is fully up and the
joint_trajectory_controller is active.

Starts the complete manipulation stack:
  1. move_group        — MoveIt 2 planning server (from tabletop_moveit_config)
  2. vlm_perception_node — Grounding DINO zero-shot detector
  3. llm_commander_node  — LLM spatial reasoner
  4. moveit_task_node    — MoveIt pick-and-place executor

Usage:
  # Terminal 1:
  ros2 launch tabletop_manipulation simulation.launch.py

  # Terminal 2 (wait ~20 s for sim to be ready):
  ros2 launch tabletop_manipulation manipulation.launch.py

  # The chatbox GUI opens automatically — type commands there.

Optional args:
  openai_api_key:=<key>   Override OPENAI_API_KEY env var
  openai_model:=gpt-4o    OpenAI model name
  ollama_model:=llama3    Local Ollama model (used when no API key is set)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RMW_ENV = {'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'}


def generate_launch_description():
    # ── Launch arguments ───────────────────────────────────────────────────
    arg_api_key = DeclareLaunchArgument(
        'openai_api_key',
        default_value=os.environ.get('OPENAI_API_KEY', ''),
        description='OpenAI API key (falls back to OPENAI_API_KEY env var)')

    arg_openai_model = DeclareLaunchArgument(
        'openai_model',
        default_value='gpt-4o',
        description='OpenAI model name')

    arg_ollama_model = DeclareLaunchArgument(
        'ollama_model',
        default_value='',
        description='Local Ollama model name (e.g. llama3); used if no API key')

    arg_rviz = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Open RViz with camera + DINO debug view')

    # ── RViz ──────────────────────────────────────────────────────────────
    manipulation_pkg = get_package_share_directory('tabletop_manipulation')
    rviz_config = os.path.join(manipulation_pkg, 'rviz', 'manipulation.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        output='screen',
        additional_env=RMW_ENV,
    )

    # ── MoveIt 2 move_group ────────────────────────────────────────────────
    moveit_config_pkg = get_package_share_directory('tabletop_moveit_config')
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_pkg, 'launch', 'move_group.launch.py')
        )
    )

    # ── VLM Perception (Grounding DINO) ───────────────────────────────────
    vlm_node = Node(
        package='tabletop_manipulation',
        executable='vlm_perception_node.py',
        name='vlm_perception_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'auto_detect_prompt': 'red cylinder. green cylinder.',
        }],
        # HF_HUB_OFFLINE=1: use local model cache, skip HuggingFace network
        # check on every startup (eliminates the "unauthenticated requests"
        # warning once the model has been downloaded once).
        additional_env={**RMW_ENV, 'HF_HUB_OFFLINE': '1'},
    )

    # ── LiDAR Fusion Node ─────────────────────────────────────────────────
    lidar_fusion_node = Node(
        package='tabletop_manipulation',
        executable='lidar_fusion_node.py',
        name='lidar_fusion_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        additional_env=RMW_ENV,
    )

    # ── LLM Commander ─────────────────────────────────────────────────────
    llm_node = Node(
        package='tabletop_manipulation',
        executable='llm_commander_node.py',
        name='llm_commander_node',
        output='screen',
        parameters=[{
            'use_sim_time':    True,
            'openai_api_key':  LaunchConfiguration('openai_api_key'),
            'openai_model':    LaunchConfiguration('openai_model'),
            'ollama_model':    LaunchConfiguration('ollama_model'),
            'detection_timeout': 40.0,  # CPU inference takes ~18 s; give ample margin
            'detect_prompt':   'red cylinder. green cylinder.',
        }],
        additional_env=RMW_ENV,
    )

    # ── MoveIt Task Executor ───────────────────────────────────────────────
    task_node = Node(
        package='tabletop_manipulation',
        executable='moveit_task_node.py',
        name='moveit_task_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        additional_env=RMW_ENV,
    )

    # ── Chatbox GUI ────────────────────────────────────────────────────────
    chatbox_node = Node(
        package='tabletop_manipulation',
        executable='chatbox_node.py',
        name='chatbox_node',
        output='screen',
        parameters=[{
            'openai_api_key': LaunchConfiguration('openai_api_key'),
            'openai_model':   LaunchConfiguration('openai_model'),
            'ollama_model':   LaunchConfiguration('ollama_model'),
        }],
        additional_env=RMW_ENV,
    )

    return LaunchDescription([
        arg_api_key,
        arg_openai_model,
        arg_ollama_model,
        arg_rviz,
        rviz,
        move_group,
        vlm_node,
        lidar_fusion_node,
        llm_node,
        task_node,
        chatbox_node,
    ])
