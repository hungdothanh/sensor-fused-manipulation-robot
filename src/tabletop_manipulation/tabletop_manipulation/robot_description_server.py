#!/usr/bin/env python3
"""
robot_description_server.py
============================
Dedicated ROS 2 node that holds the robot_description parameter.

Why: gz_ros2_control calls SyncParametersClient(<robot_param_node>)
to fetch the URDF. In Humble, that internal rclcpp node is created
inside a Gazebo thread AFTER rclcpp::init() already ran — so env-var
based DDS tuning has no effect on it. Discovery of the standard
robot_state_publisher node reliably fails.

Solution: point robot_param_node at THIS node instead.
This node:
  1. Has NO use_sim_time dependency (starts serving immediately)
  2. Is launched BEFORE Gazebo, so it is fully up when the plugin tries
  3. Is named exactly "robot_desc_provider" to match the URDF plugin config
"""

import rclpy
from rclpy.node import Node


class RobotDescriptionServer(Node):
    def __init__(self):
        # Node name MUST match  <robot_param_node>  in the URDF plugin block
        super().__init__('robot_desc_provider')
        self.declare_parameter('robot_description', '')
        val = self.get_parameter('robot_description').value
        if val:
            self.get_logger().info(
                f'Serving robot_description ({len(val)} chars) — '
                'gz_ros2_control will find it here.')
        else:
            self.get_logger().error(
                'robot_description is empty! '
                'Pass: --ros-args -p robot_description:="<urdf>"')


def main(args=None):
    rclpy.init(args=args)
    node = RobotDescriptionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
