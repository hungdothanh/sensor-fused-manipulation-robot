cmake_minimum_required(VERSION 3.8)
project(tabletop_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python sub-package (enables  from tabletop_manipulation import ...)
ament_python_install_package(${PROJECT_NAME})

# Install all Python executables (enables  ros2 run tabletop_manipulation <script>)
install(PROGRAMS
  tabletop_manipulation/robot_description_server.py
  tabletop_manipulation/vlm_perception_node.py
  tabletop_manipulation/llm_commander_node.py
  tabletop_manipulation/moveit_task_node.py
  tabletop_manipulation/chatbox_node.py
  tabletop_manipulation/lidar_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install non-Python resources so  get_package_share_directory()  works
install(DIRECTORY
  launch config worlds urdf rviz materials
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
