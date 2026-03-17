"""
constants.py
============
Single source of truth for all physical scene constants shared across nodes.

Any value that describes the real robot/table geometry lives here.
Edit this file when the physical setup changes — never edit individual nodes.

Nodes that import from here:
  - lidar_fusion_node.py
  - llm_commander_node.py
  - moveit_task_node.py
"""

# ── Table ─────────────────────────────────────────────────────────────────────
TABLE_Z   = 0.765   # world Z of conveyor table surface (metres)

# ── LiDAR height filter ───────────────────────────────────────────────────────
MIN_OBJ_Z = TABLE_Z + 0.015   # discard points at/below table surface (15 mm margin)
MAX_OBJ_Z = TABLE_Z + 0.30    # discard anything taller than 30 cm above table

# ── Cylinder geometry ─────────────────────────────────────────────────────────
CYLINDER_HEIGHT      = 0.10    # m — SDF cylinder length
CYLINDER_HALF_HEIGHT = CYLINDER_HEIGHT / 2.0   # offset from top to geometric centre

# ── Gripper geometry ──────────────────────────────────────────────────────────
# Measured along the tool0 Z axis from tool0 origin to fingertip contact surface
# with gripper fully open (θ = 0):
#   ur_to_robotiq adapter joint:   +0.011
#   base_link → knuckle_joint (Z): +0.0549
#   knuckle → finger_joint (Z):   -0.0038  (θ=0)
#   finger → finger_tip_joint (Z): +0.0472
#   fingertip_joint → contact pad: +0.0165 + ~0.004 margin
#   ────────────────────────────────────────
#   Total:                          0.130
TOOL0_TO_FINGERTIP = 0.130   # metres

# ── Motion planning ───────────────────────────────────────────────────────────
APPROACH_HEIGHT = 0.25   # m — safe hover offset above object top surface

# ── Box containers ────────────────────────────────────────────────────────────
BOX_WALL_HEIGHT = 0.10   # m — SDF box wall height
BOX_CLEARANCE   = 0.05   # m — safety margin: cylinder bottom must clear wall top
