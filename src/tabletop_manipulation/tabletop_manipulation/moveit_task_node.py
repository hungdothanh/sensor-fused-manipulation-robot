#!/usr/bin/env python3
"""
moveit_task_node.py  (pymoveit2 / OMPL edition)
================================================
Replaces the manual /compute_ik + single-JTC-waypoint approach with full
MoveIt2 motion planning via pymoveit2.

Motion planning pipeline:
    pymoveit2.MoveIt2  ──►  /move_action  ──►  MoveGroup (OMPL)
        • No IK seeds
        • Automatic random-restart IK
        • Collision-aware trajectory planning

Threading note:
    pymoveit2's plan() and wait_until_executed() call rclpy.spin_once()
    internally.  A node that is already managed by a MultiThreadedExecutor
    cannot also be driven by spin_once().
    Solution: MoveIt2 is given a *dedicated* rclpy node (moveit_task_node_planner)
    that is NEVER added to any executor.  The background _pick_place thread
    drives it exclusively via pymoveit2's own spin_once() calls.
    The main node (moveit_task_node) stays in MultiThreadedExecutor and handles
    all regular subscriptions / publishers / gripper action.

Grasp depth formula (no per-object hardcoding):
    pick_xyz[2] = world z of object top surface (camera nearest visible face)
    object_half_height = (pick_z - TABLE_Z) / 2
    tip_offset = TOOL0_TO_FINGERTIP - object_half_height
    → fingertips land at the object centroid height

Gripper: /gripper_action_controller/gripper_cmd  (GripperCommand action)
Attach:  /gripper/suction_on|off  → Ignition DetachableJoint
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from control_msgs.action import GripperCommand
from pymoveit2 import MoveIt2
from std_msgs.msg import Empty, String

from tabletop_manipulation.constants import TABLE_Z, TOOL0_TO_FINGERTIP, APPROACH_HEIGHT, CYLINDER_HALF_HEIGHT


# ── Constants ─────────────────────────────────────────────────────────────────

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Matches URDF initial_value state — the pose Gazebo spawns the arm in.
# wrist_2=4.7124 (3π/2) keeps the gripper pointing DOWN toward the table.
HOME_POSITIONS = [0.0, -1.5708, 1.3090, -1.5708, 4.7124, 0.0]

PLANNING_FRAME = 'world'
EEF_LINK       = 'tool0'
MOVE_GROUP     = 'ur_manipulator'

GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.70   # rad (slightly less than max 0.7929)

# tool0 Z pointing straight down: 180° rotation about world X axis → quat [1,0,0,0]
EEF_QUAT = [1.0, 0.0, 0.0, 0.0]   # [x, y, z, w]


class MoveItTaskNode(Node):

    def __init__(self):
        super().__init__('moveit_task_node')

        # ── Dedicated node for pymoveit2 (NOT added to any executor) ──────
        # pymoveit2 internally calls rclpy.spin_once(node) inside plan() and
        # wait_until_executed().  A node already managed by MultiThreadedExecutor
        # cannot be driven by spin_once() simultaneously — deadlock or error.
        # This separate node is only ever touched by the background _pick_place
        # thread, so spin_once() is safe to call on it.
        self._moveit_node = rclpy.create_node(
            'moveit_task_node_planner',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )

        self._moveit2 = MoveIt2(
            node=self._moveit_node,
            joint_names=ARM_JOINTS,
            base_link_name='base_link',
            end_effector_name=EEF_LINK,
            group_name=MOVE_GROUP,
            # use_move_group_action=True: plan + execute via /move_action
            # MoveGroup runs OMPL internally, tries many random IK seeds,
            # returns a collision-aware interpolated trajectory.
            use_move_group_action=True,
        )

        # Slow arm motion for safe tabletop manipulation.
        # 0.3 = 30% of maximum joint velocity/acceleration — smooth and
        # controllable without swiping objects off the table.
        self._moveit2.max_velocity     = 0.3
        self._moveit2.max_acceleration = 0.3

        # ── Gripper action client (on the main node) ───────────────────────
        # Kept on the main node so the MultiThreadedExecutor processes its
        # callbacks; we wait via threading.Event (no spin_once conflict).
        self._gripper = ActionClient(
            self, GripperCommand,
            '/gripper_action_controller/gripper_cmd')

        # ── DetachableJoint publishers ─────────────────────────────────────
        self._pub_attach = self.create_publisher(Empty, '/gripper/suction_on',  1)
        self._pub_detach = self.create_publisher(Empty, '/gripper/suction_off', 1)

        # ── Status publisher ───────────────────────────────────────────────
        self._pub_status = self.create_publisher(
            String, '/manipulation/task_status', 10)

        # ── Grasp-plan subscriber ──────────────────────────────────────────
        self._busy = False
        self.create_subscription(
            String, '/manipulation/grasp_plan', self._plan_cb, 10)

        self.get_logger().info('MoveIt task node ready (pymoveit2 / OMPL).')

    # ── Grasp plan callback ───────────────────────────────────────────────────

    def _plan_cb(self, msg: String):
        if self._busy:
            self.get_logger().warn('Task node busy — ignoring grasp plan.')
            return
        try:
            plan      = json.loads(msg.data)
            pick_xyz  = plan['pick_xyz']
            place_xyz = plan['place_xyz']
            obj_name  = plan.get('object', 'unknown')
            # place_approach_z: optional tool0 z for pre-place hover and retreat.
            # Computed by llm_commander to guarantee cylinder clears box walls.
            # Falls back to lz + APPROACH_HEIGHT if not present (table placement).
            place_approach_z = plan.get('place_approach_z', None)
        except (json.JSONDecodeError, KeyError) as exc:
            self.get_logger().error(f'Malformed grasp_plan: {exc}')
            return

        self.get_logger().info(
            f'Executing: {obj_name}  {pick_xyz} → {place_xyz}'
            + (f'  approach_z={place_approach_z:.3f}' if place_approach_z else ''))
        self._busy = True
        threading.Thread(
            target=self._run_task,
            args=(pick_xyz, place_xyz, obj_name, place_approach_z),
            daemon=True,
        ).start()

    def _run_task(self, pick_xyz, place_xyz, obj_name='unknown',
                  place_approach_z=None):
        try:
            self._pick_place(pick_xyz, place_xyz, obj_name, place_approach_z)
        except Exception as exc:
            self.get_logger().error(f'Pick-and-place failed: {exc}')
            self._status('error', str(exc))
            self._gripper_cmd(GRIPPER_OPEN)
            self._pub_detach.publish(Empty())
        finally:
            self._busy = False

    # ── Pick-and-place sequence ───────────────────────────────────────────────

    def _pick_place(self, pick_xyz, place_xyz, obj_name='unknown',
                   place_approach_z=None):
        px, py, pz = pick_xyz
        lx, ly, lz = place_xyz

        # Grasp depth formula.
        # pz is the geometric centre of the cylinder (LiDAR median Z − half-height).
        # We want fingertips at pz so fingers close around the widest point.
        # tool0 must be at: pz + TOOL0_TO_FINGERTIP  →  tip_offset = TOOL0_TO_FINGERTIP
        #
        # Safety floor: fingertips must stay ≥ 30 mm above table regardless of
        # detection noise. min_tip guarantees tool0_z ≥ TABLE_Z + TOOL0_TO_FINGERTIP + 0.030.
        tip     = TOOL0_TO_FINGERTIP
        min_tip = TABLE_Z + TOOL0_TO_FINGERTIP + 0.030 - pz
        tip     = max(tip, min_tip)

        # place_approach_z: tool0 z for the pre-place hover and retreat.
        # Supplied by llm_commander when placing into a box — ensures the held
        # cylinder clears the box walls: cylinder_bottom ≥ wall_top + clearance.
        # Falls back to lz + APPROACH_HEIGHT for free (non-box) placements.
        p_hover = place_approach_z if place_approach_z is not None \
                  else lz + APPROACH_HEIGHT

        self.get_logger().info(
            f'[task] "{obj_name}"  centre_z={pz:.3f}  '
            f'tip_offset={tip:.3f}  place_hover_z={p_hover:.3f}')

        # 1. Open gripper
        self._status('opening_gripper')
        self._gripper_cmd(GRIPPER_OPEN)

        # 2. Pre-grasp hover — Cartesian straight line, OMPL fallback
        self._status('moving_to_pregrasp')
        if not self._move_cartesian_or_ompl(px, py, pz + APPROACH_HEIGHT): return

        # 3. Descend to grasp — Cartesian straight down
        self._status('descending_to_grasp')
        if not self._move_to(px, py, pz + tip, cartesian=True): return
        time.sleep(0.5)   # let arm fully settle before closing gripper

        # 4. Close gripper firmly, then attach via DetachableJoint.
        self._status('closing_gripper')
        self._gripper_cmd(GRIPPER_CLOSED)
        self._pub_attach.publish(Empty())

        # 5. Retreat — Cartesian straight up.
        # allowed_start_tolerance=0.05 in move_group config handles the small
        # joint drift (~13 mrad) caused by grasping contact forces.
        self._status('retreating_from_pick')
        if not self._move_to(px, py, pz + APPROACH_HEIGHT, cartesian=True):
            self._release(); return

        # 6. Pre-place hover — move to p_hover (tool0 z).
        # p_hover guarantees: cylinder_bottom ≥ box_wall_top + clearance
        # so the cylinder never clips the box walls during the lateral approach.
        self._status('moving_to_preplace')
        if not self._move_cartesian_or_ompl(lx, ly, p_hover):
            self._release(); return

        # 7. Descend to place — Cartesian straight down into box
        self._status('descending_to_place')
        if not self._move_to(lx, ly, lz + tip, cartesian=True):
            self._release(); return
        time.sleep(0.5)   # let arm settle at place position before releasing

        # 8. Open gripper + detach, then wait for fingers to open
        self._status('releasing_object')
        self._release()
        time.sleep(0.8)   # wait for fingers to physically open before retreating

        # 9. Retreat from place — Cartesian straight up to p_hover.
        # Use the same p_hover so the gripper exits the box opening cleanly.
        self._status('retreating_from_place')
        self._move_to(lx, ly, p_hover, cartesian=True)

        # 10. Return home
        self._status('returning_home')
        self._go_home()

        self._status('done')
        self.get_logger().info('Pick-and-place complete.')

    def _release(self):
        self._gripper_cmd(GRIPPER_OPEN)
        self._pub_detach.publish(Empty())

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _move_cartesian_or_ompl(self, x: float, y: float, z: float) -> bool:
        """Move EEF to (x, y, z) preferring a Cartesian straight-line path.
        Falls back to OMPL joint-space planning if Cartesian fails.
        This prevents the arm from swinging wildly between hover positions."""
        if self._move_to(x, y, z, cartesian=True):
            return True
        self.get_logger().warn(
            f'Cartesian failed for ({x:.3f}, {y:.3f}, {z:.3f}), '
            f'retrying with OMPL.')
        return self._move_to(x, y, z, cartesian=False)

    def _move_to(self, x: float, y: float, z: float,
                 cartesian: bool = True) -> bool:
        """Plan and execute arm motion to (x, y, z).

        cartesian=True (default): straight-line EEF motion — smooth and
            predictable, never switches IK solution mid-move.
        cartesian=False: OMPL joint-space fallback — used when no straight-line
            Cartesian path exists (e.g. arm must pass through a different
            configuration to reach the target).
        """
        self.get_logger().info(
            f'{"Cartesian" if cartesian else "OMPL"} → '
            f'({x:.3f}, {y:.3f}, {z:.3f})')
        self._moveit2.move_to_pose(
            position=[x, y, z],
            quat_xyzw=EEF_QUAT,
            frame_id=PLANNING_FRAME,
            cartesian=cartesian,
            cartesian_max_step=0.005,   # 5 mm interpolation step
        )
        success = self._moveit2.wait_until_executed()
        if not success:
            self.get_logger().error(
                f'Motion failed for ({x:.3f}, {y:.3f}, {z:.3f})')
        return success

    def _go_home(self):
        """Return arm to spawn joint configuration."""
        self._moveit2.move_to_configuration(HOME_POSITIONS)
        self._moveit2.wait_until_executed()

    # ── Gripper ───────────────────────────────────────────────────────────────

    def _gripper_cmd(self, position: float, max_effort: float = 50.0):
        """Send a GripperCommand goal and block until result (via threading.Event)."""
        if not self._gripper.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Gripper action server not available.')
            return
        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort

        fut  = self._gripper.send_goal_async(goal)
        evt1 = threading.Event()
        fut.add_done_callback(lambda _: evt1.set())
        evt1.wait(timeout=5.0)
        if not fut.done():
            return
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('Gripper goal rejected.')
            return
        res_fut = gh.get_result_async()
        evt2 = threading.Event()
        res_fut.add_done_callback(lambda _: evt2.set())
        evt2.wait(timeout=5.0)

    # ── Status helpers ────────────────────────────────────────────────────────

    def _status(self, status: str, msg: str = ''):
        self.get_logger().info(f'[task] {status}  {msg}'.rstrip())
        payload = {'status': status}
        if msg:
            payload['msg'] = msg
        self._pub_status.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = MoveItTaskNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node._moveit_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
