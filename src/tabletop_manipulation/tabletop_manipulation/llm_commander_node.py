#!/usr/bin/env python3
"""
llm_commander_node.py
=====================
Single-stage pick-and-place pipeline:

  1. Query scene — DINO scans the scene and returns 3-D object positions.
  2. LLM intent  — parse the user command to identify which object to pick
                   and which box to place it in.
  3. Build plan  — compute pick/place coordinates and publish grasp_plan.

Topic flow (with LiDAR fusion):
  /manipulation/command  (String) → node
  node  →  /perception/dino_prompt  → vlm_perception_node
  vlm_perception_node  →  /perception/dino_results  → lidar_fusion_node
  lidar_fusion_node    →  /perception/lidar_fused_results  → node  (preferred)
  vlm_perception_node  →  /perception/dino_results  → node  (fallback, camera TF)
  node  →  /manipulation/grasp_plan  (JSON) → moveit_task_node
"""

import json
import os
import threading

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from tabletop_manipulation.constants import (
    TABLE_Z, TOOL0_TO_FINGERTIP, APPROACH_HEIGHT,
    BOX_WALL_HEIGHT, BOX_CLEARANCE, CYLINDER_HALF_HEIGHT,
)

# ── Box container geometry (must match tabletop.sdf) ──────────────────────────
#
# Conveyor scene:
#   Left  box (Y-): model origin (0.80, -0.55, 0.765) — unchanged
#   Right box (Y+): model origin (0.80, +0.38, 0.765) — moved closer to centre
#
# Hover formula guarantees cylinder_bottom ≥ wall_top + BOX_CLEARANCE:
#   tool0_hover = wall_top_z + TOOL0_TO_FINGERTIP + half_h + BOX_CLEARANCE

BOX_CONTAINERS = {
    'left_box': {
        'place_xyz':  [0.80, -0.55, TABLE_Z + 0.05],
        'wall_top_z':  TABLE_Z + BOX_WALL_HEIGHT,        # 0.865 m
    },
    'right_box': {
        'place_xyz':  [0.80,  0.38, TABLE_Z + 0.05],     # updated: box moved to Y=+0.38
        'wall_top_z':  TABLE_Z + BOX_WALL_HEIGHT,
    },
}


# ── LLM system prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a robot manipulation intent parser for an industrial conveyor pick-and-place system.

The robot is a UR5 arm. It picks colored cylinders from a conveyor table and
places them into one of two box containers on the sides of the table.

Destinations:
  "left_box"  — the box container on the LEFT side of the table (Y- direction)
  "right_box" — the box container on the RIGHT side of the table (Y+ direction)

Cylinders on the table (detected by vision, keys use underscores):
  red_cylinder, green_cylinder

OUTPUT — a JSON object with exactly these two keys:

  "object"       – exact key from scene state for the cylinder to pick
                   (e.g. "red_cylinder", "green_cylinder")

  "place_target" – "left_box" or "right_box"

OBJECT NAME MATCHING
--------------------
Match user phrasing to scene keys:
  "red cylinder"   → "red_cylinder"
  "green cylinder" → "green_cylinder"

EXAMPLES
--------
Command: "move red cylinder to right box"
→ {"object": "red_cylinder", "place_target": "right_box"}

Command: "place the green one into the left container"
→ {"object": "green_cylinder", "place_target": "left_box"}

Output nothing except the JSON object.\
"""


class LLMCommanderNode(Node):
    def __init__(self):
        super().__init__('llm_commander_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('openai_api_key',
                               os.environ.get('OPENAI_API_KEY', ''))
        self.declare_parameter('openai_model', 'gpt-4o')
        self.declare_parameter('ollama_model', '')
        self.declare_parameter('detection_timeout', 8.0)
        self.declare_parameter(
            'detect_prompt',
            'red cylinder. green cylinder.'
        )

        self._api_key       = self.get_parameter('openai_api_key').value
        self._model         = self.get_parameter('openai_model').value
        self._ollama_mod    = self.get_parameter('ollama_model').value
        self._timeout       = self.get_parameter('detection_timeout').value
        self._detect_prompt = self.get_parameter('detect_prompt').value
        self.get_logger().info(f'Detect prompt: "{self._detect_prompt}"')

        # ── TF ─────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── State ──────────────────────────────────────────────────────────
        self._dino_result  = None
        self._dino_lock    = threading.Lock()
        self._fused_result = None
        self._fused_lock   = threading.Lock()
        # Events replace polling loops — set() wakes the pipeline thread
        # immediately instead of waiting up to 0.1 s per iteration.
        self._dino_event   = threading.Event()
        self._fused_event  = threading.Event()
        self._waiting_result = False
        self._waiting_fused  = False
        self._busy           = False

        # ── Publishers / Subscribers ───────────────────────────────────────
        self.sub_cmd = self.create_subscription(
            String, '/manipulation/command', self._cmd_cb, 10)
        self.sub_dino = self.create_subscription(
            String, '/perception/dino_results', self._dino_cb, 10)
        self.sub_fused = self.create_subscription(
            String, '/perception/lidar_fused_results', self._fused_cb, 10)

        self.pub_prompt   = self.create_publisher(String, '/perception/dino_prompt', 10)
        self.pub_plan     = self.create_publisher(String, '/manipulation/grasp_plan', 10)
        self.pub_status   = self.create_publisher(String, '/manipulation/commander_status', 10)

        backend = self._model if self._api_key else (
            self._ollama_mod or 'NOT CONFIGURED')
        self.get_logger().info(
            f'LLM Commander ready. Backend: {backend}. '
            f'OpenAI key set: {bool(self._api_key)}')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _dino_cb(self, msg: String):
        if self._waiting_result:
            with self._dino_lock:
                self._dino_result = msg.data
            self._dino_event.set()   # wake pipeline thread immediately

    def _fused_cb(self, msg: String):
        if self._waiting_fused:
            with self._fused_lock:
                self._fused_result = msg.data
            self._fused_event.set()  # wake pipeline thread immediately

    def _cmd_cb(self, msg: String):
        command = msg.data.strip()
        if not command:
            return
        if self._busy:
            self.get_logger().warn(
                'Commander busy with previous command, ignoring new one.')
            return
        self.get_logger().info(f'Received command: "{command}"')
        threading.Thread(
            target=self._process_command, args=(command,), daemon=True
        ).start()

    # ── Core pipeline ─────────────────────────────────────────────────────

    def _process_command(self, command: str):
        self._busy = True
        try:
            self._run_pipeline(command)
        finally:
            self._busy = False

    def _run_pipeline(self, command: str):
        # ── Stage 1: Query scene ───────────────────────────────────────────
        self._publish_status('querying_scene')
        objects = self._query_scene()

        if not objects:
            self.get_logger().error('No objects detected.')
            self._publish_status('error: no objects detected')
            return

        self.get_logger().info(f'Scene: {objects}')
        self._publish_status('calling_llm')

        # ── Stage 2: LLM intent parsing ────────────────────────────────────
        user_msg = (
            f'Scene state (world frame, metres):\n'
            f'{json.dumps(objects, indent=2)}\n\n'
            f'Command: "{command}"'
        )
        intent_json = self._call_llm(user_msg)
        if intent_json is None:
            self._publish_status('error: LLM failed')
            return

        try:
            intent       = json.loads(intent_json)
            pick_key     = intent.get('object', '')
            place_target = intent.get('place_target', '').lower().strip()

            if not pick_key or pick_key not in objects:
                pick_key = self._fuzzy_match(pick_key, objects)
            if not pick_key or pick_key not in objects:
                raise ValueError(
                    f'Pick object "{intent.get("object")}" not in scene '
                    f'{list(objects)}')

            if place_target not in BOX_CONTAINERS:
                place_target = self._fuzzy_match(place_target, BOX_CONTAINERS)
            if not place_target or place_target not in BOX_CONTAINERS:
                raise ValueError(
                    f'Unknown place_target "{intent.get("place_target")}"')

        except Exception as exc:
            self.get_logger().error(f'Intent parsing failed: {exc}')
            self._publish_status('error: intent parsing failed')
            return

        pick_xyz = objects[pick_key]
        self.get_logger().info(
            f'Intent: pick={pick_key}  target={place_target}  xyz={pick_xyz}')

        # ── Stage 3: Build and publish grasp plan ──────────────────────────
        try:
            box       = BOX_CONTAINERS[place_target]
            place_xyz = list(box['place_xyz'])

            # Hover height: guarantees cylinder bottom clears box wall.
            # pz is the geometric centre of the cylinder; use known CYLINDER_HALF_HEIGHT.
            pz         = float(pick_xyz[2])
            wall_top_z = box['wall_top_z']
            tool0_min  = wall_top_z + TOOL0_TO_FINGERTIP + CYLINDER_HALF_HEIGHT + BOX_CLEARANCE
            place_approach_z = max(place_xyz[2] + APPROACH_HEIGHT, tool0_min)

            self.get_logger().info(
                f'Box hover: pz={pz:.3f} '
                f'wall_top={wall_top_z:.3f} place_approach_z={place_approach_z:.3f}')

            plan = {
                'object':           pick_key,
                'pick_xyz':         pick_xyz,
                'place_xyz':        place_xyz,
                'place_approach_z': place_approach_z,
            }
            self.get_logger().info(
                f'Plan: pick {pick_key} @ {pick_xyz} → {place_target} '
                f'@ {place_xyz} (hover_z={place_approach_z:.3f})')
            self.pub_plan.publish(String(data=json.dumps(plan)))
            self._publish_status('plan_published')

        except Exception as exc:
            self.get_logger().error(f'Failed to build plan: {exc}')
            self._publish_status('error: plan failed')

    # ── VLM query helper ──────────────────────────────────────────────────

    def _query_scene(self):
        """
        Publish DINO prompt, wait for lidar-fused result (preferred) or fall
        back to DINO-only result transformed via camera TF.

        Returns dict {label: [x, y, z]} in world frame, or None on failure.

        Timing strategy
        ---------------
        1. Send DINO prompt and start waiting.
        2. Wait up to self._timeout for the raw DINO result.
        3. Once DINO result arrives, give lidar_fusion_node up to
           FUSION_EXTRA_WAIT extra seconds to publish its output.
        4. If the fused result arrives in time → use it (world frame, no TF).
        5. Otherwise fall back to the DINO-only result with camera→world TF.
        """
        FUSION_EXTRA_WAIT = 5.0   # seconds to wait for fusion after DINO

        with self._dino_lock:
            self._dino_result = None
        with self._fused_lock:
            self._fused_result = None
        self._dino_event.clear()
        self._fused_event.clear()
        self._waiting_result = True
        self._waiting_fused  = True

        self.pub_prompt.publish(String(data=self._detect_prompt))

        # ── Phase 1: wait for DINO raw result ─────────────────────────────
        # Event.wait() blocks with zero CPU until _dino_cb fires set(),
        # then returns immediately — no 0.1 s polling overhead.
        self._dino_event.wait(timeout=self._timeout)
        self._waiting_result = False

        with self._dino_lock:
            dino_raw = self._dino_result

        if dino_raw is None:
            self._waiting_fused = False
            self.get_logger().error('VLM did not respond within timeout.')
            return None

        # ── Phase 2: give fusion node a bit more time ─────────────────────
        self._fused_event.wait(timeout=FUSION_EXTRA_WAIT)
        self._waiting_fused = False
        with self._fused_lock:
            fused_raw = self._fused_result

        # ── Prefer fused result; fall back to DINO-only ───────────────────
        if fused_raw is not None:
            objects = self._parse_fused_results(fused_raw)
            if objects:
                self.get_logger().info(
                    f'Using LiDAR-fused centroids: {list(objects.keys())}')
                return objects
            self.get_logger().warn(
                'Fused result was empty — falling back to DINO-only.')

        self.get_logger().warn('Using DINO-only centroids (camera TF).')
        try:
            dino_data = json.loads(dino_raw)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Bad JSON from VLM: {exc}')
            return None

        objects = self._transform_detections(dino_data.get('detections', []))
        if not objects:
            self.get_logger().warn('No valid detections from VLM.')
        return objects if objects else None

    def _parse_fused_results(self, fused_json: str):
        """
        Parse /perception/lidar_fused_results JSON into {label: [x,y,z]}.

        The fused result already contains world-frame centroids
        (centroid_3d_world), so no TF transform is needed here.
        """
        try:
            data = json.loads(fused_json)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Bad fused JSON: {exc}')
            return None

        objects = {}
        for det in data.get('detections', []):
            label     = det.get('label', '').strip().replace(' ', '_')
            c3d_world = det.get('centroid_3d_world')
            source    = det.get('source', 'unknown')
            if not label or not c3d_world or len(c3d_world) != 3:
                continue
            objects[label] = [round(c3d_world[0], 3),
                              round(c3d_world[1], 3),
                              round(c3d_world[2], 3)]
            self.get_logger().info(
                f'  {label} @ {objects[label]}  [{source}]')

        return objects if objects else None

    # ── TF transform ──────────────────────────────────────────────────────

    def _transform_detections(self, detections: list) -> dict:
        """Transform centroid_3d from camera_color_optical_frame to world."""
        camera_frame  = 'camera_color_optical_frame'
        objects_world = {}

        for det in detections:
            label = det.get('label', '').strip().replace(' ', '_')
            c3d   = det.get('centroid_3d')
            if not label or not c3d or c3d[2] <= 0.05:
                continue

            pt_cam = PointStamped()
            pt_cam.header.frame_id = camera_frame
            pt_cam.header.stamp    = rclpy.time.Time().to_msg()
            pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = (
                float(c3d[0]), float(c3d[1]), float(c3d[2]))

            try:
                pt_world = self._tf_buffer.transform(
                    pt_cam, 'world',
                    timeout=rclpy.duration.Duration(seconds=1.0))
                objects_world[label] = [
                    round(pt_world.point.x, 3),
                    round(pt_world.point.y, 3),
                    round(pt_world.point.z, 3),
                ]
            except Exception as exc:
                self.get_logger().warn(
                    f'TF transform failed for "{label}": {exc}')

        return objects_world

    # ── LLM backends ──────────────────────────────────────────────────────

    def _call_llm(self, user_message: str):
        if self._api_key:
            return self._call_openai(user_message)
        elif self._ollama_mod:
            return self._call_ollama(user_message)
        else:
            self.get_logger().error(
                'No LLM backend configured. '
                'Set OPENAI_API_KEY or the ollama_model parameter.')
            return None

    def _call_openai(self, user_message: str):
        try:
            import openai
            client = openai.OpenAI(api_key=self._api_key)
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_message},
                ],
                temperature=0.0,
                response_format={'type': 'json_object'},
            )
            return resp.choices[0].message.content
        except Exception as exc:
            self.get_logger().error(f'OpenAI API call failed: {exc}')
            return None

    def _call_ollama(self, user_message: str):
        try:
            import requests
            resp = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': self._ollama_mod,
                    'messages': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': user_message},
                    ],
                    'stream': False,
                    'format': 'json',
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()['message']['content']
        except Exception as exc:
            self.get_logger().error(f'Ollama API call failed: {exc}')
            return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fuzzy_match(name, candidates):
        """Return the key in candidates whose normalised name best matches name."""
        if not name:
            return None
        n = name.lower().replace(' ', '').replace('_', '')
        for key in candidates:
            k = key.lower().replace('_', '')
            if n in k or k in n:
                return key
        return None

    def _publish_status(self, status: str):
        self.get_logger().info(f'[commander] {status}')
        self.pub_status.publish(
            String(data=json.dumps({'status': status})))


def main(args=None):
    rclpy.init(args=args)
    node = LLMCommanderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
