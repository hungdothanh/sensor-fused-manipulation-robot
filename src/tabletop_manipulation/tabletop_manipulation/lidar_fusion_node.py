#!/usr/bin/env python3
"""
lidar_fusion_node.py
====================
Fuses GPU-LiDAR point clouds with Grounding-DINO 2-D detections to compute
accurate world-frame 3-D centroids for vertical cylinders.

Algorithm
---------
For every DINO detection (label + 2-D bounding box):

  1. Transform all LiDAR points from lidar_link → world frame (via static TF).
  2. Height-filter: discard points at or below the table surface (TABLE_Z).
  3. Transform remaining points from lidar_link → camera_color_optical_frame.
  4. Project the camera-frame points into pixel space using camera intrinsics.
  5. Keep only points whose pixel projection falls *inside* the DINO 2-D bbox.
  6. Run RANSAC circle fit on the XY projection of those world-frame points
     to recover the cylinder axis position and radius.
  7. Centroid  =  (circle_cx, circle_cy,  median_Z_of_RANSAC_inliers).
  8. Publish  /perception/lidar_fused_results  (JSON, world frame).
  9. Publish  /perception/centroid_markers  (MarkerArray for RViz):
       • RED   sphere  = DINO-only centroid (camera → world via TF)
       • GREEN sphere  = LiDAR-fused centroid
       • BLUE  ring    = fitted circle projected onto cylinder mid-height

Why RANSAC circle fit on XY?
-----------------------------
A vertical cylinder has a circular cross-section in the XY plane regardless
of the camera viewing angle or LiDAR vantage point.  Fitting that circle with
RANSAC is robust to:
  • Table reflections / ground points (removed by height filter)
  • Arm / wall occlusions (RANSAC handles partial arcs down to ~40% coverage)
  • Sensor noise (±3 mm Gaussian noise in the SDF)

Subscriptions
-------------
  /lidar/points              sensor_msgs/PointCloud2   (frame: lidar_link)
  /camera/color/camera_info  sensor_msgs/CameraInfo
  /perception/dino_results   std_msgs/String            (JSON)

Publications
------------
  /perception/lidar_fused_results   std_msgs/String     (JSON, world frame)
  /perception/centroid_markers      visualization_msgs/MarkerArray
"""

import json
import threading

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped → TF support)

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image as RosImage, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import sensor_msgs_py.point_cloud2 as pc2

from tabletop_manipulation.constants import MIN_OBJ_Z, MAX_OBJ_Z, CYLINDER_HALF_HEIGHT


# ── RANSAC parameters ─────────────────────────────────────────────────────────

RANSAC_ITERATIONS  = 500     # more iterations for partial-arc robustness
RANSAC_INLIER_DIST = 0.025   # 25 mm — wider for small-radius partial-arc noise
MIN_INLIERS        = 6       # lower bar: cylinders have ~8-12 total LiDAR pts

# ── Bbox expansion for LiDAR projection masking ───────────────────────────────
# Adds BBOX_PAD pixels on each side to tolerate small camera/LiDAR projection
# offsets caused by numerical TF imprecision.
BBOX_PAD = 12


class LidarFusionNode(Node):
    """Fuses LiDAR point clouds with DINO 2-D detections."""

    def __init__(self):
        super().__init__('lidar_fusion_node')

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._bridge      = CvBridge()

        # ── Shared state (protected by _lock) ─────────────────────────────
        self._lock        = threading.Lock()
        self._cloud_world = None   # (N, 3) float32, world frame
        self._cloud_lidar = None   # (N, 3) float32, lidar_link frame (same rows)
        self._intrinsics  = None   # dict: fx fy cx cy width height

        # ── Raw camera image (for fusion overlay drawn from scratch) ──────
        self._dino_debug_lock = threading.Lock()
        self._dino_debug_msg  = None   # latest /camera/color/image_raw Image

        # lidar_mount is a URDF link, child of camera_mount (wrist-mounted).
        # robot_state_publisher publishes the full TF chain including this frame.
        self._lidar_frame = 'lidar_mount'
        self._cam_frame   = 'camera_color_optical_frame'

        # ── Subscriptions ─────────────────────────────────────────────────
        self.sub_cloud = self.create_subscription(
            PointCloud2, '/lidar/scan/points',
            self._cloud_cb, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self._info_cb, qos_profile_sensor_data)
        self.sub_dino = self.create_subscription(
            String, '/perception/dino_results',
            self._dino_cb, 10)
        self.sub_dino_debug = self.create_subscription(
            RosImage, '/camera/color/image_raw',
            self._dino_debug_cb, qos_profile_sensor_data)

        # ── Publications ──────────────────────────────────────────────────
        self.pub_fused       = self.create_publisher(
            String, '/perception/lidar_fused_results', 10)
        self.pub_markers     = self.create_publisher(
            MarkerArray, '/perception/centroid_markers', 10)
        self.pub_fusion_debug = self.create_publisher(
            RosImage, '/perception/fusion_debug', qos_profile_sensor_data)

        self.get_logger().info(
            'LiDAR Fusion Node ready (wrist-mounted LiDAR on lidar_mount). '
            'Waiting for /lidar/scan/points, /camera/color/camera_info, '
            'and /perception/dino_results …')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._intrinsics = {
                'fx': msg.p[0], 'fy': msg.p[5],
                'cx': msg.p[2], 'cy': msg.p[6],
                'width':  msg.width,
                'height': msg.height,
            }

    def _cloud_cb(self, msg: PointCloud2):
        """Convert PointCloud2 → Nx3 numpy, transform to world frame once."""
        pts = self._decode_cloud(msg)
        if pts is None or len(pts) == 0:
            return

        # Look up TF lidar_link → world once per cloud message
        try:
            tf_l2w = self._tf_buffer.lookup_transform(
                'world', self._lidar_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().warn(f'TF lidar→world: {e}', throttle_duration_sec=5.0)
            return

        pts_world = self._apply_tf(pts, tf_l2w)
        with self._lock:
            self._cloud_lidar = pts
            self._cloud_world = pts_world

    def _dino_debug_cb(self, msg: RosImage):
        with self._dino_debug_lock:
            self._dino_debug_msg = msg

    def _dino_cb(self, msg: String):
        """Triggered every time DINO publishes results — run fusion."""
        threading.Thread(
            target=self._fuse,
            args=(msg.data,),
            daemon=True,
        ).start()

    # ── Core fusion pipeline ───────────────────────────────────────────────────

    def _fuse(self, dino_json: str):
        try:
            dino_data  = json.loads(dino_json)
            detections = dino_data.get('detections', [])
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Bad DINO JSON: {e}')
            return

        # Snapshot latest cloud + intrinsics
        with self._lock:
            cloud_world = (self._cloud_world.copy()
                           if self._cloud_world is not None else None)
            cloud_lidar = (self._cloud_lidar.copy()
                           if self._cloud_lidar is not None else None)
            intr = dict(self._intrinsics) if self._intrinsics else None

        if cloud_world is None or intr is None:
            self.get_logger().warn('No LiDAR cloud or camera info yet — skipping fusion.')
            return

        # ── Height-filter: keep only above-table object points ────────────
        mask_h = (cloud_world[:, 2] > MIN_OBJ_Z) & (cloud_world[:, 2] < MAX_OBJ_Z)
        cw_obj = cloud_world[mask_h]    # world-frame object points
        cl_obj = cloud_lidar[mask_h]    # matching lidar-frame points

        if len(cw_obj) < MIN_INLIERS:
            self.get_logger().warn(
                f'Only {len(cw_obj)} above-table LiDAR points — skipping fusion.')
            return

        # ── Transform object points: lidar_link → camera_color_optical_frame
        try:
            tf_l2c = self._tf_buffer.lookup_transform(
                self._cam_frame, self._lidar_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            self.get_logger().warn(f'TF lidar→camera: {e}')
            return

        cc_obj = self._apply_tf(cl_obj, tf_l2c)   # camera-frame object pts

        # ── Project to pixel coordinates ──────────────────────────────────
        fx, fy = intr['fx'], intr['fy']
        ppx, ppy = intr['cx'], intr['cy']
        W, H = int(intr['width']), int(intr['height'])

        Zc    = cc_obj[:, 2]
        valid = Zc > 0.05
        # u_px, v_px: pixel column/row of each projected LiDAR point
        u_px = np.where(valid, (fx * cc_obj[:, 0] / Zc + ppx), -1).astype(np.int32)
        v_px = np.where(valid, (fy * cc_obj[:, 1] / Zc + ppy), -1).astype(np.int32)

        # ── Per-detection fusion (three-tier priority) ────────────────────
        #
        # Tier 1 — RANSAC circle fit (best XY accuracy, full 3D)
        # Tier 2 — LiDAR median depth + DINO pixel centroid
        #          (accurate Z; DINO XY rescaled with correct depth)
        # Tier 3 — DINO only (last resort, no LiDAR contribution)
        #
        fused_dets   = []
        marker_array = MarkerArray()
        stamp        = self.get_clock().now().to_msg()
        mid          = 0

        for det in detections:
            label   = det.get('label', '').strip()
            box_2d  = det.get('box_2d', [])
            c3d_cam = det.get('centroid_3d')   # camera-frame, from DINO
            score   = det.get('score', 0.0)

            if len(box_2d) != 4:
                continue

            x1, y1, x2, y2 = (int(b) for b in box_2d)

            # DINO-only world centroid (camera TF, Tier-3 fallback).
            # The depth pixel hits the object surface (somewhere between side and top);
            # subtract half the cylinder height to approximate the geometric centre,
            # matching the same correction applied to the LiDAR-fused Z.
            dino_world = self._cam_pt_to_world(c3d_cam) if c3d_cam else None
            if dino_world:
                dino_world = [dino_world[0], dino_world[1],
                              round(dino_world[2] - CYLINDER_HALF_HEIGHT, 4)]

            # ── Select LiDAR points whose projection falls in the 2-D bbox
            # Expand by BBOX_PAD px to tolerate small camera/LiDAR projection offsets.
            x1p = x1 - BBOX_PAD;  y1p = y1 - BBOX_PAD
            x2p = x2 + BBOX_PAD;  y2p = y2 + BBOX_PAD
            in_box = (
                valid
                & (u_px >= x1p) & (u_px <= x2p)
                & (v_px >= y1p) & (v_px <= y2p)
                & (u_px >= 0) & (u_px < W)
                & (v_px >= 0) & (v_px < H)
            )
            pts_box_world = cw_obj[in_box]
            cc_box        = cc_obj[in_box]   # same mask in camera frame

            # ── Tier-2: LiDAR median depth + DINO pixel centroid ──────────
            # Compute this regardless of point count; use as fallback when
            # RANSAC fails (partial arc, small radius, noisy arc fitting).
            lidar_depth_centroid = None
            if len(cc_box) >= 3:
                median_depth = float(np.median(cc_box[:, 2]))   # cam-frame Z
                if 0.05 < median_depth < 5.0:
                    # Deproject the DINO 2D centroid using accurate LiDAR depth
                    bbox_cx_pix = (x1 + x2) // 2
                    bbox_cy_pix = (y1 + y2) // 2
                    x_cam = (bbox_cx_pix - ppx) * median_depth / fx
                    y_cam = (bbox_cy_pix - ppy) * median_depth / fy
                    lidar_depth_centroid = self._cam_pt_to_world(
                        [x_cam, y_cam, median_depth])

            def _make_entry(centroid, source):
                return {
                    'label':              label,
                    'score':              score,
                    'centroid_3d_world':  centroid,
                    'dino_centroid_world': dino_world,
                    'source':             source,
                }

            if len(pts_box_world) < MIN_INLIERS:
                # Not enough points for RANSAC — try Tier-2, then Tier-3
                if lidar_depth_centroid:
                    self.get_logger().info(
                        f'"{label}": {len(pts_box_world)} LiDAR pts in bbox '
                        f'(< {MIN_INLIERS}). Using lidar_depth centroid.')
                    fused_dets.append(_make_entry(lidar_depth_centroid, 'lidar_depth'))
                elif dino_world:
                    self.get_logger().warn(
                        f'"{label}": 0 usable LiDAR pts. Falling back to DINO.')
                    fused_dets.append(_make_entry(dino_world, 'dino_only'))
                continue

            # ── Tier-1: RANSAC circle fit on the XY projection (world frame)
            cx_fit, cy_fit, radius, inliers = self._ransac_circle_xy(
                pts_box_world)

            if inliers is None or len(inliers) < MIN_INLIERS:
                # RANSAC failed — partial arc or noisy; try Tier-2, then Tier-3
                if lidar_depth_centroid:
                    self.get_logger().info(
                        f'"{label}": RANSAC failed. Using lidar_depth centroid.')
                    fused_dets.append(_make_entry(lidar_depth_centroid, 'lidar_depth'))
                elif dino_world:
                    self.get_logger().warn(
                        f'"{label}": RANSAC failed and no LiDAR depth. DINO fallback.')
                    fused_dets.append(_make_entry(dino_world, 'dino_only'))
                continue

            # LiDAR inliers cluster on the top surface of the cylinder.
            # Subtract half the cylinder height to recover the geometric centre.
            median_z_top   = float(np.median(inliers[:, 2]))
            median_z       = median_z_top - CYLINDER_HALF_HEIGHT
            fused_centroid = [round(cx_fit, 4), round(cy_fit, 4), round(median_z, 4)]

            self.get_logger().info(
                f'[FUSED] "{label}": cx={cx_fit:.3f} cy={cy_fit:.3f} '
                f'r={radius:.3f} z_top={median_z_top:.3f} z_centre={median_z:.3f}'
                f'  ({len(inliers)} inliers)')

            fused_dets.append({
                'label':              label,
                'score':              score,
                'centroid_3d_world':  fused_centroid,
                'dino_centroid_world': dino_world,
                'circle_radius':      round(radius, 4),
                'n_inliers':          int(len(inliers)),
                'source':             'lidar_fusion',
            })

            # ── RViz markers ──────────────────────────────────────────────

            # RED sphere: DINO-only centroid (for comparison)
            if dino_world:
                marker_array.markers.append(
                    self._sphere_marker(
                        mid, 'dino_centroid', stamp, 'world',
                        dino_world, (1.0, 0.15, 0.15), 0.04))
                mid += 1

            # GREEN sphere: LiDAR-fused centroid
            marker_array.markers.append(
                self._sphere_marker(
                    mid, 'fused_centroid', stamp, 'world',
                    fused_centroid, (0.1, 1.0, 0.1), 0.05))
            mid += 1

            # BLUE ring: fitted circle outline at cylinder mid-height
            ring_m = Marker()
            ring_m.header.stamp    = stamp
            ring_m.header.frame_id = 'world'
            ring_m.ns    = 'cylinder_ring'
            ring_m.id    = mid
            ring_m.type  = Marker.LINE_STRIP
            ring_m.action = Marker.ADD
            ring_m.scale.x = 0.005
            ring_m.color.r = 0.2
            ring_m.color.g = 0.6
            ring_m.color.b = 1.0
            ring_m.color.a = 0.9
            ring_m.pose.orientation.w = 1.0
            ring_m.points = self._circle_ring(cx_fit, cy_fit, radius, median_z)
            ring_m.lifetime.sec = 5
            marker_array.markers.append(ring_m)
            mid += 1

        # ── 3-D NMS: drop duplicate detections that map to the same object ──
        # When DINO returns multiple labels for the same physical cylinder
        # (e.g. both "blue cylinder" and "green cylinder" pointing at one spot),
        # each bbox captures the same LiDAR cluster → identical centroids.
        # Keep the higher-score detection when two centroids are < NMS_DIST apart.
        NMS_DIST = 0.12   # metres — larger than cylinder diameter (0.06 m)
        kept = []
        for fd in sorted(fused_dets, key=lambda d: d.get('score', 0.0), reverse=True):
            c = fd.get('centroid_3d_world')
            if c is None:
                kept.append(fd)
                continue
            too_close = False
            for kd in kept:
                kc = kd.get('centroid_3d_world')
                if kc is None:
                    continue
                dist = float(np.linalg.norm(np.array(c) - np.array(kc)))
                if dist < NMS_DIST:
                    too_close = True
                    self.get_logger().warn(
                        f'NMS: dropped "{fd["label"]}" (dist={dist:.3f} m to '
                        f'"{kd["label"]}", score={fd.get("score",0):.2f})')
                    break
            if not too_close:
                kept.append(fd)
        fused_dets = kept

        # ── Publish ───────────────────────────────────────────────────────
        self.pub_fused.publish(
            String(data=json.dumps({'detections': fused_dets})))
        self.pub_markers.publish(marker_array)
        self.get_logger().info(
            f'Published {len(fused_dets)} fused detections.')

        # ── Fusion debug image (overlay both centroids on DINO debug) ─────
        with self._dino_debug_lock:
            dino_debug_msg = self._dino_debug_msg
        if dino_debug_msg is not None:
            overlay = self._draw_fusion_overlay(
                dino_debug_msg, detections, fused_dets)
            if overlay is not None:
                self.pub_fusion_debug.publish(overlay)

    # ── Fusion debug overlay ──────────────────────────────────────────────────

    def _draw_fusion_overlay(self, raw_img_msg: RosImage,
                             dino_detections: list,
                             fused_dets: list):
        """
        Draw NMS-filtered detections on the raw camera image from scratch.
        Only fused_dets (post-NMS) are drawn, eliminating duplicate label overlap.

        Colour convention (matches chatbox legend):
          Cyan  filled circle = DINO bbox centroid (camera-only)
          Green filled circle = LiDAR-fused centroid (RANSAC circle)
          Teal  filled circle = LiDAR depth + DINO pixel
          Orange filled circle = DINO-only fallback

        Bottom-left legend summarises both markers.
        """
        try:
            img = self._bridge.imgmsg_to_cv2(raw_img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'imgmsg_to_cv2 failed: {e}', throttle_duration_sec=5.0)
            return None

        with self._lock:
            intr = dict(self._intrinsics) if self._intrinsics else None
        if intr is None:
            return None

        fx, fy   = intr['fx'], intr['fy']
        ppx, ppy = intr['cx'], intr['cy']
        H, W     = img.shape[:2]

        # Build lookup: normalised label → raw DINO detection (for bbox + dino centroid)
        dino_by_label = {
            d.get('label', '').strip().replace(' ', '_'): d
            for d in dino_detections
        }

        # Look up TF once for all detections; if unavailable skip centroid dots
        tf_w2c = None
        try:
            tf_w2c = self._tf_buffer.lookup_transform(
                self._cam_frame, 'world',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
        except Exception:
            pass

        def _world_to_px(xyz):
            if tf_w2c is None:
                return None
            pt = self._apply_tf(
                np.array([[xyz[0], xyz[1], xyz[2]]], dtype=np.float32), tf_w2c)[0]
            if pt[2] < 0.05:
                return None
            return int(fx * pt[0] / pt[2] + ppx), int(fy * pt[1] / pt[2] + ppy)

        for fd in fused_dets:
            label     = fd['label'].strip()
            label_key = label.replace(' ', '_')
            det       = dino_by_label.get(label_key)

            # ── Bounding box + label (from DINO bbox, NMS-filtered) ──────────
            if det is not None:
                box = det.get('box_2d', [])
                if len(box) == 4:
                    x1, y1, x2, y2 = (int(b) for b in box)
                    score  = det.get('score', 0.0)
                    c3d_cam = det.get('centroid_3d')   # camera-frame [x,y,depth]
                    dist_m = round(c3d_cam[2], 2) if c3d_cam else 0.0
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{label}: {score:.2f} ({dist_m:.2f}m)',
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1)

                # ── Cyan dot: DINO bbox centroid (camera-only) ────────────────
                dino_w = fd.get('dino_centroid_world')
                if dino_w:
                    px = _world_to_px(dino_w)
                    if px and 0 <= px[0] < W and 0 <= px[1] < H:
                        cv2.circle(img, px, 5, (255, 255, 0), -1)
                        cv2.circle(img, px, 5, (255, 255, 255), 1)

            # ── Fused centroid dot ────────────────────────────────────────────
            c3d = fd.get('centroid_3d_world')
            if c3d is None:
                continue
            px = _world_to_px(c3d)
            if px is None or not (0 <= px[0] < W and 0 <= px[1] < H):
                continue

            source = fd.get('source', 'dino_only')
            if source == 'lidar_fusion':
                dot_color = (0, 220, 0)
                tag = 'R'
            elif source == 'lidar_depth':
                dot_color = (0, 220, 220)
                tag = 'L'
            else:
                dot_color = (0, 140, 255)
                tag = 'D'

            cv2.circle(img, px, 5, dot_color, -1)
            cv2.circle(img, px, 5, (255, 255, 255), 1)
            cv2.putText(img, tag, (px[0] + 8, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1)

        # ── Legend (bottom-left, semi-transparent background) ────────────
        legend = [
            ((255, 255,   0), '●', 'DINO bbox centroid   (cyan)'),
            ((  0, 220,   0), 'R', 'LiDAR RANSAC circle  (green)'),
            ((  0, 220, 220), 'L', 'LiDAR depth + DINO px (teal)'),
            ((  0, 140, 255), 'D', 'DINO only fallback   (orange)'),
        ]
        row_h  = 16
        pad    = 6
        box_h  = len(legend) * row_h + pad * 2
        box_w  = 230
        lx, ly = 6, H - box_h - 6

        # Dark semi-transparent backing rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + box_w, ly + box_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        for i, (col, sym, text) in enumerate(legend):
            row_y = ly + pad + i * row_h + row_h // 2
            cv2.circle(img, (lx + pad + 4, row_y), 4, col, -1)
            cv2.putText(img, text,
                        (lx + pad + 13, row_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)

        try:
            out            = self._bridge.cv2_to_imgmsg(img, 'bgr8')
            out.header     = raw_img_msg.header
            return out
        except Exception:
            return None

    # ── RANSAC circle fit ──────────────────────────────────────────────────────

    @staticmethod
    def _ransac_circle_xy(pts: np.ndarray):
        """
        Fit a circle to the XY projection of *pts* using RANSAC then refine
        with algebraic least-squares on the final consensus set.

        Parameters
        ----------
        pts : (N, 3) float32 — world-frame points

        Returns
        -------
        (cx, cy, radius, inlier_pts)  or  (None, None, None, None)

        RANSAC inner loop
        -----------------
        1. Randomly sample 3 points.
        2. Compute the unique circumscribed circle (closed-form).
        3. Count points within RANSAC_INLIER_DIST of the circle perimeter.
        4. Keep the hypothesis with the highest inlier count.

        Refinement (algebraic fit)
        --------------------------
        Given the inlier set, refit with a numerically stable SVD-based
        algebraic least-squares method (equivalent to Pratt normalisation)
        that minimises the sum of squared algebraic distances.
        """
        xy = pts[:, :2]
        N  = len(xy)

        best_count = 0
        best_mask  = None
        rng = np.random.default_rng(0)

        for _ in range(RANSAC_ITERATIONS):
            idx = rng.choice(N, 3, replace=False)
            cx, cy, r = LidarFusionNode._circle_3pts(xy[idx[0]], xy[idx[1]], xy[idx[2]])
            if cx is None:
                continue

            dists = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)
            mask  = dists < RANSAC_INLIER_DIST
            count = int(mask.sum())

            if count > best_count:
                best_count = count
                best_mask  = mask

        if best_mask is None or best_count < MIN_INLIERS:
            return None, None, None, None

        inliers = pts[best_mask]
        cx_r, cy_r, r_r = LidarFusionNode._algebraic_circle(inliers[:, :2])
        if cx_r is None:
            # Fall back to 3-pt hypothesis parameters
            idx = np.where(best_mask)[0]
            cx_r, cy_r, r_r = LidarFusionNode._circle_3pts(
                xy[idx[0]], xy[idx[1]], xy[idx[2]])

        return cx_r, cy_r, r_r, inliers

    @staticmethod
    def _circle_3pts(p1, p2, p3):
        """
        Closed-form circumscribed circle through three 2-D points.

        Derivation: the centre (ux, uy) is equidistant from all three points.
        Expanding |P-U|² = R² for each pair gives two linear equations in
        (ux, uy) which are solved via Cramer's rule.

        Returns (cx, cy, r) or (None, None, None) if points are collinear.
        """
        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])

        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-9:
            return None, None, None   # collinear

        a2 = ax**2 + ay**2
        b2 = bx**2 + by**2
        c2 = cx**2 + cy**2

        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
        r  = float(np.hypot(ax - ux, ay - uy))
        return float(ux), float(uy), r

    @staticmethod
    def _algebraic_circle(xy: np.ndarray):
        """
        Algebraic (Pratt) least-squares circle fit via SVD.

        Minimises  Σ (x² + y² + Dx + Ey + F)²  (no normalisation constraint
        needed — the SVD gives the minimum-norm solution automatically).

        Design matrix column layout:  [x²+y²,  x,  y,  1]
        The right singular vector corresponding to the smallest singular value
        gives [A, B, C, D] of  A(x²+y²) + Bx + Cy + D = 0.
        Centre and radius follow from completing the square.

        Returns (cx, cy, r) or (None, None, None) on degenerate input.
        """
        n = len(xy)
        if n < 3:
            return None, None, None

        x, y = xy[:, 0], xy[:, 1]
        z    = x**2 + y**2

        # Build design matrix and solve with SVD
        A_mat = np.column_stack([z, x, y, np.ones(n)])
        _, _, Vt = np.linalg.svd(A_mat, full_matrices=False)
        coeffs = Vt[-1]   # solution: [A, B, C, D]

        A = coeffs[0]
        if abs(A) < 1e-10:
            return None, None, None

        cx = -coeffs[1] / (2.0 * A)
        cy = -coeffs[2] / (2.0 * A)
        disc = coeffs[1]**2 + coeffs[2]**2 - 4.0 * A * coeffs[3]
        if disc < 0:
            return None, None, None

        r = float(np.sqrt(disc)) / (2.0 * abs(A))
        return float(cx), float(cy), r

    # ── Geometry helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _decode_cloud(msg: PointCloud2):
        """
        Convert a PointCloud2 message to an (N, 3) float32 numpy array.

        Uses sensor_msgs_py.point_cloud2.read_points which returns a generator
        of named tuples (x, y, z).  Compatible with ROS 2 Humble.
        """
        try:
            gen  = pc2.read_points(msg, field_names=('x', 'y', 'z'),
                                   skip_nans=True)
            # Handle both generator-of-tuples (Humble) and structured-array
            # (newer) return styles from read_points.
            raw = list(gen)
            if not raw:
                return np.zeros((0, 3), dtype=np.float32)
            sample = raw[0]
            if hasattr(sample, '__len__') or hasattr(sample, '__iter__'):
                pts = np.array([(float(p[0]), float(p[1]), float(p[2]))
                                for p in raw], dtype=np.float32)
            else:
                # structured numpy array
                pts = np.column_stack(
                    [np.asarray(raw['x'], dtype=np.float32),
                     np.asarray(raw['y'], dtype=np.float32),
                     np.asarray(raw['z'], dtype=np.float32)])
            return pts
        except Exception as e:
            return None

    @staticmethod
    def _apply_tf(pts: np.ndarray, tf_stamped) -> np.ndarray:
        """
        Apply a TransformStamped to every row of an (N, 3) numpy array.

        Builds the rotation matrix from the quaternion (qx, qy, qz, qw)
        and applies  pts_out = R @ pts.T + t  in a single vectorised op.
        No ROS message objects are created per-point.
        """
        t  = tf_stamped.transform.translation
        q  = tf_stamped.transform.rotation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Quaternion → 3×3 rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [  2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
            [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

        result = (R @ pts.T).T + np.array([t.x, t.y, t.z])
        return result.astype(np.float32)

    def _cam_pt_to_world(self, c3d_cam):
        """
        Transform a camera-frame point [x, y, z] to the world frame via TF.
        Returns [x, y, z] rounded to 4 dp, or None on failure.
        """
        pt              = PointStamped()
        pt.header.frame_id = self._cam_frame
        pt.header.stamp    = rclpy.time.Time().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (
            float(c3d_cam[0]), float(c3d_cam[1]), float(c3d_cam[2]))
        try:
            pt_w = self._tf_buffer.transform(
                pt, 'world',
                timeout=rclpy.duration.Duration(seconds=0.5))
            return [round(pt_w.point.x, 4),
                    round(pt_w.point.y, 4),
                    round(pt_w.point.z, 4)]
        except Exception:
            return None

    @staticmethod
    def _sphere_marker(mid, ns, stamp, frame, xyz, rgb, scale):
        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = frame
        m.ns    = ns
        m.id    = mid
        m.type  = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        m.lifetime.sec = 5
        return m

    @staticmethod
    def _circle_ring(cx, cy, r, z, n=48):
        """Return a closed LINE_STRIP ring of *n* Points tracing the circle."""
        pts = []
        for i in range(n + 1):
            angle = 2.0 * np.pi * i / n
            p = Point()
            p.x = cx + r * np.cos(angle)
            p.y = cy + r * np.sin(angle)
            p.z = z
            pts.append(p)
        return pts


def main(args=None):
    rclpy.init(args=args)
    node = LidarFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
