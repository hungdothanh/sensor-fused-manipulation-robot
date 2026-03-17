#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
import numpy as np
import threading
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from std_msgs.msg import String
# from visualization_msgs.msg import Marker, MarkerArray  # 3D bbox RViz — disabled
# from geometry_msgs.msg import Point                      # 3D bbox RViz — disabled
import json


class VLMPerceptionNode(Node):
    def __init__(self):
        super().__init__('vlm_perception_node')

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f'Loading Grounding DINO on {self.device}...')

        # Load Grounding DINO
        # use_fast=True: silence the "slow processor" deprecation warning.
        # HF_HUB_OFFLINE is set in the launch file so transformers never
        # attempts a network check after the initial download.
        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny").to(self.device)
        self.get_logger().info('Model loaded successfully.')

        # Current Data
        self._cv_color = None
        self._cv_depth = None
        self._intrinsics = None
        self._lock = threading.Lock()
        self._detecting = False   # guard against concurrent runs

        self.declare_parameter(
            'auto_detect_prompt',
            'red cylinder. green cylinder.'
        )
        self._auto_prompt = self.get_parameter('auto_detect_prompt').value
        self.get_logger().info(f'Auto-detect prompt: "{self._auto_prompt}"')

        # Pre-parse prompt into individual phrases for label disambiguation
        self._prompt_phrases = [
            p.strip() for p in self._auto_prompt.rstrip('.').split('.')
            if p.strip()
        ]

        self.create_timer(2.0, self._auto_detect_cb)

        self.sub_prompt = self.create_subscription(String, '/perception/dino_prompt', self._prompt_cb, 10)
        self.pub_results = self.create_publisher(String, '/perception/dino_results', 10)

        self.sub_color = self.create_subscription(Image, '/camera/color/image_raw', self._color_cb, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, '/camera/depth/image_rect_raw', self._depth_cb, qos_profile_sensor_data)
        self.sub_info  = self.create_subscription(CameraInfo, '/camera/color/camera_info', self._info_cb, qos_profile_sensor_data)

        self.pub_debug  = self.create_publisher(Image, '/perception/dino_debug', qos_profile_sensor_data)
        # self.pub_bbox3d = self.create_publisher(MarkerArray, '/perception/dino_bbox_3d', 10)  # 3D bbox RViz — disabled

    def _auto_detect_cb(self):
        if self._detecting:
            return
        with self._lock:
            ready = self._cv_color is not None and self._cv_depth is not None
        if not ready:
            return
        self._detecting = True
        threading.Thread(target=self._run_auto_detect, daemon=True).start()

    def _run_auto_detect(self):
        try:
            self._run_detection(self._auto_prompt)
        finally:
            self._detecting = False
            import time; time.sleep(28.0)

    def _color_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._lock:
                self._cv_color = cv_img
        except Exception as e:
            self.get_logger().error(str(e))

    def _depth_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            with self._lock:
                self._cv_depth = cv_img
        except Exception as e:
            self.get_logger().error(str(e))

    def _info_cb(self, msg):
        with self._lock:
            self._intrinsics = {
                'fx': msg.p[0],
                'fy': msg.p[5],
                'cx': msg.p[2],
                'cy': msg.p[6]
            }

    def _get_3d_point(self, u, v, depth_m):
        """Deproject 2D pixel to 3D point in camera frame."""
        if not self._intrinsics:
            return None
        x = (u - self._intrinsics['cx']) * depth_m / self._intrinsics['fx']
        y = (v - self._intrinsics['cy']) * depth_m / self._intrinsics['fy']
        return [float(x), float(y), float(depth_m)]

    # ── Composite-label resolver ──────────────────────────────────────────────
    # Grounding DINO concatenates all text tokens that score above text_threshold
    # for a single box (e.g. "red cylinder blue cylinder" from one detection).
    # Resolve by sampling the dominant hue in the bbox and matching it to which
    # phrase's colour word best fits the actual pixels.
    _COLOUR_BGR = {
        'red':    (0,   0,   200),
        'green':  (0,   180, 0),
        'blue':   (200, 0,   0),
        'yellow': (0,   200, 200),
        'orange': (0,   100, 220),
        'purple': (180, 0,   180),
        'cyan':   (200, 200, 0),
    }

    def _resolve_label(self, label: str, img_bgr, box) -> str:
        """
        If *label* is a DINO composite (multiple prompt phrases joined by space),
        pick the phrase whose colour word best matches the median BGR in the bbox.
        Falls back to the first matching phrase if no colour word is found.
        """
        label_lower = label.lower().strip()
        matches = [p for p in self._prompt_phrases if p.lower() in label_lower]
        if len(matches) <= 1:
            return label   # already a single clean label

        self.get_logger().warn(
            f'Composite DINO label "{label}" — resolving via pixel colour …')

        x1, y1, x2, y2 = (max(0, int(b)) for b in box[:4])
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return matches[0]

        median_bgr = np.median(roi.reshape(-1, 3), axis=0)  # [B, G, R]

        best_phrase, best_score = matches[0], -1.0
        for phrase in matches:
            for colour_word, ref_bgr in self._COLOUR_BGR.items():
                if colour_word in phrase.lower():
                    # Cosine similarity between median pixel and reference colour
                    med = np.array(median_bgr, dtype=np.float32)
                    ref = np.array(ref_bgr,    dtype=np.float32)
                    denom = (np.linalg.norm(med) * np.linalg.norm(ref))
                    score = float(np.dot(med, ref) / denom) if denom > 1e-6 else 0.0
                    if score > best_score:
                        best_score  = score
                        best_phrase = phrase

        self.get_logger().info(
            f'  → resolved to "{best_phrase}" (colour score {best_score:.3f})')
        return best_phrase

    # ── Colour map for RViz marker colours — disabled (3D bbox RViz only) ────
    # _COLOUR_VOCAB = {
    #     'red':    (1.0, 0.15, 0.15),
    #     'green':  (0.1, 0.9,  0.1),
    #     'blue':   (0.1, 0.4,  1.0),
    #     'yellow': (1.0, 1.0,  0.0),
    #     'orange': (1.0, 0.5,  0.0),
    #     'purple': (0.7, 0.0,  1.0),
    #     'cyan':   (0.0, 0.9,  0.9),
    #     'pink':   (1.0, 0.4,  0.7),
    #     'white':  (0.9, 0.9,  0.9),
    #     'black':  (0.2, 0.2,  0.2),
    # }
    # def _label_colour(self, label: str):
    #     lower = label.lower()
    #     for word, rgb in self._COLOUR_VOCAB.items():
    #         if word in lower:
    #             return rgb
    #     return (0.8, 0.8, 0.0)

    # ── 3D bbox helpers — disabled (RViz only, not used by chatbox pipeline) ──
    # def _compute_bbox_3d(self, x1, y1, x2, y2, depth_img):
    #     """Back-project all valid depth pixels inside the 2D bbox to 3D."""
    #     if not self._intrinsics:
    #         return None, None, None
    #     fx = self._intrinsics['fx']; fy = self._intrinsics['fy']
    #     cx = self._intrinsics['cx']; cy = self._intrinsics['cy']
    #     h, w = depth_img.shape[:2]
    #     x1c, y1c = max(0, x1), max(0, y1)
    #     x2c, y2c = min(w - 1, x2), min(h - 1, y2)
    #     if x2c <= x1c or y2c <= y1c:
    #         return None, None, None
    #     roi = depth_img[y1c:y2c + 1, x1c:x2c + 1]
    #     grid_y, grid_x = np.mgrid[y1c:y2c + 1, x1c:x2c + 1]
    #     zs = roi.flatten().astype(np.float32)
    #     xs = grid_x.flatten().astype(np.float32)
    #     ys = grid_y.flatten().astype(np.float32)
    #     valid = np.isfinite(zs) & (zs > 0.05) & (zs < 5.0)
    #     if valid.sum() < 5:
    #         return None, None, None
    #     zs, xs, ys = zs[valid], xs[valid], ys[valid]
    #     x3 = (xs - cx) * zs / fx
    #     y3 = (ys - cy) * zs / fy
    #     pts = np.stack([x3, y3, zs], axis=1)
    #     min_pt  = pts.min(axis=0)
    #     max_pt  = pts.max(axis=0)
    #     centroid = (min_pt + max_pt) / 2.0
    #     return centroid, min_pt, max_pt

    # def _bbox3d_to_line_list(self, min_pt, max_pt):
    #     x0, y0, z0 = (float(v) for v in min_pt)
    #     x1, y1, z1 = (float(v) for v in max_pt)
    #     corners = [
    #         (x0,y0,z0),(x1,y0,z0),(x0,y1,z0),(x1,y1,z0),
    #         (x0,y0,z1),(x1,y0,z1),(x0,y1,z1),(x1,y1,z1),
    #     ]
    #     edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    #     pts = []
    #     for a, b in edges:
    #         pa = Point(); pa.x, pa.y, pa.z = corners[a]
    #         pb = Point(); pb.x, pb.y, pb.z = corners[b]
    #         pts.extend([pa, pb])
    #     return pts

    def _prompt_cb(self, msg):
        text_prompt = msg.data.lower()
        self.get_logger().info(f'Received DINO query: "{text_prompt}"')
        threading.Thread(
            target=self._run_detection, args=(text_prompt,), daemon=True
        ).start()

    def _run_detection(self, text_prompt: str):
        with self._lock:
            color_img = self._cv_color.copy() if self._cv_color is not None else None
            depth_img = self._cv_depth.copy() if self._cv_depth is not None else None

        if color_img is None or depth_img is None:
            self.get_logger().warn('No images received yet. Cannot run DINO.')
            return

        image_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image_rgb, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image_rgb.shape[:2]])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.37,
            target_sizes=target_sizes
        )[0]

        detections = []
        debug_img = color_img.copy()
        img_h, img_w = color_img.shape[:2]

        label_key = "text_labels" if "text_labels" in results else "labels"
        for score, label, box in zip(results["scores"], results[label_key], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            score = score.item()
            # Resolve composite labels ("red cylinder blue cylinder" → single phrase)
            label = self._resolve_label(label, color_img, box)

            x1, y1, x2, y2 = map(int, box)
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(img_w - 1, x2); y2c = min(img_h - 1, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            # Centroid from 2D bounding box center
            cx = (x1c + x2c) // 2
            cy = (y1c + y2c) // 2

            # Sample depth at centroid
            depth_m = float(depth_img[cy, cx])

            if not np.isfinite(depth_m) or depth_m < 0.05 or depth_m > 5.0:
                self.get_logger().warn(
                    f'Bad depth {depth_m:.3f} m for "{label}", skipping.')
                continue

            p3d = self._get_3d_point(cx, cy, depth_m)

            if p3d:
                detections.append({
                    "label": label,
                    "score": round(score, 2),
                    "box_2d": box,
                    "centroid_3d": p3d
                })

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw mask centroid in cyan so it's visually distinct from the
                # old bbox-centre dot
                cv2.circle(debug_img, (cx, cy), 5, (255, 255, 0), -1)
                cv2.putText(debug_img, f"{label}: {score:.2f} ({depth_m:.2f}m)",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── 3-D bounding boxes (MarkerArray) — disabled (RViz only) ──────────
        # ma = MarkerArray()
        # stamp = self.get_clock().now().to_msg()
        # cam_frame = 'camera_color_optical_frame'
        # for mid, det in enumerate(detections):
        #     x1, y1, x2, y2 = map(int, det['box_2d'])
        #     centroid3d, min_pt, max_pt = self._compute_bbox_3d(
        #         x1, y1, x2, y2, depth_img)
        #     if min_pt is None:
        #         continue
        #     r, g, b = self._label_colour(det['label'])
        #     box_m = Marker()
        #     box_m.header.stamp    = stamp
        #     box_m.header.frame_id = cam_frame
        #     box_m.ns   = 'dino_box'
        #     box_m.id   = mid * 2
        #     box_m.type = Marker.LINE_LIST
        #     box_m.action = Marker.ADD
        #     box_m.scale.x = 0.006
        #     box_m.color.r = r; box_m.color.g = g; box_m.color.b = b; box_m.color.a = 1.0
        #     box_m.pose.orientation.w = 1.0
        #     box_m.points = self._bbox3d_to_line_list(min_pt, max_pt)
        #     box_m.lifetime.sec = 0
        #     ma.markers.append(box_m)
        #     txt_m = Marker()
        #     txt_m.header.stamp    = stamp
        #     txt_m.header.frame_id = cam_frame
        #     txt_m.ns   = 'dino_label'
        #     txt_m.id   = mid * 2 + 1
        #     txt_m.type = Marker.TEXT_VIEW_FACING
        #     txt_m.action = Marker.ADD
        #     txt_m.pose.position.x = float(centroid3d[0])
        #     txt_m.pose.position.y = float(centroid3d[1])
        #     txt_m.pose.position.z = float(max_pt[2]) + 0.05
        #     txt_m.pose.orientation.w = 1.0
        #     txt_m.scale.z = 0.05
        #     txt_m.color.r = 1.0; txt_m.color.g = 1.0; txt_m.color.b = 1.0; txt_m.color.a = 1.0
        #     txt_m.text = f'{det["label"]} ({det["score"]:.2f})'
        #     txt_m.lifetime.sec = 0
        #     ma.markers.append(txt_m)
        # self.pub_bbox3d.publish(ma)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.pub_debug.publish(debug_msg)
        except Exception:
            pass

        res_json = json.dumps({"detections": detections})
        self.pub_results.publish(String(data=res_json))
        self.get_logger().info(f'Published {len(detections)} detections.')

def main(args=None):
    rclpy.init(args=args)
    node = VLMPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
