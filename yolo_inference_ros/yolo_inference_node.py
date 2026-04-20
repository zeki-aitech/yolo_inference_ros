import math
import time
from typing import List, Dict, Optional, Tuple, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.duration import Duration

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs

from vision_msgs.msg import (
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    LabelInfo, VisionClass
)
from visualization_msgs.msg import Marker, MarkerArray

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

import numpy as np

from yolo_inference_ros.depth_processor import DepthProcessor


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        # --- Parameter Declaration ---
        self.declare_parameter("enable_3d", False)
        self.declare_parameter("enable_debug", False)
        
        # TF Frame Parameters
        self.declare_parameter("target_frame", "base_link")
        
        # Tracker Parameters
        self.declare_parameter("enable_tracker", False) 
        self.declare_parameter("tracker_cfg", "bytetrack.yaml")
        
        # Temporal Filter Parameters (Sliding Window + Hysteresis)
        self.declare_parameter("enable_temporal_filter", False)
        self.declare_parameter("temp_window_size", 5)
        self.declare_parameter("temp_enter_thresh", 0.8)
        self.declare_parameter("temp_exit_thresh", 0.3)
        
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("task", "detect")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # --- Get Parameter Values ---
        self.enable_3d = self.get_parameter("enable_3d").get_parameter_value().bool_value
        self.enable_debug = self.get_parameter("enable_debug").get_parameter_value().bool_value
        
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        
        self.enable_tracker = self.get_parameter("enable_tracker").get_parameter_value().bool_value
        self.tracker_cfg = self.get_parameter("tracker_cfg").get_parameter_value().string_value
        
        self.enable_temporal_filter = self.get_parameter("enable_temporal_filter").get_parameter_value().bool_value
        self.temp_window_size = self.get_parameter("temp_window_size").get_parameter_value().integer_value
        self.temp_enter_thresh = self.get_parameter("temp_enter_thresh").get_parameter_value().double_value
        self.temp_exit_thresh = self.get_parameter("temp_exit_thresh").get_parameter_value().double_value
        
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = self.get_parameter("fuse_model").get_parameter_value().bool_value
        self.yolo_encoding = self.get_parameter("yolo_encoding").get_parameter_value().string_value

        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
    
        image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        
        self.image_qos_profile = QoSProfile(
            reliability=image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.cv_bridge = CvBridge()
        
        # --- Internal States for Temporal Filtering ---
        self.track_history: Dict[int, List[float]] = {}
        self.active_tracks: Set[int] = set()

        # Inference speed profiling: accumulate then log every 1s
        self._speed_last_log_time = time.monotonic()
        self._speed_preprocess_ms_list: List[float] = []
        self._speed_inference_ms_list: List[float] = []
        self._speed_postprocess_ms_list: List[float] = []
        self._speed_3d_ms_list: List[float] = []
        self._speed_callback_ms_list: List[float] = []

        if self.enable_temporal_filter and not self.enable_tracker:
            self.get_logger().warn("Temporal Filter requires Tracker! Forcing enable_temporal_filter to False.")
            self.enable_temporal_filter = False

        # --- Publishers ---
        self.pub_2d = self.create_publisher(Detection2DArray, 'detections_2d', 10)
        latch_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.pub_label_info = self.create_publisher(LabelInfo, 'labels', latch_qos)
        
        if self.enable_debug:
            self.get_logger().info("Debug Mode Enabled: Publishing to ~/debug_image and ~/debug_markers_3d")
            self.pub_debug = self.create_publisher(Image, '~/debug_image', 10)
            if self.enable_3d:
                self.pub_markers = self.create_publisher(MarkerArray, '~/debug_markers_3d', 10)

        # --- Subscribers & 3D Setup ---
        if self.enable_3d:
            self.get_logger().info(f"Mode: 3D (RGB + Depth + Info) -> Target Frame: {self.target_frame}")
            self.init_3d()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            self.pub_3d = self.create_publisher(Detection3DArray, 'detections_3d', 10)
            
            self.image_sub = message_filters.Subscriber(
                self, Image, '/camera/image_raw', qos_profile=self.image_qos_profile)
            self.depth_sub = message_filters.Subscriber(
                self, Image, '/camera/depth/image_raw', qos_profile=self.depth_qos_profile)
            self.depth_info_sub = message_filters.Subscriber(
                self, CameraInfo, '/camera/depth/camera_info', qos_profile=self.depth_info_qos_profile)
            
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.depth_info_sub], 
                queue_size=10, slop=0.1) 
            self.ts.registerCallback(self.callback_3d)
            
        else:
            self.get_logger().info("Mode: 2D (RGB Only)")
            self.sub_2d = self.create_subscription(
                Image, 
                '/camera/image_raw', 
                self.callback_2d, 
                self.image_qos_profile
            )
        
        # --- Initialize Model ---
        self.init_yolo_model()
        self.publish_label_info()
        
        self.get_logger().info('YoloInferenceNode initialized')
        if self.enable_tracker:
            self.get_logger().info(f"YOLO Tracker Enabled: {self.tracker_cfg}")
        if self.enable_temporal_filter:
            self.get_logger().info(f"Temporal Filter Enabled (Window: {self.temp_window_size}, "
                                   f"Enter: {self.temp_enter_thresh}, Exit: {self.temp_exit_thresh})")
        
    def init_yolo_model(self):
        self.get_logger().info(f"Loading YOLO model: {self.model} ...")
        
        try:
            self.yolo_model = YOLO(model=self.model, task=self.task)
            
            if 'cuda' in self.device and not torch.cuda.is_available():
                self.get_logger().warn("CUDA requested but not available! Fallback to CPU.")
                self.device = 'cpu'
            
            if self.model.endswith('.pt'):
                 self.yolo_model.to(self.device)

            self.get_logger().info(f"Model loaded on {self.device}")

        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load YOLO model: {e}")
            raise e
        
        if self.fuse_model and self.model.endswith('.pt'):
            try:
                self.get_logger().info("Fusing model layers for faster inference...")
                self.yolo_model.fuse()
            except Exception as e:
                self.get_logger().warn(f"Could not fuse model: {e}")

        try:
            self.get_logger().info("Warming up model...")
            dummy_input = torch.zeros((1, 3, self.imgsz_height, self.imgsz_width)).to(self.device)
            self.yolo_model(dummy_input, verbose=False, device=self.device)
            self.get_logger().info("Warmup complete! System ready.")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed (non-critical): {e}")

    def init_3d(self):
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("depth_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        depth_reliability = self.get_parameter("depth_reliability").get_parameter_value().integer_value
        depth_info_reliability = self.get_parameter("depth_info_reliability").get_parameter_value().integer_value
        
        self.depth_qos_profile = QoSProfile(
            reliability=depth_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.depth_info_qos_profile = QoSProfile(
            reliability=depth_info_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.depth_processor = DepthProcessor(
            depth_image_units_divisor=self.depth_image_units_divisor
        )

    def publish_label_info(self):
        msg = LabelInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "yolo_model" 
        
        for cls_id, cls_name in self.yolo_model.names.items():
            vc = VisionClass()
            vc.class_id = int(cls_id)
            vc.class_name = str(cls_name)
            msg.class_map.append(vc)
            
        msg.threshold = float(self.threshold)
        self.pub_label_info.publish(msg)
        self.get_logger().info(f"Published LabelInfo mapping for {len(self.yolo_model.names)} classes to /labels")

    def _apply_temporal_filter(self, results: Results) -> List[int]:
        if not results.boxes or results.boxes.id is None:
            self._decay_lost_tracks(set())
            return []

        valid_indices = []
        current_frame_tids = set()
        
        track_ids = results.boxes.id.int().tolist()
        confs = results.boxes.conf.float().tolist()

        for i, (tid, conf) in enumerate(zip(track_ids, confs)):
            current_frame_tids.add(tid)

            # 1. Sliding Window Logic
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append(conf)
            
            if len(self.track_history[tid]) > self.temp_window_size:
                self.track_history[tid].pop(0)

            history = self.track_history[tid]
            max_conf = max(history)
            mean_conf = sum(history) / len(history)

            # 2. Hysteresis Logic
            if tid not in self.active_tracks:
                min_hits = min(3, self.temp_window_size)
                if max_conf >= self.temp_enter_thresh and len(history) >= min_hits:
                    self.active_tracks.add(tid)
                    valid_indices.append(i)
            else:
                if mean_conf <= self.temp_exit_thresh:
                    self.active_tracks.remove(tid)
                else:
                    valid_indices.append(i)

        self._decay_lost_tracks(current_frame_tids)
        return valid_indices

    def _decay_lost_tracks(self, current_frame_tids: Set[int]):
        lost_tids = set(self.track_history.keys()) - current_frame_tids
        
        for tid in list(lost_tids):
            self.track_history[tid].append(0.0)
            
            if len(self.track_history[tid]) > self.temp_window_size:
                self.track_history[tid].pop(0)

            if sum(self.track_history[tid]) / len(self.track_history[tid]) <= self.temp_exit_thresh:
                if tid in self.active_tracks:
                    self.active_tracks.remove(tid)
                del self.track_history[tid]
        
    def callback_2d(self, image_msg: Image):
        self._process_data(image_msg)
        
    def callback_3d(self, image_msg: Image, depth_msg: Image, depth_info_msg: CameraInfo):
        self._process_data(image_msg, depth_msg, depth_info_msg)
    
    def _process_data(
        self, 
        image_msg: Image, 
        depth_msg: Optional[Image] = None, 
        depth_info_msg: Optional[CameraInfo] = None
    ):
        t_callback_start = time.perf_counter()
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                image_msg, desired_encoding=self.yolo_encoding
            )
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
        
        # --- TF Lookup for 3D Projection ---
        target_transform = None
        if self.enable_3d:
            try:
                # Use Time() (zero time) to fetch the latest transform, avoiding Extrapolation errors
                target_transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    image_msg.header.frame_id,
                    rclpy.time.Time()
                )
            except Exception as ex:
                self.get_logger().warn(f"TF Lookup failed from {image_msg.header.frame_id} to {self.target_frame}: {ex}", throttle_duration_sec=2.0)
                # If TF fails, skip 3D processing to avoid bad data
                pass

        # --- Inference / Tracking Execution ---
        if self.enable_tracker:
            inference_results = self.yolo_model.track(
                source=cv_image,
                tracker=self.tracker_cfg,
                persist=True,
                verbose=False,
                stream=False,
                conf=0.1 if self.enable_temporal_filter else self.threshold, 
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                device=self.device,
            )
        else:
            inference_results = self.yolo_model.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                max_det=self.max_det,
                augment=self.augment,
                agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks,
                device=self.device,
            )
        results: Results = inference_results[0].cpu()

        # --- Record model speed (preprocess | inference | postprocess) ---
        r0 = inference_results[0]
        pre_ms = inf_ms = post_ms = 0.0
        if hasattr(r0, 'speed') and isinstance(getattr(r0, 'speed'), dict):
            pre_ms = r0.speed.get('preprocess', 0.0)
            inf_ms = r0.speed.get('inference', 0.0)
            post_ms = r0.speed.get('postprocess', 0.0)
        elif hasattr(r0, 'speed') and hasattr(r0.speed, 'get'):
            pre_ms = r0.speed.get('preprocess', 0.0)
            inf_ms = r0.speed.get('inference', 0.0)
            post_ms = r0.speed.get('postprocess', 0.0)
        self._speed_preprocess_ms_list.append(pre_ms)
        self._speed_inference_ms_list.append(inf_ms)
        self._speed_postprocess_ms_list.append(post_ms)

        # --- Filter Execution ---
        if self.enable_temporal_filter and self.enable_tracker:
            valid_indices = self._apply_temporal_filter(results)
        else:
            valid_indices = list(range(len(results.boxes))) if results.boxes else []

        detections_2d_msg = Detection2DArray()
        detections_2d_msg.header = image_msg.header # 2D pixel coordinates stay in camera frame
        
        if self.enable_3d:
            detections_3d_msg = Detection3DArray()
            detections_3d_msg.header.stamp = image_msg.header.stamp
            detections_3d_msg.header.frame_id = self.target_frame # 3D coordinates are published in the target frame

        frame_3d_ms = 0.0
        if results.boxes:
            hypothesis_list = self.parse_hypothesis(results)
            boxes_2d_list = self.parse_boxes(results) 

            for i in valid_indices:
                # --- 2D Detection Processing ---
                det_2d = Detection2D()
                det_2d.header = image_msg.header
                det_2d.bbox = boxes_2d_list[i]
                
                det_2d.id = hypothesis_list[i]["track_id"]
                
                obj_hyp = ObjectHypothesisWithPose()
                obj_hyp.hypothesis.class_id = hypothesis_list[i]["class_id"]
                obj_hyp.hypothesis.score = hypothesis_list[i]["score"]
                
                det_2d.results.append(obj_hyp)
                detections_2d_msg.detections.append(det_2d)

                # --- 3D Detection Processing ---
                if self.enable_3d and depth_msg is not None and depth_info_msg is not None and target_transform is not None:
                    t_3d_start = time.perf_counter()
                    try:
                        depth_image = self.cv_bridge.imgmsg_to_cv2(
                            depth_msg, desired_encoding="passthrough")
                    except Exception as e:
                        self.get_logger().error(f"Depth processing failed: {e}")
                        frame_3d_ms += (time.perf_counter() - t_3d_start) * 1000.0
                        continue
                    
                    box_3d_data = self.depth_processor.convert_to_3d_bbox(
                        depth_image=depth_image,
                        depth_info=depth_info_msg,
                        center_x=det_2d.bbox.center.position.x,
                        center_y=det_2d.bbox.center.position.y,
                        size_x=det_2d.bbox.size_x,
                        size_y=det_2d.bbox.size_y
                    )
                    
                    if box_3d_data is not None:
                        # 1. Package as PoseStamped in the Camera Frame
                        pose_cam = PoseStamped()
                        pose_cam.header = image_msg.header
                        pose_cam.pose.position.x = box_3d_data.x
                        pose_cam.pose.position.y = box_3d_data.y
                        pose_cam.pose.position.z = box_3d_data.z
                        pose_cam.pose.orientation.w = 1.0 # Default unrotated camera optical frame

                        # 2. Apply TF2 Transform to Target Frame (e.g., base_link)
                        try:
                            pose_target = tf2_geometry_msgs.do_transform_pose(pose_cam.pose, target_transform)
                        except Exception as e:
                            self.get_logger().warn(f"Failed to apply TF transform: {e}")
                            frame_3d_ms += (time.perf_counter() - t_3d_start) * 1000.0
                            continue

                        # 3. Build the Detection3D Message
                        det_3d_msg = Detection3D()
                        det_3d_msg.header.stamp = image_msg.header.stamp
                        det_3d_msg.header.frame_id = self.target_frame
                        det_3d_msg.id = hypothesis_list[i]["track_id"]
                        
                        det_3d_msg.bbox.center = pose_target
                        
                        # Sizes remain identical. The orientation quaternion from do_transform_pose 
                        # automatically rotates the local axes of this bounding box to match the target frame!
                        det_3d_msg.bbox.size.x = box_3d_data.w
                        det_3d_msg.bbox.size.y = box_3d_data.h
                        det_3d_msg.bbox.size.z = box_3d_data.d
                        
                        det_3d_msg.results.append(obj_hyp)
                        detections_3d_msg.detections.append(det_3d_msg)
                    frame_3d_ms += (time.perf_counter() - t_3d_start) * 1000.0
        if self.enable_3d:
            self._speed_3d_ms_list.append(frame_3d_ms)

        # Publish Results
        self.pub_2d.publish(detections_2d_msg)
        if self.enable_3d and detections_3d_msg is not None:
            self.pub_3d.publish(detections_3d_msg)
            
        # Publish Debug Image & Markers
        if self.enable_debug:
            try:
                # 1. Publish Annotated 2D Image
                annotated_frame = results.plot()
                debug_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                debug_msg.header = image_msg.header
                self.pub_debug.publish(debug_msg)
                
                # 2. Publish 3D Markers to RViz2
                if self.enable_3d and detections_3d_msg is not None:
                    marker_array = MarkerArray()
                    
                    delete_marker = Marker()
                    delete_marker.action = Marker.DELETEALL
                    marker_array.markers.append(delete_marker)

                    for i, det_3d in enumerate(detections_3d_msg.detections):
                        new_markers = self.create_bb_markers(det_3d, color=(0, 255, 0), base_id=i)
                        marker_array.markers.extend(new_markers)
                        
                    self.pub_markers.publish(marker_array)
                    
            except Exception as e:
                self.get_logger().error(f"Failed to publish debug visualizations: {e}")

        # --- Full callback latency + log every 1s ---
        t_callback_end = time.perf_counter()
        self._speed_callback_ms_list.append((t_callback_end - t_callback_start) * 1000.0)
        now = time.monotonic()
        if now - self._speed_last_log_time >= 1.0 and self._speed_callback_ms_list:
            n = len(self._speed_callback_ms_list)
            avg_pre = sum(self._speed_preprocess_ms_list) / n
            avg_inf = sum(self._speed_inference_ms_list) / n
            avg_post = sum(self._speed_postprocess_ms_list) / n
            avg_cb = sum(self._speed_callback_ms_list) / n
            fps_cb = 1000.0 / avg_cb if avg_cb > 0 else 0.0
            msg = f"[Speed] frames={n} | model process: pre={avg_pre:.1f} inf={avg_inf:.1f} post={avg_post:.1f} ms"
            if self.enable_3d and self._speed_3d_ms_list:
                avg_3d = sum(self._speed_3d_ms_list) / len(self._speed_3d_ms_list)
                msg += f" | 3d process: {avg_3d:.1f} ms"
            msg += f" | full latency: {avg_cb:.1f} ms ({fps_cb:.1f} FPS)"
            self.get_logger().info(msg)
            self._speed_last_log_time = now
            self._speed_preprocess_ms_list.clear()
            self._speed_inference_ms_list.clear()
            self._speed_postprocess_ms_list.clear()
            self._speed_3d_ms_list.clear()
            self._speed_callback_ms_list.clear()

    def create_bb_markers(self, detection: Detection3D, color: Tuple[int, int, int], base_id: int) -> List[Marker]:
        markers = []
        bbox = detection.bbox
        lifetime = Duration(seconds=0.5).to_msg()

        box_marker = Marker()
        box_marker.header = detection.header
        box_marker.ns = "yolo_3d_boxes"
        box_marker.id = base_id * 2  
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.frame_locked = False

        box_marker.pose.position.x = bbox.center.position.x
        box_marker.pose.position.y = bbox.center.position.y
        box_marker.pose.position.z = bbox.center.position.z
        
        box_marker.pose.orientation.x = bbox.center.orientation.x
        box_marker.pose.orientation.y = bbox.center.orientation.y
        box_marker.pose.orientation.z = bbox.center.orientation.z
        box_marker.pose.orientation.w = bbox.center.orientation.w
        
        box_marker.scale.x = bbox.size.x
        box_marker.scale.y = bbox.size.y
        box_marker.scale.z = bbox.size.z

        box_marker.color.r = color[0] / 255.0
        box_marker.color.g = color[1] / 255.0
        box_marker.color.b = color[2] / 255.0
        box_marker.color.a = 0.4
        box_marker.lifetime = lifetime

        markers.append(box_marker)

        if detection.results:
            text_marker = Marker()
            text_marker.header = detection.header
            text_marker.ns = "yolo_3d_labels"
            text_marker.id = (base_id * 2) + 1  
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.frame_locked = False

            text_marker.pose.position.x = bbox.center.position.x
            text_marker.pose.position.y = bbox.center.position.y - (bbox.size.y / 2.0) - 0.1
            text_marker.pose.position.z = bbox.center.position.z
            
            text_marker.scale.z = 0.15 
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            class_id_str = detection.results[0].hypothesis.class_id
            raw_cls_id = int(class_id_str)
            class_name = self.yolo_model.names[raw_cls_id]
            
            score = detection.results[0].hypothesis.score
            track_id_str = detection.id
            
            if track_id_str != "-1":
                text_marker.text = f"{class_name}-[ID:{track_id_str}]-({score:.2f})"
            else:
                text_marker.text = f"{class_name}-({score:.2f})"
            
            text_marker.lifetime = lifetime
            markers.append(text_marker)

        return markers

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []
        if results.boxes:
            for box_data in results.boxes:
                cls_id = int(box_data.cls)
                track_id = int(box_data.id) if box_data.id is not None else -1

                hypothesis = {
                    "class_id": str(cls_id), 
                    "track_id": str(track_id),
                    "class_name": self.yolo_model.names[cls_id],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list = []
        if results.boxes:
            for box_data in results.boxes:
                msg = BoundingBox2D()
                box = box_data.xywh[0]
                
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = 0.0 
                
                msg.size_x = float(box[2])
                msg.size_y = float(box[3])
                boxes_list.append(msg)
        return boxes_list
    

def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()