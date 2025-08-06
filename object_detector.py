import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import time
from typing import List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    A comprehensive object detection class using YOLOv8
    Supports image, video, and real-time webcam detection
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        logger.info(f"Loading YOLO model: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        self.model = YOLO(model_path)
        
        # Get class names
        self.class_names = self.model.names
        
        # Colors for different classes (BGR format)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        logger.info(f"Model loaded successfully. Available classes: {len(self.class_names)}")
    
    def detect_image(self, image_path: str, save_path: Optional[str] = None, show_result: bool = True) -> List[dict]:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            show_result: Whether to display the result
            
        Returns:
            List of detection dictionaries
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run inference
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Store detection
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    
                    # Draw bounding box and label
                    color = [int(c) for c in self.colors[class_id]]
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save result if requested
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            logger.info(f"Annotated image saved to {save_path}")
        
        # Show result if requested
        if show_result:
            cv2.imshow('Object Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        logger.info(f"Detected {len(detections)} objects in image")
        return detections
    
    def detect_video(self, video_path: str, save_path: Optional[str] = None, show_result: bool = True) -> None:
        """
        Detect objects in a video file
        
        Args:
            video_path: Path to input video
            save_path: Path to save annotated video (optional)
            show_result: Whether to display the result in real-time
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if saving
        out = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, results)
            
            # Save frame if requested
            if out is not None:
                out.write(annotated_frame)
            
            # Show frame if requested
            if show_result:
                cv2.imshow('Video Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log progress every 30 frames
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                          f"FPS: {frame_count/elapsed:.1f}")
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        logger.info(f"Video processing completed in {elapsed:.2f}s, "
                   f"Average FPS: {frame_count/elapsed:.1f}")
    
    def detect_webcam(self, camera_id: int = 0) -> None:
        """
        Real-time object detection using webcam
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        logger.info("Starting real-time detection. Press 'q' to quit.")
        
        # FPS calculation
        prev_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, results)
            
            # Calculate and display FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:  # Update FPS every second
                fps_display = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            cv2.putText(annotated_frame, f"FPS: {fps_display}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Object Detection', annotated_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time detection stopped")
    
    def _annotate_frame(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Annotate a frame with detection results
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Draw bounding box and label
                    color = [int(c) for c in self.colors[class_id]]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.class_names),
            'class_names': list(self.class_names.values())
        }