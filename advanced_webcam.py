#!/usr/bin/env python3
"""
Advanced Webcam Object Detection with Interactive Features
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict, deque
import threading
import queue

from object_detector import ObjectDetector

class AdvancedWebcamDetector:
    """Enhanced webcam detector with interactive features"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.detector = ObjectDetector(model_path, conf_threshold)
        self.recording = False
        self.screenshot_mode = False
        self.tracking_objects = True
        self.show_stats = True
        
        # Statistics tracking
        self.object_counts = defaultdict(int)
        self.detection_history = deque(maxlen=100)  # Last 100 frames
        self.session_stats = {
            'total_detections': 0,
            'unique_objects': set(),
            'start_time': time.time()
        }
        
        # Alert system
        self.alerts = {
            'person': True,  # Alert when person detected
            'car': False,
            'animal': False  # Alert for animals (cat, dog, etc.)
        }
        self.last_alert_time = 0
        
        # Create output directories
        Path("screenshots").mkdir(exist_ok=True)
        Path("recordings").mkdir(exist_ok=True)
    
    def webcam_with_features(self, camera_id=0):
        """Enhanced webcam detection with interactive features"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Video recording setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        
        print("🎥 Advanced Webcam Detection Started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Take screenshot")
        print("  'r' - Start/Stop recording")
        print("  't' - Toggle object tracking")
        print("  'h' - Toggle stats display")
        print("  'a' - Toggle person alerts")
        print("  'c' - Clear statistics")
        
        fps_counter = 0
        fps_display = 0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.detector.model(frame, conf=self.detector.conf_threshold)
            
            # Process detections
            detections = self._extract_detections(results)
            self._update_statistics(detections)
            
            # Check for alerts
            self._check_alerts(detections)
            
            # Annotate frame
            annotated_frame = self._annotate_advanced_frame(frame, results, detections)
            
            # Add statistics overlay
            if self.show_stats:
                annotated_frame = self._add_stats_overlay(annotated_frame)
            
            # Calculate FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            # Add FPS to frame
            cv2.putText(annotated_frame, f"FPS: {fps_display}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Recording indicator
            if self.recording:
                cv2.circle(annotated_frame, (annotated_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (annotated_frame.shape[1] - 60, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Advanced Object Detection', annotated_frame)
            
            # Record if enabled
            if self.recording and out is not None:
                out.write(annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._take_screenshot(annotated_frame)
            elif key == ord('r'):
                out = self._toggle_recording(cap, out, fourcc)
            elif key == ord('t'):
                self.tracking_objects = not self.tracking_objects
                print(f"Object tracking: {'ON' if self.tracking_objects else 'OFF'}")
            elif key == ord('h'):
                self.show_stats = not self.show_stats
                print(f"Stats display: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('a'):
                self.alerts['person'] = not self.alerts['person']
                print(f"Person alerts: {'ON' if self.alerts['person'] else 'OFF'}")
            elif key == ord('c'):
                self._clear_statistics()
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Save session summary
        self._save_session_summary()
        print("📊 Session completed! Check session_summary.json for details.")
    
    def _extract_detections(self, results):
        """Extract detection data from YOLO results"""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.detector.class_names[class_id]
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'area': (x2 - x1) * (y2 - y1)
                    })
        return detections
    
    def _update_statistics(self, detections):
        """Update detection statistics"""
        frame_objects = []
        for det in detections:
            class_name = det['class_name']
            self.object_counts[class_name] += 1
            self.session_stats['unique_objects'].add(class_name)
            frame_objects.append(class_name)
        
        self.detection_history.append({
            'timestamp': time.time(),
            'objects': frame_objects,
            'count': len(detections)
        })
        self.session_stats['total_detections'] += len(detections)
    
    def _check_alerts(self, detections):
        """Check for alert conditions"""
        current_time = time.time()
        if current_time - self.last_alert_time < 2.0:  # Cooldown period
            return
        
        for det in detections:
            class_name = det['class_name']
            
            # Person alert
            if self.alerts['person'] and class_name == 'person':
                self._trigger_alert(f"Person detected! (confidence: {det['confidence']:.2f})")
                self.last_alert_time = current_time
            
            # Animal alert
            if self.alerts['animal'] and class_name in ['cat', 'dog', 'bird', 'horse']:
                self._trigger_alert(f"{class_name.title()} detected!")
                self.last_alert_time = current_time
    
    def _trigger_alert(self, message):
        """Trigger an alert (could be extended to send notifications)"""
        print(f"🚨 ALERT: {message}")
        # Could add: sound alert, email, push notification, etc.
    
    def _annotate_advanced_frame(self, frame, results, detections):
        """Enhanced frame annotation with additional features"""
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Color based on object type
            color = self._get_object_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Enhanced label with additional info
            label = f"{class_name}: {confidence:.2f}"
            if self.tracking_objects:
                area = det['area']
                label += f" (area: {area}px)"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point for tracking
            if self.tracking_objects:
                center = det['center']
                cv2.circle(annotated_frame, center, 3, color, -1)
        
        return annotated_frame
    
    def _get_object_color(self, class_name):
        """Get color based on object category"""
        if class_name == 'person':
            return (0, 255, 0)  # Green for people
        elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
            return (255, 0, 0)  # Blue for vehicles
        elif class_name in ['cat', 'dog', 'bird', 'horse']:
            return (0, 255, 255)  # Yellow for animals
        else:
            return (0, 0, 255)  # Red for other objects
    
    def _add_stats_overlay(self, frame):
        """Add statistics overlay to frame"""
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 60), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Current session stats
        y_offset = 80
        cv2.putText(frame, "Session Stats:", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        session_time = time.time() - self.session_stats['start_time']
        cv2.putText(frame, f"Time: {session_time:.0f}s", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Total detections: {self.session_stats['total_detections']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Top detected objects
        y_offset += 20
        top_objects = sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for obj, count in top_objects:
            y_offset += 15
            cv2.putText(frame, f"{obj}: {count}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _take_screenshot(self, frame):
        """Save a screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def _toggle_recording(self, cap, out, fourcc):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/detection_{timestamp}.avi"
            
            # Get frame dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.recording = True
            print(f"🎬 Recording started: {filename}")
        else:
            # Stop recording
            if out is not None:
                out.release()
            self.recording = False
            print("⏹️ Recording stopped")
            out = None
        
        return out
    
    def _clear_statistics(self):
        """Clear session statistics"""
        self.object_counts.clear()
        self.detection_history.clear()
        self.session_stats = {
            'total_detections': 0,
            'unique_objects': set(),
            'start_time': time.time()
        }
        print("📊 Statistics cleared!")
    
    def _save_session_summary(self):
        """Save session summary to JSON"""
        summary = {
            'session_duration': time.time() - self.session_stats['start_time'],
            'total_detections': self.session_stats['total_detections'],
            'unique_objects': list(self.session_stats['unique_objects']),
            'object_counts': dict(self.object_counts),
            'detection_rate': self.session_stats['total_detections'] / len(self.detection_history) if self.detection_history else 0
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_summary_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Demo the advanced webcam features"""
    print("🚀 Advanced Webcam Object Detection")
    print("=" * 40)
    
    try:
        detector = AdvancedWebcamDetector(model_path='yolov8n.pt', conf_threshold=0.3)
        detector.webcam_with_features(camera_id=0)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and not used by another application.")

if __name__ == '__main__':
    main()