#!/usr/bin/env python3
"""
Creative Applications for Object Detection Pipeline
Various cool use cases and interactive demos
"""

import cv2
import numpy as np
import time
import json
import random
from pathlib import Path
from collections import defaultdict
import pygame
import threading

from object_detector import ObjectDetector

class ObjectCountingGame:
    """Interactive game: Count specific objects in real-time"""
    
    def __init__(self, detector):
        self.detector = detector
        self.target_objects = ['person', 'bottle', 'cup', 'cell phone', 'book']
        self.current_target = random.choice(self.target_objects)
        self.score = 0
        self.countdown = 30  # 30 seconds per round
        self.start_time = time.time()
    
    def play_counting_game(self):
        """Play the object counting game"""
        cap = cv2.VideoCapture(0)
        
        print("🎮 Object Counting Game Started!")
        print(f"Find and show: {self.current_target.upper()}")
        print("You have 30 seconds per round!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.detector.model(frame, conf=0.5)  # Higher confidence for game
            
            # Process detections
            detections = self._extract_detections(results)
            target_count = sum(1 for det in detections if det['class_name'] == self.current_target)
            
            # Update score
            if target_count > 0:
                self.score += target_count
            
            # Check time
            elapsed = time.time() - self.start_time
            remaining = max(0, self.countdown - elapsed)
            
            # Annotate frame
            annotated_frame = self._annotate_game_frame(frame, detections, target_count, remaining)
            
            cv2.imshow('Object Counting Game', annotated_frame)
            
            # Next round
            if remaining <= 0:
                self._next_round()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"🏆 Final Score: {self.score}")
    
    def _extract_detections(self, results):
        """Extract detections from YOLO results"""
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
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
        return detections
    
    def _annotate_game_frame(self, frame, detections, target_count, remaining):
        """Annotate frame for the game"""
        annotated_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            
            # Highlight target objects
            if class_name == self.current_target:
                color = (0, 255, 0)  # Green for target
                thickness = 4
            else:
                color = (128, 128, 128)  # Gray for others
                thickness = 2
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Game UI
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Target: {self.current_target.upper()}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Score: {self.score}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Time: {remaining:.1f}s", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Found: {target_count}", (250, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _next_round(self):
        """Start next round with new target"""
        self.current_target = random.choice(self.target_objects)
        self.start_time = time.time()
        print(f"🎯 New target: {self.current_target.upper()}")

class SecurityMonitor:
    """Smart security monitoring system"""
    
    def __init__(self, detector):
        self.detector = detector
        self.motion_threshold = 10000  # Pixel area threshold
        self.alert_cooldown = 5  # seconds
        self.last_alert = 0
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.intrusion_zones = []  # Define zones to monitor
    
    def monitor_security(self):
        """Run security monitoring"""
        cap = cv2.VideoCapture(0)
        
        print("🔒 Security Monitor Active")
        print("Controls:")
        print("  'z' - Define intrusion zone (click and drag)")
        print("  'c' - Clear zones")
        print("  'q' - Quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Motion detection
            fg_mask = self.background_subtractor.apply(frame)
            
            # Object detection
            results = self.detector.model(frame, conf=0.3)
            detections = self._extract_detections(results)
            
            # Check for security alerts
            self._check_security_alerts(detections, fg_mask)
            
            # Annotate frame
            annotated_frame = self._annotate_security_frame(frame, detections, fg_mask)
            
            cv2.imshow('Security Monitor', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.intrusion_zones.clear()
                print("🗑️ Intrusion zones cleared")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _extract_detections(self, results):
        """Extract detections"""
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
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    })
        return detections
    
    def _check_security_alerts(self, detections, fg_mask):
        """Check for security violations"""
        current_time = time.time()
        if current_time - self.last_alert < self.alert_cooldown:
            return
        
        # Check for people in frame
        people = [det for det in detections if det['class_name'] == 'person']
        
        if people:
            # Check motion level
            motion_area = cv2.countNonZero(fg_mask)
            if motion_area > self.motion_threshold:
                self._trigger_security_alert("Motion detected with person present!", people)
                self.last_alert = current_time
        
        # Check for unusual objects
        unusual_objects = [det for det in detections 
                          if det['class_name'] in ['knife', 'scissors', 'umbrella']]
        if unusual_objects:
            self._trigger_security_alert("Unusual objects detected!", unusual_objects)
            self.last_alert = current_time
    
    def _trigger_security_alert(self, message, objects):
        """Trigger security alert"""
        print(f"🚨 SECURITY ALERT: {message}")
        for obj in objects:
            print(f"   - {obj['class_name']} (confidence: {obj['confidence']:.2f})")
        # Could add: save snapshot, send notification, activate alarm, etc.
    
    def _annotate_security_frame(self, frame, detections, fg_mask):
        """Annotate frame for security monitoring"""
        annotated_frame = frame.copy()
        
        # Draw detections with security-focused colors
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            
            # Color coding for security
            if class_name == 'person':
                color = (0, 0, 255)  # Red for people
            elif class_name in ['knife', 'scissors']:
                color = (0, 0, 255)  # Red for weapons
            else:
                color = (0, 255, 255)  # Yellow for other objects
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"{class_name} {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show motion mask in corner
        motion_display = cv2.resize(fg_mask, (160, 120))
        annotated_frame[10:130, 10:170] = cv2.cvtColor(motion_display, cv2.COLOR_GRAY2BGR)
        
        # Security status
        status_color = (0, 255, 0) if len([d for d in detections if d['class_name'] == 'person']) == 0 else (0, 0, 255)
        cv2.putText(annotated_frame, "SECURE" if status_color == (0, 255, 0) else "ALERT", 
                   (annotated_frame.shape[1]-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        return annotated_frame

class VirtualPetGame:
    """Virtual pet that reacts to detected objects"""
    
    def __init__(self, detector):
        self.detector = detector
        self.pet_mood = 50  # 0-100 scale
        self.pet_energy = 100
        self.last_interaction = time.time()
        self.pet_position = [320, 240]  # Center of screen
        
    def run_virtual_pet(self):
        """Run the virtual pet game"""
        cap = cv2.VideoCapture(0)
        
        print("🐱 Virtual Pet Started!")
        print("Show objects to interact with your pet:")
        print("  - Bottle/Cup: Pet gets thirsty")
        print("  - Food items: Pet gets hungry") 
        print("  - Person: Pet gets happy")
        print("  - Sports ball: Pet wants to play")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.detector.model(frame, conf=0.4)
            detections = self._extract_detections(results)
            
            # Update pet based on detections
            self._update_pet_state(detections)
            
            # Draw everything
            annotated_frame = self._draw_pet_world(frame, detections)
            
            cv2.imshow('Virtual Pet', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _extract_detections(self, results):
        """Extract detections"""
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
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
        return detections
    
    def _update_pet_state(self, detections):
        """Update pet state based on detected objects"""
        current_time = time.time()
        
        # Decay over time
        if current_time - self.last_interaction > 10:
            self.pet_mood = max(0, self.pet_mood - 1)
            self.pet_energy = max(0, self.pet_energy - 1)
        
        for det in detections:
            class_name = det['class_name']
            
            # React to different objects
            if class_name == 'person':
                self.pet_mood = min(100, self.pet_mood + 5)
                self.last_interaction = current_time
            elif class_name in ['bottle', 'cup']:
                self.pet_energy = min(100, self.pet_energy + 3)
                self.last_interaction = current_time
            elif class_name in ['pizza', 'banana', 'apple', 'sandwich']:
                self.pet_mood = min(100, self.pet_mood + 8)
                self.pet_energy = min(100, self.pet_energy + 5)
                self.last_interaction = current_time
            elif class_name in ['sports ball', 'frisbee']:
                self.pet_mood = min(100, self.pet_mood + 10)
                self.last_interaction = current_time
    
    def _draw_pet_world(self, frame, detections):
        """Draw the pet and its world"""
        annotated_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, det['class_name'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw pet (simple emoji-style)
        pet_x, pet_y = self.pet_position
        
        # Pet face based on mood
        if self.pet_mood > 70:
            pet_color = (0, 255, 0)  # Happy - green
            pet_emoji = "😊"
        elif self.pet_mood > 30:
            pet_color = (0, 255, 255)  # Neutral - yellow
            pet_emoji = "😐"
        else:
            pet_color = (0, 0, 255)  # Sad - red
            pet_emoji = "😢"
        
        # Draw pet as circle with emoji
        cv2.circle(annotated_frame, (pet_x, pet_y), 30, pet_color, -1)
        cv2.putText(annotated_frame, pet_emoji, (pet_x-15, pet_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Pet stats UI
        cv2.rectangle(annotated_frame, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Mood: {self.pet_mood}/100", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pet_color, 2)
        cv2.putText(annotated_frame, f"Energy: {self.pet_energy}/100", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mood bar
        bar_width = int((self.pet_mood / 100) * 200)
        cv2.rectangle(annotated_frame, (20, 75), (20 + bar_width, 85), pet_color, -1)
        cv2.rectangle(annotated_frame, (20, 75), (220, 85), (255, 255, 255), 1)
        
        return annotated_frame

def main():
    """Demo menu for creative applications"""
    print("🎨 Creative Object Detection Applications")
    print("=" * 50)
    print("Choose an application:")
    print("1. Object Counting Game")
    print("2. Security Monitor")
    print("3. Virtual Pet")
    print("4. Advanced Webcam (from previous file)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    try:
        detector = ObjectDetector('yolov8n.pt', conf_threshold=0.25)
        
        if choice == '1':
            game = ObjectCountingGame(detector)
            game.play_counting_game()
        elif choice == '2':
            monitor = SecurityMonitor(detector)
            monitor.monitor_security()
        elif choice == '3':
            pet = VirtualPetGame(detector)
            pet.run_virtual_pet()
        elif choice == '4':
            from advanced_webcam import AdvancedWebcamDetector
            advanced = AdvancedWebcamDetector()
            advanced.webcam_with_features()
        else:
            print("Invalid choice!")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and dependencies are installed.")

if __name__ == '__main__':
    main()