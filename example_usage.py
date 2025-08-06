#!/usr/bin/env python3
"""
Object Detection Pipeline - Example Usage
Demonstrates various features of the object detection system
"""

import os
import cv2
import numpy as np
from pathlib import Path
import requests

from object_detector import ObjectDetector
from utils import (
    analyze_detections,
    save_detections_json,
    create_detection_summary,
    filter_detections,
    plot_detection_statistics,
    visualize_detections_matplotlib
)

def download_sample_image(url: str, filename: str) -> str:
    """Download a sample image for testing"""
    if not os.path.exists(filename):
        print(f"Downloading sample image: {filename}")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    return filename

def create_sample_images():
    """Create or download sample images for testing"""
    # Create samples directory
    Path("samples").mkdir(exist_ok=True)
    
    # Sample images URLs (using free images from various sources)
    sample_urls = {
        "samples/street_scene.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Busy_street_scene_in_Tokyo.jpg/640px-Busy_street_scene_in_Tokyo.jpg",
        "samples/animals.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Elephants_drinking.jpg/640px-Elephants_drinking.jpg",
        "samples/office.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Office_desktop.jpg/640px-Office_desktop.jpg"
    }
    
    # Try to download sample images
    downloaded_images = []
    for filename, url in sample_urls.items():
        try:
            downloaded_images.append(download_sample_image(url, filename))
        except Exception as e:
            print(f"Could not download {filename}: {e}")
    
    # If downloads failed, create a simple test image
    if not downloaded_images:
        print("Creating a simple test image...")
        # Create a simple image with shapes
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw some simple objects
        cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(img, (400, 100), 80, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(img, (300, 300), (550, 400), (0, 0, 255), -1)  # Red rectangle
        
        # Add some text
        cv2.putText(img, "Test Image", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        test_image_path = "samples/test_image.jpg"
        cv2.imwrite(test_image_path, img)
        downloaded_images.append(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    return downloaded_images

def example_basic_detection():
    """Example 1: Basic object detection on an image"""
    print("=" * 60)
    print("Example 1: Basic Object Detection")
    print("=" * 60)
    
    # Create detector with default YOLOv8 nano model
    detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.25)
    
    # Get or create sample images
    sample_images = create_sample_images()
    
    if not sample_images:
        print("No sample images available for this example.")
        return
    
    # Use the first available sample image
    image_path = sample_images[0]
    print(f"Processing image: {image_path}")
    
    # Detect objects
    detections = detector.detect_image(
        image_path=image_path,
        save_path="output/example1_result.jpg",
        show_result=False  # Set to True to display the image
    )
    
    # Print results
    if detections:
        print(f"Detected {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']} (confidence: {det['confidence']:.3f})")
    else:
        print("No objects detected.")
    
    return detections

def example_detection_analysis():
    """Example 2: Advanced detection analysis and visualization"""
    print("=" * 60)
    print("Example 2: Detection Analysis and Visualization")
    print("=" * 60)
    
    # Run detection
    detections = example_basic_detection()
    
    if not detections:
        print("No detections to analyze.")
        return
    
    # Analyze detections
    analysis = analyze_detections(detections)
    print(f"\nDetection Analysis:")
    print(f"  Total objects: {analysis['total_objects']}")
    print(f"  Unique classes: {analysis['unique_classes']}")
    print(f"  Average confidence: {analysis['confidence_stats']['mean']:.3f}")
    
    # Save results to JSON
    Path("output").mkdir(exist_ok=True)
    save_detections_json(detections, "output/detections.json")
    
    # Create summary report
    sample_images = create_sample_images()
    if sample_images:
        summary_path = create_detection_summary(sample_images[0], detections, "output")
        print(f"Summary report created at: {summary_path}")

def example_filtering():
    """Example 3: Filtering detections"""
    print("=" * 60)
    print("Example 3: Filtering Detections")
    print("=" * 60)
    
    # Run detection
    detections = example_basic_detection()
    
    if not detections:
        print("No detections to filter.")
        return
    
    print(f"Original detections: {len(detections)}")
    
    # Filter by confidence
    high_conf_detections = filter_detections(
        detections, 
        min_confidence=0.5
    )
    print(f"High confidence detections (>0.5): {len(high_conf_detections)}")
    
    # Filter by specific classes (example: only people and cars)
    specific_classes = filter_detections(
        detections,
        allowed_classes=['person', 'car', 'truck', 'bicycle']
    )
    print(f"Vehicle/person detections: {len(specific_classes)}")
    
    # Filter by minimum area
    large_objects = filter_detections(
        detections,
        min_area=1000  # Minimum 1000 pixels area
    )
    print(f"Large objects (>1000px area): {len(large_objects)}")

def example_different_models():
    """Example 4: Comparing different YOLO models"""
    print("=" * 60)
    print("Example 4: Comparing Different YOLO Models")
    print("=" * 60)
    
    # Get sample image
    sample_images = create_sample_images()
    if not sample_images:
        print("No sample images available.")
        return
    
    image_path = sample_images[0]
    
    # Different YOLO models (from smallest/fastest to largest/most accurate)
    models = [
        ('yolov8n.pt', 'YOLOv8 Nano'),
        ('yolov8s.pt', 'YOLOv8 Small'),
        ('yolov8m.pt', 'YOLOv8 Medium'),
    ]
    
    results = {}
    
    for model_path, model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            # Create detector
            detector = ObjectDetector(model_path=model_path, conf_threshold=0.25)
            
            # Time the detection
            import time
            start_time = time.time()
            detections = detector.detect_image(
                image_path=image_path,
                save_path=f"output/{model_path.replace('.pt', '_result.jpg')}",
                show_result=False
            )
            end_time = time.time()
            
            # Store results
            results[model_name] = {
                'detections': len(detections),
                'time': end_time - start_time,
                'model_size': model_path
            }
            
            print(f"  Detections: {len(detections)}")
            print(f"  Time: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison
    print(f"\nModel Comparison Summary:")
    print(f"{'Model':<15} {'Detections':<12} {'Time (s)':<10}")
    print("-" * 40)
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name:<15} {result['detections']:<12} {result['time']:<10.3f}")
        else:
            print(f"{model_name:<15} {'Error':<12} {'-':<10}")

def example_webcam_detection():
    """Example 5: Real-time webcam detection (interactive)"""
    print("=" * 60)
    print("Example 5: Real-time Webcam Detection")
    print("=" * 60)
    
    print("This example will start real-time object detection using your webcam.")
    print("Press 'q' to quit the webcam view.")
    
    # Ask user if they want to proceed
    response = input("Do you want to start webcam detection? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Skipping webcam detection example.")
        return
    
    try:
        # Create detector
        detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.3)
        
        # Start webcam detection
        detector.detect_webcam(camera_id=0)
        
    except Exception as e:
        print(f"Webcam detection failed: {e}")
        print("This might happen if no webcam is available.")

def example_batch_processing():
    """Example 6: Batch processing multiple images"""
    print("=" * 60)
    print("Example 6: Batch Processing Multiple Images")
    print("=" * 60)
    
    # Get sample images
    sample_images = create_sample_images()
    
    if len(sample_images) < 2:
        print("Need at least 2 images for batch processing example.")
        return
    
    # Create detector
    detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.25)
    
    # Process all images
    all_detections = {}
    total_objects = 0
    
    print(f"Processing {len(sample_images)} images...")
    
    for i, image_path in enumerate(sample_images, 1):
        print(f"\nProcessing image {i}/{len(sample_images)}: {image_path}")
        
        # Detect objects
        detections = detector.detect_image(
            image_path=image_path,
            save_path=f"output/batch_result_{i}.jpg",
            show_result=False
        )
        
        all_detections[image_path] = detections
        total_objects += len(detections)
        
        print(f"  Detected: {len(detections)} objects")
    
    # Summary
    print(f"\nBatch Processing Summary:")
    print(f"  Total images processed: {len(sample_images)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Average objects per image: {total_objects / len(sample_images):.1f}")
    
    # Find most common classes across all images
    all_classes = {}
    for image_detections in all_detections.values():
        for det in image_detections:
            class_name = det['class_name']
            all_classes[class_name] = all_classes.get(class_name, 0) + 1
    
    if all_classes:
        print(f"  Most common classes:")
        for class_name, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {class_name}: {count}")

def main():
    """Run all examples"""
    print("Object Detection Pipeline - Example Usage")
    print("=" * 60)
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    # Run examples
    try:
        example_basic_detection()
        example_detection_analysis()
        example_filtering()
        example_different_models()
        example_batch_processing()
        
        # Interactive examples
        print("\n" + "=" * 60)
        print("Interactive Examples")
        print("=" * 60)
        example_webcam_detection()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"Error during examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll examples completed!")
    print("Check the 'output' directory for generated files.")

if __name__ == '__main__':
    main()