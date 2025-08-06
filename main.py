#!/usr/bin/env python3
"""
Object Detection Pipeline - Main Application
A comprehensive object detection system using YOLOv8
"""

import argparse
import sys
import os
from pathlib import Path
import logging

from object_detector import ObjectDetector
from utils import (
    analyze_detections, 
    save_detections_json, 
    create_detection_summary,
    filter_detections,
    plot_detection_statistics,
    benchmark_model
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def detect_image_command(args):
    """Handle image detection command"""
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    print(f"Processing image: {args.input}")
    
    # Detect objects
    detections = detector.detect_image(
        image_path=args.input,
        save_path=args.output,
        show_result=not args.no_display
    )
    
    # Print detection summary
    if detections:
        print(f"\nDetection Results:")
        print(f"Total objects detected: {len(detections)}")
        
        # Group by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  - {class_name}: {count}")
        
        # Save results if requested
        if args.save_json:
            json_path = args.save_json
            save_detections_json(detections, json_path)
        
        # Create summary report if requested
        if args.save_summary:
            summary_path = create_detection_summary(args.input, detections, args.save_summary)
            print(f"Summary saved to: {summary_path}")
    else:
        print("No objects detected in the image.")

def detect_video_command(args):
    """Handle video detection command"""
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    print(f"Processing video: {args.input}")
    
    detector.detect_video(
        video_path=args.input,
        save_path=args.output,
        show_result=not args.no_display
    )
    
    print("Video processing completed.")

def detect_webcam_command(args):
    """Handle webcam detection command"""
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    print(f"Starting real-time detection on camera {args.camera}")
    print("Press 'q' to quit")
    
    detector.detect_webcam(camera_id=args.camera)

def benchmark_command(args):
    """Handle benchmark command"""
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    # Get test images
    if os.path.isdir(args.input):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        test_images = []
        for ext in image_extensions:
            test_images.extend(Path(args.input).glob(f'*{ext}'))
            test_images.extend(Path(args.input).glob(f'*{ext.upper()}'))
        test_images = [str(img) for img in test_images]
    else:
        # Single image
        test_images = [args.input]
    
    if not test_images:
        print(f"No images found in {args.input}")
        return
    
    print(f"Benchmarking with {len(test_images)} images...")
    
    results = benchmark_model(detector, test_images, num_runs=args.runs)
    
    # Save benchmark results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {args.output}")

def info_command(args):
    """Handle info command"""
    detector = ObjectDetector(model_path=args.model)
    info = detector.get_model_info()
    
    print("Model Information:")
    print(f"  Model Path: {info['model_path']}")
    print(f"  Device: {info['device']}")
    print(f"  Confidence Threshold: {info['conf_threshold']}")
    print(f"  IoU Threshold: {info['iou_threshold']}")
    print(f"  Number of Classes: {info['num_classes']}")
    print(f"  Available Classes:")
    
    # Print classes in columns
    classes = info['class_names']
    for i in range(0, len(classes), 4):
        row = classes[i:i+4]
        print(f"    {' | '.join(f'{j:2d}: {cls:15s}' for j, cls in enumerate(row, i))}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Object Detection Pipeline using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect objects in an image
  python main.py image input.jpg --output output.jpg
  
  # Process video with custom confidence threshold
  python main.py video input.mp4 --confidence 0.5 --output output.mp4
  
  # Real-time webcam detection
  python main.py webcam
  
  # Benchmark performance on a directory of images
  python main.py benchmark ./test_images/ --runs 5
  
  # Show model information
  python main.py info --model yolov8s.pt
        """
    )
    
    # Global arguments
    parser.add_argument('--model', default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Image detection
    image_parser = subparsers.add_parser('image', help='Detect objects in an image')
    image_parser.add_argument('input', help='Input image path')
    image_parser.add_argument('--output', '-o', help='Output image path')
    image_parser.add_argument('--no-display', action='store_true',
                             help='Don\'t display the result')
    image_parser.add_argument('--save-json', help='Save detections to JSON file')
    image_parser.add_argument('--save-summary', help='Save summary to directory')
    
    # Video detection
    video_parser = subparsers.add_parser('video', help='Detect objects in a video')
    video_parser.add_argument('input', help='Input video path')
    video_parser.add_argument('--output', '-o', help='Output video path')
    video_parser.add_argument('--no-display', action='store_true',
                             help='Don\'t display the result')
    
    # Webcam detection
    webcam_parser = subparsers.add_parser('webcam', help='Real-time webcam detection')
    webcam_parser.add_argument('--camera', type=int, default=0,
                              help='Camera device ID (default: 0)')
    
    # Benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    benchmark_parser.add_argument('input', help='Input image or directory')
    benchmark_parser.add_argument('--runs', type=int, default=3,
                                 help='Number of benchmark runs (default: 3)')
    benchmark_parser.add_argument('--output', '-o', help='Save benchmark results to JSON')
    
    # Model info
    info_parser = subparsers.add_parser('info', help='Show model information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == 'image':
            if not os.path.exists(args.input):
                print(f"Error: Input image '{args.input}' not found.")
                sys.exit(1)
            detect_image_command(args)
        
        elif args.command == 'video':
            if not os.path.exists(args.input):
                print(f"Error: Input video '{args.input}' not found.")
                sys.exit(1)
            detect_video_command(args)
        
        elif args.command == 'webcam':
            detect_webcam_command(args)
        
        elif args.command == 'benchmark':
            if not os.path.exists(args.input):
                print(f"Error: Input path '{args.input}' not found.")
                sys.exit(1)
            benchmark_command(args)
        
        elif args.command == 'info':
            info_command(args)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()