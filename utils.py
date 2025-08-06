import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os
from pathlib import Path
import json
import pandas as pd

def visualize_detections_matplotlib(image_path: str, detections: List[Dict]) -> None:
    """
    Visualize detections using matplotlib (good for Jupyter notebooks)
    
    Args:
        image_path: Path to the image
        detections: List of detection dictionaries
    """
    # Load image (convert BGR to RGB for matplotlib)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    
    # Draw bounding boxes
    ax = plt.gca()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        width = x2 - x1
        height = y2 - y1
        
        # Use different colors for different classes
        color = colors[det['class_id'] % len(colors)]
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        plt.text(x1, y1-5, label, fontsize=10, color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.title(f"Object Detection Results - {len(detections)} objects detected")
    plt.tight_layout()
    plt.show()

def save_detections_json(detections: List[Dict], output_path: str) -> None:
    """
    Save detection results to JSON file
    
    Args:
        detections: List of detection dictionaries
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"Detections saved to {output_path}")

def load_detections_json(json_path: str) -> List[Dict]:
    """
    Load detection results from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of detection dictionaries
    """
    with open(json_path, 'r') as f:
        detections = json.load(f)
    return detections

def analyze_detections(detections: List[Dict]) -> Dict:
    """
    Analyze detection results and provide statistics
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    if not detections:
        return {"total_objects": 0}
    
    # Count objects by class
    class_counts = {}
    confidence_scores = []
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidence_scores.append(confidence)
    
    analysis = {
        "total_objects": len(detections),
        "unique_classes": len(class_counts),
        "class_distribution": class_counts,
        "confidence_stats": {
            "mean": np.mean(confidence_scores),
            "median": np.median(confidence_scores),
            "min": np.min(confidence_scores),
            "max": np.max(confidence_scores),
            "std": np.std(confidence_scores)
        }
    }
    
    return analysis

def plot_detection_statistics(detections: List[Dict]) -> None:
    """
    Plot statistics about detections
    
    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        print("No detections to analyze")
        return
    
    analysis = analyze_detections(detections)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution
    classes = list(analysis['class_distribution'].keys())
    counts = list(analysis['class_distribution'].values())
    
    ax1.bar(classes, counts)
    ax1.set_title('Object Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Confidence distribution
    confidences = [det['confidence'] for det in detections]
    ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Confidence Score Distribution')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
    ax2.legend()
    
    # Box plot of confidence by class
    class_confidences = {}
    for det in detections:
        class_name = det['class_name']
        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(det['confidence'])
    
    box_data = [class_confidences[cls] for cls in classes]
    ax3.boxplot(box_data, labels=classes)
    ax3.set_title('Confidence Distribution by Class')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Confidence Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    stats_data = [
        ['Total Objects', analysis['total_objects']],
        ['Unique Classes', analysis['unique_classes']],
        ['Mean Confidence', f"{analysis['confidence_stats']['mean']:.3f}"],
        ['Median Confidence', f"{analysis['confidence_stats']['median']:.3f}"],
        ['Min Confidence', f"{analysis['confidence_stats']['min']:.3f}"],
        ['Max Confidence', f"{analysis['confidence_stats']['max']:.3f}"],
        ['Std Confidence', f"{analysis['confidence_stats']['std']:.3f}"]
    ]
    
    table = ax4.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Detection Statistics Summary')
    
    plt.tight_layout()
    plt.show()

def filter_detections(detections: List[Dict], 
                     min_confidence: float = 0.0,
                     max_confidence: float = 1.0,
                     allowed_classes: List[str] = None,
                     min_area: float = 0.0) -> List[Dict]:
    """
    Filter detections based on various criteria
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold
        allowed_classes: List of allowed class names (None means all)
        min_area: Minimum bounding box area
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    
    for det in detections:
        # Check confidence
        if not (min_confidence <= det['confidence'] <= max_confidence):
            continue
        
        # Check class
        if allowed_classes and det['class_name'] not in allowed_classes:
            continue
        
        # Check area
        x1, y1, x2, y2 = det['bbox']
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        
        filtered.append(det)
    
    return filtered

def create_detection_summary(image_path: str, detections: List[Dict], output_dir: str = "output") -> str:
    """
    Create a comprehensive summary of detection results
    
    Args:
        image_path: Path to the analyzed image
        detections: List of detection dictionaries
        output_dir: Directory to save output files
        
    Returns:
        Path to the summary file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename
    base_name = Path(image_path).stem
    
    # Save detections as JSON
    json_path = f"{output_dir}/{base_name}_detections.json"
    save_detections_json(detections, json_path)
    
    # Create summary report
    analysis = analyze_detections(detections)
    
    summary_path = f"{output_dir}/{base_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Object Detection Summary Report\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Detection Summary:\n")
        f.write(f"- Total objects detected: {analysis['total_objects']}\n")
        f.write(f"- Unique classes: {analysis['unique_classes']}\n\n")
        
        if analysis['total_objects'] > 0:
            f.write(f"Class Distribution:\n")
            for class_name, count in analysis['class_distribution'].items():
                percentage = (count / analysis['total_objects']) * 100
                f.write(f"- {class_name}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nConfidence Statistics:\n")
            stats = analysis['confidence_stats']
            f.write(f"- Mean: {stats['mean']:.3f}\n")
            f.write(f"- Median: {stats['median']:.3f}\n")
            f.write(f"- Range: {stats['min']:.3f} - {stats['max']:.3f}\n")
            f.write(f"- Standard Deviation: {stats['std']:.3f}\n")
            
            f.write(f"\nDetailed Results:\n")
            for i, det in enumerate(detections, 1):
                f.write(f"{i}. {det['class_name']} (confidence: {det['confidence']:.3f}, "
                       f"bbox: {det['bbox']})\n")
    
    print(f"Summary report saved to {summary_path}")
    return summary_path

def benchmark_model(detector, test_images: List[str], num_runs: int = 3) -> Dict:
    """
    Benchmark detection performance on a set of test images
    
    Args:
        detector: ObjectDetector instance
        test_images: List of test image paths
        num_runs: Number of runs for averaging
        
    Returns:
        Benchmark results dictionary
    """
    import time
    
    results = {
        'total_images': len(test_images),
        'num_runs': num_runs,
        'times': [],
        'avg_time_per_image': 0,
        'total_detections': 0,
        'fps': 0
    }
    
    print(f"Benchmarking on {len(test_images)} images, {num_runs} runs each...")
    
    for run in range(num_runs):
        run_start = time.time()
        run_detections = 0
        
        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found, skipping...")
                continue
            
            start_time = time.time()
            detections = detector.detect_image(img_path, show_result=False)
            end_time = time.time()
            
            results['times'].append(end_time - start_time)
            run_detections += len(detections)
        
        run_end = time.time()
        print(f"Run {run + 1}: {run_end - run_start:.2f}s, {run_detections} detections")
        results['total_detections'] += run_detections
    
    # Calculate statistics
    if results['times']:
        results['avg_time_per_image'] = np.mean(results['times'])
        results['fps'] = 1.0 / results['avg_time_per_image']
        results['min_time'] = np.min(results['times'])
        results['max_time'] = np.max(results['times'])
        results['std_time'] = np.std(results['times'])
    
    results['avg_detections_per_image'] = results['total_detections'] / (len(test_images) * num_runs)
    
    print(f"\nBenchmark Results:")
    print(f"Average time per image: {results['avg_time_per_image']:.3f}s")
    print(f"Average FPS: {results['fps']:.1f}")
    print(f"Average detections per image: {results['avg_detections_per_image']:.1f}")
    
    return results