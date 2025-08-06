# Object Detection Pipeline with YOLOv8

A comprehensive, production-ready object detection system built with YOLOv8, supporting images, videos, and real-time webcam detection.

## Features

🎯 **Multiple Input Types**
- Single image detection
- Video file processing  
- Real-time webcam detection
- Batch processing for multiple images

🚀 **Performance Options**
- Multiple YOLOv8 model sizes (nano to extra-large)
- GPU acceleration support
- Configurable confidence and IoU thresholds
- Real-time FPS monitoring

📊 **Advanced Analysis**
- Detection statistics and visualizations
- Confidence score analysis
- Class distribution plots
- Filtering and post-processing tools

💾 **Export & Integration**
- JSON export of detection results
- Comprehensive summary reports
- Command-line interface
- Jupyter notebook for interactive exploration

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd object-detection-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from object_detector import ObjectDetector

# Initialize detector
detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.25)

# Detect objects in an image
detections = detector.detect_image('input.jpg', save_path='output.jpg')

# Print results
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

### Command Line Interface

```bash
# Detect objects in an image
python main.py image input.jpg --output output.jpg --confidence 0.3

# Process a video
python main.py video input.mp4 --output output.mp4

# Real-time webcam detection
python main.py webcam

# Benchmark performance
python main.py benchmark ./test_images/ --runs 3

# Show model information
python main.py info --model yolov8s.pt
```

## File Structure

```
object-detection-pipeline/
├── object_detector.py          # Main detection class
├── utils.py                   # Utility functions and analysis tools
├── main.py                    # Command-line interface
├── example_usage.py           # Example scripts and demonstrations
├── object_detection_demo.ipynb # Interactive Jupyter notebook
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Core Components

### ObjectDetector Class

The main class providing object detection functionality:

```python
detector = ObjectDetector(
    model_path='yolov8n.pt',    # Model size: n/s/m/l/x
    conf_threshold=0.25,         # Confidence threshold
    iou_threshold=0.45          # IoU threshold for NMS
)

# Methods
detector.detect_image(image_path, save_path, show_result)
detector.detect_video(video_path, save_path, show_result)
detector.detect_webcam(camera_id)
detector.get_model_info()
```

### Utility Functions

Located in `utils.py`, providing advanced analysis capabilities:

- `analyze_detections()` - Statistical analysis of results
- `visualize_detections_matplotlib()` - Matplotlib-based visualization
- `plot_detection_statistics()` - Comprehensive statistics plots
- `filter_detections()` - Filter by confidence, class, or area
- `save_detections_json()` - Export results to JSON
- `create_detection_summary()` - Generate comprehensive reports
- `benchmark_model()` - Performance benchmarking

## Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | ~6MB | Fastest | Good | Real-time applications |
| YOLOv8s | ~22MB | Fast | Better | Balanced speed/accuracy |
| YOLOv8m | ~52MB | Medium | Very Good | High accuracy needs |
| YOLOv8l | ~87MB | Slow | Excellent | Maximum accuracy |
| YOLOv8x | ~136MB | Slowest | Best | Research/offline processing |

## Examples

### Basic Image Detection

```python
from object_detector import ObjectDetector
from utils import analyze_detections, visualize_detections_matplotlib

# Initialize detector
detector = ObjectDetector('yolov8n.pt')

# Detect objects
detections = detector.detect_image('street_scene.jpg', 'output.jpg')

# Analyze results
analysis = analyze_detections(detections)
print(f"Found {analysis['total_objects']} objects")
print(f"Average confidence: {analysis['confidence_stats']['mean']:.2f}")

# Visualize with matplotlib
visualize_detections_matplotlib('street_scene.jpg', detections)
```

### Video Processing

```python
# Process entire video
detector.detect_video('input_video.mp4', 'output_video.mp4')

# Real-time webcam (press 'q' to quit)
detector.detect_webcam(camera_id=0)
```

### Filtering Detections

```python
from utils import filter_detections

# Filter by confidence
high_conf = filter_detections(detections, min_confidence=0.7)

# Filter by specific classes
people_cars = filter_detections(
    detections, 
    allowed_classes=['person', 'car']
)

# Filter by object size
large_objects = filter_detections(detections, min_area=5000)
```

### Batch Processing

```python
import os
from pathlib import Path

# Process all images in a directory
image_dir = Path("test_images")
results = {}

for img_path in image_dir.glob("*.jpg"):
    detections = detector.detect_image(
        str(img_path), 
        f"output/{img_path.stem}_detected.jpg",
        show_result=False
    )
    results[str(img_path)] = detections

print(f"Processed {len(results)} images")
```

## Interactive Notebook

Launch the Jupyter notebook for interactive exploration:

```bash
jupyter notebook object_detection_demo.ipynb
```

The notebook includes:
- Step-by-step tutorials
- Visualization examples
- Model comparison
- Performance analysis
- Interactive parameter tuning

## Command Line Options

### Global Options
- `--model`: YOLO model path (default: yolov8n.pt)
- `--confidence`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--verbose`: Enable verbose logging

### Image Detection
```bash
python main.py image INPUT_IMAGE [options]
```
Options:
- `--output, -o`: Output image path
- `--no-display`: Don't show result window
- `--save-json`: Save detections to JSON file
- `--save-summary`: Save summary report to directory

### Video Detection
```bash
python main.py video INPUT_VIDEO [options]
```
Options:
- `--output, -o`: Output video path
- `--no-display`: Don't show real-time preview

### Webcam Detection
```bash
python main.py webcam [options]
```
Options:
- `--camera`: Camera device ID (default: 0)

### Benchmarking
```bash
python main.py benchmark INPUT_PATH [options]
```
Options:
- `--runs`: Number of benchmark runs (default: 3)
- `--output, -o`: Save results to JSON file

## Performance Tips

1. **Model Selection**: Use YOLOv8n for real-time applications, larger models for accuracy
2. **GPU Acceleration**: Ensure CUDA is available for significant speedup
3. **Confidence Threshold**: Higher values reduce false positives but may miss objects
4. **Input Resolution**: Smaller images process faster but may reduce accuracy
5. **Batch Processing**: Process multiple images together for efficiency

## Supported Classes

YOLOv8 is trained on the COCO dataset and can detect 80 object classes including:

**People & Animals**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Objects**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture**: chair, couch, potted plant, bed, dining table, toilet

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone

And many more! Use `python main.py info` to see the complete list.

## Output Formats

### Detection Dictionary
```python
{
    'class_id': 0,
    'class_name': 'person',
    'confidence': 0.85,
    'bbox': [x1, y1, x2, y2]  # Bounding box coordinates
}
```

### JSON Export
```json
[
  {
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.8543,
    "bbox": [245, 123, 456, 678]
  }
]
```

### Summary Report
- Detection statistics
- Class distribution
- Confidence analysis
- Detailed object list
- Processing metadata

## Troubleshooting

### Common Issues

**"No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**"Could not load image"**
- Check file path and format
- Supported formats: JPG, PNG, BMP, TIFF

**"CUDA out of memory"**
- Use a smaller model (yolov8n instead of yolov8x)
- Reduce input image size
- Lower batch size for video processing

**"No camera found"**
- Check camera permissions
- Try different camera IDs (0, 1, 2...)
- Ensure camera is not used by another application

### Performance Issues

- **Slow detection**: Use GPU acceleration or smaller model
- **High memory usage**: Process images individually instead of batching
- **Poor accuracy**: Use larger model or adjust confidence threshold

## Advanced Usage

### Custom Training

You can train YOLOv8 on custom datasets:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(data='custom_dataset.yaml', epochs=100)

# Use custom model
detector = ObjectDetector('runs/detect/train/weights/best.pt')
```

### Integration with Other Systems

The detection pipeline can be easily integrated:

```python
# Web API integration
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = ObjectDetector()

@app.route('/detect', methods=['POST'])
def detect_api():
    # Process uploaded image
    file = request.files['image']
    detections = detector.detect_image(file.filename, show_result=False)
    return jsonify(detections)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ultralytics** for the YOLOv8 implementation
- **OpenCV** for computer vision utilities
- **PyTorch** for deep learning framework
- **COCO Dataset** for training data

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description
4. Include system information and error messages

---

**Happy Detecting! 🎯**