# Slouching Detection System (Flask + ML)

An end-to-end slouching posture detection demo using a lightweight ML classifier and pose keypoints from MediaPipe. Includes a Flask backend for inference and a web UI that streams webcam frames for live feedback.

## Features
- Live webcam detection in the browser
- Flask API endpoint for inference (`/detect`)
- MediaPipe Pose for keypoint extraction
- Simple ML pipeline (scikit-learn) trained on synthetic features
- Clear status with confidence score and angles

## Quick Start
1. Python 3.10+ recommended. Install system deps:
   - Linux: `sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6`
2. Create venv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Train or auto-generate the ML model (optional):
   ```bash
   python train_model.py
   ```
   If you skip this, the server will auto-create a model at first run.
4. Run the server:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## Production (Gunicorn)
```bash
export PORT=8080
export HOST=0.0.0.0
# Ensure model exists
python train_model.py
# Start
gunicorn -w 2 -b ${HOST}:${PORT} app:app
```

## How It Works
- The browser captures webcam frames and posts them periodically to `/detect` as JPEG base64.
- The server decodes the image, runs MediaPipe Pose to extract landmarks, computes geometric features (neck angle, back angle, forward-head offset, shoulder slope), then feeds them into a small scikit-learn pipeline.
- The classifier outputs a probability of slouching and a label.

## Files
- `app.py`: Flask app and inference endpoint
- `posture/feature_extractor.py`: Pose extraction and feature engineering
- `posture/model.py`: Model load/predict utilities
- `train_model.py`: Generates synthetic data and trains a logistic regression model
- `templates/index.html`: Web UI
- `static/js/app.js`: Webcam capture and API integration
- `static/css/style.css`: Basic styling

## Notes
- This is a demo. Real-world calibration requires a diverse dataset and per-user baselines.
- Lighting and camera angle affect accuracy. Prefer a front-facing webcam at shoulder height.

## Acknowledgements
- MediaPipe Pose by Google
- scikit-learn for the ML pipeline