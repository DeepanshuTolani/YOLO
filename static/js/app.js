let poseDetector = null;
let videoEl = document.getElementById('video');
let canvasEl = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusText = document.getElementById('statusText');
const probText = document.getElementById('probText');
const neckVal = document.getElementById('neckVal');
const backVal = document.getElementById('backVal');
const slopeVal = document.getElementById('slopeVal');
const fhVal = document.getElementById('fhVal');

let streaming = false;
let intervalId = null;

async function loadPose() {
  // Dynamically load MediaPipe Tasks Vision
  if (!window.FilesetResolver) {
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  const filesetResolver = await window.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );
  poseDetector = await window.PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  videoEl.srcObject = stream;
  await videoEl.play();
  return { width: videoEl.videoWidth || 640, height: videoEl.videoHeight || 480 };
}

function angleBetween(v1, v2) {
  const n1 = v1.map((x, i, a) => x / (Math.hypot(...a) + 1e-8));
  const n2 = v2.map((x, i, a) => x / (Math.hypot(...a) + 1e-8));
  const dot = Math.max(-1, Math.min(1, n1[0]*n2[0] + n1[1]*n2[1]));
  return (Math.acos(dot) * 180) / Math.PI;
}

function computeFeatures(landmarks, width, height) {
  if (!landmarks || landmarks.length === 0) return null;
  const lm = landmarks[0];
  const idx = window.PoseLandmarker.POSE_LANDMARKS;
  const toPx = (p) => [p.x * width, p.y * height];

  const left_sh = toPx(lm[idx.LEFT_SHOULDER]);
  const right_sh = toPx(lm[idx.RIGHT_SHOULDER]);
  const left_hip = toPx(lm[idx.LEFT_HIP]);
  const right_hip = toPx(lm[idx.RIGHT_HIP]);
  const nose = toPx(lm[idx.NOSE]);

  const shoulder_mid = [(left_sh[0] + right_sh[0]) / 2, (left_sh[1] + right_sh[1]) / 2];
  const hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2];

  const vertical = [0, -1];
  const back_vec = [shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]];
  const back_angle = angleBetween(back_vec, vertical);

  const neck_vec = [nose[0] - shoulder_mid[0], nose[1] - shoulder_mid[1]];
  const neck_angle = angleBetween(neck_vec, vertical);

  const horiz = [1, 0];
  const shoulder_vec = [right_sh[0] - left_sh[0], right_sh[1] - left_sh[1]];
  const shoulder_slope = angleBetween(shoulder_vec, horiz);

  const shoulder_width = Math.hypot(...shoulder_vec) + 1e-8;
  const forward_head_norm = Math.abs(nose[0] - shoulder_mid[0]) / shoulder_width;

  return {
    neck_angle_deg: neck_angle,
    back_angle_deg: back_angle,
    shoulder_slope_deg: shoulder_slope,
    forward_head_norm: forward_head_norm
  };
}

async function detectOnce() {
  if (!poseDetector) return;
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if (!w || !h) return;
  if (!canvasEl.width) { canvasEl.width = w; canvasEl.height = h; }

  const res = await poseDetector.detectForVideo(videoEl, performance.now());
  const features = computeFeatures(res.landmarks, w, h);
  if (!features) {
    statusText.textContent = 'NO PERSON';
    probText.textContent = '-';
    neckVal.textContent = backVal.textContent = slopeVal.textContent = fhVal.textContent = '-';
    return;
  }
  try {
    const resp = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    const json = await resp.json();
    if (!json.ok) { statusText.textContent = 'Error'; return; }
    const label = json.label || 'unknown';
    statusText.textContent = label.toUpperCase();
    statusText.className = label === 'slouch' ? 'bad' : (label === 'upright' ? 'good' : '');

    probText.textContent = (json.slouch_probability !== undefined) ? (json.slouch_probability * 100).toFixed(1) + '%' : '-';
    const f = json.features || {};
    neckVal.textContent = f.neck_angle_deg !== undefined ? f.neck_angle_deg.toFixed(1) : '-';
    backVal.textContent = f.back_angle_deg !== undefined ? f.back_angle_deg.toFixed(1) : '-';
    slopeVal.textContent = f.shoulder_slope_deg !== undefined ? f.shoulder_slope_deg.toFixed(1) : '-';
    fhVal.textContent = f.forward_head_norm !== undefined ? f.forward_head_norm.toFixed(2) : '-';
  } catch (e) {
    statusText.textContent = 'Network error';
  }
}

startBtn.addEventListener('click', async () => {
  if (streaming) return;
  startBtn.disabled = true;
  statusText.textContent = 'Starting...';
  try {
    await loadPose();
    await setupCamera();
    streaming = true;
    stopBtn.disabled = false;
    statusText.textContent = 'Running';
    intervalId = setInterval(detectOnce, 300);
  } catch (e) {
    statusText.textContent = 'Init error';
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', () => {
  if (!streaming) return;
  streaming = false;
  stopBtn.disabled = true;
  startBtn.disabled = false;
  statusText.textContent = 'Stopped';
  if (intervalId) clearInterval(intervalId);
  const stream = videoEl.srcObject;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    videoEl.srcObject = null;
  }
});