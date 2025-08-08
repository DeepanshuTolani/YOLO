import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


PoseLandmark = mp.solutions.pose.PoseLandmark


class PoseFeatureExtractor:
    def __init__(self) -> None:
        self._mp_pose = mp.solutions.pose
        # Enable segmentation for better landmark stability in some cases
        self._pose = self._mp_pose.Pose(static_image_mode=False,
                                        model_complexity=1,
                                        enable_segmentation=False,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

    def close(self) -> None:
        self._pose.close()

    @staticmethod
    def _to_numpy_landmarks(results) -> Optional[np.ndarray]:
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return arr

    def extract_landmarks(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        return self._to_numpy_landmarks(results)

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        v1n = v1 / (np.linalg.norm(v1) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2) + 1e-8)
        dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
        return math.degrees(math.acos(dot))

    def compute_features(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Optional[Dict[str, float]]:
        if landmarks is None or landmarks.shape[0] < len(PoseLandmark):
            return None
        h, w = image_shape[:2]
        # Convert normalized [0,1] to pixel coords for scale-aware features
        pts = landmarks.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h

        def p(idx: PoseLandmark) -> np.ndarray:
            return pts[idx.value, :2]

        left_sh = p(PoseLandmark.LEFT_SHOULDER)
        right_sh = p(PoseLandmark.RIGHT_SHOULDER)
        left_hip = p(PoseLandmark.LEFT_HIP)
        right_hip = p(PoseLandmark.RIGHT_HIP)
        nose = p(PoseLandmark.NOSE)
        left_ear = p(PoseLandmark.LEFT_EAR)
        right_ear = p(PoseLandmark.RIGHT_EAR)

        shoulder_mid = (left_sh + right_sh) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0

        # Vertical direction in image coordinates is up = negative y
        vertical_vec = np.array([0.0, -1.0])

        # Back angle: line hip_mid -> shoulder_mid vs vertical
        back_vec = shoulder_mid - hip_mid
        back_angle_deg = self._angle_between(back_vec, vertical_vec)

        # Neck/head proxy: shoulder_mid -> nose vs vertical
        neck_vec = nose - shoulder_mid
        neck_angle_deg = self._angle_between(neck_vec, vertical_vec)

        # Shoulder slope: angle of right_sh - left_sh vs horizontal
        horiz_vec = np.array([1.0, 0.0])
        shoulder_vec = right_sh - left_sh
        shoulder_slope_deg = self._angle_between(shoulder_vec, horiz_vec)

        # Forward head: horizontal offset of nose from shoulder_mid line normalized by shoulder width
        shoulder_width = np.linalg.norm(shoulder_vec) + 1e-8
        forward_head_norm = abs(nose[0] - shoulder_mid[0]) / shoulder_width

        features = {
            "neck_angle_deg": float(neck_angle_deg),
            "back_angle_deg": float(back_angle_deg),
            "shoulder_slope_deg": float(shoulder_slope_deg),
            "forward_head_norm": float(forward_head_norm),
        }
        return features


def draw_overlays(image_bgr: np.ndarray, landmarks: Optional[np.ndarray]) -> np.ndarray:
    annotated = image_bgr.copy()
    if landmarks is None:
        return annotated
    h, w = annotated.shape[:2]
    pts = landmarks.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Re-run a light drawing by constructing a fake landmark object
    # This avoids re-invoking the model just for drawing
    class Landmarks:
        pass
    lm_list = []
    for i in range(pts.shape[0]):
        x, y = float(pts[i, 0]) / w, float(pts[i, 1]) / h
        lm = mp.framework.formats.landmark_pb2.NormalizedLandmark(x=x, y=y, z=0.0, visibility=0.99)
        lm_list.append(lm)
    landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark=lm_list)

    mp_drawing.draw_landmarks(
        annotated,
        landmark_list,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2, circle_radius=2),
    )
    return annotated