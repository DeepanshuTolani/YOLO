import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "posture_model.pkl")

FEATURE_ORDER = [
    "neck_angle_deg",
    "back_angle_deg",
    "shoulder_slope_deg",
    "forward_head_norm",
]


@dataclass
class Prediction:
    label: str
    slouch_probability: float
    features: Dict[str, float]


class PostureModel:
    def __init__(self, model: Optional[BaseEstimator] = None) -> None:
        self.model = model

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        if not os.path.exists(MODEL_PATH):
            # Lazy training using the training script when missing
            from train_model import train_and_save
            os.makedirs(MODEL_DIR, exist_ok=True)
            train_and_save(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features: Dict[str, float]) -> Prediction:
        self.ensure_model()
        x = np.array([[features[k] for k in FEATURE_ORDER]], dtype=np.float32)
        proba = float(self.model.predict_proba(x)[0, 1])  # probability of slouching (class 1)
        label = "slouch" if proba >= 0.5 else "upright"
        return Prediction(label=label, slouch_probability=proba, features=features)

    def to_dict(self, pred: Prediction) -> Dict:
        return {
            "label": pred.label,
            "slouch_probability": round(pred.slouch_probability, 4),
            "features": {k: round(float(v), 3) for k, v in pred.features.items()},
        }