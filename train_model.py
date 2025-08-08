import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "posture_model.pkl")


def synthesize_dataset(n: int = 4000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Features: [neck_angle_deg, back_angle_deg, shoulder_slope_deg, forward_head_norm]
    neck = rng.normal(loc=18.0, scale=10.0, size=n)  # typical upright 10-25
    back = rng.normal(loc=10.0, scale=7.0, size=n)   # typical upright 5-15
    slope = rng.normal(loc=5.0, scale=7.0, size=n)   # small slope
    fh = rng.normal(loc=0.25, scale=0.12, size=n)    # normalized head forward

    X = np.vstack([neck, back, slope, fh]).T

    # Heuristic labeling rule; add noise for realism
    logits = (
        0.12 * (neck - 22.0) +
        0.12 * (back - 15.0) +
        2.0 * (fh - 0.35) +
        0.03 * (slope - 8.0)
    )
    noise = rng.normal(0.0, 0.7, size=n)
    score = logits + noise
    y = (score > 0.0).astype(int)

    return X.astype(np.float32), y


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X, y)
    return pipe


def train_and_save(path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X, y = synthesize_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    model = train_model(Xtr, ytr)
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    print("[train_model] Training posture model...")
    train_and_save(MODEL_PATH)
    print(f"[train_model] Saved model to {MODEL_PATH}")