import os
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from posture.model import PostureModel

app = Flask(__name__)

model = PostureModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}

    # Expect features directly from the client (computed via MediaPipe Tasks JS)
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify({"ok": False, "error": "missing features"}), 400

    required = {"neck_angle_deg", "back_angle_deg", "shoulder_slope_deg", "forward_head_norm"}
    if not required.issubset(features.keys()):
        return jsonify({"ok": False, "error": "incomplete features"}), 400

    pred = model.predict(features)
    return jsonify({"ok": True, **model.to_dict(pred)})


@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)