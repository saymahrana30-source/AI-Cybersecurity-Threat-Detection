"""REST API for real-time threat scoring.

    python -m src.api                      # http://127.0.0.1:5000
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
         -d '{"unique_dst_ports": 120, "conn_per_min": 60, "syn_ratio": 0.5, "conn_state": "REJ"}'

Only send the fields you know - the rest default to a typical normal flow.
"""
from __future__ import annotations
import os
import joblib
from flask import Flask, jsonify, request

NUMERIC = ["duration", "src_bytes", "dst_bytes", "packet_count", "syn_ratio", "unique_dst_ports",
           "conn_per_min", "login_attempts", "failed_logins"]
CATEGORICAL = ["protocol", "service", "conn_state"]
ALLOWED = set(NUMERIC + CATEGORICAL)


def _validate(flow):
    if not isinstance(flow, dict) or not flow:
        return "body must be a non-empty JSON object"
    unknown = sorted(set(flow) - ALLOWED)
    if unknown:
        return f"unknown field(s) {unknown}; allowed: {sorted(ALLOWED)}"
    for k in NUMERIC:
        if k in flow and (isinstance(flow[k], bool) or not isinstance(flow[k], (int, float)) or flow[k] < 0):
            return f"'{k}' must be a non-negative number"
    for k in CATEGORICAL:
        if k in flow and not isinstance(flow[k], str):
            return f"'{k}' must be a string"
    return None


def create_app(model_path="models/threat_detector.joblib", detector=None):
    app = Flask(__name__)
    det = detector or joblib.load(model_path)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "classes": det.classes})

    @app.post("/predict")
    def predict():
        flow = request.get_json(silent=True)
        err = _validate(flow)
        if err:
            return jsonify({"error": err}), 400
        return jsonify(det.predict_one(flow))

    @app.post("/predict_batch")
    def predict_batch():
        flows = request.get_json(silent=True)
        if not isinstance(flows, list) or not flows or len(flows) > 1000:
            return jsonify({"error": "body must be a JSON list of 1-1000 flow objects"}), 400
        for f in flows:
            err = _validate(f)
            if err:
                return jsonify({"error": err}), 400
        return jsonify([det.predict_one(f) for f in flows])

    return app


if __name__ == "__main__":
    if not os.path.exists("models/threat_detector.joblib"):
        raise SystemExit("Train first:  python main.py")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
