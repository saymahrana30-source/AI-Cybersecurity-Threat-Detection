"""Score a few hand-written flows with the trained model:  python predict.py"""
import joblib
import pandas as pd

SAMPLES = {
    "Normal web browsing": {"service": "https", "src_bytes": 600, "dst_bytes": 9000, "conn_per_min": 4, "duration": 1.2},
    "SYN flood": {"service": "http", "conn_per_min": 300, "syn_ratio": 0.85, "conn_state": "S0", "duration": 0.1,
                  "src_bytes": 120, "dst_bytes": 20, "packet_count": 35},
    "Port scan": {"unique_dst_ports": 150, "conn_per_min": 60, "syn_ratio": 0.5, "conn_state": "REJ", "duration": 0.05,
                  "src_bytes": 70, "dst_bytes": 0, "service": "other"},
    "SSH brute force": {"service": "ssh", "login_attempts": 40, "failed_logins": 38, "conn_per_min": 30,
                        "duration": 3.0, "src_bytes": 420, "dst_bytes": 330, "packet_count": 24},
    "Large upload to odd host": {"service": "other", "src_bytes": 40_000_000, "dst_bytes": 200, "duration": 90,
                                 "packet_count": 30000, "conn_per_min": 2},
    "Long-lived covert channel": {"service": "https", "duration": 650, "src_bytes": 420, "dst_bytes": 240,
                                  "packet_count": 35, "conn_per_min": 2},
}


def main():
    det = joblib.load("models/threat_detector.joblib")
    rows = []
    for name, flow in SAMPLES.items():
        r = det.predict_one(flow)
        rows.append({"flow": name, "verdict": r["verdict"], "threat?": "YES" if r["threat_detected"] else "no",
                     "confidence": r["confidence"], "anomaly_score": r["anomaly_score"]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
