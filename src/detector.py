"""ThreatDetector: the deployable object - supervised classifier + anomaly detector in one."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import NORMAL, UNKNOWN
from .models import anomaly_scores


class ThreatDetector:
    def __init__(self, clf, iso, iso_threshold, raw_columns, defaults, classes):
        self.clf, self.iso, self.iso_threshold = clf, iso, iso_threshold
        self.raw_columns, self.defaults, self.classes = raw_columns, defaults, list(classes)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one verdict row per flow."""
        proba = self.clf.predict_proba(df)
        idx = proba.argmax(axis=1)
        pred = np.array(self.clf.classes_)[idx]
        score = anomaly_scores(self.iso, df)
        is_anom = score > self.iso_threshold
        conf = proba.max(axis=1)
        shaky = (pred != NORMAL) & (conf < 0.6) & is_anom          # attack label but low confidence + odd -> unknown
        verdict = np.where(shaky, UNKNOWN, np.where(pred != NORMAL, pred, np.where(is_anom, UNKNOWN, NORMAL)))
        p_normal = proba[:, list(self.clf.classes_).index(NORMAL)]
        return pd.DataFrame({"pred_label": pred, "confidence": proba.max(axis=1), "threat_score": 1 - p_normal,
                             "anomaly_score": score, "is_anomaly": is_anom, "verdict": verdict}, index=df.index)

    def predict_one(self, flow: dict) -> dict:
        """Score a single flow given as a (possibly partial) dict - used by the API."""
        row = {**self.defaults, **flow}
        out = self.predict(pd.DataFrame([row], columns=self.raw_columns)).iloc[0]
        return {"threat_detected": bool(out["verdict"] != NORMAL), "verdict": str(out["verdict"]),
                "attack_type": str(out["pred_label"]), "confidence": round(float(out["confidence"]), 4),
                "threat_score": round(float(out["threat_score"]), 4),
                "anomaly_score": round(float(out["anomaly_score"]), 4), "anomalous": bool(out["is_anomaly"])}
