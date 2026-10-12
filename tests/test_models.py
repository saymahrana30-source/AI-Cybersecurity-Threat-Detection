import dataclasses
import numpy as np
import pandas as pd
from src.config import NORMAL, NOVEL_ATTACK, Config
from src.pipeline import train_and_evaluate
from src.simulator import gen_attack, simulate_dataset

CFG = dataclasses.replace(Config(), rf_trees=60, iso_trees=80)
_cache = {}


def _trained():
    if not _cache:
        df = simulate_dataset(9000, seed=3)
        _cache["out"] = train_and_evaluate(df, CFG, cv=False, log=lambda *_: None)
    return _cache["out"]


def test_random_forest_detects_most_attacks_with_few_false_alarms():
    _, res, _ = _trained()
    rf = res["Random Forest"]
    assert rf["f1"] > 0.85 and rf["precision"] > 0.95 and rf["false_positive_rate"] < 0.01


def test_random_forest_beats_logistic_regression_on_precision():
    _, res, _ = _trained()
    assert res["Random Forest"]["precision"] > res["Logistic Regression"]["precision"]


def test_unseen_attack_is_mostly_flagged_by_the_anomaly_detector():
    _, res, _ = _trained()
    z = res["zero_day_StealthC2"]
    assert z["flagged_by_isolation_forest"] > 0.5
    assert z["full_detector_missed"] < 0.4


def test_detector_returns_one_verdict_per_flow_and_single_flow_api_works():
    det, _, art = _trained()
    out = det.predict(art["X_test"].head(50))
    assert len(out) == 50 and {"verdict", "confidence", "anomaly_score"} <= set(out.columns)
    r = det.predict_one({"unique_dst_ports": 200, "conn_per_min": 60, "syn_ratio": 0.5, "conn_state": "REJ",
                         "src_bytes": 70, "dst_bytes": 0, "duration": 0.05})
    assert r["threat_detected"] is True


def test_detector_accepts_normal_defaults():
    det, _, _ = _trained()
    assert det.predict_one({})["verdict"] == NORMAL
