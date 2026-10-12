"""Train + evaluate everything and build the deployable ThreatDetector."""
from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from .config import NORMAL, NOVEL_ATTACK, UNKNOWN, Config
from .detector import ThreatDetector
from .evaluate import binary_metrics
from .models import anomaly_scores, fit_isolation_forest, make_supervised
from .simulator import COLUMNS as SIM_COLUMNS, gen_attack


def train_and_evaluate(df: pd.DataFrame, cfg: Config = Config(), cv: bool = True, log=print):
    """Returns (detector, results dict, artefacts for plotting)."""
    train, test = train_test_split(df, test_size=cfg.test_size, stratify=df["label"], random_state=cfg.seed)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    X_train, y_train = train.drop(columns="label"), train["label"]
    X_test, y_test = test.drop(columns="label"), test["label"]
    log(f"  train={len(train):,}  test={len(test):,}  attack share={100 * (df['label'] != NORMAL).mean():.1f}%")

    results, models = {}, make_supervised(train, cfg)
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)
        score = 1 - proba[:, list(m.classes_).index(NORMAL)]
        results[name] = binary_metrics(y_test, pred != NORMAL, score)
        log(f"  {name:<20} acc={results[name]['accuracy']:.3f} prec={results[name]['precision']:.3f} "
            f"recall={results[name]['recall']:.3f} f1={results[name]['f1']:.3f} auc={results[name]['roc_auc']:.3f}")
    rf = models["Random Forest"]

    iso, thr = fit_isolation_forest(train, cfg)
    sc = anomaly_scores(iso, X_test)
    results["Isolation Forest (anomaly)"] = binary_metrics(y_test, sc > thr, sc)
    r = results["Isolation Forest (anomaly)"]
    log(f"  {'Isolation Forest':<20} acc={r['accuracy']:.3f} prec={r['precision']:.3f} recall={r['recall']:.3f} "
        f"f1={r['f1']:.3f} auc={r['roc_auc']:.3f}   (trained on normal traffic only)")

    classes = sorted(set(df["label"]))
    defaults = X_train[X_train.index.isin(y_train[y_train == NORMAL].index)].select_dtypes("number").median().round(3).to_dict()
    for c in X_train.select_dtypes(exclude="number").columns:
        defaults[c] = X_train[c].mode().iloc[0]
    detector = ThreatDetector(rf, iso, thr, list(X_train.columns), defaults, classes)
    # zero-day test: an attack type neither model has ever seen (only possible with the simulator's columns)
    nv_scores = None
    if set(SIM_COLUMNS) - {"label"} <= set(df.columns):
        rng = np.random.default_rng(cfg.seed + 1)
        novel = gen_attack(rng, NOVEL_ATTACK, 800)
        novel.insert(0, "timestamp", pd.Timestamp("2026-01-15")); novel.insert(1, "src_ip", "203.0.113.66")
        nv_X = novel.drop(columns="label")
        nv_sup = float((rf.predict(nv_X) != NORMAL).mean())
        nv_scores = anomaly_scores(iso, nv_X)
        nv_iso = float((nv_scores > thr).mean())
        nv_verd = detector.predict(nv_X)["verdict"]
        nv_conf = float(detector.predict(nv_X)["confidence"].mean())
        results["zero_day_StealthC2"] = {
            "flagged_by_random_forest": round(nv_sup, 3), "flagged_by_isolation_forest": round(nv_iso, 3),
            "full_detector_unknown_anomaly": round(float((nv_verd == UNKNOWN).mean()), 3),
            "full_detector_missed": round(float((nv_verd == NORMAL).mean()), 3)}
        log(f"  unseen attack (StealthC2): Random Forest flags {nv_sup:.0%} (but with only {nv_conf:.2f} avg confidence, "
            f"as wrong attack types); Isolation Forest flags {nv_iso:.0%}")
        log(f"  full detector verdict on it: {results['zero_day_StealthC2']['full_detector_unknown_anomaly']:.0%} UnknownAnomaly, "
            f"{results['zero_day_StealthC2']['full_detector_missed']:.0%} missed")

    results["per_class_report"] = classification_report(y_test, rf.predict(X_test), output_dict=True, zero_division=0)
    if cv:
        f1 = cross_val_score(rf, X_train, (y_train != NORMAL).astype(int), cv=3, scoring="f1", n_jobs=1)
        results["cv_f1_random_forest"] = {"mean": round(float(f1.mean()), 4), "std": round(float(f1.std()), 4)}
        log(f"  3-fold CV F1 (Random Forest): {f1.mean():.3f} +/- {f1.std():.3f}")

    art = {"X_test": X_test, "y_test": y_test, "rf": rf, "models": models, "iso": iso, "thr": thr,
           "iso_test_scores": sc, "novel_scores": nv_scores, "train": train}
    return detector, results, art


def save_detector(detector, results, model_dir="models", out_dir="outputs"):
    os.makedirs(model_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    joblib.dump(detector, os.path.join(model_dir, "threat_detector.joblib"), compress=3)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    rep = results["per_class_report"]
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"{'class':<18}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}\n")
        for k, v in rep.items():
            if isinstance(v, dict):
                f.write(f"{k:<18}{v['precision']:>10.3f}{v['recall']:>10.3f}{v['f1-score']:>10.3f}{int(v['support']):>10}\n")
