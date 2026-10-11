"""Model zoo: supervised classifiers (known attacks) + Isolation Forest (unknown anomalies)."""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import NORMAL, Config
from .preprocessing import build_preprocessor


def make_supervised(df_train, cfg: Config):
    pre = build_preprocessor(df_train)
    lr = Pipeline([("pre", clone(pre)), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    rf = Pipeline([("pre", clone(pre)), ("model", RandomForestClassifier(
        n_estimators=cfg.rf_trees, class_weight="balanced_subsample", min_samples_leaf=5,
        n_jobs=-1, random_state=cfg.seed))])
    return {"Logistic Regression": lr, "Random Forest": rf}


def fit_isolation_forest(df_train, cfg: Config):
    """Train on NORMAL traffic only - anything unlike normal gets a high anomaly score."""
    normal = df_train[df_train["label"] == NORMAL]
    iso = Pipeline([("pre", build_preprocessor(df_train, categorical=False)),
                    ("model", IsolationForest(n_estimators=cfg.iso_trees, random_state=cfg.seed, n_jobs=-1))])
    iso.fit(normal)
    scores = anomaly_scores(iso, normal)
    threshold = float(np.percentile(scores, 100 * (1 - cfg.iso_fpr)))   # allow ~3% false positives
    return iso, threshold


def anomaly_scores(iso, df) -> np.ndarray:
    """Higher = more anomalous."""
    return -iso.score_samples(df)
