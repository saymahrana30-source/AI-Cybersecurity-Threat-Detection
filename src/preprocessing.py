"""Cleaning + feature engineering + scaling, packaged as an sklearn Pipeline.

Because preprocessing lives INSIDE the saved model, predictions at deployment time get exactly the same
treatment as training data (a classic bug in hand-rolled scripts)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import ID_COLUMNS


def signed_log1p(x):
    """Compress heavy-tailed counts/bytes while keeping sign (some real datasets contain -1)."""
    return np.sign(x) * np.log1p(np.abs(x))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behaviour features that make attacks easier to separate (only when source columns exist)."""
    df = df.copy()
    cols = set(df.columns)
    if {"src_bytes", "dst_bytes"} <= cols:
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
        if "duration" in cols:
            df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / df["duration"].clip(lower=1e-3)
        if "packet_count" in cols:
            df["avg_packet_size"] = (df["src_bytes"] + df["dst_bytes"]) / df["packet_count"].clip(lower=1)
    if {"failed_logins", "login_attempts"} <= cols:
        df["failed_login_ratio"] = df["failed_logins"] / df["login_attempts"].clip(lower=1)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features, drop identifiers/labels, clean infinities."""
    df = engineer_features(df)
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    return df.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(sample_df: pd.DataFrame, categorical: bool = True) -> Pipeline:
    """Create the (unfitted) preprocessing pipeline from the columns found in `sample_df`.
    categorical=False keeps only numeric behaviour features (better for Isolation Forest)."""
    prepared = prepare(sample_df.head(50))
    num = prepared.select_dtypes(include="number").columns.tolist()
    cat = [c for c in prepared.columns if c not in num] if categorical else []
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    parts = [("num", numeric, num)] + ([("cat", cat_pipe, cat)] if cat else [])
    ct = ColumnTransformer(parts)
    return Pipeline([("prepare", FunctionTransformer(prepare, validate=False)), ("columns", ct)])
