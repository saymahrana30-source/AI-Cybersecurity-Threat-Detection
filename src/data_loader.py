"""Load the simulated dataset - or a real one (CICIDS2017 / NSL-KDD / UNSW-NB15 CSV)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .config import NORMAL
from .simulator import simulate_dataset

BENIGN_WORDS = {"benign", "normal", "0", "normal."}


def load_dataset(path=None, label_col="label", n=30000, seed=42, max_rows=150000):
    """Return a DataFrame with a 'label' column ('Normal' or an attack name)."""
    if path is None:
        return simulate_dataset(n, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if label_col not in df.columns:
        raise KeyError(f"label column '{label_col}' not in {list(df.columns)[:10]}...")
    df = df.rename(columns={label_col: "label"})
    df = df.replace([np.inf, -np.inf], np.nan)
    lab = df["label"].astype(str).str.strip()
    df["label"] = np.where(lab.str.lower().isin(BENIGN_WORDS), NORMAL, lab.str.replace(r"\W+", "", regex=True))
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df
