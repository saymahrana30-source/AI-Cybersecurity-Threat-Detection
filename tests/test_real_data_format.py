"""The pipeline must also work on CICIDS-style CSVs: different columns, 'BENIGN' labels, inf/NaN values."""
import dataclasses
import numpy as np
import pandas as pd
from src.config import NORMAL, Config
from src.data_loader import load_dataset
from src.pipeline import train_and_evaluate


def _fake_cicids(path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    attack = rng.random(n) < 0.2
    df = pd.DataFrame({" Flow Duration": rng.exponential(1e5, n) * np.where(attack, 0.05, 1),
                       " Total Fwd Packets": rng.poisson(np.where(attack, 40, 8)),
                       "Flow Bytes/s": rng.exponential(2000, n) * np.where(attack, 50, 1),
                       " Destination Port": rng.choice([22, 80, 443, 8080], n),
                       " Label": np.where(attack, "DDoS", "BENIGN")})
    df.loc[3, "Flow Bytes/s"] = np.inf
    df.loc[5, " Total Fwd Packets"] = np.nan
    df.to_csv(path, index=False)


def test_loader_maps_labels_and_cleans_infinities(tmp_path):
    p = tmp_path / "cic.csv"; _fake_cicids(p)
    df = load_dataset(str(p), label_col="Label")
    assert set(df["label"]) == {NORMAL, "DDoS"} and not np.isinf(df.select_dtypes("number")).any().any()


def test_pipeline_trains_on_a_different_feature_set(tmp_path):
    p = tmp_path / "cic.csv"; _fake_cicids(p)
    df = load_dataset(str(p), label_col="Label")
    cfg = dataclasses.replace(Config(), rf_trees=40, iso_trees=40)
    det, res, _ = train_and_evaluate(df, cfg, cv=False, log=lambda *_: None)
    assert res["Random Forest"]["f1"] > 0.9
