"""Look at the data and the preprocessing output:  python explore.py"""
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import build_preprocessor, prepare

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
df = load_dataset(n=30000)
print("=== DATASET PREVIEW (first 5 network flows) ===")
print(df.head(5).assign(timestamp=lambda d: d["timestamp"].dt.strftime("%H:%M:%S")).drop(columns="src_ip").to_string(index=False))
print(f"\nshape: {df.shape}   missing values: {int(df.isna().sum().sum())}")
print("\n=== CLASS BALANCE ===")
print(df["label"].value_counts().to_string())
print("\n=== PREPROCESSING ===")
prepared = prepare(df.head(1000))
print(f"raw columns      : {df.shape[1] - 1}   -> after feature engineering: {prepared.shape[1]}")
print("engineered cols  :", [c for c in prepared.columns if c not in df.columns])
pre = build_preprocessor(df).fit(df)
X = pre.transform(df)
print(f"model-ready matrix: {X.shape[0]:,} rows x {X.shape[1]} features (log-scaled, standardised, one-hot)")
