import numpy as np
from src.preprocessing import build_preprocessor, engineer_features, prepare, signed_log1p
from src.simulator import simulate_dataset


def test_engineered_features_are_added():
    df = engineer_features(simulate_dataset(300, seed=1))
    assert {"bytes_per_sec", "avg_packet_size", "byte_ratio", "failed_login_ratio"} <= set(df.columns)


def test_prepare_drops_identifiers_and_labels():
    out = prepare(simulate_dataset(300, seed=1))
    assert not ({"timestamp", "src_ip", "label"} & set(out.columns))


def test_signed_log_handles_negatives_and_zero():
    out = signed_log1p(np.array([-1.0, 0.0, 10.0]))
    assert np.isfinite(out).all() and out[0] < 0 < out[2]


def test_preprocessor_outputs_finite_matrix_and_tolerates_unseen_category():
    df = simulate_dataset(1000, seed=1)
    pre = build_preprocessor(df).fit(df)
    X = pre.transform(df)
    assert X.shape[0] == len(df) and np.isfinite(X).all()
    odd = df.head(3).copy(); odd["service"] = "gopher"; odd.loc[odd.index[0], "duration"] = np.nan
    assert np.isfinite(pre.transform(odd)).all()
