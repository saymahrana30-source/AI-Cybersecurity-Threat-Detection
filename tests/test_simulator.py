import numpy as np
from src.config import NORMAL, NOVEL_ATTACK
from src.simulator import COLUMNS, simulate_dataset, simulate_stream


def test_dataset_has_expected_columns_and_no_missing_values():
    df = simulate_dataset(2000, seed=1)
    assert list(df.columns) == COLUMNS and df.isna().sum().sum() == 0


def test_dataset_is_imbalanced_and_excludes_novel_attack():
    df = simulate_dataset(5000, seed=1)
    share = (df["label"] != NORMAL).mean()
    assert 0.10 < share < 0.14
    assert NOVEL_ATTACK not in set(df["label"])


def test_simulation_is_reproducible_with_a_seed():
    a, b = simulate_dataset(500, seed=7), simulate_dataset(500, seed=7)
    assert a.drop(columns="src_ip").equals(b.drop(columns="src_ip"))


def test_attacks_look_different_from_normal_on_average():
    df = simulate_dataset(8000, seed=2)
    g = df.groupby("label")
    assert g["conn_per_min"].median()["DoS"] > 10 * g["conn_per_min"].median()[NORMAL]
    assert g["unique_dst_ports"].median()["PortScan"] > 3 * g["unique_dst_ports"].median()[NORMAL]
    assert g["failed_logins"].mean()["BruteForce"] > 10 * g["failed_logins"].mean()[NORMAL]
    assert g["src_bytes"].median()["DataExfiltration"] > 100 * g["src_bytes"].median()[NORMAL]


def test_stream_contains_all_campaigns_in_time_order():
    flows, truth = simulate_stream(1, 60)
    assert flows["timestamp"].is_monotonic_increasing and len(truth) == 6
    assert {t["src_ip"] for t in truth} <= set(flows["src_ip"])
