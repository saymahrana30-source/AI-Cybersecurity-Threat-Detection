"""Central configuration - every tunable number lives here."""
from dataclasses import dataclass

NORMAL = "Normal"
KNOWN_ATTACKS = ["DoS", "PortScan", "BruteForce", "DataExfiltration"]
NOVEL_ATTACK = "StealthC2"            # never shown to the supervised model (zero-day demo)
UNKNOWN = "UnknownAnomaly"               # verdict for "looks odd, but not a known attack"
ID_COLUMNS = ["timestamp", "src_ip", "dst_ip", "label"]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_flows: int = 30000                  # size of the simulated training dataset
    test_size: float = 0.25
    rf_trees: int = 150
    iso_trees: int = 200
    iso_fpr: float = 0.03                 # allowed false-positive rate of the anomaly detector
    # alerting
    incident_gap_s: int = 120             # flows from one source closer than this form one incident
    min_flows_known: int = 3              # flows needed to raise an alert for a known attack type
    min_flows_unknown: int = 5            # stricter for "unknown anomaly" (noisier)
    replay_minutes: int = 60
    replay_seed: int = 2024
