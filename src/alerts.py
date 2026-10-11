"""Alert generation: turn per-flow verdicts into SOC-style incidents with severity + response advice."""
from __future__ import annotations
import pandas as pd

from .config import NORMAL, UNKNOWN, Config

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PLAYBOOK = {   # attack -> (base severity index, MITRE ATT&CK technique, recommended response)
    "DoS": (2, "T1498 Network Denial of Service", "Rate-limit / block source at firewall, enable SYN cookies"),
    "PortScan": (1, "T1046 Network Service Discovery", "Block source, review exposed services"),
    "BruteForce": (2, "T1110 Brute Force", "Block source, enforce MFA, lock affected accounts"),
    "DataExfiltration": (3, "T1048 Exfiltration Over Alternative Protocol", "Isolate host, inspect transferred data, rotate credentials"),
    UNKNOWN: (1, "T1071 Application Layer Protocol (suspected C2)", "Investigate host, check for beaconing / malware"),
}


def severity(kind: str, flows: int, confidence: float) -> str:
    level = PLAYBOOK.get(kind, (1,))[0]
    level += 1 if flows >= 100 else 0
    level -= 1 if confidence < 0.6 else 0
    return LEVELS[max(0, min(3, level))]


def generate_alerts(flows: pd.DataFrame, verdicts: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Group suspicious flows by (source IP, verdict); flows < `incident_gap_s` apart form one incident."""
    d = flows[["timestamp", "src_ip"]].join(verdicts[["verdict", "confidence", "threat_score"]])
    d = d[d["verdict"] != NORMAL].sort_values("timestamp")
    rows = []
    for (ip, kind), g in d.groupby(["src_ip", "verdict"]):
        gap = g["timestamp"].diff().dt.total_seconds().fillna(0) > cfg.incident_gap_s
        for _, inc in g.groupby(gap.cumsum()):
            need = cfg.min_flows_unknown if kind == UNKNOWN else cfg.min_flows_known
            if len(inc) < need:
                continue
            conf = float(inc["confidence"].mean())
            lvl, mitre, action = PLAYBOOK[kind]
            rows.append({"src_ip": ip, "attack_type": kind, "severity": severity(kind, len(inc), conf),
                         "flows": len(inc), "avg_confidence": round(conf, 3),
                         "first_seen": inc["timestamp"].min(), "last_seen": inc["timestamp"].max(),
                         "mitre": mitre, "recommended_action": action})
    out = pd.DataFrame(rows, columns=["src_ip", "attack_type", "severity", "flows", "avg_confidence", "first_seen",
                                      "last_seen", "mitre", "recommended_action"])
    out = out.sort_values("first_seen").reset_index(drop=True)
    out.insert(0, "alert_id", [f"ALERT-{i + 1:03d}" for i in range(len(out))])
    return out


def score_alerts(alerts: pd.DataFrame, truth: list) -> dict:
    """Compare alerts with the scripted attack campaigns (ground truth of the simulation)."""
    attackers = {t["src_ip"] for t in truth}
    detected, correct, delays = 0, 0, []
    for t in truth:
        a = alerts[alerts["src_ip"] == t["src_ip"]]
        if len(a):
            detected += 1
            first = a["first_seen"].min()
            delays.append(max(0.0, (first - t["start"]).total_seconds()))
            ok_type = {"StealthC2": UNKNOWN}.get(t["attack"], t["attack"])
            correct += int((a["attack_type"] == ok_type).any())
    return {"campaigns": len(truth), "campaigns_detected": detected, "correct_attack_type": correct,
            "false_alerts": int((~alerts["src_ip"].isin(attackers)).sum()), "total_alerts": len(alerts),
            "mean_time_to_detect_s": round(sum(delays) / len(delays), 1) if delays else None}
