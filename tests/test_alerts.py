import pandas as pd
from src.alerts import generate_alerts, score_alerts, severity
from src.config import Config


def _frame(ip, verdict, n, start="2026-01-01 10:00:00", step_s=5):
    ts = pd.Timestamp(start) + pd.to_timedelta(range(0, n * step_s, step_s), unit="s")
    flows = pd.DataFrame({"timestamp": ts, "src_ip": ip})
    verdicts = pd.DataFrame({"verdict": verdict, "confidence": 0.95, "threat_score": 0.95}, index=flows.index)
    return flows, verdicts


def test_incident_is_raised_for_enough_flows_and_ignored_for_too_few():
    f, v = _frame("1.1.1.1", "PortScan", 10)
    assert len(generate_alerts(f, v, Config())) == 1
    f, v = _frame("1.1.1.1", "PortScan", 2)
    assert len(generate_alerts(f, v, Config())) == 0


def test_flows_far_apart_in_time_become_separate_incidents():
    f1, v1 = _frame("2.2.2.2", "DoS", 5, "2026-01-01 10:00:00")
    f2, v2 = _frame("2.2.2.2", "DoS", 5, "2026-01-01 10:30:00")
    f, v = pd.concat([f1, f2], ignore_index=True), pd.concat([v1, v2], ignore_index=True)
    assert len(generate_alerts(f, v, Config())) == 2


def test_normal_flows_never_alert():
    f, v = _frame("3.3.3.3", "Normal", 50)
    assert generate_alerts(f, v, Config()).empty


def test_severity_rules():
    assert severity("DataExfiltration", 10, 0.9) == "CRITICAL"
    assert severity("PortScan", 10, 0.9) == "MEDIUM"
    assert severity("PortScan", 200, 0.9) == "HIGH"
    assert severity("PortScan", 10, 0.4) == "LOW"


def test_scoring_counts_detected_campaigns_and_false_alerts():
    f, v = _frame("9.9.9.9", "BruteForce", 10)
    alerts = generate_alerts(f, v, Config())
    truth = [{"src_ip": "9.9.9.9", "attack": "BruteForce", "start": pd.Timestamp("2026-01-01 09:59:50"),
              "end": pd.Timestamp("2026-01-01 10:10:00")},
             {"src_ip": "8.8.8.8", "attack": "DoS", "start": pd.Timestamp("2026-01-01 10:00:00"),
              "end": pd.Timestamp("2026-01-01 10:05:00")}]
    s = score_alerts(alerts, truth)
    assert s["campaigns_detected"] == 1 and s["correct_attack_type"] == 1 and s["false_alerts"] == 0
