# Architecture

```
 ┌──────────────┐   ┌────────────────┐   ┌───────────────────────────────────────┐
 │ DATA SOURCE  │   │ PREPROCESSING  │   │ DETECTION  (ThreatDetector)            │
 │ simulator.py │──►│ preprocessing  │──►│  Random Forest   → known attack class   │
 │ or real CSV  │   │ .py (Pipeline) │   │  Isolation Forest→ "how strange is it?" │
 └──────────────┘   └────────────────┘   └───────────────┬───────────────────────┘
                                                           │ verdict per flow
        ┌──────────────────────────────────────────────────▼───────────────────────┐
        │ ALERTING (alerts.py): group by source IP + time gap → incident              │
        │ severity (type, volume, confidence) + MITRE technique + recommended action  │
        └───────────────┬──────────────────────────────┬──────────────────────────────┘
                        ▼                              ▼
              outputs/alerts.csv|json           REST API (api.py)  /predict  /predict_batch
              figures (visualize.py)
```

## Data flow
1. **Flow records** - each row = one network connection: protocol, service, state, duration, bytes each way, packets, SYN ratio, distinct ports touched, connection rate, login attempts/failures.
2. **Preprocess** (inside one sklearn `Pipeline`, so training and the API treat data identically): add engineered features (bytes/sec, average packet size, upload/download ratio, failed-login ratio) → drop identifiers → median-impute → signed log transform (heavy-tailed bytes/counts) → standardise → one-hot encode categories.
3. **Random Forest** (class-balanced) classifies Normal / DoS / PortScan / BruteForce / DataExfiltration.
4. **Isolation Forest** is trained on **normal traffic only** (numeric features) and scores how unusual each flow is; the alert threshold gives ~3% false positives on normal training data.
5. **Verdict rule:** attack class if the Random Forest is confident → `UnknownAnomaly` if it says Normal (or is unsure, confidence < 0.6) but the Isolation Forest finds it anomalous → otherwise `Normal`.
6. **Incident builder:** suspicious flows from the same source with gaps < 120 s form one incident; an alert needs ≥ 3 flows (≥ 5 for unknown anomalies, which are noisier).
7. **Severity:** base level by attack type (exfiltration = CRITICAL, DoS/brute force = HIGH, scan/unknown = MEDIUM), +1 if ≥ 100 flows, −1 if average confidence < 0.6.

## Module map
| File | Role |
|---|---|
| `simulator.py` | normal traffic + 4 known attacks + 1 held-out "unknown" attack; 1-hour live replay with scripted campaigns |
| `data_loader.py` | simulator or real CSV (labels mapped to `Normal` / attack name, infinities cleaned) |
| `preprocessing.py` | feature engineering + scaling pipeline |
| `models.py` | Logistic Regression, Random Forest, Isolation Forest + threshold |
| `evaluate.py` | metrics |
| `detector.py` | combines supervised + anomaly into one deployable object (`joblib`) |
| `alerts.py` | incidents, severity, MITRE playbook, scoring against ground-truth campaigns |
| `api.py` | Flask API with input validation |
| `visualize.py` | charts |

## Why two models?
A supervised model is precise on attacks it has seen but blind to new ones; an anomaly detector catches the new and strange but is noisier. Together: high precision on known threats **and** coverage of the unknown - the same layered idea used in commercial detection stacks.
