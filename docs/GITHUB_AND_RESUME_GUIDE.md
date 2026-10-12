# GitHub, resume and interview guide

## 1. Publish on GitHub
Repo name: **`AI-Cybersecurity-Threat-Detection`**
Description: *ML-based network threat detection: Random Forest + Isolation Forest, SOC-style alerts with MITRE ATT&CK mapping, Flask API (Python, scikit-learn).*
Topics: `cybersecurity` `machine-learning` `intrusion-detection` `anomaly-detection` `isolation-forest` `random-forest` `scikit-learn` `python` `flask` `network-security` `soc` `mitre-attack`

```bash
cd AI-Cybersecurity-Threat-Detection
git init
git add .
git commit -m "feat: project structure, simulator and preprocessing"
git branch -M main
# create an EMPTY repo on github.com (no README/license), then:
git remote add origin https://github.com/<your-username>/AI-Cybersecurity-Threat-Detection.git
git push -u origin main
```
`models/*.joblib` and `data/raw/` are git-ignored on purpose (re-create with `python main.py`). Pin the repo on your profile.

## 2. Honest 7-day commit plan
Add the files day by day so history matches your real work.

| Day | Commit | Proof to save | Commit message |
|---|---|---|---|
| 1 | `.gitignore`, `requirements.txt`, `src/config.py`, `src/simulator.py` | `02_dataset_and_preprocessing.png` | `feat: cyber-range traffic simulator` |
| 2 | `src/data_loader.py`, `explore.py`, `data/sample/` | dataset preview screenshot | `feat: data loading + exploration` |
| 3 | `src/preprocessing.py`, `tests/test_preprocessing.py` | preprocessing output | `feat: feature engineering + scaling pipeline` |
| 4 | `src/models.py`, `src/pipeline.py`, `src/evaluate.py` | training log | `feat: train RF, LR, Isolation Forest` |
| 5 | `src/detector.py`, `src/alerts.py`, tests | confusion matrix, alert board | `feat: threat verdicts + SOC alerts` |
| 6 | `src/visualize.py`, `src/api.py`, `predict.py`, `main.py` | all figures, API screenshot | `feat: figures + REST API` |
| 7 | README, docs, screenshots, terminal_logs | repo + README preview | `docs: README, architecture, results` |

## 3. Resume bullets
- Built an ML threat-detection system on simulated network-flow data: class-balanced Random Forest (**F1 0.91, precision 0.99, false-positive rate 0.08%**) for known attacks plus an Isolation Forest trained only on normal traffic to flag unseen behaviour.
- Designed an alerting layer that groups flows into incidents with severity and MITRE ATT&CK mapping; in a 1-hour live replay it detected **6/6 attack campaigns with 0 false alerts** (17 s mean time-to-detect), including a stealth C2 channel absent from training.
- Packaged the model with preprocessing in a single scikit-learn pipeline and a validated Flask REST API; 26 automated tests, reproducible CLI and documented GitHub repo.

## 4. LinkedIn post
> Built an AI-powered threat detection system in Python. Random Forest catches known attacks (DoS, port scans, brute force, exfiltration) at 99% precision; an Isolation Forest trained only on normal traffic flags behaviour it has never seen. Suspicious flows become SOC-style alerts with severity and MITRE ATT&CK mapping. In a simulated 1-hour attack replay it caught all 6 campaigns with zero false alerts. Includes a REST API and 26 tests. Data is simulated - next step: CICIDS2017. Code: <link> #cybersecurity #machinelearning #python

## 5. Interview Q&A (matches this repo)
1. **What does it do?** Classifies network flows as normal or one of four attack types, flags unknown anomalies, and turns suspicious flows into prioritised alerts.
2. **What data?** A simulator I wrote that produces flow records (protocol, bytes, duration, SYN ratio, ports touched, connection rate, login failures) with ~12% attacks, including stealthy and camouflaged traffic. The pipeline also loads CICIDS-style CSVs.
3. **Why simulated data - isn't that cheating?** I had no access to real corporate traffic, so I built a controlled range. It proves the pipeline and lets me test specific scenarios (like a never-seen attack). I state clearly that metrics are from simulation and the next step is real datasets.
4. **Why Random Forest + Isolation Forest?** Supervised is precise on known attacks; unsupervised catches the unknown. Together they cover both.
5. **Why not just accuracy?** 88% of flows are normal, so a model that says "normal" always scores 88%. I report precision (alert quality), recall (attacks caught), F1 and false-positive rate.
6. **Your recall is 0.84 - why?** About 15% of each attack's flows are deliberately camouflaged and 40% are low-rate stealth. Individual flows slip through, which is why I aggregate into incidents - all 6 campaigns were still caught.
7. **How do you handle imbalance?** `class_weight="balanced_subsample"`, stratified split, and metrics beyond accuracy.
8. **How do you avoid train/serve skew?** Preprocessing is inside the sklearn Pipeline that gets saved and used by the API.
9. **How do you detect zero-days?** The Isolation Forest is trained on normal traffic only; flows that look unlike it get high scores. It flagged 80% of an attack type it never saw.
10. **What would you change for production?** Real datasets, streaming ingestion, SIEM integration, drift monitoring and retraining, per-alert explanations (SHAP), analyst feedback loop for the threshold.

**Explain to HR:** "I built software that watches network activity, learns what normal looks like, and raises prioritised alerts when something looks like an attack - even a type it hasn't seen before."

## 6. Troubleshooting
| Problem | Reason | Fix |
|---|---|---|
| `ModuleNotFoundError: sklearn` | not installed in this venv | activate venv, `pip install -r requirements.txt` |
| `FileNotFoundError: models/threat_detector.joblib` | model not trained yet | run `python main.py` first |
| API returns 400 | unknown field or non-numeric value | check allowed fields in the error message |
| Unpickling error after upgrading scikit-learn | model saved with another version | re-run `python main.py` |
| Real CSV crashes on memory | file too big | loader samples 150k rows; raise/lower `max_rows` |
| Real CSV has no `Label` column | different name | pass `--label-col <name>` (leading spaces are stripped) |
| Metrics are 100% on a real dataset | duplicate rows / leakage | drop duplicates and ID-like columns before training |
| `git push` rejected | remote has a README | `git pull origin main --rebase`, push again |

## 7. Problems in the original guide (fixed here)
- It trained on all columns but its API sent only 3 features → the model would crash. Here preprocessing is saved with the model and missing API fields get sensible defaults.
- It scaled data but never saved the scaler, so predictions would be on a different scale. Here the scaler is part of the pipeline.
- `drop("Timestamp")` and `StandardScaler` fail on the string label/categorical columns of real datasets. Here columns are handled by type.
- Plain accuracy on imbalanced data is misleading; precision/recall/F1 are used instead.
- `debug=True` in Flask exposes a remote-code-execution debugger - disabled here.
