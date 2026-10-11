# Using a real dataset

The pipeline is column-agnostic: it engineers features only when the source columns exist, picks numeric/categorical columns automatically, and maps labels (`BENIGN`/`normal` → `Normal`, everything else keeps its name).

```bash
python main.py --data path/to/file.csv --label-col Label --no-replay
```
`--no-replay` skips the simulated attack replay and zero-day demo (they need the simulator's columns). Replace the models/figures as usual in `outputs/`.

| Dataset | Where | Label column | Notes |
|---|---|---|---|
| CICIDS2017 | Canadian Institute for Cybersecurity (one CSV per day) | `Label` | big files - the loader samples up to 150,000 rows |
| NSL-KDD | Kaggle / UNB | usually the 42nd column; add headers first | classic benchmark, small |
| UNSW-NB15 | UNSW Canberra | `attack_cat` (multi-class) or `label` (0/1) | training/testing CSVs provided |

Tips
- Download into `data/raw/` (git-ignored). Never commit multi-GB files.
- If a dataset has no header row, add one first (NSL-KDD).
- For `0/1` labels the loader treats `0` as normal.
- The pipeline was verified on a CICIDS-style file with different column names, spaces in headers, `inf` and NaN values (see `tests/test_real_data_format.py`), but **I could not download the real datasets while building this**, so check the first run on your machine and watch the per-class report.
- Expect lower, more honest numbers than on the simulator: real data has far more overlap.
