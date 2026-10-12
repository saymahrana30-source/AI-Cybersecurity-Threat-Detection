"""All figures (Matplotlib + Seaborn). Every function saves one PNG and returns its path."""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import KNOWN_ATTACKS, NORMAL, NOVEL_ATTACK, UNKNOWN

sns.set_theme(style="whitegrid", font_scale=0.95)
COLORS = {NORMAL: "#90a4ae", "DoS": "#e53935", "PortScan": "#fb8c00", "BruteForce": "#8e24aa",
          "DataExfiltration": "#1e88e5", UNKNOWN: "#00897b", NOVEL_ATTACK: "#00897b"}
SEV = {"LOW": "#9ccc65", "MEDIUM": "#fbc02d", "HIGH": "#fb8c00", "CRITICAL": "#d32f2f"}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def class_distribution(df, out):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(counts.index, counts.values, color=[COLORS.get(k, "#607d8b") for k in counts.index])
    ax.set_yscale("log"); ax.set_ylabel("flows (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v * 1.1, f"{v:,}\n({100 * v / len(df):.1f}%)", ha="center", fontsize=9)
    ax.set_title("Dataset class distribution - heavily imbalanced, like real networks"); ax.set_ylim(top=counts.max() * 6)
    return _save(fig, out, "01_class_distribution.png")


def feature_distributions(df, out):
    feats = [("conn_per_min", "connections / minute"), ("unique_dst_ports", "distinct ports touched"),
             ("failed_logins", "failed logins"), ("src_bytes", "bytes sent")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (f, title) in zip(axs.ravel(), feats):
        for lab, g in df.groupby("label"):
            ax.hist(np.log1p(g[f]), bins=40, histtype="step", density=True, lw=1.8, label=lab, color=COLORS.get(lab))
        ax.set_title(title + "  (log1p scale)"); ax.set_yticks([])
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("How attacks differ from normal traffic", fontweight="bold")
    return _save(fig, out, "02_feature_distributions.png")


def confusion_matrix_plot(cm, labels, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axs[0], cbar=False)
    axs[0].set_title("Confusion matrix (counts)")
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Greens", xticklabels=labels,
                yticklabels=labels, ax=axs[1], cbar=False)
    axs[1].set_title("Row-normalised (= recall per class)")
    for ax in axs:
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Random Forest - multi-class attack classification (test set)", fontweight="bold")
    return _save(fig, out, "03_confusion_matrix.png")


def model_comparison(results, out):
    names = [k for k in results if isinstance(results[k], dict) and "f1" in results[k]]
    metrics = ["precision", "recall", "f1"]
    x, w = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m.upper() if m == "f1" else m.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.12); ax.legend(ncol=3, loc="lower center")
    ax.set_title("Model comparison on held-out test flows (attack = positive class)")
    return _save(fig, out, "04_model_comparison.png")


def feature_importance(rf, out, top=12):
    names = rf.named_steps["pre"].named_steps["columns"].get_feature_names_out()
    imp = rf.named_steps["model"].feature_importances_
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i].split("__")[-1] for i in idx], imp[idx], color="#1e88e5")
    ax.set_title("What the Random Forest looks at (top features)"); ax.set_xlabel("importance")
    return _save(fig, out, "05_feature_importance.png")


def anomaly_scores(art, out):
    y = art["y_test"].values
    sc, thr = art["iso_test_scores"], art["thr"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(min(sc.min(), art["novel_scores"].min()), max(sc.max(), art["novel_scores"].max()), 70)
    ax.hist(sc[y == NORMAL], bins=bins, alpha=.7, color=COLORS[NORMAL], label="normal", density=True)
    ax.hist(sc[y != NORMAL], bins=bins, alpha=.6, color=COLORS["DoS"], label="known attacks", density=True)
    ax.hist(art["novel_scores"], bins=bins, alpha=.6, color=COLORS[UNKNOWN], label="UNSEEN attack (StealthC2)", density=True)
    ax.axvline(thr, color="k", ls="--", label=f"alert threshold (~3% false positives)")
    ax.set_xlabel("Isolation Forest anomaly score  (higher = stranger)"); ax.set_yticks([]); ax.legend()
    ax.set_title("Anomaly detection: trained on normal traffic only, still flags the unseen attack")
    return _save(fig, out, "06_anomaly_score_distribution.png")


def threat_timeline(flows, verdicts, truth, thr, out):
    """3 panels: what really happened / what the detector flagged / anomaly scores."""
    t0 = flows["timestamp"].min()
    mins = (flows["timestamp"] - t0).dt.total_seconds().values / 60
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.3, 1.3]})
    # 1) ground truth campaigns (what the attackers did)
    for i, t in enumerate(truth):
        a, b = (t["start"] - t0).total_seconds() / 60, (t["end"] - t0).total_seconds() / 60
        c = COLORS.get(t["attack"], "#00897b")
        axs[0].barh(i, b - a, left=a, color=c, height=.6)
        label = f"{t['attack']}  ({t['src_ip']})"
        if b - a > 25:
            axs[0].text(a + 0.5, i, label, va="center", fontsize=8, color="white", fontweight="bold")
        elif a > 40:
            axs[0].text(a - 0.5, i, label, va="center", ha="right", fontsize=8)
        else:
            axs[0].text(b + 0.5, i, label, va="center", fontsize=8)
    axs[0].set_yticks([]); axs[0].invert_yaxis()
    axs[0].set_title("Ground truth: scripted attack campaigns hidden inside normal traffic", fontsize=10)
    # 2) what the detector flagged
    verd = verdicts["verdict"].values
    bins = np.arange(0, int(mins.max()) + 2)
    bottom = np.zeros(len(bins) - 1)
    for cat in KNOWN_ATTACKS + [UNKNOWN]:
        cnt, _ = np.histogram(mins[verd == cat], bins=bins)
        axs[1].bar(bins[:-1] + .5, cnt, bottom=bottom, width=.9, color=COLORS[cat], label=cat)
        bottom += cnt
    axs[1].set_ylabel("flagged flows / min"); axs[1].legend(fontsize=8, ncol=5, loc="upper left")
    axs[1].set_title("Detector output: flows flagged per minute, by verdict", fontsize=10)
    # 3) anomaly score of every flow
    sc = verdicts["anomaly_score"].values
    is_att = flows["label"].values != NORMAL
    axs[2].scatter(mins[~is_att], sc[~is_att], s=3, color=COLORS[NORMAL], alpha=.4, label="normal flow")
    axs[2].scatter(mins[is_att], sc[is_att], s=7, color=COLORS["DoS"], alpha=.7, label="attack flow (ground truth)")
    axs[2].axhline(thr, color="k", ls="--", lw=1, label="anomaly threshold")
    axs[2].set_ylabel("anomaly score"); axs[2].set_xlabel("minutes since start of capture")
    axs[2].legend(fontsize=8, ncol=3, loc="upper left")
    axs[2].set_title("Isolation Forest anomaly score per flow", fontsize=10)
    axs[2].set_xlim(0, mins.max() + 1)
    return _save(fig, out, "07_threat_timeline.png")


def alert_board(alerts, score, out):
    fig, ax = plt.subplots(figsize=(13, 0.9 + 0.55 * max(1, len(alerts))))
    ax.axis("off")
    cols = ["alert_id", "severity", "attack_type", "src_ip", "flows", "avg_confidence", "first_seen", "mitre"]
    cell = alerts[cols].copy()
    cell["first_seen"] = cell["first_seen"].dt.strftime("%H:%M:%S")
    cell["mitre"] = cell["mitre"].str.split().str[0]            # technique ID only (full text is in alerts.csv)
    tbl = ax.table(cellText=cell.values, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#263238"); tbl[0, j].set_text_props(color="white", weight="bold")
    for i, sev in enumerate(alerts["severity"], start=1):
        tbl[i, 1].set_facecolor(SEV[sev]); tbl[i, 1].set_text_props(weight="bold")
    ax.set_title(f"SOC alert board  |  {score['campaigns_detected']}/{score['campaigns']} attack campaigns detected  |  "
                 f"{score['false_alerts']} false alerts  |  mean time-to-detect {score['mean_time_to_detect_s']} s",
                 fontweight="bold", fontsize=11)
    return _save(fig, out, "08_alert_board.png")
