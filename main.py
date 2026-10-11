"""AI-Powered Cybersecurity Threat Detection - run the whole pipeline.

    python main.py                       # simulate data -> train -> evaluate -> replay attacks -> alerts + figures
    python main.py --data cicids.csv --label-col Label     # use a real dataset instead of the simulator
    python main.py --flows 60000 --seed 1
"""
import argparse
import dataclasses
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="AI-Powered Cybersecurity Threat Detection")
    p.add_argument("--data", help="path to a real dataset CSV (default: built-in simulator)")
    p.add_argument("--label-col", default="label", help="name of the label column in --data")
    p.add_argument("--flows", type=int, default=30000, help="number of simulated training flows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cv", action="store_true", help="skip cross-validation (faster)")
    p.add_argument("--no-replay", action="store_true", help="skip the live-attack replay (needed for real datasets)")
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    from src import visualize as viz
    from src.alerts import generate_alerts, score_alerts
    from src.config import Config, NORMAL
    from src.data_loader import load_dataset
    from src.evaluate import multiclass_confusion
    from src.pipeline import save_detector, train_and_evaluate
    from src.simulator import simulate_stream

    cfg = dataclasses.replace(Config(), seed=args.seed, n_flows=args.flows)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print(" AI-Powered Cybersecurity Threat Detection System")
    print("=" * 74)

    print("[1/5] Loading data ...")
    df = load_dataset(args.data, args.label_col, cfg.n_flows, cfg.seed)
    if args.data is None:
        os.makedirs("data/raw", exist_ok=True); os.makedirs("data/sample", exist_ok=True)
        df.to_csv("data/raw/network_traffic.csv", index=False)
        df.sample(500, random_state=1).sort_values("timestamp").to_csv("data/sample/network_traffic_sample.csv", index=False)
    print(f"      {len(df):,} flows, {df.shape[1] - 1} columns | classes: {df['label'].value_counts().to_dict()}")

    print("[2/5] Training + evaluating models ...")
    detector, results, art = train_and_evaluate(df, cfg, cv=not args.no_cv)
    save_detector(detector, results, "models", args.out)

    print("[3/5] Drawing evaluation figures ...")
    labels = sorted(set(df["label"]), key=lambda x: (x != NORMAL, x))
    cm = multiclass_confusion(art["y_test"], art["rf"].predict(art["X_test"]), labels)
    viz.class_distribution(df, fig_dir); viz.feature_distributions(df, fig_dir)
    viz.confusion_matrix_plot(cm, labels, fig_dir); viz.model_comparison(results, fig_dir)
    viz.feature_importance(art["rf"], fig_dir)
    if args.data is None:
        viz.anomaly_scores(art, fig_dir)

    summary = {"models": {k: v for k, v in results.items() if isinstance(v, dict) and "f1" in v}}
    if not args.no_replay and args.data is None:
        print("[4/5] Replaying one hour of live traffic with hidden attacks ...")
        flows, truth = simulate_stream(cfg.replay_seed, cfg.replay_minutes)
        verdicts = detector.predict(flows)
        alerts = generate_alerts(flows, verdicts, cfg)
        score = score_alerts(alerts, truth)
        print(f"      {len(flows):,} flows scored, {int((verdicts['verdict'] != NORMAL).sum()):,} flagged\n")
        print(" ---------------------------- SOC ALERT FEED ----------------------------")
        for _, a in alerts.iterrows():
            print(f" [{a['alert_id']}] {a['severity']:<8} {a['attack_type']:<17} src={a['src_ip']:<15} flows={a['flows']:<4} "
                  f"conf={a['avg_confidence']:.2f}  {a['first_seen']:%H:%M:%S}-{a['last_seen']:%H:%M:%S}")
            print(f"            MITRE {a['mitre']}  ->  {a['recommended_action']}")
        print(" -------------------------------------------------------------------------")
        alerts.to_csv(os.path.join(args.out, "alerts.csv"), index=False)
        alerts.assign(first_seen=alerts["first_seen"].astype(str), last_seen=alerts["last_seen"].astype(str)) \
              .to_json(os.path.join(args.out, "alerts.json"), orient="records", indent=2)
        viz.threat_timeline(flows, verdicts, truth, detector.iso_threshold, fig_dir)
        viz.alert_board(alerts, score, fig_dir)
        summary["replay"] = score
        print(f"      campaigns detected: {score['campaigns_detected']}/{score['campaigns']}  |  correct attack type: "
              f"{score['correct_attack_type']}/{score['campaigns']}  |  false alerts: {score['false_alerts']}  |  "
              f"mean time-to-detect: {score['mean_time_to_detect_s']} s")
    else:
        print("[4/5] Replay skipped.")

    print("[5/5] Saving ...")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    rf = results["Random Forest"]
    print("-" * 74)
    print(f" Random Forest  accuracy={rf['accuracy']:.3f}  precision={rf['precision']:.3f}  recall={rf['recall']:.3f}  F1={rf['f1']:.3f}")
    print(f" model  -> models/threat_detector.joblib   |   figures -> {fig_dir}/   |   alerts -> {args.out}/alerts.csv")
    print(f" finished in {time.time() - t0:.1f}s")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
