"""Stage 6 — Calibration Study (Reliability Diagram, ECE, Brier Score).

Analyzes the calibration of Clef decision-model probabilities against TREC-COVID
gold relevance judgements (qrels), fulfilling Section 14 of the Team Handbook:
- Evaluates both Judged-Only and All-Candidates (unjudged-as-zero) assumptions
- Compares S5 (noul), S6 (score: both expected-level/2 and P(rel)=P(1)+P(2)),
  and S4 cross-encoder sigmoid baseline
- Computes Expected Calibration Error (ECE) and Brier Score
- Evaluates threshold transfer across topics (mean, std, min, max per-topic F1)
- Generates side-by-side reliability diagrams and confidence histograms in results/calibration.png

Inputs:
  artifacts/clef_raw.jsonl   (Contract B raw decisions from Stage 4)
  runs/S4_ce.trec            (Cross-encoder baseline run)
  qrels from ir_datasets

Outputs:
  results/calibration.png    (2-panel reliability diagram with bin histograms beneath)
  results/calibration_metrics.json (ECE, Brier score, optimal threshold, topic transfer stats)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import log, load_config, load_jsonl, dump_json
from load_data import load_dataset
from evaluate import _parse_trec_scores


def compute_calibration_stats(probs, labels, n_bins=10):
    """Compute ECE, Brier score, and bin stats for reliability diagram.

    probs: array-like in [0, 1]
    labels: array-like in {0, 1}
    n_bins: number of equal-width bins (default: 10)
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return {"ece": float("nan"), "brier": float("nan"), "bins": []}

    brier = float(np.mean((probs - labels) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_stats = []
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(probs[mask]))
            mean_obs = float(np.mean(labels[mask]))
            gap = abs(mean_pred - mean_obs)
            ece += (count / n) * gap
        else:
            mean_pred = float(bin_centers[i])
            mean_obs = 0.0
            gap = 0.0

        bin_stats.append({
            "bin_idx": i,
            "range": [round(low, 2), round(high, 2)],
            "center": round(bin_centers[i], 3),
            "count": count,
            "mean_pred": round(mean_pred, 4),
            "mean_obs": round(mean_obs, 4),
        })

    return {
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "bins": bin_stats
    }


def find_optimal_threshold(probs, labels):
    """Find the single fixed probability threshold that maximizes F1 across pooled items."""
    best_f1 = -1.0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return round(best_thresh, 3), round(best_f1, 4)


def evaluate_per_topic_threshold(items, threshold, prob_key="p_noul", label_key="label_judged_only"):
    """Evaluate performance of a single fixed threshold per topic to test threshold transfer."""
    by_qid = {}
    for it in items:
        lbl = it.get(label_key)
        if lbl is not None:
            by_qid.setdefault(it["qid"], []).append((it[prob_key], lbl))

    topic_f1s = {}
    for qid in sorted(by_qid.keys(), key=lambda x: int(x) if x.isdigit() else x):
        pairs = by_qid[qid]
        probs = np.array([p for p, l in pairs])
        labels = np.array([l for p, l in pairs])
        preds = (probs >= threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        topic_f1s[qid] = round(float(f1), 4)

    vals = list(topic_f1s.values())
    return {
        "fixed_threshold": round(threshold, 3),
        "mean_f1": round(float(np.mean(vals)), 4) if vals else 0.0,
        "std_f1": round(float(np.std(vals)), 4) if vals else 0.0,
        "median_f1": round(float(np.median(vals)), 4) if vals else 0.0,
        "min_f1": round(float(np.min(vals)), 4) if vals else 0.0,
        "max_f1": round(float(np.max(vals)), 4) if vals else 0.0,
        "per_topic_f1": topic_f1s
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    art_dir = root / "artifacts"
    runs_dir = root / "runs"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = art_dir / "clef_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run Stage 4 decision re-ranking first.")

    raw_decisions = load_jsonl(raw_path)
    log(f"Loaded {len(raw_decisions)} decisions from {raw_path}")

    # Load qrels
    log("Loading qrels...")
    _, _, qrels = load_dataset(cfg)

    # Load S4 cross-encoder scores for comparison curve if available
    s4_path = runs_dir / "S4_ce.trec"
    s4_scores = {}
    if s4_path.exists():
        s4_scores = _parse_trec_scores(s4_path)
        log(f"Loaded S4 scores for {len(s4_scores)} topics")

    # Match each decision to ground truth
    items = []
    n_judged = 0
    n_total = len(raw_decisions)

    for r in raw_decisions:
        qid = str(r["qid"])
        docid = str(r["docid"])
        prob_noul = float(r.get("probability", 0.0))

        # Expected score in [0, 2] -> normalized score in [0, 1]
        score_val = float(r.get("score", 0.0))
        prob_score_div2 = min(max(score_val / 2.0, 0.0), 1.0)

        # Graded score probabilities: P(relevant) = P(grade >= 1) = P(1) + P(2)
        score_probs = r.get("score_probabilities", {})
        p_1 = float(score_probs.get("1", 0.0))
        p_2 = float(score_probs.get("2", 0.0))
        prob_score_binary = min(max(p_1 + p_2, 0.0), 1.0)

        # S4 logit -> sigmoid prob
        s4_prob = None
        if qid in s4_scores and docid in s4_scores[qid]:
            s4_logit = s4_scores[qid][docid]
            s4_prob = float(sigmoid(s4_logit))

        topic_qrels = qrels.get(qid, {})
        is_judged = docid in topic_qrels
        if is_judged:
            n_judged += 1
            grade = topic_qrels[docid]
            rel_binary = 1 if grade >= 1 else 0
        else:
            grade = -1
            rel_binary = 0  # Default assumption: unjudged treated as not-relevant

        items.append({
            "qid": qid,
            "docid": docid,
            "p_noul": prob_noul,
            "p_score": prob_score_div2,
            "p_score_binary": prob_score_binary,
            "p_s4": s4_prob,
            "grade": grade,
            "is_judged": is_judged,
            "label_unjudged_zero": rel_binary,
            "label_judged_only": rel_binary if is_judged else None
        })

    judged_pct = (n_judged / n_total) * 100 if n_total > 0 else 0.0
    log(f"Judged decisions: {n_judged}/{n_total} ({judged_pct:.1f}%)")

    # -------------------------------------------------------------
    # Calibration analysis 1: Judged-only (excluding unjudged docs)
    # -------------------------------------------------------------
    judged_items = [it for it in items if it["is_judged"]]
    p_noul_judged = np.array([it["p_noul"] for it in judged_items])
    p_score_judged = np.array([it["p_score"] for it in judged_items])
    p_score_bin_judged = np.array([it["p_score_binary"] for it in judged_items])
    labels_judged = np.array([it["label_judged_only"] for it in judged_items])

    stats_s5_judged = compute_calibration_stats(p_noul_judged, labels_judged)
    stats_s6_judged = compute_calibration_stats(p_score_judged, labels_judged)
    stats_s6_bin_judged = compute_calibration_stats(p_score_bin_judged, labels_judged)
    thresh_s5_j, f1_s5_j = find_optimal_threshold(p_noul_judged, labels_judged)
    transfer_s5_j = evaluate_per_topic_threshold(judged_items, thresh_s5_j, "p_noul", "label_judged_only")

    stats_s4_judged = None
    if s4_scores:
        s4_p_judged = np.array([it["p_s4"] for it in judged_items if it["p_s4"] is not None])
        s4_l_judged = np.array([it["label_judged_only"] for it in judged_items if it["p_s4"] is not None])
        if len(s4_p_judged) > 0:
            stats_s4_judged = compute_calibration_stats(s4_p_judged, s4_l_judged)

    log(f"[Judged-Only] S5 Clef noul:       ECE={stats_s5_judged['ece']:.4f}, Brier={stats_s5_judged['brier']:.4f}, Best F1={f1_s5_j:.4f} @ thr={thresh_s5_j}")
    log(f"[Judged-Only] Threshold Transfer: Topic F1 Mean={transfer_s5_j['mean_f1']:.4f}, Std={transfer_s5_j['std_f1']:.4f}, Min={transfer_s5_j['min_f1']:.4f}, Max={transfer_s5_j['max_f1']:.4f}")
    log(f"[Judged-Only] S6 Clef score/2:    ECE={stats_s6_judged['ece']:.4f}, Brier={stats_s6_judged['brier']:.4f}")
    log(f"[Judged-Only] S6 Clef P(1)+P(2):  ECE={stats_s6_bin_judged['ece']:.4f}, Brier={stats_s6_bin_judged['brier']:.4f}")
    if stats_s4_judged:
        log(f"[Judged-Only] S4 Cross-enc:       ECE={stats_s4_judged['ece']:.4f}, Brier={stats_s4_judged['brier']:.4f}")

    # -------------------------------------------------------------
    # Calibration analysis 2: All candidates (unjudged as zero)
    # -------------------------------------------------------------
    p_noul_all = np.array([it["p_noul"] for it in items])
    p_score_all = np.array([it["p_score"] for it in items])
    p_score_bin_all = np.array([it["p_score_binary"] for it in items])
    labels_all = np.array([it["label_unjudged_zero"] for it in items])

    stats_s5_all = compute_calibration_stats(p_noul_all, labels_all)
    stats_s6_all = compute_calibration_stats(p_score_all, labels_all)
    stats_s6_bin_all = compute_calibration_stats(p_score_bin_all, labels_all)
    thresh_s5_all, f1_s5_all = find_optimal_threshold(p_noul_all, labels_all)
    transfer_s5_all = evaluate_per_topic_threshold(items, thresh_s5_all, "p_noul", "label_unjudged_zero")

    stats_s4_all = None
    if s4_scores:
        s4_p_all = np.array([it["p_s4"] for it in items if it["p_s4"] is not None])
        s4_l_all = np.array([it["label_unjudged_zero"] for it in items if it["p_s4"] is not None])
        if len(s4_p_all) > 0:
            stats_s4_all = compute_calibration_stats(s4_p_all, s4_l_all)

    log(f"[Unjudged=0] S5 Clef noul:        ECE={stats_s5_all['ece']:.4f}, Brier={stats_s5_all['brier']:.4f}, Best F1={f1_s5_all:.4f} @ thr={thresh_s5_all}")
    log(f"[Unjudged=0] Threshold Transfer:  Topic F1 Mean={transfer_s5_all['mean_f1']:.4f}, Std={transfer_s5_all['std_f1']:.4f}")
    log(f"[Unjudged=0] S6 Clef score/2:     ECE={stats_s6_all['ece']:.4f}, Brier={stats_s6_all['brier']:.4f}")
    log(f"[Unjudged=0] S6 Clef P(1)+P(2):   ECE={stats_s6_bin_all['ece']:.4f}, Brier={stats_s6_bin_all['brier']:.4f}")

    # -------------------------------------------------------------
    # Generate Publication-Quality Dual Reliability Diagram (2x2)
    # Left column: Judged-Only pool; Right column: Unjudged as 0 pool
    # -------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(14, 10.5),
        gridspec_kw={"height_ratios": [3, 1.2], "wspace": 0.25, "hspace": 0.28},
        dpi=300
    )
    (ax_c_j, ax_c_a), (ax_h_j, ax_h_a) = axes

    # --- Left Panel (Judged Only) ---
    ax_c_j.plot([0, 1], [0, 1], "k--", label="Perfect Calibration ($y = x$)", linewidth=1.5, alpha=0.7)

    s5_bins_j = stats_s5_judged["bins"]
    ax_c_j.plot([b["mean_pred"] for b in s5_bins_j if b["count"] > 0],
                [b["mean_obs"] for b in s5_bins_j if b["count"] > 0],
                "s-", color="#1f77b4", linewidth=2, markersize=7,
                label=f"S5 Clef (noul) — ECE: {stats_s5_judged['ece']:.3f}, Brier: {stats_s5_judged['brier']:.3f}")

    s6_bin_bins_j = stats_s6_bin_judged["bins"]
    ax_c_j.plot([b["mean_pred"] for b in s6_bin_bins_j if b["count"] > 0],
                [b["mean_obs"] for b in s6_bin_bins_j if b["count"] > 0],
                "D-", color="#d62728", linewidth=2, markersize=6.5,
                label=f"S6 Clef P(1)+P(2) — ECE: {stats_s6_bin_judged['ece']:.3f}, Brier: {stats_s6_bin_judged['brier']:.3f}")

    s6_bins_j = stats_s6_judged["bins"]
    ax_c_j.plot([b["mean_pred"] for b in s6_bins_j if b["count"] > 0],
                [b["mean_obs"] for b in s6_bins_j if b["count"] > 0],
                "^--", color="#ff7f0e", linewidth=1.8, markersize=6.5, alpha=0.85,
                label=f"S6 Clef (score/2) — ECE: {stats_s6_judged['ece']:.3f}")

    if stats_s4_judged:
        s4_bins_j = stats_s4_judged["bins"]
        ax_c_j.plot([b["mean_pred"] for b in s4_bins_j if b["count"] > 0],
                    [b["mean_obs"] for b in s4_bins_j if b["count"] > 0],
                    "o:", color="#2ca02c", linewidth=1.8, markersize=6, alpha=0.85,
                    label=rf"S4 Cross-Encoder ($\sigma(z)$) — ECE: {stats_s4_judged['ece']:.3f}")

    ax_c_j.set_xlim(-0.02, 1.02)
    ax_c_j.set_ylim(-0.02, 1.02)
    ax_c_j.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax_c_j.set_ylabel("Observed Relevant Fraction (TP)", fontsize=11, fontweight="bold")
    ax_c_j.set_title(f"(a) Judged-Only Candidates (N={n_judged}, {judged_pct:.1f}% pool)", fontsize=12, pad=10, fontweight="bold")
    ax_c_j.legend(loc="upper left", frameon=True, fontsize=8.8)
    ax_c_j.grid(True, linestyle=":", alpha=0.6)

    # Histogram Left
    bin_centers = [b["center"] for b in s5_bins_j]
    width = 0.035
    ax_h_j.bar([c - width/2 for c in bin_centers], [b["count"] for b in s5_bins_j], width=width,
               color="#1f77b4", alpha=0.85, label="S5 (noul)")
    ax_h_j.bar([c + width/2 for c in bin_centers], [b["count"] for b in s6_bin_bins_j], width=width,
               color="#d62728", alpha=0.85, label="S6 P(1)+P(2)")
    ax_h_j.set_xlim(-0.02, 1.02)
    ax_h_j.set_xlabel("Probability Bin", fontsize=10)
    ax_h_j.set_ylabel("Candidates", fontsize=10)
    ax_h_j.set_title("Candidate Distribution across Bins (Judged Set)", fontsize=9.5, pad=5)
    ax_h_j.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.2)
    ax_h_j.grid(True, linestyle=":", alpha=0.5)

    # --- Right Panel (Unjudged Treated as Zero) ---
    ax_c_a.plot([0, 1], [0, 1], "k--", label="Perfect Calibration ($y = x$)", linewidth=1.5, alpha=0.7)

    s5_bins_a = stats_s5_all["bins"]
    ax_c_a.plot([b["mean_pred"] for b in s5_bins_a if b["count"] > 0],
                [b["mean_obs"] for b in s5_bins_a if b["count"] > 0],
                "s-", color="#1f77b4", linewidth=2, markersize=7,
                label=f"S5 Clef (noul) — ECE: {stats_s5_all['ece']:.3f}, Brier: {stats_s5_all['brier']:.3f}")

    s6_bin_bins_a = stats_s6_bin_all["bins"]
    ax_c_a.plot([b["mean_pred"] for b in s6_bin_bins_a if b["count"] > 0],
                [b["mean_obs"] for b in s6_bin_bins_a if b["count"] > 0],
                "D-", color="#d62728", linewidth=2, markersize=6.5,
                label=f"S6 Clef P(1)+P(2) — ECE: {stats_s6_bin_all['ece']:.3f}, Brier: {stats_s6_bin_all['brier']:.3f}")

    s6_bins_a = stats_s6_all["bins"]
    ax_c_a.plot([b["mean_pred"] for b in s6_bins_a if b["count"] > 0],
                [b["mean_obs"] for b in s6_bins_a if b["count"] > 0],
                "^--", color="#ff7f0e", linewidth=1.8, markersize=6.5, alpha=0.85,
                label=f"S6 Clef (score/2) — ECE: {stats_s6_all['ece']:.3f}")

    if stats_s4_all:
        s4_bins_a = stats_s4_all["bins"]
        ax_c_a.plot([b["mean_pred"] for b in s4_bins_a if b["count"] > 0],
                    [b["mean_obs"] for b in s4_bins_a if b["count"] > 0],
                    "o:", color="#2ca02c", linewidth=1.8, markersize=6, alpha=0.85,
                    label=rf"S4 Cross-Encoder ($\sigma(z)$) — ECE: {stats_s4_all['ece']:.3f}")

    ax_c_a.set_xlim(-0.02, 1.02)
    ax_c_a.set_ylim(-0.02, 1.02)
    ax_c_a.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax_c_a.set_ylabel("Observed Relevant Fraction (TP)", fontsize=11, fontweight="bold")
    ax_c_a.set_title(f"(b) All Candidates: Unjudged as Zero (N={n_total})", fontsize=12, pad=10, fontweight="bold")
    ax_c_a.legend(loc="upper left", frameon=True, fontsize=8.8)
    ax_c_a.grid(True, linestyle=":", alpha=0.6)

    # Histogram Right
    ax_h_a.bar([c - width/2 for c in bin_centers], [b["count"] for b in s5_bins_a], width=width,
               color="#1f77b4", alpha=0.85, label="S5 (noul)")
    ax_h_a.bar([c + width/2 for c in bin_centers], [b["count"] for b in s6_bin_bins_a], width=width,
               color="#d62728", alpha=0.85, label="S6 P(1)+P(2)")
    ax_h_a.set_xlim(-0.02, 1.02)
    ax_h_a.set_xlabel("Probability Bin", fontsize=10)
    ax_h_a.set_ylabel("Candidates", fontsize=10)
    ax_h_a.set_title("Candidate Distribution across Bins (Full Pool)", fontsize=9.5, pad=5)
    ax_h_a.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.2)
    ax_h_a.grid(True, linestyle=":", alpha=0.5)

    plot_path = results_dir / "calibration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved reliability diagram -> {plot_path}")

    # -------------------------------------------------------------
    # Persist Comprehensive Calibration Deliverables
    # -------------------------------------------------------------
    metrics_summary = {
        "dataset": cfg.get("dataset", {}).get("ir_id", "cord19/trec-covid"),
        "total_candidates": n_total,
        "judged_candidates": n_judged,
        "judged_fraction": round(judged_pct / 100.0, 4),
        "judged_only": {
            "S5_clef_noul": {
                "ece": stats_s5_judged["ece"],
                "brier": stats_s5_judged["brier"],
                "optimal_threshold": thresh_s5_j,
                "optimal_f1": f1_s5_j,
                "threshold_transfer_analysis": transfer_s5_j,
                "bins": stats_s5_judged["bins"]
            },
            "S6_clef_score": {
                "ece": stats_s6_judged["ece"],
                "brier": stats_s6_judged["brier"],
                "bins": stats_s6_judged["bins"]
            },
            "S6_clef_score_binary": {
                "ece": stats_s6_bin_judged["ece"],
                "brier": stats_s6_bin_judged["brier"],
                "bins": stats_s6_bin_judged["bins"]
            },
            "S4_cross_encoder": {
                "ece": stats_s4_judged["ece"] if stats_s4_judged else None,
                "brier": stats_s4_judged["brier"] if stats_s4_judged else None
            } if stats_s4_judged else None
        },
        "all_candidates_unjudged_as_zero": {
            "S5_clef_noul": {
                "ece": stats_s5_all["ece"],
                "brier": stats_s5_all["brier"],
                "optimal_threshold": thresh_s5_all,
                "optimal_f1": f1_s5_all,
                "threshold_transfer_analysis": transfer_s5_all,
                "bins": stats_s5_all["bins"]
            },
            "S6_clef_score": {
                "ece": stats_s6_all["ece"],
                "brier": stats_s6_all["brier"],
                "bins": stats_s6_all["bins"]
            },
            "S6_clef_score_binary": {
                "ece": stats_s6_bin_all["ece"],
                "brier": stats_s6_bin_all["brier"],
                "bins": stats_s6_bin_all["bins"]
            },
            "S4_cross_encoder": {
                "ece": stats_s4_all["ece"] if stats_s4_all else None,
                "brier": stats_s4_all["brier"] if stats_s4_all else None
            } if stats_s4_all else None
        }
    }
    dump_json(results_dir / "calibration_metrics.json", metrics_summary)
    log(f"Saved calibration metrics -> {results_dir / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
