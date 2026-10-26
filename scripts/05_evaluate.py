"""Stage 5 — Evaluate all systems, significance, and persist results.

Reads every run file that exists, evaluates against the qrels, writes:
  results/metrics.csv      (long format: system, metric, value)
  results/significance.txt (paired Student's t over topics, per ranx.compare)

Run systems are discovered from files present in runs/ so the same script works
for Stage 1 (S1-S3) and Stage 4 (add S4-S7) without edits.
"""
import csv
from pathlib import Path

from common import log, load_config, ROOT, dump_json
from load_data import load_dataset
from evaluate import build_qrels, load_runs, evaluate_system, compare_systems, print_metrics_table

# (system id, run file) — only those whose files exist get evaluated.
SYSTEMS = [
    ("S1 BM25", "S1_bm25.trec"),
    ("S2 Dense", "S2_dense.trec"),
    ("S3 HyDE", "S3_hyde.trec"),
    ("S4 Cross-encoder", "S4_ce.trec"),
    ("S5 Clef noul", "S5_clef_noul.trec"),
    ("S6 Clef score", "S6_clef_score.trec"),
    ("S7 Clef flash", "S7_clef_flash.trec"),
]


def main():
    cfg = load_config()
    ROOT = Path(__file__).resolve().parent.parent
    runs_dir = ROOT / "runs"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _, _, qrels = load_dataset(cfg)
    qrels_run = build_qrels(qrels)

    active = [(sid, runs_dir / fname) for sid, fname in SYSTEMS
              if (runs_dir / fname).exists()]
    if not active:
        log("No run files found in runs/. Nothing to evaluate.")
        return

    runs = load_runs(active)
    results = {}
    for sid, run in runs.items():
        m = evaluate_system(qrels_run, run, cfg["eval"]["metrics"])
        results[sid] = {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
    log("=== metrics table ===")
    print_metrics_table(results)

    # ---- significance across all active systems (paired t over topics) ----
    report = compare_systems(qrels_run, list(runs.values()),
                             [m for m in cfg["eval"]["metrics"] if m != "recall@100"],
                             cfg["eval"]["max_p"])
    (results_dir / "significance.txt").write_text(
        str(report) if not hasattr(report, "write") else "", encoding="utf-8")
    log("wrote results/significance.txt")

    # ---- persist metrics (long format) ----
    metrics_path = results_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "value"])
        for sid in results:
            for metric, value in results[sid].items():
                w.writerow([sid, metric, value])
    log(f"wrote {metrics_path} ({len(results)} systems)")


if __name__ == "__main__":
    main()
