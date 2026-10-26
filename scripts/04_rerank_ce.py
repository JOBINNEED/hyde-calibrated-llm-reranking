"""Stage 4a — Supervised Cross-Encoder Re-Ranking Baseline (S4).

Re-ranks the 5,000 frozen candidates in artifacts/candidates.jsonl using
cross-encoder/ms-marco-MiniLM-L-6-v2.

Outputs:
  runs/S4_ce.trec (Contract C standard TREC format: qid Q0 docno rank score S4_ce)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from common import log, load_config, load_jsonl, write_trec_6col
from load_data import load_dataset
from evaluate import build_qrels, load_runs, evaluate_system, print_metrics_table, summarize


def main():
    t0 = time.time()
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    art_dir = root / "artifacts"
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cand_path = art_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing {cand_path}. Run Stage 3 first.")

    candidates = load_jsonl(cand_path)
    log(f"Loaded {len(candidates)} candidates from {cand_path}")

    # Load corpus documents for text lookup
    log("Loading corpus documents...")
    docs, queries, qrels = load_dataset(cfg)

    # Initialize cross-encoder
    model_name = cfg.get("cross_encoder", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CrossEncoder '{model_name}' on {device}...")
    ce = CrossEncoder(model_name, max_length=512, device=device)

    # Build (query, doc_text) pairs
    pairs = []
    missing_docs = 0
    for c in candidates:
        q_text = c.get("query", "")
        docno = c["docno"]
        d_text = docs.get(docno, "")
        if not d_text:
            missing_docs += 1
        pairs.append((q_text, d_text))

    if missing_docs > 0:
        log(f"WARN: {missing_docs} candidates had empty document text")

    log(f"Scoring {len(pairs)} pairs with CrossEncoder (batch_size=128)...")
    scores = ce.predict(pairs, batch_size=128, show_progress_bar=True)

    # Attach scores to candidate dicts and group by qid
    by_qid = {}
    for c, score in zip(candidates, scores):
        qid = str(c["qid"])
        item = {
            "qid": qid,
            "docno": str(c["docno"]),
            "score": float(score),
            "stage_a_rank": int(c.get("stage_a_rank", 9999)),
        }
        by_qid.setdefault(qid, []).append(item)

    # Sort each topic descending by score; break ties by stage_a_rank
    run_records = []
    for qid in sorted(by_qid.keys(), key=lambda x: int(x) if x.isdigit() else x):
        topic_cands = by_qid[qid]
        topic_cands.sort(key=lambda x: (-x["score"], x["stage_a_rank"]))

        for rank, c in enumerate(topic_cands, start=1):
            run_records.append((qid, c["docno"], rank, c["score"], "S4_ce"))

    out_trec = runs_dir / "S4_ce.trec"
    n_written = write_trec_6col(out_trec, run_records)
    dur = time.time() - t0
    log(f"Wrote {n_written} lines to {out_trec} in {dur:.1f}s")

    # Sanity checks
    assert n_written == len(candidates), f"Expected {len(candidates)} lines, wrote {n_written}"
    assert len(by_qid) == 50, f"Expected 50 topics, got {len(by_qid)}"
    for qid, items in by_qid.items():
        assert len(items) == 100, f"Topic {qid} has {len(items)} items, expected 100"

    # Evaluate against qrels
    qrels_run = build_qrels(qrels)
    s4_run = load_runs([("S4 Cross-encoder", out_trec)])["S4 Cross-encoder"]
    metrics = evaluate_system(qrels_run, s4_run, cfg["eval"]["metrics"])
    log("=== S4 Cross-Encoder Metrics ===")
    print_metrics_table({"S4 Cross-encoder": summarize(metrics)})


if __name__ == "__main__":
    main()
