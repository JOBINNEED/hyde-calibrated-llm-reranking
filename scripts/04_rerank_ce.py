"""Stage 4a — Supervised Cross-Encoder Re-Ranking Baseline (S4).

Re-ranks the 5,000 frozen candidates in artifacts/candidates.jsonl using
cross-encoder/ms-marco-MiniLM-L-6-v2.

INFORMATION PARITY (see CHANGELOG 2026-10-27, CL-001)
-----------------------------------------------------
The decision model in Stage 4b/4c builds its state from BOTH topic fields:

    TOPIC: {query}          <- terse keyword title, "coronavirus origin"
    QUESTION: {question}    <- natural language, "what is the origin of COVID-19"
    DOCUMENT TITLE: {title}
    DOCUMENT: {abstract}

The original version of this script passed only `query` to the cross-encoder,
so S4 was scored on strictly less information than S5/S6 -- and on the input
distribution ms-marco-MiniLM handles worst, since it was trained on
natural-language questions. Any S5/S6-over-S4 margin measured that way is
confounded with the input difference rather than attributable to the model.

The CE query side is now composed from `cross_encoder.query_fields` (default:
both fields), making S4 an information-matched baseline. Single-field variants
remain runnable as an explicit ablation:

    python 04_rerank_ce.py                                   # S4_ce.trec   (query+question)
    python 04_rerank_ce.py --query-fields query    --tag query_only
    python 04_rerank_ce.py --query-fields question --tag question_only

Report the STRONGEST cross-encoder configuration as the baseline. A claim to
beat a supervised re-ranker has to beat its best setting, not a crippled one.

Outputs:
  runs/S4_ce.trec (Contract C standard TREC format: qid Q0 docno rank score S4_ce)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from common import log, load_config, load_jsonl, write_trec_6col
from load_data import load_dataset
from evaluate import build_qrels, load_runs, evaluate_system, print_metrics_table, summarize

VALID_FIELDS = ("query", "question", "narrative")


def build_query_text(candidate: dict, fields) -> str:
    """Compose the CE query side from the named candidate fields, in order.

    Empty/missing fields are skipped so a topic lacking `question` degrades to
    `query` alone rather than emitting a dangling separator.
    """
    parts = []
    for f in fields:
        v = (candidate.get(f) or "").strip()
        if v:
            parts.append(v)
    return ". ".join(parts)


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 4a cross-encoder re-ranking baseline (S4).")
    ap.add_argument("--query-fields", nargs="+", default=None, metavar="FIELD",
                    help=f"Candidate fields forming the CE query side, in order. "
                         f"Choices: {', '.join(VALID_FIELDS)}. "
                         f"Default: cross_encoder.query_fields from config.yaml.")
    ap.add_argument("--tag", default=None,
                    help="Variant suffix for the output file: runs/S4_ce_<tag>.trec. "
                         "Omit for the canonical runs/S4_ce.trec.")
    return ap.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
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

    ce_cfg = cfg.get("cross_encoder", {})

    # ---- resolve the query-side composition (CLI overrides config) ----
    query_fields = args.query_fields or ce_cfg.get("query_fields", ["query", "question"])
    bad = [f for f in query_fields if f not in VALID_FIELDS]
    if bad:
        raise ValueError(f"Unknown query field(s) {bad}. Valid: {list(VALID_FIELDS)}")
    log(f"CE query side composed from fields: {query_fields}")

    # ---- output path: canonical, or a tagged variant ----
    out_name = f"S4_ce_{args.tag}.trec" if args.tag else "S4_ce.trec"
    run_tag = f"S4_ce_{args.tag}" if args.tag else "S4_ce"
    out_trec = runs_dir / out_name
    log(f"Output -> {out_trec} (run tag '{run_tag}')")

    # Load corpus documents for text lookup
    log("Loading corpus documents...")
    docs, queries, qrels = load_dataset(cfg)

    # Initialize cross-encoder
    model_name = ce_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_length = int(ce_cfg.get("max_length", 512))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading CrossEncoder '{model_name}' (max_length={max_length}) on {device}...")
    ce = CrossEncoder(model_name, max_length=max_length, device=device)

    # Build (query, doc_text) pairs
    pairs = []
    missing_docs = 0
    empty_queries = 0
    for c in candidates:
        q_text = build_query_text(c, query_fields)
        if not q_text:
            empty_queries += 1
        docno = c["docno"]
        d_text = docs.get(docno, "")
        if not d_text:
            missing_docs += 1
        pairs.append((q_text, d_text))

    if missing_docs:
        log(f"WARN: {missing_docs} candidates had empty document text")
    if empty_queries:
        log(f"WARN: {empty_queries} candidates produced an empty query side")

    # Show exactly what the model is being fed, so a field-composition mistake
    # is visible in the log instead of silently changing the baseline.
    log(f"EXAMPLE CE query side -> {pairs[0][0][:200]!r}")
    log(f"EXAMPLE CE doc side   -> {pairs[0][1][:150]!r}")

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
            run_records.append((qid, c["docno"], rank, c["score"], run_tag))

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
    sid = f"S4 Cross-encoder ({'+'.join(query_fields)})"
    s4_run = load_runs([(sid, out_trec)])[sid]
    metrics = evaluate_system(qrels_run, s4_run, cfg["eval"]["metrics"])
    log("=== S4 Cross-Encoder Metrics ===")
    print_metrics_table({sid: summarize(metrics)})


if __name__ == "__main__":
    main()
