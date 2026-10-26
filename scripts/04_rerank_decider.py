"""Stage 4b/4c — Decision-Model Re-Ranking (S5 noul, S6 score).

Queries the live vLLM decision endpoint (/v1/systemone) for the 5,000 frozen
candidates in artifacts/candidates.jsonl, evaluating both 'noul' and 'score'
questions simultaneously in concurrent batches.

Outputs:
  artifacts/clef_raw.jsonl   (Contract B schema: raw decision probabilities)
  runs/S5_clef_noul.trec     (Standard TREC: qid Q0 docno rank score S5_clef_noul)
  runs/S6_clef_score.trec    (Standard TREC: qid Q0 docno rank score S6_clef_score)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import ir_datasets
from common import (log, load_config, load_jsonl, write_trec_6col,
                    append_jsonl)
from evaluate import build_qrels, load_runs, evaluate_system, print_metrics_table, summarize


def get_requests_session(pool_connections=64, pool_maxsize=64):
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retries
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_doc_metadata(dataset_id: str):
    log(f"Loading document metadata (title, abstract) from ir_datasets: {dataset_id}...")
    t0 = time.time()
    ds = ir_datasets.load(dataset_id)
    doc_meta = {}
    for d in ds.docs_iter():
        doc_id = getattr(d, "docno", None) or getattr(d, "doc_id", None)
        if doc_id:
            title = getattr(d, "title", "") or ""
            abstract = getattr(d, "abstract", "") or ""
            if not title and not abstract:
                abstract = getattr(d, "text", "") or ""
            doc_meta[doc_id] = (title.strip(), abstract.strip())
    log(f"Loaded {len(doc_meta)} documents in {time.time() - t0:.2f}s")
    return doc_meta


def build_state(query: str, question: str, title: str, abstract: str, max_state_chars: int):
    truncated = False
    if len(abstract) > max_state_chars:
        abstract = abstract[:max_state_chars]
        truncated = True

    state = (
        f"TOPIC: {query}\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENT TITLE: {title}\n\n"
        f"DOCUMENT: {abstract}"
    )
    return state, truncated, len(state)


def query_candidate(session, endpoint, model_name, candidate, doc_meta, max_chars, questions_spec):
    qid = str(candidate["qid"])
    docno = str(candidate["docno"])
    stage_a_rank = int(candidate.get("stage_a_rank", 9999))
    query = candidate.get("query", "")
    question = candidate.get("question", "")

    title, abstract = doc_meta.get(docno, ("", ""))
    state, truncated, state_chars = build_state(query, question, title, abstract, max_chars)

    payload = {
        "model": model_name,
        "state": state,
        "questions": questions_spec
    }

    t0 = time.time()
    resp = session.post(endpoint, json=payload, timeout=60)
    latency_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"Endpoint returned {resp.status_code} for ({qid}, {docno}): {resp.text[:200]}")

    data = resp.json()
    answers = data.get("answers", {})

    noul_ans = answers.get("noul", {})
    prob_noul = noul_ans.get("noul")
    if prob_noul is None:
        prob_noul = noul_ans.get("probabilities", {}).get("yes", 0.0)
    conf_noul = noul_ans.get("confidence", prob_noul)

    score_ans = answers.get("score", {})
    val_score = score_ans.get("score", 0.0)
    probs_score = score_ans.get("probabilities", {})

    # Record conforming to Contract B schema
    record = {
        "qid": qid,
        "docid": docno,
        "model": model_name,
        "question_type": "noul",
        "probability": float(prob_noul),
        "confidence": float(conf_noul),
        "score": float(val_score),
        "score_probabilities": probs_score,
        "stage_a_rank": stage_a_rank,
        "latency_ms": round(latency_ms, 1),
        "state_chars": state_chars,
        "truncated": bool(truncated)
    }
    return record


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent.parent
    art_dir = root / "artifacts"
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    cand_path = art_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing {cand_path}. Run Stage 3 first.")

    candidates = load_jsonl(cand_path)
    log(f"Loaded {len(candidates)} candidates from {cand_path}")

    # Configuration tunables
    endpoint = cfg.get("reranker", {}).get("base_url", "http://localhost:8000/v1").rstrip("/") + "/systemone"
    model_name = cfg.get("reranker", {}).get("model_active", "local-active")
    max_chars = cfg.get("reranker", {}).get("max_state_chars", 6000)
    num_workers = 32

    # Questions specification querying both noul and score
    questions_spec = {
        "noul": {
            "type": "noul",
            "instructions": "Does this document provide relevant information to answer the question?"
        },
        "score": {
            "type": "score",
            "instructions": "Rate how relevant this document is to the question.",
            "criteria": [
                "0: Not relevant",
                "1: Partially relevant",
                "2: Fully relevant"
            ]
        }
    }

    # Load doc metadata
    ds_id = cfg.get("dataset", {}).get("ir_id", "cord19/trec-covid")
    doc_meta = load_doc_metadata(ds_id)

    # Check for existing clef_raw.jsonl to support checkpoint / resume
    raw_path = art_dir / "clef_raw.jsonl"
    existing_records = []
    done_keys = set()
    if raw_path.exists():
        try:
            existing_records = load_jsonl(raw_path)
            for r in existing_records:
                done_keys.add((str(r["qid"]), str(r["docid"])))
            log(f"Found {len(existing_records)} existing decisions in {raw_path}")
        except Exception as e:
            log(f"Could not load existing {raw_path}: {e}")

    remaining_candidates = [
        c for c in candidates
        if (str(c["qid"]), str(c["docno"])) not in done_keys
    ]
    log(f"Need to process {len(remaining_candidates)} of {len(candidates)} candidates")

    session = get_requests_session(pool_connections=num_workers, pool_maxsize=num_workers * 2)

    all_records = list(existing_records)
    if remaining_candidates:
        log(f"Executing decision model queries at {endpoint} with {num_workers} concurrent workers...")
        t_start = time.time()
        completed = 0
        batch_buffer = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_cand = {
                executor.submit(
                    query_candidate,
                    session,
                    endpoint,
                    model_name,
                    c,
                    doc_meta,
                    max_chars,
                    questions_spec
                ): c for c in remaining_candidates
            }

            for future in as_completed(future_to_cand):
                c = future_to_cand[future]
                try:
                    record = future.result()
                    all_records.append(record)
                    batch_buffer.append(record)
                    completed += 1

                    if len(batch_buffer) >= 100:
                        append_jsonl(raw_path, batch_buffer)
                        batch_buffer = []
                        elapsed = time.time() - t_start
                        rate = completed / elapsed
                        log(f"Progress: {completed}/{len(remaining_candidates)} ({rate:.1f} decisions/sec)")

                except Exception as exc:
                    log(f"ERROR querying ({c.get('qid')}, {c.get('docno')}): {exc}")
                    raise

            # Flush remaining buffer
            if batch_buffer:
                append_jsonl(raw_path, batch_buffer)

        tot_time = time.time() - t_start
        log(f"Completed {len(remaining_candidates)} decisions in {tot_time:.1f}s ({len(remaining_candidates)/tot_time:.1f} req/s)")

    # Sanity check decisions
    log(f"Total decisions collected: {len(all_records)}")
    assert len(all_records) == 5000, f"Expected 5000 decisions, got {len(all_records)}"

    # Group decisions by qid
    by_qid = {}
    for r in all_records:
        qid = str(r["qid"])
        by_qid.setdefault(qid, []).append(r)

    # Produce runs/S5_clef_noul.trec
    log("Building runs/S5_clef_noul.trec...")
    s5_records = []
    for qid in sorted(by_qid.keys(), key=lambda x: int(x) if x.isdigit() else x):
        topic_cands = list(by_qid[qid])
        # Sort by noul probability descending, break ties by stage_a_rank ascending
        topic_cands.sort(key=lambda x: (-x["probability"], x["stage_a_rank"]))
        for rank, c in enumerate(topic_cands, start=1):
            s5_records.append((qid, c["docid"], rank, c["probability"], "S5_clef_noul"))

    out_s5 = runs_dir / "S5_clef_noul.trec"
    n5 = write_trec_6col(out_s5, s5_records)
    log(f"Wrote {n5} lines to {out_s5}")

    # Produce runs/S6_clef_score.trec
    log("Building runs/S6_clef_score.trec...")
    s6_records = []
    for qid in sorted(by_qid.keys(), key=lambda x: int(x) if x.isdigit() else x):
        topic_cands = list(by_qid[qid])
        # Sort by expected score descending, break ties by stage_a_rank ascending
        topic_cands.sort(key=lambda x: (-x["score"], x["stage_a_rank"]))
        for rank, c in enumerate(topic_cands, start=1):
            s6_records.append((qid, c["docid"], rank, c["score"], "S6_clef_score"))

    out_s6 = runs_dir / "S6_clef_score.trec"
    n6 = write_trec_6col(out_s6, s6_records)
    log(f"Wrote {n6} lines to {out_s6}")

    # Evaluate S5 and S6 against qrels
    from load_data import load_dataset
    _, _, qrels = load_dataset(cfg)
    qrels_run = build_qrels(qrels)

    runs = load_runs([
        ("S5 Clef noul", out_s5),
        ("S6 Clef score", out_s6)
    ])
    results = {}
    for sid, run in runs.items():
        m = evaluate_system(qrels_run, run, cfg["eval"]["metrics"])
        results[sid] = summarize(m)

    log("=== S5 and S6 Evaluation Metrics ===")
    print_metrics_table(results)


if __name__ == "__main__":
    main()
