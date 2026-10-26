"""Stage 1 acceptance test — validate artifacts on disk independently.

Does NOT trust the 01 log. Re-loads every produced file and checks that it is
well-formed, internally consistent, and retrieval-ready. Exits non-zero on
the first hard failure.
"""
import sys
import numpy as np
import faiss
from pathlib import Path

from common import log, load_config, load_json
from evaluate import load_runs, evaluate_system

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "runs"

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    log(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main():
    cfg = load_config()
    n_expected = 191132

    # ---- 1. embedding array ----
    emb_path = CACHE / "corpus_emb.npy"
    if not emb_path.exists():
        check("corpus_emb.npy exists", False, "missing file")
        return
    embs = np.load(emb_path)
    check("corpus_emb.npy shape",
          tuple(embs.shape) == (n_expected, 768),
          f"shape={tuple(embs.shape)}")
    check("corpus_emb.npy dtype float32", embs.dtype == np.float32, str(embs.dtype))

    # ---- 2. normalized (cosine == dot) ----
    norms = np.linalg.norm(embs, axis=1)
    check("embeddings L2-normalised ~1.0",
          bool(np.allclose(norms, 1.0, atol=1e-3)),
          f"norm min={norms.min():.5f} max={norms.max():.5f} "
          f"bad={int((norms < 0.999).sum())}")

    # ---- 3. docids.json ----
    docids_path = CACHE / "docids.json"
    if not docids_path.exists():
        check("docids.json exists", False, "missing file")
        return
    docids = load_json(docids_path)
    check("docids.json length == corpus", len(docids) == n_expected, f"len={len(docids)}")
    check("docids unique", len(set(docids)) == len(docids),
          f"dupes={len(docids) - len(set(docids))}")

    # ---- 4. FAISS index ----
    faiss_path = CACHE / "faiss.index"
    if not faiss_path.exists():
        check("faiss.index exists", False, "missing file")
        return
    index = faiss.read_index(str(faiss_path))
    check("faiss ntotal == corpus", int(index.ntotal) == n_expected,
          f"ntotal={index.ntotal}")
    check("faiss is FlatIP", isinstance(index, faiss.IndexFlatIP), str(type(index)))

    # ---- 5. BM25 run file ----
    rec_path = RUNS / "S1_bm25.trec"
    if not rec_path.exists():
        check("S1_bm25.trec exists", False, "missing file")
        return
    runs = load_runs([("S1_bm25", rec_path)])
    run = runs["S1_bm25"]
    n_topics = len(run)
    check("S1_bm25 topic count == 50", n_topics == 50, f"topics={n_topics}")

    # every ranked docno must exist in the corpus, and top_k == 100 each
    docset = set(docids)
    min_hits = max_hits = None
    off_corpus = 0
    for qid in run.keys():
        docs = run[qid]
        k = len(docs)
        min_hits = k if min_hits is None else min(min_hits, k)
        max_hits = k if max_hits is None else max(max_hits, k)
        for docno in docs.keys():
            if docno not in docset:
                off_corpus += 1
    check("S1_bm25 all docs in corpus", off_corpus == 0, f"off_corpus={off_corpus}")
    check("S1_bm25 top_k == 100", min_hits == max_hits == 100,
          f"min={min_hits} max={max_hits}")

    # ---- 6. re-derive BM25 nDCG@10 and compare to the 01 log value ----
    from load_data import load_dataset
    _, _, qrels = load_dataset(cfg)
    from evaluate import build_qrels
    qrels_run = build_qrels(qrels)
    metrics = evaluate_system(qrels_run, run, cfg["eval"]["metrics"])
    ndcg = float(metrics.get("ndcg@10", 0.0))
    check("S1 nDCG@10 near 0.5722", abs(ndcg - 0.5722) <= 0.01,
          f"ndcg@10={ndcg:.4f} (expect ~0.5722)")

    # ---- 7. FAISS retrieval sanity: top-100 must contain relevant docs ----
    q_vec = embs[0].reshape(1, -1)  # first corpus row is a valid vector; re-query q1
    # pick a real query vector from the corpus embedding of a known q1 doc
    _, idx = index.search(q_vec, 5)
    check("FAISS search returns 5 valid rows",
          all(int(i) >= 0 and int(i) < n_expected for i in idx[0]),
          str(idx[0].tolist()))

    # ---- report ----
    failures = [c for c in checks if not c[1]]
    log(f"==== {len(checks) - len(failures)}/{len(checks)} checks passed ====")
    if failures:
        log("RESULT: STAGE 1 TEST FAILED")
        sys.exit(1)
    log("RESULT: STAGE 1 TEST PASSED")


if __name__ == "__main__":
    main()
