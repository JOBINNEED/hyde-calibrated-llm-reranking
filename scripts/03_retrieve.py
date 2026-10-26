"""Stage 3 — Embed, average, and retrieve.

Produces:
  runs/S2_dense.trec     (query-only dense retrieval, reference)
  runs/S3_hyde.trec      (HyDE: mean of query + N generated-passage vectors)
  artifacts/candidates.jsonl   frozen top-k candidates per topic + query text

The frozen candidate set is the controlled input to every re-ranker in Stage 4.
Only one model (Contriever) is resident here; exit frees the GPU.
"""
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from common import (log, load_config, write_trec, append_jsonl,
                    load_jsonl, load_json)
from load_data import load_dataset
from evaluate import build_qrels, load_runs, evaluate_system, print_metrics_table


def hyde_vector(model, query_text, gen_texts):
    """Mean of the query vector and the N generated-passage vectors, L2-normalised."""
    parts = [model.encode([query_text], normalize_embeddings=True)[0]]
    for t in gen_texts:
        parts.append(model.encode([t], normalize_embeddings=True)[0])
    v = np.mean(parts, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def main():
    cfg = load_config()
    ROOT = Path(__file__).resolve().parent.parent
    cache_dir = ROOT / "data" / "cache"
    runs_dir = ROOT / "runs"
    art_dir = ROOT / "artifacts"
    for d in (runs_dir, art_dir):
        d.mkdir(parents=True, exist_ok=True)

    _, queries, qrels = load_dataset(cfg)

    embs = np.load(cache_dir / "corpus_emb.npy")
    docids = load_json(cache_dir / "docids.json")
    index = faiss.read_index(str(cache_dir / "faiss.index"))
    log(f"corpus index rows: {len(docids)}, dim: {embs.shape[1]}")

    device = "cuda" if _cuda_ok() else "cpu"
    model = SentenceTransformer(cfg["encoder"]["model"], device=device)
    model.max_length = int(cfg["encoder"]["max_length"])

    hyde_docs = load_jsonl(art_dir / "hyde_docs.jsonl")
    gens_by_qid = {}
    for r in hyde_docs:
        gens_by_qid.setdefault(r["qid"], []).append(r["text"])

    top_k = cfg["retrieval"]["top_k"]
    rankings_s2, rankings_s3, candidates = [], [], []

    for qid, q in sorted(queries.items()):
        q_vec = model.encode([q["query"]], normalize_embeddings=True)[0].reshape(1, -1)

        # S2 — query-only dense
        _, i2 = index.search(q_vec, top_k)
        for rank, row in enumerate(i2[0]):
            score = float(np.dot(q_vec[0], embs[row]))
            rankings_s2.append((qid, docids[row], score))

        # S3 — HyDE
        gen_texts = gens_by_qid.get(qid, [])
        if gen_texts:
            hv = hyde_vector(model, q["query"], gen_texts).reshape(1, -1)
            _, i3 = index.search(hv, top_k)
            for rank, row in enumerate(i3[0]):
                score = float(np.dot(hv[0], embs[row]))
                rankings_s3.append((qid, docids[row], score))
                candidates.append({
                    "qid": qid,
                    "docno": docids[row],
                    "stage_a_rank": rank + 1,
                    "query": q["query"],
                    "question": q["question"],
                })
        else:
            log(f"WARN q={qid}: no generations — S3 skipped")

    write_trec(runs_dir / "S2_dense.trec", rankings_s2)
    write_trec(runs_dir / "S3_hyde.trec", rankings_s3)
    append_jsonl(art_dir / "candidates.jsonl", candidates)

    # ---- evaluate S2 vs S3 ----
    qrels_run = build_qrels(qrels)
    results = {}
    for tag, runpath in (
        ("S2 dense", runs_dir / "S2_dense.trec"),
        ("S3 HyDE", runs_dir / "S3_hyde.trec"),
    ):
        r = load_runs([(tag, runpath)])[tag]
        m = evaluate_system(qrels_run, r, cfg["eval"]["metrics"])
        results[tag] = {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
        log(f"{tag}: {results[tag]}")
    print_metrics_table(results)

    log(f"frozen top-{top_k} candidates -> {art_dir / 'candidates.jsonl'}")


def _cuda_ok():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
