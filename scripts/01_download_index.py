"""Stage 1 — download index the corpus, run BM25 baseline + acceptance gate,
then encode the full corpus with Contriever into a FAISS index.

Gate: S1 (BM25) nDCG@10 must land near the published BEIR figure before we
generate a single hypothetical document. This validates the loader, the qrels
alignment, and the evaluation call all at once.

Outputs:
  data/cache/corpus_emb.npy     (N, 768) float32  -- normalised embeddings
  data/cache/docids.json        row index -> docno
  data/cache/faiss.index        FAISS Flat inner-product index
  runs/S1_bm25.trec             BM25 run (acceptance)
"""
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pathlib import Path

from common import log, load_config, write_trec, load_json, dump_json, tokenize
from load_data import load_dataset
from evaluate import build_qrels, load_runs, evaluate_system, summarize, print_metrics_table


def build_bm25(docs):
    """Return (BM25 model, list of tokenized docs, docno list)."""
    docno_list = list(docs)
    tokenized = [tokenize(docs[d]) for d in docno_list]
    bm25 = BM25Okapi(tokenized)
    return bm25, docno_list


def bm25_query(qid, bm25, docno_list, query_tokens, top_k):
    scores = bm25.get_scores(query_tokens)
    top = np.argsort(scores)[::-1][:top_k]
    out = []
    for i in top:
        if scores[i] > 0:
            out.append((qid, docno_list[i], float(scores[i])))
    return out


def main():
    cfg = load_config()
    ROOT = Path(__file__).resolve().parent.parent
    runs_dir = ROOT / "runs"
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    docs, queries, qrels = load_dataset(cfg)

    # ---- shape sanity (handbook Stage 1 acceptance) ----
    n_expected_docs = 171332
    n_expected_q = 50
    if len(docs) != n_expected_docs:
        log(f"WARN corpus has {len(docs)} docs (expected ~{n_expected_docs})")
    if len(queries) != n_expected_q:
        log(f"WARN queries has {len(queries)} (expected {n_expected_q})")

    # ---- BM25 baseline + acceptance ----
    bm25, docno_list = build_bm25(docs)
    log(f"BM25 over {len(docno_list)} docs")

    rankings = []
    for qid, q in queries.items():
        toks = tokenize(q["query"])
        for _, docno, score in bm25_query(qid, bm25, docno_list, toks, cfg["retrieval"]["top_k"]):
            rankings.append((qid, docno, score))
    write_trec(runs_dir / "S1_bm25.trec", rankings)

    qrels_run = build_qrels(qrels)
    run_bm25 = load_runs([(f"{cfg['dataset']['name']}_bm25", runs_dir / 'S1_bm25.trec')])
    metrics = evaluate_system(qrels_run, run_bm25[f"{cfg['dataset']['name']}_bm25"],
                              cfg["eval"]["metrics"])
    log(f"S1 BM25 metrics: {metrics}")
    print_metrics_table({f"S1 BM25": summarize(metrics)})

    ref = cfg["eval"]["bm25_reference_ndcg10"]
    got = metrics.get("ndcg@10")
    if got is not None:
        if abs(got - ref) <= 0.03:
            log(f"ACCEPTANCE OK: BM25 nDCG@10={got:.4f} ~ reference {ref}")
        else:
            log(f"!! BM25 nDCG@10={got:.4f} is off from reference {ref}. "
                f"Check qrels alignment / docid mapping / split.")

    # ---- encode corpus with Contriever (GPU) -> FAISS ----
    enc = cfg["encoder"]
    device = "cuda" if _cuda_ok() else "cpu"
    log(f"encoder device: {device}")
    model = SentenceTransformer(enc["model"], device=device)
    # ST>=6 no longer takes max_length on encode(); set it on the model/tokenizer.
    model.max_length = int(enc["max_length"])

    docno_all = list(docs)
    texts_all = [docs[d] for d in docno_all]
    log(f"encoding {len(texts_all)} documents, batch={enc['batch_size']}")
    embs = model.encode(texts_all, batch_size=enc["batch_size"],
                        normalize_embeddings=True,
                        show_progress_bar=True)
    embs = embs.astype("float32")
    log(f"embeddings shape: {embs.shape}")

    # ---- persist (write both files in the same loop == same order) ----
    np.save(cache_dir / "corpus_emb.npy", embs)
    dump_json(cache_dir / "docids.json", docno_all)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(cache_dir / "faiss.index"))
    log(f"saved corpus_emb.npy, docids.json, faiss.index to {cache_dir}")


def _cuda_ok():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
