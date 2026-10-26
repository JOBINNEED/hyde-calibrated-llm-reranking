"""Load TREC-COVID corpus, queries, and qrels via ir_datasets.

Returns plain dicts (docno->text, qid->query dict, qrels dict) so no other
stage depends on ir_datasets at runtime.
"""
import ir_datasets
from common import log, load_config


def resolve_dataset(cfg):
    """Return an ir_datasets dataset object, trying the configured id, then
    listing available ids so a typo surfaces as a clear error."""
    wanted = cfg["dataset"]["ir_id"]
    try:
        ds = ir_datasets.load(wanted)
        log(f"loaded dataset id={wanted}")
        return ds
    except Exception as e:
        known = [
            "cord19/trec-covid",           # complete: 50 topics, deep 0/1/2 judgments
            "cord19/fulltext/trec-covid",  # same, with full text
            "cord19/trec-covid/round1", "cord19/trec-covid/round2",
            "cord19/trec-covid/round3", "cord19/trec-covid/round4",
        ]
        raise RuntimeError(
            f"Could not load ir_datasets id '{wanted}'. {e}\n"
            f"Known COVID dataset ids: {known}"
        )


def doc_text(doc):
    """Build retrieval text from a corpus document (title + abstract preferred)."""
    title = getattr(doc, "title", "") or ""
    abstract = getattr(doc, "abstract", "") or ""
    text = getattr(doc, "text", None)
    parts = [p.strip() for p in (title, abstract) if p.strip()]
    if parts:
        return " ".join(parts)
    # Some datasets only expose .text
    return text or ""


def load_dataset(cfg):
    ds = resolve_dataset(cfg)

    docs = {}
    for d in ds.docs_iter():
        docno = getattr(d, "docno", None) or getattr(d, "doc_id", None)
        if docno is None:
            continue
        text = doc_text(d)
        if text.strip():
            docs[docno] = text
    log(f"corpus docs with text: {len(docs)}")

    queries = {}
    for q in ds.queries_iter():
        qid = getattr(q, "query_id", None) or getattr(q, "qid", None)
        q_text = ""
        try:
            q_text = q.default_text() or ""
        except Exception:
            q_text = ""
        if not q_text:
            q_text = getattr(q, "title", "") or getattr(q, "text", "") or getattr(q, "description", "")
        queries[qid] = {
            "query": q_text,
            "question": getattr(q, "description", "") or "",
            "narrative": getattr(q, "narrative", "") or "",
        }
    log(f"queries: {len(queries)}")

    qrels = {}
    for re_ in ds.qrels_iter():
        qid = getattr(re_, "query_id", None)
        docno = getattr(re_, "doc_id", None)
        grade = getattr(re_, "relevance", None)
        if qid is None or docno is None or grade is None:
            continue
        qrels.setdefault(qid, {})[docno] = int(grade)
    log(f"qrels topics: {len(qrels)}")

    return docs, queries, qrels


def _extra(q, field):
    """Some query objects carry extra fields (question/narrative) not in .text."""
    try:
        extra = q.get_extra_fields()
    except Exception:
        extra = {}
    return extra.get(field, "") if isinstance(extra, dict) else ""
