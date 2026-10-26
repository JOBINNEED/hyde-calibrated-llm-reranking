"""Shared helpers for the HyDE + pplx-decider Stage 1 pipeline.

One model on the GPU at a time; every stage reads/writes files on disk.
"""
import json
import re
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CONFIG_PATH = ROOT / "config.yaml"


# ---- Frozen HyDE generation prompt (handbook PROMPT_V1) ----
# Changed only via a decision-log entry; a silent change invalidates every run.
PROMPT_V1 = (
    "Write a short scientific abstract that answers the following question.\n"
    "Write it in the style of a COVID-19 research paper abstract: formal,\n"
    "specific, and using domain terminology. Do not refuse, do not add\n"
    "disclaimers, and do not mention that this is hypothetical. Write 80-120 words.\n"
    "Question: {query}\n"
    "Abstract:"
)


# --------------------------------------------------------------------- config
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------- tokens
# Cheap, deterministic tokenizer used for BM25. Lowercase + alnum runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------- logging
def log(msg: str) -> None:
    print(f"[{Path(__file__).name}] {msg}", flush=True)


# -------------------------------------------------------------- TREC run files
# Contract C: standard 6-column ("qid Q0 docno rank score tag") or 5-column ("qid Q0 docno score tag")
def write_trec(path: Path, rankings, tag: str = "system") -> int:
    """rankings: iterable of (qid, docno, score) or (qid, docno, rank, score, tag).
    Higher score = better rank.
    Returns number of lines written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in rankings:
            if len(item) == 5:
                qid, docno, rank, score, item_tag = item
                f.write(f"{qid} Q0 {docno} {rank} {float(score):.6f} {item_tag}\n")
            elif len(item) == 4:
                qid, docno, rank, score = item
                f.write(f"{qid} Q0 {docno} {rank} {float(score):.6f} {tag}\n")
            else:
                qid, docno, score = item[:3]
                f.write(f"{qid} Q0 {docno} {float(score):.6f} {tag}\n")
            n += 1
    return n


def write_trec_6col(path: Path, rankings) -> int:
    """rankings: iterable of (qid, docno, rank, score, tag). Standard TREC format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for qid, docno, rank, score, tag in rankings:
            f.write(f"{qid} Q0 {docno} {rank} {float(score):.6f} {tag}\n")
            n += 1
    return n


# --------------------------------------------------------------- JSONL artifacts
def append_jsonl(path: Path, rows) -> int:
    """Append rows (dicts) to a JSONL file. Append-only by contract A/B."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
