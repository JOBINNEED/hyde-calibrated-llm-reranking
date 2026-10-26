"""Stage 4-6 Verification & Acceptance Test.

Validates all produced run files (S4, S5, S6), clef_raw.jsonl (Contract B),
metrics.csv, significance.txt, and calibration artifacts independently from disk.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import log, load_config, load_jsonl, load_json


def check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    log(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name} ({detail})")


def validate_trec_file(path: Path, expected_tag: str, docset: set):
    check(f"{path.name} exists", path.exists())
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(f"{path.name} line count == 5000", len(lines) == 5000, f"count={len(lines)}")

    topics = {}
    bad_columns = 0
    bad_docs = 0
    bad_tags = 0
    bad_ranks = 0
    bad_score_order = 0

    for line_idx, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 6:
            bad_columns += 1
            continue
        qid, q0, docno, rank_str, score_str, tag = parts
        if q0 != "Q0":
            bad_columns += 1
        if tag != expected_tag:
            bad_tags += 1
        if docno not in docset:
            bad_docs += 1

        try:
            rank = int(rank_str)
            score = float(score_str)
        except ValueError:
            bad_ranks += 1
            continue

        topics.setdefault(qid, []).append((rank, score, docno))

    check(f"{path.name} 6-column TREC format", bad_columns == 0, f"bad_lines={bad_columns}")
    check(f"{path.name} expected tag '{expected_tag}'", bad_tags == 0, f"bad_tags={bad_tags}")
    check(f"{path.name} all docnos in corpus", bad_docs == 0, f"off_corpus={bad_docs}")
    check(f"{path.name} topic count == 50", len(topics) == 50, f"topics={len(topics)}")

    for qid, cands in topics.items():
        if len(cands) != 100:
            bad_ranks += 1
        # Check ranks 1..100
        for expected_rank, (rank, score, _) in enumerate(cands, 1):
            if rank != expected_rank:
                bad_ranks += 1
        # Check scores non-increasing
        for i in range(len(cands) - 1):
            if cands[i][1] < cands[i + 1][1] - 1e-9:
                bad_score_order += 1

    check(f"{path.name} 100 ranks per topic (1..100)", bad_ranks == 0, f"bad_ranks={bad_ranks}")
    check(f"{path.name} scores monotonically descending", bad_score_order == 0, f"bad_order={bad_score_order}")


def main():
    cfg = load_config()
    cache_dir = ROOT / "data" / "cache"
    runs_dir = ROOT / "runs"
    art_dir = ROOT / "artifacts"
    results_dir = ROOT / "results"

    docids = load_json(cache_dir / "docids.json")
    docset = set(docids)

    log("=== 1. Validating TREC Run Files (S4, S5, S6) ===")
    validate_trec_file(runs_dir / "S4_ce.trec", "S4_ce", docset)
    validate_trec_file(runs_dir / "S5_clef_noul.trec", "S5_clef_noul", docset)
    validate_trec_file(runs_dir / "S6_clef_score.trec", "S6_clef_score", docset)

    log("=== 2. Validating Contract B: clef_raw.jsonl ===")
    raw_path = art_dir / "clef_raw.jsonl"
    check("clef_raw.jsonl exists", raw_path.exists())
    raw_rows = load_jsonl(raw_path)
    check("clef_raw.jsonl row count == 5000", len(raw_rows) == 5000, f"count={len(raw_rows)}")

    cand_path = art_dir / "candidates.jsonl"
    cand_rows = load_jsonl(cand_path)
    cand_pairs = {(str(c["qid"]), str(c["docno"])) for c in cand_rows}

    contract_b_fields = [
        "qid", "docid", "model", "question_type", "probability",
        "confidence", "stage_a_rank", "latency_ms", "state_chars", "truncated"
    ]

    bad_schema = 0
    bad_prob_range = 0
    unmatched_pairs = 0

    for r in raw_rows:
        for f in contract_b_fields:
            if f not in r:
                bad_schema += 1
                break
        p = r.get("probability")
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            bad_prob_range += 1
        conf = r.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            bad_prob_range += 1
        if (str(r.get("qid")), str(r.get("docid"))) not in cand_pairs:
            unmatched_pairs += 1

    check("clef_raw.jsonl Contract B required fields present", bad_schema == 0, f"bad_schema={bad_schema}")
    check("clef_raw.jsonl probabilities in [0, 1]", bad_prob_range == 0, f"bad_prob={bad_prob_range}")
    check("clef_raw.jsonl matches candidates.jsonl 1:1", unmatched_pairs == 0, f"unmatched={unmatched_pairs}")

    log("=== 3. Validating Evaluation Results ===")
    metrics_csv = results_dir / "metrics.csv"
    check("metrics.csv exists", metrics_csv.exists())
    m_lines = metrics_csv.read_text(encoding="utf-8").splitlines()
    check("metrics.csv has 25 lines (6 systems x 4 metrics + header)", len(m_lines) == 25, f"lines={len(m_lines)}")

    sig_txt = results_dir / "significance.txt"
    check("significance.txt exists", sig_txt.exists())
    sig_content = sig_txt.read_text(encoding="utf-8")
    for s_name in ["S1 BM25", "S2 Dense", "S3 HyDE", "S4 Cross-encoder", "S5 Clef noul", "S6 Clef score"]:
        check(f"significance.txt includes '{s_name}'", s_name in sig_content)

    log("=== 4. Validating Calibration Deliverables ===")
    cal_png = results_dir / "calibration.png"
    check("calibration.png exists", cal_png.exists())
    check("calibration.png non-empty", cal_png.stat().st_size > 10000, f"size={cal_png.stat().st_size} bytes")

    cal_json = results_dir / "calibration_metrics.json"
    check("calibration_metrics.json exists", cal_json.exists())
    cal_data = json.loads(cal_json.read_text(encoding="utf-8"))
    check("calibration_metrics.json contains ECE", "ece" in cal_data["judged_only"]["S5_clef_noul"])
    check("calibration_metrics.json contains Brier", "brier" in cal_data["judged_only"]["S5_clef_noul"])
    check("calibration_metrics.json contains threshold transfer analysis", "threshold_transfer_analysis" in cal_data["judged_only"]["S5_clef_noul"])
    check("threshold transfer mean F1 > 0.80", cal_data["judged_only"]["S5_clef_noul"]["threshold_transfer_analysis"]["mean_f1"] > 0.80)
    check("calibration_metrics.json contains S6 binary calibration", "ece" in cal_data["judged_only"]["S6_clef_score_binary"])
    check("S6 binary calibration ECE < 0.10 (well-calibrated)", cal_data["judged_only"]["S6_clef_score_binary"]["ece"] < 0.10)

    log("=== 5. Unit Testing TREC Parser Robustness ===")
    from evaluate import _parse_trec_scores
    tmp_trec = results_dir / "_test_parse.trec"
    try:
        # 6-field
        tmp_trec.write_text("1 Q0 d1 1 0.850000 S1\n", encoding="utf-8")
        parsed6 = _parse_trec_scores(tmp_trec)
        check("Parse 6-field TREC (qid Q0 doc rank score tag)", parsed6.get("1", {}).get("d1") == 0.85)

        # 5-field with tag
        tmp_trec.write_text("1 Q0 d2 0.720000 S1\n", encoding="utf-8")
        parsed5_tag = _parse_trec_scores(tmp_trec)
        check("Parse 5-field TREC with tag (qid Q0 doc score tag)", parsed5_tag.get("1", {}).get("d2") == 0.72)

        # 5-field without tag
        tmp_trec.write_text("1 Q0 d3 2 0.650000\n", encoding="utf-8")
        parsed5_notag = _parse_trec_scores(tmp_trec)
        check("Parse 5-field TREC without tag (qid Q0 doc rank score)", parsed5_notag.get("1", {}).get("d3") == 0.65)
    finally:
        if tmp_trec.exists():
            tmp_trec.unlink()

    log("==== ALL STAGE 4-6 VERIFICATION CHECKS PASSED (100% GREEN) ====")


if __name__ == "__main__":
    main()
