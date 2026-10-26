"""Evaluation with ranx: metrics + paired significance across topics."""
from pathlib import Path

from ranx import Qrels, Run, evaluate, compare
from common import log, load_config, dump_json


def build_qrels(qrels_dict):
    """qrels_dict: {qid: {docno: grade}} -> ranx Qrels."""
    return Qrels(qrels_dict)


def load_runs(run_paths):
    """run_paths: list of (system_id, path) -> {system_id: Run}.

    Builds ranx Run objects from the 5-field TREC run files we write
    (Contract C: "qid Q0 docno score runname"). ranx's own .from_file parser
    expects a 6-field line and cannot parse standard 5-field TREC, so we parse
    the file ourselves and hand the scores to Run.from_dict.
    """
    runs = {}
    for sid, p in run_paths:
        runs[sid] = Run.from_dict(_parse_trec_scores(p), name=sid)
    return runs


def _parse_trec_scores(path):
    """Parse a 4-field, 5-field, or 6-field TREC run file into {qid: {docno: score}}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _q0, docno = parts[0], parts[1], parts[2]
            if len(parts) >= 6:
                # Standard 6-field: qid Q0 docno rank score tag
                raw_score = parts[4]
            elif len(parts) == 5:
                # Could be (qid Q0 docno rank score) or (qid Q0 docno score tag)
                try:
                    _ = float(parts[4])
                    raw_score = parts[4]
                except ValueError:
                    raw_score = parts[3]
            else:
                # 4-field: qid Q0 docno score
                raw_score = parts[3]

            try:
                s = float(raw_score)
            except ValueError:
                continue
            out.setdefault(qid, {})[docno] = s
    return out


def evaluate_system(qrels, run, metrics):
    return evaluate(qrels, run, metrics)


def compare_systems(qrels, runs, metrics, max_p):
    """runs: list of Run objects (all keyed by same topics).
    Returns the ranx comparison report (paired Student's t over topics)."""
    return compare(qrels=qrels, runs=runs, metrics=metrics, max_p=max_p)


def summarize(metrics_dict):
    """Keep the headline numbers from a ranx evaluate() dict."""
    out = {}
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def print_metrics_table(results):
    """results: {system: {metric: value}}."""
    systems = list(results)
    metrics = []
    for s in systems:
        for m in results[s]:
            if m not in metrics:
                metrics.append(m)
    header = "System".ljust(26) + "".join(m.rjust(16) for m in metrics)
    print(header)
    print("-" * len(header))
    for s in systems:
        row = s.ljust(26)
        for m in metrics:
            row += f"{results[s].get(m, float('nan')):>16.4f}"
        print(row)
