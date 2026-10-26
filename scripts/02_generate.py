"""Stage 2 — Generate hypothetical documents (HyDE) with qwen3.8/Ornith.

50 topics x 8 sequential single-sequence generations = 400 passages.
Writes artifacts/hyde_docs.jsonl (Contract A, append-only).

Because the generator is a hybrid reasoning model, every call sends a system
prompt that forces the abstract into message.content instead of a reasoning
trace. Each generation uses its own seed for diversity (batch=1 by design).
"""
from datetime import datetime, timezone
from pathlib import Path

import re

from openai import OpenAI

from common import (log, load_config, append_jsonl, PROMPT_V1, tokenize)
from load_data import load_dataset

NAMES = {
    "artifacts": "hyde_docs.jsonl",
}


def make_client(cfg):
    g = cfg["generator"]
    client = OpenAI(base_url=g["base_url"], api_key=g.get("api_key", "not-needed"))
    return client


def generate_one(client, cfg, user):
    """One single-sequence generation. Returns (text, truncated, used)."""
    g = cfg["generator"]
    resp = client.chat.completions.create(
        model=g["model"],
        messages=[{"role": "user", "content": user}],
        max_completion_tokens=g["max_tokens"],
        temperature=g["temperature"],
        n=1,
    )
    choice = resp.choices[0]
    content = choice.message.content
    truncated = choice.finish_reason == "length"
    used = getattr(resp.usage, "completion_tokens", None)
    return content, truncated, used


def clean_generated(text):
    """Strip spurious markdown / section headers the model may prepend before
    the abstract body.

    The generator is a hybrid reasoning model. When it does not use the
    reasoning channel it can still open the response with a markdown or section
    header -- e.g. ``**Abstract**``, ``## Abstract``, ``# Abstract``,
    ``Abstract:`` or a *stacked* ``Abstract**\n\nBackground:**``. The prompt
    already ends with the ``Abstract:`` label, so any such leading header(s) are
    noise. The strip is repeated so that multiple stacked headers are all
    removed; it never removes a body word (e.g. "Abstract reasoning" is kept).
    Returns the abstract body unchanged when no such header is present.
    """
    if not text:
        return text
    # Loop so stacked headers ("Abstract**\n\nBackground:**") are all stripped.
    # Each pass strips leading markdown, then one leading section-header token.
    # A header is only stripped when followed by punctuation/markdown or a
    # newline -- never by a real word, which is genuine body text.
    while True:
        stripped = re.sub(r'^[`#*_\s]*', '', text)
        m = re.match(
            r'(?:Abstract|Background|Introduction|Methods|Results|'
            r'Conclusion|Summary)'
            r'(?:[^A-Za-z\s][^A-Za-z]*|[\t ]*\n)[ \t]*',
            stripped, flags=re.I)
        if not m:
            break
        text = stripped[m.end():]
    return text.strip()


def main():
    cfg = load_config()
    ROOT = Path(__file__).resolve().parent.parent
    art_dir = ROOT / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / NAMES["artifacts"]

    _, queries, _ = load_dataset(cfg)
    log(f"topics to expand: {len(queries)}")

    client = make_client(cfg)
    g = cfg["generator"]

    total = 0
    failures = 0
    sample_show = []
    for qid, q in sorted(queries.items()):
        prompt = PROMPT_V1.format(query=q["query"])
        gens_for_topic = 0
        for i in range(g["n_generations"]):
            seed = 12345 + i
            attempts = 0
            while attempts < 3:
                attempts += 1
                try:
                    text, truncated, used = generate_one(
                        client, cfg, prompt)
                except Exception as e:
                    log(f"  q={qid} gen={i} attempt {attempts} error: {e}")
                    continue
                if text and text.strip():
                    text = clean_generated(text)
                    if text and text.strip():
                        break
                log(f"  q={qid} gen={i} empty content, retry")
            else:
                failures += 1
                log(f"!! q={qid} gen={i}: all attempts failed/empty — skipped")
                continue

            n_tokens = len(tokenize(text)) if text else 0
            row = {
                "qid": qid,
                "gen_idx": i,
                "model": g["model"],
                "prompt_version": g["prompt_version"],
                "temperature": g["temperature"],
                "seed": seed,
                "text": text,
                "n_tokens": n_tokens,
                "truncated": truncated,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            append_jsonl(out_path, [row])
            total += 1
            gens_for_topic += 1
            if gens_for_topic <= 2:
                sample_show.append((qid, text[:160]))

    log(f"generated {total} passages ({failures} failures) -> {out_path}")
    print("\n=== first 3 generations (inspect by hand) ===")
    for qid, preview in sample_show:
        print(f"[qid {qid}] {preview}")


if __name__ == "__main__":
    main()
