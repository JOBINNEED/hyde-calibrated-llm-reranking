# Change Log

Every change to the method, the data, or the evaluation gets an entry here, with
a date and the reason. This is the operational companion to Section 16 of
`HyDE_Clef_Project_Handbook.pdf` — it is what makes the final report writable in
week 8 without trying to reconstruct October from memory.

**Entry format:** what changed, why it mattered, what it affects, what to re-run.
A change that invalidates existing numbers says so explicitly.

---

## 2026-10-27

### CL-001 — Cross-encoder baseline given information parity with the decision model

**Status:** code fixed, **re-run required**
**Files:** `scripts/04_rerank_ce.py`, `config.yaml`, `scripts/05_evaluate.py`
**Invalidates:** the published S4 number (nDCG@10 0.638) and every S5/S6-vs-S4 comparison

**What was wrong.** The two Stage 4 re-rankers were not being fed the same
information. The decision model builds its state from *both* topic fields:

```
TOPIC: coronavirus origin                      <- `query`    (terse keyword title)
QUESTION: what is the origin of COVID-19       <- `question` (natural language)
DOCUMENT TITLE: ...
DOCUMENT: ...
```

while `04_rerank_ce.py` passed only `query`:

```python
q_text = c.get("query", "")      # "coronavirus origin" and nothing else
pairs.append((q_text, d_text))
```

**Why it matters.** Two compounding problems, not one.

1. **Strictly less information.** S4 never saw the natural-language question,
   which is the more informative of the two fields. Any S5/S6 margin over S4 is
   therefore confounded with the input difference — it cannot be attributed to
   the model, which is the entire claim the project rests on.
2. **Worst-case input distribution.** `ms-marco-MiniLM-L-6-v2` was trained on
   MS MARCO natural-language questions. A bare keyword string is precisely the
   input it handles worst, so the handicap is larger than "one missing field"
   suggests.

The headline result (S5 0.750 / S6 0.760 vs S4 0.638) is the project's central
claim. Published as it stood, the first competent reviewer would find this, and
the finding would be fatal rather than correctable. **A claim to beat a
supervised re-ranker has to beat its best configuration, not a crippled one.**

**What changed.**

- The CE query side is now composed from `cross_encoder.query_fields`
  (default `[query, question]`), joined with `". "`, skipping empty fields.
- `cross_encoder.max_length` is now read from config instead of hard-coded.
- Added `--query-fields` and `--tag` so single-field variants run as an explicit
  ablation without overwriting the canonical run.
- The script now logs the exact query and document strings it feeds the model,
  so a future composition mistake is visible in the log rather than silent.
- `05_evaluate.py` picks up the ablation run files when they exist.
- The pre-fix run was preserved as `runs/S4_ce_query_only.trec` rather than
  discarded — it is now the evidence for the input-sensitivity ablation.

**What to re-run** (Windows GPU box, in this order):

```bash
python scripts/04_rerank_ce.py                                     # S4_ce.trec  (query+question)
python scripts/04_rerank_ce.py --query-fields question --tag question_only
python scripts/05_evaluate.py                                      # metrics + significance
python scripts/06_calibration.py                                   # CE calibration uses S4 scores
```

Cheap: 5,000 CE pairs is seconds on the RTX 5090. No re-indexing, no
re-generation, no new decision-model calls.

**Expected outcome.** S4 climbs by a meaningful margin. The honest possibilities:

- *S5/S6 still win* — the result survives and is now defensible. Most likely.
- *The margin narrows sharply* — still a result, reported with the narrower gap.
- *S4 overtakes* — then the original finding was an artefact, and catching it
  here rather than in the viva is the whole point of this entry.

Report whichever CE configuration scores highest as the baseline, and show the
ablation table alongside it. The input-sensitivity result is itself worth a
paragraph: it quantifies how much a supervised re-ranker depends on getting
natural-language queries.

---

### CL-002 — Decision-model provenance recorded; model identified as d3

**Status:** code fixed, **re-stamp required** (existing decisions carry a wrong model field)
**Files:** `config.yaml`, `scripts/04_rerank_decider.py`
**Invalidates:** the `model` field in all 5,000 rows of `artifacts/clef_raw.jsonl` — not the probabilities

**What was wrong.**

```python
model_name = cfg.get("reranker", {}).get("model_active", "local-active")
```

`config.yaml` has no `model_active` key — it had `model: pplx-decider-v1.1-27b`.
The lookup silently fell back to the default, so the config value was never used
and **all 5,000 decisions recorded `"model": "local-active"`**.

**Why it matters.** The project's stated contribution is the first IR evaluation
of this decision model. An artifact that cannot say which model produced it does
not support that claim. A silent `.get()` fallback is also the worst failure
mode available: no error, no warning, and the wrong value looks plausible.

**The model.** We are using **d3**, a decision-model family from vLLM Semantic
Router, Apache 2.0, published at `ollama.com/library/d3` in 2b / 4b / 9b / 27b
sizes. It scores options you supply — category choice, yes/no (`noul`), or a
rating (`score`) — over the `/v1/systemone` endpoint, and does not generate
conversational text. We serve the 27B checkpoint through vLLM, where it is
registered under a served-model-name.

**What changed.** The config now carries two keys, because they are genuinely
two different things:

| Key | Value | Purpose |
|---|---|---|
| `served_model_name` | `local-active` | what the vLLM deployment answers to; goes in the payload |
| `model_id` | `d3:27b` | the real checkpoint; stamped into every record |

Each decision row now carries both. `main()` reads the correct keys and logs the
resolved pair at startup.

**What to re-run.** Nothing expensive. The probabilities are unaffected — only
the provenance field is wrong. Either re-stamp `clef_raw.jsonl` in place (map
`"local-active"` → `"d3:27b"`, documented as a CL-002 correction), or delete it
and re-run `04_rerank_decider.py` for a clean 5,000. Prefer the re-run if the
endpoint is up; prefer the re-stamp if it is not, and say which you did.

**Two things the d3 model card gives the project for free:**

1. **A documented quality ladder.** The publisher's own Decision Index (text):
   2B 38.9 · 4B 52.6 · 9B 55.8 · 27B 66.3. This is a better RQ4 than the
   original clef-vs-clef-flash plan — run d3:4b and d3:9b on the same frozen
   candidates and you get a size-vs-accuracy-vs-latency curve against a
   published reference ladder. Each extra size is one script invocation.
2. **An explicit invitation to do the calibration study.** The model card states
   that scores are model outputs, *not guaranteed real-world frequencies*, and
   recommends testing on your own examples before setting thresholds. Our
   calibration analysis answers a question the model's own authors flagged as
   open. Quote this in the report — it reframes the study from "something we
   thought of" to "the publisher's stated caveat, measured".

**Also note:** the d3 Ollama packages are documented at a 4096-token context,
despite the 256K in the model listing. `max_state_chars: 6000` (~1500 tokens)
fits comfortably either way, but verify against the vLLM deployment before
raising it.

---

## Open items — logged, not yet actioned

Recorded so they are not rediscovered later. Ordered by impact on the result.

| ID | Issue | Why it matters | Cost to fix |
|---|---|---|---|
| OI-01 | **Naming mismatch: artifacts say `clef`, the model is `d3`.** `runs/S5_clef_noul.trec`, `S6_clef_score.trec`, `artifacts/clef_raw.jsonl`, and the handbook all say Clef. | The report must name the model actually used. Renaming cascades through `05_evaluate.py`, `06_calibration.py`, and the handbook, so it is a deliberate decision, not a silent edit. | Medium — decide once, change everywhere in one commit |
| OI-02 | **BM25 acceptance gate failed and was not treated as a gate.** 0.572 vs the 0.656 reference, outside the ±0.03 tolerance; the script logs a warning and continues. | Two legitimate causes: we use `cord19/trec-covid` (191,132 docs) not BEIR's filtered 171,332, and `tokenize()` is bare `[a-z0-9]+` with no stemming or stopword removal where the reference uses Anserini's Porter stemming. Either fix the tokenizer or write the explanation into the report — but do not leave a failed gate unexplained. | Low (add stemming) or zero (document it) |
| OI-03 | **S2 dense at 0.122 is partly self-inflicted.** Three compounding causes: (a) the terse `query` field feeds dense retrieval, its worst input; (b) `normalize_embeddings=True` makes it cosine while Contriever is trained for dot product; (c) `model.max_length = 350` is a **no-op** — the sentence-transformers attribute is `max_seq_length`, so the configured truncation never applied. | A stronger S2 makes the HyDE gain *more* credible, not less. Right now a reviewer can argue the 3.6× improvement is measured against a broken baseline. | Low — three one-line changes, one re-index |
| OI-04 | **The frozen candidate pool is worse than BM25's.** S3 Recall@100 = 0.071 vs BM25's 0.098. | Pool recall is the hard ceiling on every re-ranker. We are handing S4–S6 a weaker set than BM25 alone would produce. Fusing BM25 with HyDE-dense via RRF (~10 lines with `ranx.fuse`) raises the ceiling for every downstream system at once. | Low — highest value-per-line change available |
| OI-05 | **`confidence` is not independent information.** It equals `max(p, 1−p)` exactly — the 1,971 rows where `confidence == probability` are precisely those with p ≥ 0.5. | Analysing it as a separate signal would be a reporting error. | Zero — just do not use it |
| OI-06 | **Judged-only calibration needs careful framing.** Optimal threshold 0.05 with F1 0.92, and even the lowest probability bin is 47% relevant. | That is TREC pooling bias, not model skill: judged ⇒ was retrieved by a pooled system ⇒ likely relevant. The all-candidates treatment is the honest one. Both are already reported — the write-up must explain *why* they disagree. | Zero — a paragraph |
| OI-07 | Missing S7; `__pycache__` committed; no `.gitignore`; one-line README; `significance.txt` write logic is fragile (`str(report) if not hasattr(report,"write") else ""` silently writes an empty file if the object ever grows a `.write`). | Housekeeping. The README matters most — it is what a reader sees first. | Low |

---

## Re-run order after these fixes

Nothing above requires re-encoding the corpus or regenerating the 400
hypothetical documents. Both remain valid.

```bash
# CL-001
python scripts/04_rerank_ce.py
python scripts/04_rerank_ce.py --query-fields question --tag question_only

# CL-002 (if re-running rather than re-stamping)
python scripts/04_rerank_decider.py

# always last
python scripts/05_evaluate.py
python scripts/06_calibration.py
```
