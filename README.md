# MET: Mixed-Evidence Trust

Code, data and model outputs for the paper *"The Trust Frontier in Retrieval-Augmented Generation: Robustness to Corpus Poisoning Trades Off Against Following Legitimate Evidence"* (under review).

Defences against RAG corpus poisoning are usually scored only on how well they resist injected passages. MET scores every method on two axes at once: **robustness** (accuracy when the retrieved evidence is poisoned) and **utility** (accuracy when the retrieved evidence is a legitimate update the model should follow). The repository also contains **MET-Vote**, the paper's frontier-tunable aggregator, a prior-aligned adaptive attack against it, a MADAM-RAG baseline, and a multi-hop component.

Every table in the paper is regenerated from the stored model outputs in `results/v3/` **without a GPU**.

## Quick start: reproduce the paper's tables

```bash
pip install -r requirements.txt
python mktables.py tables          # writes every LaTeX table of the paper into tables/
python mkfigure.py                 # trust-frontier figure  -> figures/frontier.pdf
python mkfigure_adaptive.py        # adaptive-attack figure -> figures/adaptive_gamma.pdf
```

Analysis scripts behind specific claims in the text:

| Script | What it reproduces |
|---|---|
| `madam_report.py l8` / `m7` | MADAM-RAG rescoring, paired tests, aggregator overrides |
| `multihop_report.py q7` / `l8` / `m7` | multi-hop component: accuracy, R/U, isolated-reading diagnostics |
| `adaptive_report.py q7` / `l8` / `m7` | prior-aligned adaptive attack vs the standard attack on the same questions |
| `conf_gate.py` | prior-confidence gate (can confidence separate exploitable priors?) |
| `select_gamma.py` | prior-weight sweep on the development / test split |
| `verify_claims.py` | parametric-leakage filter and related analysis numbers |
| `fixnums.py`, `neg_numbers.py`, `explore_*.py` | override counts and the rejected variants ("negative results") |

Model tags: `q7` = Qwen2.5-7B-Instruct, `l8` = Llama-3.1-8B-Instruct, `m7` = Mistral-7B-Instruct-v0.3.

## Repository layout

```
met/            methods: standard RAG, CAD, AdaCAD, ARR, SDAG, isolate-then-vote, RobustRAG
                (keyword / decoding), MET-Vote (prior_vote), MADAM-RAG port (madam.py); scoring (text.py)
run.py          runs any set of methods on a data file; resumable, one JSON line per instance
data/           benchmark instances (JSONL); see below
results/v3/     stored generations and scores for every model and component
attack/         builds the prior-aligned adaptive attack and the MADAM-RAG subsets
builders/       scripts that built the benchmark components from their sources
notebooks/      the Kaggle notebooks used for the GPU runs (2x T4)
annotation/     human validation of the abstention classifier (sheet, scorer, hidden key)
```

Scripts in `attack/` and `builders/` import the `met` package, so run them from the repository root with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python builders/build_multihop.py <hotpotqa_dev_distractor.parquet>`.

## Benchmark components (`data/`)

| File | Component | Questions | Notes |
|---|---|---|---|
| `met_known.jsonl` | Known facts | 100 | m in {0,2,3,5} of 5 passages poisoned |
| `met.jsonl` (part `mixed`) | PoisonedRAG mixed evidence | 90 evaluated | m in {0,1,2,3,5} |
| `met_retrieval.jsonl` | Realistic retrieval (Contriever, BEIR) | 259 | 0/1/3/5 injected passages, re-retrieved top-5 |
| `met.jsonl` (part `counterfactual`) | Single-document counterfactual edits | 300 | from Faithfulness-QA |
| `met_edits.jsonl` | Multi-document legitimate edits | 200 | orig / edit / update |
| `met_multihop.jsonl` | Multi-hop (HotpotQA bridge) | 100 | orig / edit / p1 / p2 |
| `met_adaptive_*.jsonl` | Prior-aligned adaptive attack | per model | targets each model's own wrong closed-book answer |
| `met_madam*.jsonl` | Paired subsets for MADAM-RAG | 840 instances | fixed seed, drawn before any MADAM output |

## Running experiments (GPU)

```bash
python run.py --model meta-llama/Llama-3.1-8B-Instruct --data data/met_known.jsonl \
              --out results/l8_known.jsonl --dtype float16 --device auto
```

`--methods` selects methods (default: closed_book, rag, cad, adacad, arr, sdag, vote, pvote; add `rrag_kw,rrag_dec` for RobustRAG and `madam` for MADAM-RAG); `--max_hours` stops cleanly before a time limit, and re-running with the same `--out` resumes. The notebooks in `notebooks/` contain the exact presets used for each run. Qwen2.5 must run in `bfloat16`.

**MADAM-RAG note.** `met/madam.py` ports the authors' reference implementation with its prompts verbatim. One change was required: the aggregator's reply is prefilled with `All Correct Answers: [` (see the paper, Section 6), because Llama-3.1-8B otherwise continues the prompt instead of answering.

## Licence

Code: MIT (see `LICENSE`). The data files are derived from SQuAD and HotpotQA (CC BY-SA 4.0), Natural Questions (CC BY-SA 3.0), MS MARCO (research, non-commercial terms), the BEIR corpora, the PoisonedRAG adversarial texts and Faithfulness-QA, and remain subject to the licences of those sources.

## Citation

```bibtex
@article{met2026,
  title  = {The Trust Frontier in Retrieval-Augmented Generation: Robustness to Corpus Poisoning Trades Off Against Following Legitimate Evidence},
  note   = {Under review. Code and data: https://github.com/alidotkhalid/met-benchmark},
  year   = {2026}
}
```
