"""Paired evaluation subset for the MADAM-RAG baseline.

MADAM-RAG makes up to ~18 generation calls of up to 128 tokens per instance (agents x rounds plus an
aggregator per round), roughly 20-40x the cost of the other baselines, so it is evaluated on a fixed
random subset of every MET component. The subset is drawn with a fixed seed, before any MADAM output
exists, and every other method is compared to it on exactly these instances (paired).

Writes data/met_madam.jsonl (standard components) and data/met_madam_adaptive_{tag}.jsonl
(prior-aligned adaptive attack; the attack depends on the model, so one file per model).
"""
import json, random

SEED = 0
rng = random.Random(SEED)


def load(p):
    return [json.loads(l) for l in open(p)]


def interleave(rows, seed=1):
    """Order the instances so that ANY prefix is a balanced random sample: all conditions of a question stay
    together, questions are shuffled within each source, and sources are interleaved in proportion to their
    size. A run stopped by a time limit then still covers every component (set unchanged, order only)."""
    r2 = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault(r["source"], {}).setdefault(r["qid"], []).append(r)
    keyed = []
    for src in sorted(groups):
        qs = sorted(groups[src]); r2.shuffle(qs)
        keyed += [((i + 0.5) / len(qs), src, groups[src][q]) for i, q in enumerate(qs)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, g in keyed for r in g]


def pick(rows, n_q, keep=lambda r: True):
    qids = sorted({r["qid"] for r in rows})
    rng.shuffle(qids)
    chosen = set(qids[:n_q])
    return [r for r in rows if r["qid"] in chosen and keep(r)]


out = []
out += pick(load("data/met_known.jsonl"), 50)                                         # 50 facts x m in {0,2,3,5}
ret = load("data/met_retrieval.jsonl")
for src in ["ret-nq", "ret-hotpotqa", "ret-msmarco"]:                                  # 20 per corpus x inject {0,1,3,5}
    out += pick([r for r in ret if r["source"] == src], 20)
out += pick(load("data/met_edits.jsonl"), 100, keep=lambda r: r["cond"] in ("orig", "edit"))
out += pick([r for r in load("data/met.jsonl") if r["part"] == "counterfactual"], 100)  # single-document, cheap
out = interleave(out)
with open("data/met_madam.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print("met_madam.jsonl:", len(out), "instances", Counter(r["part"] for r in out))

for tag in ["q7", "l8", "m7"]:
    a = [r for r in load(f"data/met_adaptive_{tag}.jsonl") if r["m"] in (1, 2, 3)]
    b = [r for r in load(f"data/met_adaptive_ret_{tag}.jsonl") if r["n_inj"] in (1, 3)]
    with open(f"data/met_madam_adaptive_{tag}.jsonl", "w") as f:
        for r in interleave(a + b):
            f.write(json.dumps(r) + "\n")
    print(f"met_madam_adaptive_{tag}.jsonl:", len(a) + len(b), "instances")
