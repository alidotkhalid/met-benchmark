"""Pilot: does weighting each document's vote by the model's confidence in its isolated answer help?"""
import json, sys
from met.text import same_answer, norm, score, is_nonanswer


def clusters(iso, w):
    cl = []  # [rep, weight]
    for a, c in zip(iso, w):
        if not norm(a) or is_nonanswer(a):
            continue
        for x in cl:
            if same_answer(a, x[0]):
                x[1] += c; break
        else:
            cl.append([a, c])
    return cl


def agg(iso, w, prior, pconf, gamma, rag, grounded=True):
    cl = clusters(iso, w)
    if not cl:
        return prior if norm(prior) else rag
    if len(cl) == 1:
        return cl[0][0]
    sc = [x[1] for x in cl]
    hit = [i for i, x in enumerate(cl) if same_answer(prior, x[0])]
    if hit:
        sc[hit[0]] += gamma * pconf
    elif not grounded and gamma * pconf > max(sc):
        return prior
    best = max(sc); tied = [i for i, s in enumerate(sc) if abs(s - best) < 1e-9]
    r = [i for i in tied if same_answer(rag, cl[i][0])]
    return cl[(r or tied)[0]][0]


R = [json.loads(l) for l in open(sys.argv[1])]
AL = {}
for df in ["data/met_known.jsonl", "data/met_edits.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl"]:
    for l in open(df):
        d = json.loads(l); AL[d["id"]] = d.get("gold_aliases")
for r in R: r["gold_aliases"] = AL.get(r["id"])
key = lambda r: (r["part"], r.get("cond") or r["m"])
lv = sorted({key(r) for r in R}, key=str)
V = {}
for g in [0, 1, 1.5, 2, 2.5, 3]:
    V[f"count/g{g}"] = lambda r, g=g: agg(r["isolated"], [1] * len(r["isolated"]), r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"])
    V[f"conf/g{g}"] = lambda r, g=g: agg(r["isolated"], r["iso_conf"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"])
    V[f"conf2/g{g}"] = lambda r, g=g: agg(r["isolated"], [c * c for c in r["iso_conf"]], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"])
print(f"{'variant':<14}" + "".join(f"{str(l[1]):>8}" for l in lv) + "    mean")
for b in ["rag", "arr"]:
    v = [100 * sum(r["res"][b]["acc"] for r in R if key(r) == l) / sum(key(r) == l for r in R) for l in lv]
    print(f"{b:<14}" + "".join(f"{x:8.1f}" for x in v) + f"{sum(v)/len(v):8.1f}")
for n, f in V.items():
    v = [100 * sum(score(f(r), r["gold"], r["wrong"], r.get("gold_aliases"))["acc"] for r in R if key(r) == l) / sum(key(r) == l for r in R) for l in lv]
    print(f"{n:<14}" + "".join(f"{x:8.1f}" for x in v) + f"{sum(v)/len(v):8.1f}")
# confidence of answer-doc vs distractor answers
import collections
