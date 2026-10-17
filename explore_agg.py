"""Offline design study of the answer-level aggregator (no model calls). Selection on the dev split only."""
import json, hashlib, sys
from met.methods import cluster_votes
from met.text import same_answer, norm, score

alias = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for l in open(df):
        d = json.loads(l); alias[d["id"]] = d.get("gold_aliases")


def agg(iso, prior, conf, gamma, rag, tie="first", prior_mode="free"):
    cl = cluster_votes(iso)
    if not cl:
        return prior if norm(prior) else rag
    if len(cl) == 1:
        return cl[0][0]
    sc = [c[1] for c in cl]
    hit = [i for i, c in enumerate(cl) if same_answer(prior, c[0])]
    if hit:
        sc[hit[0]] += gamma * conf
    elif prior_mode == "free" and gamma * conf > max(sc):
        return prior
    best = max(sc)
    tied = [i for i, s in enumerate(sc) if abs(s - best) < 1e-9]
    if tie == "rag" and len(tied) > 1:
        r = [i for i in tied if same_answer(rag, cl[i][0])]
        if r:
            return cl[r[0]][0]
    return cl[tied[0]][0]


FILES = {"Qwen-7B": ["q7_known_all", "q7_met", "q7_ret_all", "q7_edits"], "Llama-8B": ["l8_known_all", "l8_met", "l8_ret_all", "l8_edits"],
         "Mistral-7B": ["m7_known_all", "m7_met", "m7_ret_all"]}
VARIANTS = {f"{pm}/{tie}/g{g:g}": dict(prior_mode=pm, tie=tie, gamma=g)
            for pm in ["free", "grounded"] for tie in ["first", "rag"] for g in [0, 1.5, 2, 2.5, 3]}


def dev(q): return int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 3


def setname(r):
    return {"mixed": "prnd", "retrieval": "retr", "edits": "edit"}.get(r["part"], "known") if not r["id"].startswith("tri") else "known"


rows = []
for m, fs in FILES.items():
    for f in fs:
        for l in open(f"results/v3/{f}.jsonl"):
            r = json.loads(l)
            if r.get("isolated") and r["part"] != "counterfactual":
                r["model"] = m; r["set"] = "known" if "known" in f else setname(r); rows.append(r)
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
rows = [r for r in rows if dev(r["qid"]) == (split == "dev")]
res = {}
for name, kw in VARIANTS.items():
    cells = {}
    for r in rows:
        p = agg(r["isolated"], r["prior_answer"], r["prior_conf"], kw["gamma"], r["pred"].get("rag", ""), kw["tie"], kw["prior_mode"])
        key = (r["model"], r["set"], r.get("cond") or r.get("n_inj", r["m"]))
        cells.setdefault(key, []).append(score(p, r["gold"], r["wrong"], alias.get(r["id"]))["acc"])
    per_set = {}
    for (m, s, lv), v in cells.items():
        per_set.setdefault(s, []).append(sum(v) / len(v))
    res[name] = {s: 100 * sum(v) / len(v) for s, v in per_set.items()}
sets = ["known", "prnd", "retr", "edit"]
print(f"split={split}  (mean accuracy over models x levels)")
print(f"{'variant':<22}" + "".join(f"{s:>8}" for s in sets) + f"{'mean':>8}")
for name, d in sorted(res.items(), key=lambda x: -sum(x[1].values()) / len(x[1])):
    print(f"{name:<22}" + "".join(f"{d.get(s, float('nan')):8.1f}" for s in sets) + f"{sum(d.values())/len(d):8.1f}")
