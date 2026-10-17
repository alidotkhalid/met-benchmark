"""Offline variants of our method computed from saved per-document answers (no model calls):
  pvote_gX      : ours, isolated answers from the plain prompt (as run)
  pvabs_gX      : ours on abstention-prompted isolated answers (RobustRAG's iso prompt); 'I don't know'
                  answers are dropped before voting; if every document abstains, the closed-book answer is returned.
Also prints paired McNemar exact tests between chosen methods."""
import json, sys, math
from met.methods import prior_vote
from met.text import score

IDK_PAT = ("i don't know", "i do not know")


def is_idk(a):
    return any(p in a.lower() for p in IDK_PAT)


def load(path, datafiles):
    alias = {}
    for df in datafiles:
        for l in open(df):
            d = json.loads(l); alias[d["id"]] = d.get("gold_aliases")
    R = [json.loads(l) for l in open(path)]
    for r in R:
        r["gold_aliases"] = alias.get(r["id"])
        if r.get("iso_abst"):
            kept = [a for a in r["iso_abst"] if not is_idk(a)]
            for g in [1, 1.5, 2, 3]:
                p = prior_vote(kept, r["prior_answer"], r["prior_conf"], g, r["prior_answer"]) if kept else r["prior_answer"]
                r["pred"][f"pvabs_g{g:g}"] = p
                r["res"][f"pvabs_g{g:g}"] = score(p, r["gold"], r["wrong"], r["gold_aliases"])
    return R


def mcnemar(R, a, b, metric="acc"):
    """two-sided exact McNemar; returns (a_only, b_only, p)"""
    x = sum(1 for r in R if r["res"][a][metric] and not r["res"][b][metric])
    y = sum(1 for r in R if r["res"][b][metric] and not r["res"][a][metric])
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p


if __name__ == "__main__":
    path, key = sys.argv[1], sys.argv[2]  # key = m (known) or n_inj (retrieval)
    R = load(path, ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl"])
    meths = ["closed_book", "rag", "arr", "vote", "pvote_g2", "pvote_g3", "pvabs_g1.5", "pvabs_g2", "pvabs_g3", "rrag_kw", "rrag_dec"]
    vals = sorted({r[key] for r in R})
    for metric in ["acc", "asr"]:
        print(f"\n{metric} by {key}")
        print(f"{'method':<13}" + "".join(f"{key}={v:<5}" for v in vals) + "  mean")
        for me in meths:
            row = [100 * sum(r["res"][me][metric] for r in R if r[key] == v) / sum(r[key] == v for r in R) for v in vals]
            print(f"{me:<13}" + "".join(f"{x:8.1f}" for x in row) + f"{sum(row)/len(row):8.1f}")
    print("\nMcNemar (accuracy): wins_A / wins_B / p")
    for a, b in [("pvote_g2", "rrag_kw"), ("pvote_g3", "rrag_kw"), ("pvabs_g2", "rrag_kw"), ("pvabs_g3", "rrag_kw"),
                 ("pvabs_g2", "pvote_g2"), ("pvabs_g2", "arr"), ("pvote_g2", "rrag_dec")]:
        print(f"{a:>10} vs {b:<9}" + "".join(f" | {key}={v}: {'%d/%d p=%.2g' % mcnemar([r for r in R if r[key]==v], a, b)}" for v in vals))
