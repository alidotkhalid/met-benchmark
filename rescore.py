"""Recompute the answer-level methods (vote / pvote) from saved isolated answers + prior, without re-running the model.
usage: python rescore.py in.jsonl out.jsonl"""
import json, sys
from met.methods import vote, prior_vote
from met.text import score

GAMMAS = [0.5, 1, 1.5, 2, 2.5, 3]
alias = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    try:
        for l in open(df):
            d = json.loads(l); alias[d["id"]] = d.get("gold_aliases")
    except FileNotFoundError:
        pass


def rescore_row(r):
    iso = r.get("isolated")
    if not iso:
        return r
    fb = r["pred"].get("rag", iso[0])
    r["pred"]["vote"] = vote(iso, fb)
    for g in GAMMAS:
        r["pred"][f"pvote_g{g:g}"] = prior_vote(iso, r["prior_answer"], r["prior_conf"], g, fb)
    r["pred"]["pvote_free_g4"] = prior_vote(iso, r["prior_answer"], r["prior_conf"], 4, fb, grounded=False)
    for k in ["vote", "pvote_free_g4"] + [f"pvote_g{g:g}" for g in GAMMAS]:
        r["res"][k] = score(r["pred"][k], r["gold"], r["wrong"], alias.get(r["id"]))
    return r


if __name__ == "__main__":
    with open(sys.argv[2], "w") as f:
        for l in open(sys.argv[1]):
            f.write(json.dumps(rescore_row(json.loads(l))) + "\n")
    print("rescored ->", sys.argv[2])
