"""Cross-model summary on rescored results (results/v3): accuracy per set x level, McNemar vs ours (pvote_g2.5)."""
import json, math, os
from variants import mcnemar

D = "results/v3"
MODELS = {"Qwen-7B": "q7", "Llama-8B": "l8", "Mistral-7B": "m7"}
SETS = [("known", "{}_known{}.jsonl", "m", [0, 2, 3, 5]), ("poisonedrag", "{}_met.jsonl", "m", [0, 1, 2, 3, 5]),
        ("retrieval", "{}_ret_all.jsonl", "n_inj", [0, 1, 3, 5]), ("edits", "{}_edits.jsonl", "cond", ["orig", "edit", "update"]),
        ("counterfactual", "{}_cf.jsonl", "m", [0, 1])]
METH = ["closed_book", "rag", "cad", "adacad", "arr", "sdag", "vote", "rrag_kw", "rrag_dec", "pvote_g2.5", "pvote_free_g4"]
OURS = "pvote_g2.5"


def load(m, pat):
    for suf in ["_all", ""]:
        p = os.path.join(D, pat.format(m, suf) if "{}" in pat[3:] else pat.format(m))
        if os.path.exists(p):
            return [json.loads(l) for l in open(p)]
    return None


for mname, m in MODELS.items():
    print(f"\n==================== {mname} ====================")
    for sname, pat, key, lv in SETS:
        R = load(m, pat)
        if not R:
            continue
        R = [r for r in R if r["part"] != "counterfactual"] if sname == "poisonedrag" else R
        if sname == "poisonedrag":
            R = [r for r in R if r["part"] == "mixed"]
        lv = [v for v in lv if any(r[key] == v for r in R)]
        print(f"\n{sname} (n q = {len({r['qid'] for r in R})}), accuracy % by {key}")
        print(f"{'method':<15}" + "".join(f"{str(v):>8}" for v in lv))
        for me in METH:
            if me not in R[0]["res"]:
                continue
            print(f"{me:<15}" + "".join(f"{100*sum(r['res'][me]['acc'] for r in R if r[key]==v)/sum(r[key]==v for r in R):8.1f}" for v in lv))
        print("McNemar ours vs: (ours-only / other-only / p)")
        for b in ["rag", "arr", "vote", "rrag_kw", "rrag_dec"]:
            if b not in R[0]["res"]:
                continue
            cells = []
            for v in lv:
                x, y, p = mcnemar([r for r in R if r[key] == v], OURS, b)
                cells.append(f"{v}:{x}/{y} {'p=%.1g' % p}")
            print(f"   {b:<9}" + " | ".join(cells))
