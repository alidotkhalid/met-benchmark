"""One SQuAD answer carries Wikipedia citation residue ('1984[needs' for '1984'), affecting one question in
the edits and counterfactual components (5 instances). Clean it in the data and rescore every stored
prediction of the affected instances. MADAM files are rescored by madam_report.py, which applies the same fix."""
import json, glob
from met.text import score
FIX = {"1984[needs": "1984"}
for f in ["data/met.jsonl", "data/met_edits.jsonl", "data/met_madam.jsonl"] + \
         [p for p in glob.glob("results/v3/*.jsonl") if "_madam" not in p]:
    rows = [json.loads(l) for l in open(f)]
    n = 0
    for r in rows:
        if r.get("gold") in FIX or r.get("wrong") in FIX:
            r["gold"], r["wrong"] = FIX.get(r["gold"], r["gold"]), FIX.get(r["wrong"], r["wrong"])
            if "pred" in r:
                r["res"] = {k: score(v, r["gold"], r["wrong"], r.get("gold_aliases")) for k, v in r["pred"].items()}
            n += 1
    if n:
        open(f, "w").write("".join(json.dumps(r) + "\n" for r in rows)); print(f, n)
