"""Summarise a results file: accuracy / attack-success by method x number of poisoned docs (m),
split by whether the model's closed-book prior already knows the truth, plus counterfactual following."""
import json, sys
from collections import defaultdict

path = sys.argv[1]
R = [json.loads(l) for l in open(path)]
methods = list(R[0]["res"].keys())


def table(rows, metric, title):
    ms = sorted({r["m"] for r in rows})
    print(f"\n{title}  (n questions = {len({r['qid'] for r in rows})})")
    print(f"{'method':<14}" + "".join(f"m={m:<7}" for m in ms))
    for me in methods:
        line = f"{me:<14}"
        for m in ms:
            v = [r["res"][me][metric] for r in rows if r["m"] == m and me in r["res"]]
            line += f"{100*sum(v)/len(v):6.1f}  " if v else "   -    "
        print(line)


mixed = [r for r in R if r["part"] == "mixed"]
if mixed:
    table(mixed, "acc", "MIXED EVIDENCE - accuracy on true answer (%) by # poisoned docs out of 5")
    table(mixed, "asr", "MIXED EVIDENCE - attack success, outputs poisoned answer (%)")
    for known in [1, 0]:
        sub = [r for r in mixed if r["prior_knows_truth"] == known]
        if sub:
            table(sub, "acc", f"MIXED - accuracy, prior {'KNOWS' if known else 'does NOT know'} the truth")
ret = [r for r in R if r["part"] == "retrieval"]
if ret:
    ns = sorted({r["n_inj"] for r in ret})
    print(f"\nREAL RETRIEVAL (Contriever top-{ret[0]['k']} over BEIR corpus), n questions = {len({r['qid'] for r in ret})}")
    print("avg poisoned passages retrieved: " + "  ".join(f"inject {n}: {sum(r['m'] for r in ret if r['n_inj']==n)/max(1,sum(r['n_inj']==n for r in ret)):.2f}" for n in ns))
    for metric, name in [("acc", "accuracy (%)"), ("asr", "attack success (%)")]:
        print(f"\n{name} by number of injected passages")
        print(f"{'method':<14}" + "".join(f"inj={n:<6}" for n in ns))
        for me in methods:
            line = f"{me:<14}"
            for n in ns:
                v = [r["res"][me][metric] for r in ret if r["n_inj"] == n and me in r["res"]]
                line += f"{100*sum(v)/len(v):6.1f}  " if v else "   -    "
            print(line)
ed = [r for r in R if r["part"] == "edits"]
if ed:
    conds = [c for c in ["orig", "edit", "update"] if any(r["cond"] == c for r in ed)]
    print(f"\nLEGITIMATE EDITS in 5-doc context (edit/original paragraph + same-article distractors), n={len({r['qid'] for r in ed})}")
    print("acc = gives the answer of the (edited) answer paragraph; for 'update' the other answer is the stale one")
    print(f"{'method':<14}" + "".join(f"{c+' acc':>13}" for c in conds) + f"{'update->stale':>15}")
    for me in methods:
        line = f"{me:<14}"
        for c in conds:
            v = [r["res"][me]["acc"] for r in ed if r["cond"] == c and me in r["res"]]
            line += f"{100*sum(v)/len(v):13.1f}" if v else "            -"
        v = [r["res"][me]["asr"] for r in ed if r["cond"] == "update" and me in r["res"]]
        print(line + (f"{100*sum(v)/len(v):15.1f}" if v else ""))
cf = [r for r in R if r["part"] == "counterfactual"]
if cf:
    print(f"\nCOUNTERFACTUAL (Faithfulness-QA, single doc) n={len({r['qid'] for r in cf})}")
    print(f"{'method':<14}{'orig acc':>10}{'cf follow':>11}{'cf reverts to prior':>21}")
    for me in methods:
        o = [r["res"][me]["acc"] for r in cf if r["m"] == 0 and me in r["res"]]
        c = [r["res"][me]["acc"] for r in cf if r["m"] == 1 and me in r["res"]]
        rv = [r["res"][me]["asr"] for r in cf if r["m"] == 1 and me in r["res"]]
        if o and c:
            print(f"{me:<14}{100*sum(o)/len(o):10.1f}{100*sum(c)/len(c):11.1f}{100*sum(rv)/len(rv):21.1f}")
tm = defaultdict(list)
for r in R:
    for k, v in r["time"].items():
        tm[k].append(v)
print("\nmean seconds per instance:", {k: round(sum(v)/len(v), 2) for k, v in tm.items()})
