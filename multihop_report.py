"""Multi-hop component: accuracy / ASR by condition, R/U, paired tests, and isolated-reading diagnostics."""
import json, sys
from variants import mcnemar
from met.text import contains, is_nonanswer

TAG = sys.argv[1] if len(sys.argv) > 1 else "l8"
D = {json.loads(l)["id"]: json.loads(l) for l in open("data/met_multihop.jsonl")}
rows = [json.loads(l) for l in open(f"results/v3/{TAG}_multihop.jsonl")]
M = ["closed_book", "rag", "cad", "adacad", "arr", "sdag", "vote", "rrag_dec", "rrag_kw", "pvote_g0", "pvote_g2.5", "pvote_free_g4"]
M = [m for m in M if m in rows[0]["res"]]
C = ["orig", "edit", "poison1", "poison2"]
pc = lambda rs, m, k="acc": 100 * sum(r["res"][m][k] for r in rs) / len(rs)
print(f"== {TAG}: {len(rows)} instances, {len({r['qid'] for r in rows})} questions   (acc / ASR)")
print(f"{'':14}" + "".join(f"{c:>14}" for c in C) + f"{'U':>7}{'R':>7}")
for m in M:
    v = {c: [r for r in rows if r["cond"] == c] for c in C}
    U = (pc(v["orig"], m) + pc(v["edit"], m)) / 2
    R = (pc(v["poison1"], m) + pc(v["poison2"], m)) / 2
    print(f"{m:14}" + "".join(f"{pc(v[c], m):6.1f}/{pc(v[c], m, 'asr'):5.1f}  " for c in C) + f"{U:7.1f}{R:7.1f}")
print("\npaired (MET-Vote g2.5 only-correct / other only-correct, p):")
for c in C:
    rs = [r for r in rows if r["cond"] == c]
    print(" ", c, {b: mcnemar(rs, "pvote_g2.5", b)[:3] for b in ["rag", "vote", "rrag_kw", "arr"]})
# isolated readings
print("\nisolated readings (orig): by document role")
stat = {}
for r in rows:
    if r["cond"] != "orig" or not r["isolated"]:
        continue
    for d, a in zip(D[r["id"]]["docs"], r["isolated"]):
        s = stat.setdefault(d["role"], [0, 0, 0])
        s[0] += 1; s[1] += contains(a, r["gold"]); s[2] += not is_nonanswer(a)
for k, (n, g, ans) in stat.items():
    print(f"  {k:10} n={n:4}  contains gold {100*g/n:5.1f}%   substantive answer {100*ans/n:5.1f}%")
o = [r for r in rows if r["cond"] == "orig"]
anyg = sum(any(contains(a, r["gold"]) for a in r["isolated"]) for r in o)
print(f"  orig: some isolated reading contains gold in {anyg}/{len(o)}; joint RAG correct {sum(r['res']['rag']['acc'] for r in o)}")
