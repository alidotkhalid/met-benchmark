"""Prior-aligned adaptive attack: accuracy and attack success, compared on the SAME questions with
the standard attack (original PoisonedRAG target)."""
import json, math, sys
from variants import mcnemar

M = ["closed_book", "rag", "cad", "adacad", "arr", "sdag", "vote", "rrag_dec", "rrag_kw", "pvote_g2.5", "pvote_free_g4"]
NAME = {"pvote_g2.5": "MET-Vote", "pvote_free_g4": "MET-Vote free", "rrag_kw": "RRAG-KW", "rrag_dec": "RRAG-Dec",
        "closed_book": "closed-book", "vote": "vote"}


def load(p): return [json.loads(l) for l in open(p)]


def table(A, B, key, levels, title):
    print(f"\n{title}")
    print(f"{'':<14}" + "".join(f"{key}={lv:<11}" for lv in levels))
    print(f"{'':<14}" + "".join(f"{'std':>5} {'adapt':>6}  " for _ in levels))
    for me in M:
        if me not in A[0]["res"]:
            continue
        line = f"{NAME.get(me, me):<14}"
        for lv in levels:
            a = [r for r in A if r[key] == lv]; b = [r for r in B if r[key] == lv]
            va = 100 * sum(r["res"][me]["acc"] for r in a) / len(a) if a else float("nan")
            vb = 100 * sum(r["res"][me]["acc"] for r in b) / len(b) if (b and me in b[0]["res"]) else float("nan")
            line += f"{vb:5.1f} {va:6.1f}  "
        print(line)


tag = sys.argv[1]
A = load(f"results/v3/{tag}_adaptive.jsonl"); AR = load(f"results/v3/{tag}_adaptive_ret.jsonl")
print(f"== {tag}: mixed {len(A)} rows / {len({r['qid'] for r in A})} q ; retrieval {len(AR)} rows / {len({r['qid'] for r in AR})} q")
print("nan prior_conf:", sum(not math.isfinite(r['prior_conf']) for r in A + AR),
      "| closed-book scores (should be 0 by construction):",
      sum(r['res']['closed_book']['acc'] for r in A + AR))
qa = {r["qid"][4:] for r in A}                                   # adv-<qid>
B = [r for r in load(f"results/v3/{tag}_met.jsonl") if r["part"] == "mixed" and r["qid"] in qa]
qr = {"ret-" + r["qid"][7:] for r in AR}                          # advret-<ds-id>
BR = [r for r in load(f"results/v3/{tag}_ret_all.jsonl") if r["qid"] in qr]
table(A, B, "m", [0, 1, 2, 3], "MIXED evidence, accuracy %: standard attack (std) vs prior-aligned attack (adapt), same questions")
table(AR, BR, "n_inj", [0, 1, 3, 5], "REAL RETRIEVAL, accuracy %: standard vs prior-aligned, same questions")
for lab, X, key, lvs in [("mixed", A, "m", [1, 2, 3]), ("retrieval", AR, "n_inj", [1, 3])]:
    print(f"\nMcNemar under the adaptive attack ({lab}): MET-Vote vs baseline  [MET-only/base-only, p]")
    for b in ["vote", "rag", "arr", "rrag_kw", "rrag_dec"]:
        if b in X[0]["res"]:
            print(f"   {NAME.get(b, b):<9}" + "  ".join(f"{key}={lv}: {mcnemar([r for r in X if r[key]==lv], 'pvote_g2.5', b)[0]}/{mcnemar([r for r in X if r[key]==lv], 'pvote_g2.5', b)[1]} p={mcnemar([r for r in X if r[key]==lv], 'pvote_g2.5', b)[2]:.1g}" for lv in lvs))
print("\nattack success (outputs the model's own wrong answer), adaptive, mixed m=2 / retrieval inj=3:")
for me in M:
    s1 = [r for r in A if r["m"] == 2]; s2 = [r for r in AR if r["n_inj"] == 3]
    print(f"   {NAME.get(me, me):<14} {100*sum(r['res'][me]['asr'] for r in s1)/len(s1):5.1f}   {100*sum(r['res'][me]['asr'] for r in s2)/len(s2):5.1f}")
