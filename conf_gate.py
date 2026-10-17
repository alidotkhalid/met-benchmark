"""Prior-confidence gate (Section 9): can the model's confidence in its closed-book answer separate the priors an
attacker can exploit (wrong closed-book answers, the prior-aligned attack's targets) from correct ones?
Wrong priors: the questions of the adaptive attack (mixed + retrieval), one per question.
Right priors: questions of known facts, PoisonedRAG mixed and realistic retrieval whose closed-book answer is correct.
Threshold: the one maximising Youden's J = (share of right priors kept) - (share of wrong priors admitted)."""
import json, statistics
L = lambda f: [json.loads(l) for l in open(f)]
for t in ["q7", "l8", "m7"]:
    wrong = {}
    for f in [f"results/v3/{t}_adaptive.jsonl", f"results/v3/{t}_adaptive_ret.jsonl"]:
        for r in L(f):
            wrong.setdefault(r["qid"], r["prior_conf"])
    right = {}
    for f in [f"results/v3/{t}_known_all.jsonl", f"results/v3/{t}_met.jsonl", f"results/v3/{t}_ret_all.jsonl"]:
        for r in L(f):
            if r.get("part", "") == "counterfactual":
                continue
            if r["res"]["closed_book"]["acc"]:
                right.setdefault(r["qid"], r["prior_conf"])
    W, Rt = list(wrong.values()), list(right.values())
    best = max(sorted(set(W + Rt)), key=lambda th: sum(x >= th for x in Rt) / len(Rt) - sum(x >= th for x in W) / len(W))
    keep = 100 * sum(x >= best for x in Rt) / len(Rt); adm = 100 * sum(x >= best for x in W) / len(W)
    print(f"{t}: wrong n={len(W)} median {statistics.median(W):.2f} | right n={len(Rt)} median {statistics.median(Rt):.2f} | "
          f"best threshold {best:.3f} keeps {keep:.0f}% of right priors, admits {adm:.0f}% of wrong")
