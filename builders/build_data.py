"""Build the Mixed-Evidence Trust benchmark (MET) from public sources.

Part A (mixed evidence, k=5 docs): PoisonedRAG adversarial passages (NQ, HotpotQA, MS MARCO).
  Each question has 5 fluent passages supporting a wrong answer. We make style-matched
  *benign* passages by swapping the wrong answer back to the correct one, so benign and
  poisoned passages differ only in the answer (no stylistic shortcut).
  Config m = number of poisoned passages among k=5 (disjoint passage indices, shuffled order).

Part B (single-doc counterfactual, Faithfulness-QA SQuAD test): original vs. entity-substituted
  context. Measures whether a method still follows a coherent context that contradicts the prior.
"""
import json, random, re, os, argparse

p = argparse.ArgumentParser()
p.add_argument("--repos", required=True)
p.add_argument("--out", default="data")
p.add_argument("--configs", default="0,1,2,3,5")
p.add_argument("--n_fqa", type=int, default=300)
p.add_argument("--seed", type=int, default=13)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
K = 5
CONFIGS = [int(x) for x in a.configs.split(",")]


def wb(s):
    return re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.I)


met, dropped = [], {}
for ds in ["nq", "hotpotqa", "msmarco"]:
    d = json.load(open(f"{a.repos}/PoisonedRAG/results/adv_targeted_results/{ds}.json"))
    for qid, x in d.items():
        gold, wrong = x["correct answer"].strip(), x["incorrect answer"].strip()
        why = None
        if gold.lower() in {"yes", "no"} or wrong.lower() in {"yes", "no"}:
            why = "yesno"
        elif len(gold) < 2 or len(wrong) < 2 or gold.lower() in wrong.lower() or wrong.lower() in gold.lower():
            why = "overlap"
        advs = x["adv_texts"][:K]
        if not why and (len(advs) < K or not all(wb(wrong).search(t) for t in advs)):
            why = "wrong_not_in_text"
        if not why and any(wb(gold).search(t) for t in advs):
            why = "gold_in_adv"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        benign = [wb(wrong).sub(gold, t) for t in advs]
        for m in CONFIGS:
            docs = [{"text": advs[i], "poison": True} for i in range(m)] + \
                   [{"text": benign[i], "poison": False} for i in range(m, K)]
            rng.shuffle(docs)
            met.append({"id": f"{ds}-{qid}-m{m}", "qid": f"{ds}-{qid}", "source": ds, "part": "mixed",
                        "m": m, "k": K, "question": x["question"], "gold": gold, "wrong": wrong,
                        "docs": docs})

fqa = [json.loads(l) for l in open(f"{a.repos}/faithfulness-qa-dataset/data/faithfulness_qa_squad_test.jsonl")]
fqa = [x for x in fqa if len(x["original_context"].split()) <= 180 and x["faithful_answer"] in x["modified_context"]
       and x["original_answer"] in x["original_context"]]
rng.shuffle(fqa)
for x in fqa[: a.n_fqa]:
    for variant, ctx, tgt, oth in [("orig", x["original_context"], x["original_answer"], x["faithful_answer"]),
                                   ("cf", x["modified_context"], x["faithful_answer"], x["original_answer"])]:
        met.append({"id": f"fqa-{x['id']}-{variant}", "qid": f"fqa-{x['id']}", "source": "fqa-squad",
                    "part": "counterfactual", "m": 1 if variant == "cf" else 0, "k": 1,
                    "question": x["question"], "gold": tgt, "wrong": oth,
                    "docs": [{"text": ctx, "poison": variant == "cf"}]})

with open(f"{a.out}/met.jsonl", "w") as f:
    for r in met:
        f.write(json.dumps(r) + "\n")
nq = len({r["qid"] for r in met if r["part"] == "mixed"})
print(f"mixed questions kept: {nq}  dropped: {dropped}")
print(f"instances: {len(met)}  (mixed={sum(r['part']=='mixed' for r in met)}, cf={sum(r['part']=='counterfactual' for r in met)})")
