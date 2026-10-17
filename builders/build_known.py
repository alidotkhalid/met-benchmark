"""Part C of MET: well-known facts (TriState-Bench fact library), so that the model's prior is usually right.
Poisoned passages = original misleading passage + LLM rewordings (tristate/paraphrase.py).
Benign passages   = the same passages with the wrong answer swapped back to the true one (style-matched),
                    including bare-surname mentions for person answers (e.g. "Adams" -> "Washington")."""
import json, re, random, argparse

p = argparse.ArgumentParser()
p.add_argument("--passages", default="tristate/passages.jsonl")
p.add_argument("--out", default="data/met_known.jsonl")
p.add_argument("--configs", default="0,1,2,3,5")
p.add_argument("--seed", type=int, default=13)
a = p.parse_args()
rng, K = random.Random(a.seed), 5
CONF = [int(x) for x in a.configs.split(",")]


def rx(s):
    return re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.I)


def swap(t, wrong, gold, typ):
    t = rx(wrong).sub(gold, t)
    if typ == "person" and len(wrong.split()) > 1 and len(gold.split()) > 0:
        wl, gl = wrong.split()[-1], gold.split()[-1]
        t = re.sub(r"(?<![\w])" + re.escape(wl) + r"(?=('s)?(?![\w]))", gl, t)
    return t


rows, kept, dropped = [], 0, 0
for l in open(a.passages):
    x = json.loads(l)
    gold, wrong, P = x["gold"][0], x["wrong"], x["passages"]
    if len(P) < K:
        dropped += 1; continue
    benign = [swap(t, wrong, gold, x["type"]) for t in P[:K]]
    wl = wrong.split()[-1]
    if any(rx(wrong).search(b) or (x["type"] == "person" and rx(wl).search(b)) for b in benign):
        dropped += 1; continue
    kept += 1
    for m in CONF:
        docs = [{"text": P[i], "poison": True} for i in range(m)] + [{"text": benign[i], "poison": False} for i in range(m, K)]
        rng.shuffle(docs)
        rows.append({"id": f"tri-{x['fact_id']}-m{m}", "qid": f"tri-{x['fact_id']}", "source": "tristate", "part": "mixed",
                     "m": m, "k": K, "question": x["question"], "gold": gold, "gold_aliases": x["gold"], "wrong": wrong,
                     "docs": docs})
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"facts kept {kept}, dropped {dropped}; instances {len(rows)} -> {a.out}")
