"""Part E of MET: legitimate context edits inside a realistic multi-document context (k=5).

Faithfulness-QA gives, for a SQuAD question, the original paragraph and a counterfactual paragraph in which
the answer entity is substituted. Real retrieved sets rarely contain only the answer passage, so we surround it
with hard distractors: other paragraphs of the same Wikipedia article, ranked by BM25 similarity to the
question, that contain neither the original nor the edited answer.
Conditions per question (document order shuffled, seed fixed):
  orig   : original paragraph + 4 distractors          (gold = original answer)       - control
  edit   : edited paragraph   + 4 distractors          (gold = edited answer)         - must follow the edit
  update : edited paragraph + original (stale) paragraph + 3 distractors
           (gold = edited answer, wrong = original)   - a newer source contradicts an older one; 1 vs 1
The model sees nothing that marks a document as edited; the edit is only 'legitimate' by construction.
"""
import argparse, json, math, random, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--met", default="data/met.jsonl")
p.add_argument("--out", default="data/met_edits.jsonl")
p.add_argument("--n", type=int, default=200)
p.add_argument("--k", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rnd = random.Random(a.seed)

pairs = {}
for l in open(a.met):
    r = json.loads(l)
    if r["part"] == "counterfactual":
        pairs.setdefault(r["qid"], {})["cf" if r["m"] == 1 else "orig"] = r
sq = pq.read_table(hf_hub_download("rajpurkar/squad", "plain_text/train-00000-of-00001.parquet",
                                   repo_type="dataset")).to_pandas()
title_of = dict(zip(sq["id"], sq["title"]))
paras = sq.groupby("title")["context"].apply(lambda s: list(dict.fromkeys(s))).to_dict()

STOP = set("a an the of in on at to for from by with and or but is are was were be been what which who whom whose when where why how did does do that this it its as".split())


def toks(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP]


def bm25(query, docs, k1=1.5, b=0.75):
    D = [toks(d) for d in docs]; avg = sum(map(len, D)) / len(D); N = len(D)
    df = {}
    for d in D:
        for w in set(d): df[w] = df.get(w, 0) + 1
    out = []
    for d in D:
        tf = {}
        for w in d: tf[w] = tf.get(w, 0) + 1
        out.append(sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
                       for w in set(toks(query)) if w in tf))
    return out


def has(text, ans):
    return re.search(re.escape(ans.lower()), text.lower()) is not None


rows, qids = [], sorted(pairs)
rnd.shuffle(qids)
for qid in qids:
    if len(rows) >= 3 * a.n:
        break
    o, c = pairs[qid]["orig"], pairs[qid]["cf"]
    orig_txt, cf_txt, ans_o, ans_e = o["docs"][0]["text"], c["docs"][0]["text"], o["gold"], c["gold"]
    cand = [t for t in paras[title_of[qid[4:]]] if t.strip() != orig_txt.strip()
            and not has(t, ans_o) and not has(t, ans_e) and len(t.split()) <= 180]
    if len(cand) < a.k - 1:
        continue
    s = bm25(o["question"], cand)
    dis = [t for _, t in sorted(zip(s, cand), key=lambda z: -z[0])][:a.k - 1]
    for cond, docs, gold, wrong in [
            ("orig", [(orig_txt, "answer")] + [(d, "distractor") for d in dis], ans_o, ans_e),
            ("edit", [(cf_txt, "edit")] + [(d, "distractor") for d in dis], ans_e, ans_o),
            ("update", [(cf_txt, "edit"), (orig_txt, "stale")] + [(d, "distractor") for d in dis[:a.k - 2]], ans_e, ans_o)]:
        docs = docs[:]; rnd.shuffle(docs)
        rows.append({"id": f"edit-{qid[4:]}-{cond}", "qid": f"edit-{qid[4:]}", "source": "edit-squad", "part": "edits",
                     "cond": cond, "m": {"orig": 0, "edit": 1, "update": 2}[cond], "k": a.k,
                     "question": o["question"], "gold": gold, "wrong": wrong,
                     "docs": [{"text": t, "role": role, "poison": role in ("edit",)} for t, role in docs]})
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} instances ({len(rows)//3} questions) -> {a.out}")
