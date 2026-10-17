"""Adaptive attack A1-R: prior-aligned targeting in the realistic retrieval setting.

Same threat model as attack/build_prior_aligned.py -- the attacker spends one closed-book query per
question and adopts the model's own (wrong) answer as its target -- but now the poisoned passages
are injected into the real BEIR corpus and compete for the Contriever top-5 against the real
benign passages. How many of them reach the context is decided by the retriever.

Per model, since the target depends on that model's prior.
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _equiv import equivalent
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel

p = argparse.ArgumentParser()
p.add_argument("--repos", required=True)
p.add_argument("--results", default="results/v3")
p.add_argument("--outdir", default="data")
p.add_argument("--k", type=int, default=5)
p.add_argument("--inject", default="0,1,3,5")
p.add_argument("--datasets", default="nq,hotpotqa,msmarco")
a = p.parse_args()
INJ = [int(x) for x in a.inject.split(",")]
MODELS = {"q7": "Qwen2.5-7B", "l8": "Llama-3.1-8B", "m7": "Mistral-7B"}
BAD = re.compile(r"^(i |no |not |none|unknown|unclear|n/?a\b|there is no|cannot)", re.I)


def wb(s):
    return re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.I)


def hf_file(repo):
    return glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{repo.replace('/', '--')}/snapshots/*/corpus/*.parquet"))[0]


tok = AutoTokenizer.from_pretrained("facebook/contriever")
enc = AutoModel.from_pretrained("facebook/contriever", use_safetensors=False).eval()
torch.set_num_threads(os.cpu_count())


@torch.no_grad()
def embed(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append((h * m).sum(1) / m.sum(1))
    return torch.cat(out)


# the model's closed-book answer on each retrieval question, and whether it was right
prior = {tag: {} for tag in MODELS}
for tag in MODELS:
    for l in open(f"{a.results}/{tag}_ret_all.jsonl"):
        r = json.loads(l)
        prior[tag][r["qid"]] = (r["prior_answer"], r["prior_knows_truth"])

rows = {tag: [] for tag in MODELS}
stats = {tag: {"kept": 0, "prior_correct": 0, "degenerate": 0, "collides": 0, "no_subst": 0} for tag in MODELS}
for ds in a.datasets.split(","):
    adv = json.load(open(f"{a.repos}/PoisonedRAG/results/adv_targeted_results/{ds}.json"))
    res = json.load(open(f"{a.repos}/PoisonedRAG/results/beir_results/{ds}-contriever.json"))
    targets = [x for x in adv.values() if x["correct answer"].strip().lower() not in {"yes", "no"}]
    need = {d for x in targets for d in list(res[x["id"]])[:100]}
    t = pq.read_table(hf_file(f"BeIR/{ds}"), columns=["_id", "text"], filters=[("_id", "in", list(need))])
    corpus = dict(zip(t.column("_id").to_pylist(), t.column("text").to_pylist()))
    print(f"{ds}: {len(targets)} targets, {len(corpus)} benign passages", flush=True)
    for n, x in enumerate(targets):
        qid = f"ret-{ds}-{x['id']}"
        q, gold, wrong = x["question"], x["correct answer"].strip(), x["incorrect answer"].strip()
        ids = [d for d in list(res[x["id"]])[:100] if d in corpus]
        benign = [corpus[d] for d in ids]
        sb = [res[x["id"]][d] for d in ids]
        qemb = None
        for tag in MODELS:
            pa, knows = prior[tag].get(qid, (None, 1))
            if pa is None:
                continue
            P = pa.strip().rstrip(".").strip()
            if knows:
                stats[tag]["prior_correct"] += 1; continue
            if not P or len(P.split()) > 6 or BAD.match(P):
                stats[tag]["degenerate"] += 1; continue
            if equivalent(P, gold) or equivalent(P, wrong) or wb(gold).search(P) or wb(P).search(gold) \
                or wb(P).search(wrong) or wb(wrong).search(P):
                stats[tag]["collides"] += 1; continue
            advs = [t_ for t_ in x["adv_texts"][:max(INJ)] if wb(wrong).search(t_)]
            if len(advs) < max(INJ):
                stats[tag]["no_subst"] += 1; continue
            poison = [q + "." + wb(wrong).sub(P, t_) for t_ in advs]
            if qemb is None:
                qemb = embed([q])[0]
            pe = embed(poison)
            sp = [float(pe[j] @ qemb) for j in range(len(poison))]
            stats[tag]["kept"] += 1
            for N in INJ:
                pool = [(sb[i], benign[i], False) for i in range(len(benign))] + \
                       [(sp[j], poison[j], True) for j in range(N)]
                top = sorted(pool, key=lambda z: -z[0])[:a.k]
                rows[tag].append({"id": f"advret-{ds}-{x['id']}-n{N}", "qid": f"advret-{ds}-{x['id']}",
                                  "source": f"ret-{ds}", "part": "retrieval", "attack": "prior_aligned",
                                  "n_inj": N, "m": sum(z[2] for z in top), "k": a.k,
                                  "question": q, "gold": gold, "wrong": P,
                                  "prior_target": P, "orig_wrong": wrong,
                                  "gold_in_benign_topk": any(gold.lower() in z[1].lower() for z in top if not z[2]),
                                  "docs": [{"text": z[1], "poison": z[2], "score": round(z[0], 4)} for z in top]})
        if n % 25 == 0:
            print(f"  {ds} {n}/{len(targets)}", flush=True)

for tag, full in MODELS.items():
    path = os.path.join(a.outdir, f"met_adaptive_ret_{tag}.jsonl")
    with open(path, "w") as f:
        for r in rows[tag]:
            f.write(json.dumps(r) + "\n")
    sub = [r for r in rows[tag] if r["n_inj"] == max(INJ)]
    mp = sum(r["m"] for r in sub) / max(len(sub), 1)
    print(f"{full}: {stats[tag]} -> {len(rows[tag])} instances ({stats[tag]['kept']} questions), "
          f"mean poisoned in top-{a.k} at inject {max(INJ)} = {mp:.2f} -> {path}")
