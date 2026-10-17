"""Part D of MET: realistic retrieval with corpus poisoning (PoisonedRAG black-box setting).

For each PoisonedRAG target question we take the real Contriever top-100 passages from the BEIR corpus,
inject N in {0,1,3,5} attacker passages (question + "." + misleading text, exactly as in PoisonedRAG),
re-rank everything with the same Contriever retriever (dot product) and keep the top-k.
How many poisoned passages reach the top-k is decided by the retriever, not by us.
"""
import argparse, json, os, glob
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModel

p = argparse.ArgumentParser()
p.add_argument("--repos", required=True)
p.add_argument("--out", default="../data/met_retrieval.jsonl")
p.add_argument("--k", type=int, default=5)
p.add_argument("--inject", default="0,1,3,5")
p.add_argument("--datasets", default="nq,hotpotqa,msmarco")
p.add_argument("--verify", type=int, default=5, help="questions per dataset used to verify stored scores")
a = p.parse_args()
INJ = [int(x) for x in a.inject.split(",")]


def hf_file(repo):
    return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/datasets--{repo.replace('/', '--')}/snapshots/*/corpus/*.parquet"))[0]


tok = AutoTokenizer.from_pretrained("facebook/contriever")
enc = AutoModel.from_pretrained("facebook/contriever").eval()
torch.set_num_threads(os.cpu_count())


@torch.no_grad()
def embed(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append((h * m).sum(1) / m.sum(1))  # Contriever mean pooling
    return torch.cat(out)


rows, check = [], []
for ds in a.datasets.split(","):
    adv = json.load(open(f"{a.repos}/PoisonedRAG/results/adv_targeted_results/{ds}.json"))
    res = json.load(open(f"{a.repos}/PoisonedRAG/results/beir_results/{ds}-contriever.json"))
    targets = [x for x in adv.values() if x["correct answer"].strip().lower() not in {"yes", "no"}]
    need = {d for x in targets for d in list(res[x["id"]])[:100]}
    t = pq.read_table(hf_file(f"BeIR/{ds}"), columns=["_id", "title", "text"], filters=[("_id", "in", list(need))])
    corpus = dict(zip(t.column("_id").to_pylist(), t.column("text").to_pylist()))
    titles = dict(zip(t.column("_id").to_pylist(), t.column("title").to_pylist()))
    print(f"{ds}: {len(targets)} targets, {len(need)} benign passages needed, {len(corpus)} found", flush=True)
    for n, x in enumerate(targets):
        q, ids = x["question"], [d for d in list(res[x["id"]])[:100] if d in corpus]
        benign = [corpus[d] for d in ids]
        sb = [res[x["id"]][d] for d in ids]  # PoisonedRAG's stored Contriever dot-product scores (same model)
        poison = [q + "." + t_ for t_ in x["adv_texts"][:max(INJ)]]
        if n < a.verify:  # verify: recompute top-10 benign scores and compare to the stored ones
            E = embed([q] + [titles[d] + " " + corpus[d] for d in ids[:10]] + poison)  # BEIR scores title + text
            check.extend(abs(float(E[1 + i] @ E[0]) - sb[i]) for i in range(min(10, len(benign))))
        else:
            E = embed([q] + poison); E = torch.cat([E[:1], torch.zeros(10, E.shape[1]), E[1:]])
        sp = [float(E[11 + j] @ E[0]) for j in range(len(poison))]
        gold, wrong = x["correct answer"].strip(), x["incorrect answer"].strip()
        for N in INJ:
            pool = [(sb[i], benign[i], False) for i in range(len(benign))] + [(sp[j], poison[j], True) for j in range(N)]
            top = sorted(pool, key=lambda z: -z[0])[:a.k]
            rows.append({"id": f"ret-{ds}-{x['id']}-n{N}", "qid": f"ret-{ds}-{x['id']}", "source": f"ret-{ds}",
                         "part": "retrieval", "n_inj": N, "m": sum(z[2] for z in top), "k": a.k,
                         "question": q, "gold": gold, "wrong": wrong,
                         "gold_in_benign_topk": any(gold.lower() in z[1].lower() for z in top if not z[2]),
                         "docs": [{"text": z[1], "poison": z[2], "score": round(z[0], 4)} for z in top]})
        if n % 20 == 0:
            print(f"  {ds} {n}/{len(targets)}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} instances -> {a.out}")
print(f"verification: max |recomputed - stored| score over {len(check)} passages = {max(check):.4f} (mean {sum(check)/len(check):.5f})")
for N in INJ:
    sub = [r for r in rows if r["n_inj"] == N]
    print(f"inject {N}: mean poisoned in top-{a.k} = {sum(r['m'] for r in sub)/len(sub):.2f}; "
          f"gold present in benign docs = {sum(r['gold_in_benign_topk'] for r in sub)/len(sub):.2f}")
