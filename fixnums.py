import json, hashlib
from met.methods import prior_vote, cluster_votes
from met.text import score, norm, same_answer, is_nonanswer, contains
from variants import mcnemar

AL, DOCS = {}, {}
for df in ["data/met_known.jsonl","data/met_retrieval.jsonl","data/met.jsonl","data/met_edits.jsonl"]:
    for l in open(df):
        d=json.loads(l); AL[d["id"]]=d.get("gold_aliases"); DOCS[d["id"]]=[x["text"] for x in d["docs"]]
M=[("q7","Qwen"),("l8","Llama"),("m7","Mistral")]
ld=lambda t,f:[json.loads(l) for l in open(f"results/v3/{t}_{f}.jsonl")]

print("== (a) rrag_kw correct with NO per-passage reading giving gold")
for t,n in M:
    for f,key,lv in [("known_all","m",5),("ret_all","n_inj",5)]:
        R=[r for r in ld(t,f) if r[key]==lv]
        ok=[r for r in R if r["res"]["rrag_kw"]["acc"]]
        golds=lambda r:[r["gold"]]+(AL.get(r["id"]) or [])
        un=[r for r in ok if not any(contains(a,g) for a in (r["iso_abst"] or []) for g in golds(r))]
        print(f"  {n} {f}{lv}: correct {len(ok)}, none of the per-passage readings gave gold: {len(un)}")

print("== (b) distinct questions in benchmark")
ids=set()
for f,part in [("data/met_known.jsonl",None),("data/met.jsonl","mixed"),("data/met_retrieval.jsonl",None),("data/met_edits.jsonl",None),("data/met.jsonl","counterfactual")]:
    for l in open(f):
        d=json.loads(l)
        if part and d["part"]!=part: continue
        q=d["qid"]
        for p in ("edit-","fqa-","ret-nq-","ret-hotpotqa-","ret-msmarco-"):
            if q.startswith(p): q=q[len(p):]
        ids.add(q)
print("  distinct underlying question ids:",len(ids))

print("== (c) edits orig/edit gap ours vs rag")
for t,n in M:
    E=ld(t,"edits")
    for c in ["orig","edit"]:
        s=[r for r in E if r["cond"]==c]
        o=100*sum(r["res"]["pvote_g2.5"]["acc"] for r in s)/len(s); g=100*sum(r["res"]["rag"]["acc"] for r in s)/len(s)
        print(f"  {n} {c}: ours {o:.1f} rag {g:.1f} gap {g-o:.1f}")

print("== (d) no-abstention-filter ablation, edits orig (final method minus the filter)")
import met.text as T
orig_fn=T.is_nonanswer
for t,n in M:
    E=[r for r in ld(t,"edits") if r["cond"]=="orig"]
    T.is_nonanswer=lambda s: not norm(s)
    import met.methods as MM; MM.is_nonanswer=T.is_nonanswer
    v=[score(prior_vote(r["isolated"],r["prior_answer"],r["prior_conf"],2.5,r["pred"]["rag"]),r["gold"],r["wrong"],AL.get(r["id"]))["acc"] for r in E]
    T.is_nonanswer=orig_fn; MM.is_nonanswer=orig_fn
    print(f"  {n}: {100*sum(v)/len(v):.1f}")

print("== (e) p-values, known m=5, ours vs rrag_kw")
for t,n in M:
    R=[r for r in ld(t,"known_all") if r["m"]==5]
    print(f"  {n}: {mcnemar(R,'pvote_g2.5','rrag_kw')}")
