import json, hashlib
from met.methods import prior_vote
from met.text import score, norm, same_answer
AL, DOCS = {}, {}
for df in ["data/met_known.jsonl","data/met_retrieval.jsonl","data/met.jsonl","data/met_edits.jsonl"]:
    for l in open(df):
        d=json.loads(l); AL[d["id"]]=d.get("gold_aliases"); DOCS[d["id"]]=[x["text"] for x in d["docs"]]
POIS={("known",2),("known",3),("known",5),("retr",1),("retr",3),("retr",5)}
LEGIT={("known",0),("retr",0),("edit","orig"),("edit","edit"),("cf",0),("cf",1)}
STOP=set("a an the of in on at to for from by with and or is are was were be it its as that this his her their".split())
def sup(a,d,thr=0.5):
    t=[w for w in norm(a).split() if w not in STOP]; dn=set(norm(d).split())
    return bool(t) and sum(w in dn for w in t)/len(t)>=thr
def dev(q): return int(hashlib.md5(q.encode()).hexdigest(),16)%10<3
def rows(tag):
    out=[]
    for s,f in {"known":"known_all","retr":"ret_all","edit":"edits","cf":"cf"}.items():
        for l in open(f"results/v3/{tag}_{f}.jsonl"):
            r=json.loads(l); r["set"]=s; r["lv"]=r.get("cond") or (r["n_inj"] if s=="retr" else r["m"]); out.append(r)
    return out
def RU(rs,fn):
    c={}
    for r in rs:
        k=(r["set"],r["lv"])
        if k in POIS or k in LEGIT: c.setdefault(k,[]).append(score(fn(r),r["gold"],r["wrong"],AL.get(r["id"]))["acc"])
    R=[sum(v)/len(v) for k,v in c.items() if k in POIS]; U=[sum(v)/len(v) for k,v in c.items() if k in LEGIT]
    return 100*sum(R)/len(R),100*sum(U)/len(U),{k:round(100*sum(v)/len(v),1) for k,v in c.items()}
G=2.5
def base(r): return prior_vote(r["isolated"],r["prior_answer"],r["prior_conf"],G,r["pred"]["rag"]) if r.get("isolated") else r["pred"]["rag"]
def gate(r):
    if not r.get("isolated"): return r["pred"]["rag"]
    iso=[a if sup(a,d) else "" for a,d in zip(r["isolated"],DOCS[r["id"]])]
    return prior_vote(iso,r["prior_answer"],r["prior_conf"],G,r["pred"]["rag"])
def leak(r):
    if not r.get("isolated"): return r["pred"]["rag"]
    iso=[a if not (same_answer(a,r["prior_answer"]) and not sup(a,d)) else "" for a,d in zip(r["isolated"],DOCS[r["id"]])]
    return prior_vote(iso,r["prior_answer"],r["prior_conf"],G,r["pred"]["rag"])
def abst(r):
    if not r.get("isolated"): return r["pred"]["rag"]
    return prior_vote(r.get("iso_abst") or r["isolated"],r["prior_answer"],r["prior_conf"],G,r["pred"]["rag"])
for split in ["dev","full"]:
    print("==",split)
    for t,n in [("q7","Qwen"),("l8","Llama"),("m7","Mistral")]:
        rs=rows(t)
        if split=="dev": rs=[r for r in rs if dev(r["qid"])]
        b=RU(rs,base)
        for nm,fn in [("gate",gate),("leak",leak),("abst",abst)]:
            v=RU(rs,fn)
            print(f"  {n:<8}{nm:<6} dR {v[0]-b[0]:+5.1f} dU {v[1]-b[1]:+5.1f} | edit-orig {v[2][('edit','orig')]-b[2][('edit','orig')]:+5.1f} edit-edit {v[2][('edit','edit')]-b[2][('edit','edit')]:+5.1f}")
