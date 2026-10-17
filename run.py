"""Run all methods on the Mixed-Evidence Trust benchmark. Resumable; writes one JSON line per instance.

Example (laptop, CPU):
  python run.py --model Qwen/Qwen2.5-1.5B-Instruct --limit_q 40 --out results/qwen1.5b.jsonl
"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from met.methods import LM, rule_cad, rule_adacad, rule_arr, vote, prior_vote, robustrag_keyword, robustrag_decoding
from met.text import score
from met.madam import madam_rag

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", default="data/met.jsonl")
p.add_argument("--out", required=True)
p.add_argument("--limit_q", type=int, default=0, help="max questions per source (0 = all)")
p.add_argument("--parts", default="mixed,counterfactual,retrieval,edits")
p.add_argument("--methods", default="closed_book,rag,cad,adacad,arr,sdag,vote,pvote")
p.add_argument("--gammas", default="0.5,1,1.5,2,2.5,3")
p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
p.add_argument("--device", default="cpu", help="cpu | cuda | auto (spread over all GPUs)")
p.add_argument("--threads", type=int, default=0)
p.add_argument("--max_new", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_hours", type=float, default=0, help="stop cleanly (exit 0) once this much wall time has passed; 0 = no limit")
a = p.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
methods = a.methods.split(",")
gammas = [float(g) for g in a.gammas.split(",")]

rows = [json.loads(l) for l in open(a.data)]
rows = [r for r in rows if r["part"] in a.parts.split(",")]
if a.limit_q:
    keep, per = set(), {}
    for r in rows:
        if r["qid"] not in keep and per.get(r["source"], 0) < a.limit_q:
            keep.add(r["qid"]); per[r["source"]] = per.get(r["source"], 0) + 1
    rows = [r for r in rows if r["qid"] in keep]

done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

tok = AutoTokenizer.from_pretrained(a.model)
kw = {} if a.device == "cpu" else {"device_map": "auto" if a.device == "auto" else a.device}
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype),
                                             attn_implementation="eager", **kw)
model.eval()
lm = LM(model, tok, max_new=a.max_new)
print(f"model={a.model} instances={len(rows)} done={len(done)} methods={methods}", flush=True)

prior_cache = {}  # qid -> (answer, confidence)


def prior(r):
    if r["qid"] not in prior_cache:
        ids = lm.prompt_noctx(r["question"])
        ans = lm.greedy(ids)
        ll = lm.answer_loglik(ids, [ans])[0] if ans else -10.0
        if not math.isfinite(ll):
            raise FloatingPointError(f"Non-finite prior log-likelihood for {r['qid']}; numeric precision problem.")
        prior_cache[r["qid"]] = (ans, math.exp(ll))
    return prior_cache[r["qid"]]


t0 = time.time()
with open(a.out, "a") as f:
    for n, r in enumerate(rows):
        if r["id"] in done:
            continue
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            print(f"TIME BUDGET REACHED after {(time.time()-t0)/3600:.2f} h: stopping cleanly; re-run with the same --out to resume",
                  flush=True)
            break
        q, docs = r["question"], [d["text"] for d in r["docs"]]
        pa, pconf = prior(r)
        pred, timing = {}, {}

        def run(name, fn):
            s = time.time(); pred[name] = fn(); timing[name] = round(time.time() - s, 2)

        ids_ctx, ids_pri = lm.prompt_ctx(q, docs), lm.prompt_noctx(q)
        if "closed_book" in methods: pred["closed_book"] = pa
        if "rag" in methods: run("rag", lambda: lm.greedy(ids_ctx))
        if "cad" in methods: run("cad", lambda: lm.contrastive(ids_ctx, ids_pri, rule_cad(1.0)))
        if "adacad" in methods: run("adacad", lambda: lm.contrastive(ids_ctx, ids_pri, rule_adacad()))
        if "arr" in methods: run("arr", lambda: lm.contrastive(ids_ctx, ids_pri, rule_arr()))
        if "sdag" in methods and len(docs) > 1: run("sdag", lambda: lm.sdag(q, docs))
        elif "sdag" in methods: pred["sdag"] = pred.get("rag") or lm.greedy(ids_ctx)
        iso = iso_conf = None
        if "vote" in methods or "pvote" in methods:
            s = time.time()
            iso_c = [lm.greedy_conf(lm.prompt_ctx(q, [d])) for d in docs]
            iso, iso_conf = [x[0] for x in iso_c], [round(x[1], 4) for x in iso_c]
            timing["isolated"] = round(time.time() - s, 2)
            fb = pred.get("rag", iso[0])
            if "vote" in methods: pred["vote"] = vote(iso, fb)
            if "pvote" in methods:
                for g in gammas:
                    pred[f"pvote_g{g:g}"] = prior_vote(iso, pa, pconf, g, fb)
                pred["pvote_free_g4"] = prior_vote(iso, pa, pconf, 4, fb, grounded=False)
        iso_abst = None
        if "rrag_kw" in methods:
            s = time.time(); pred["rrag_kw"], iso_abst = robustrag_keyword(lm, q, docs); timing["rrag_kw"] = round(time.time() - s, 2)
        if "rrag_dec" in methods: run("rrag_dec", lambda: robustrag_decoding(lm, q, docs))
        madam_rec = None
        if "madam" in methods:
            s = time.time()
            pred["madam"], pred["madam_first"], madam_rec = madam_rag(lm, q, docs)
            timing["madam"] = round(time.time() - s, 2)
        res = {k: score(v, r["gold"], r["wrong"], r.get("gold_aliases")) for k, v in pred.items()}
        cf_edit = r["part"] == "counterfactual" and r["m"] == 1   # only here is 'gold' the edited (non-true) answer
        prior_known = score(pa, r["wrong"] if cf_edit else r["gold"], r["gold"] if cf_edit else r["wrong"],
                            None if cf_edit else r.get("gold_aliases"))["acc"]
        out = {k: r[k] for k in ["id", "qid", "source", "part", "m", "k", "gold", "wrong"]}
        out.update({k: r[k] for k in ["n_inj", "gold_in_benign_topk", "cond"] if k in r})
        out.update(prior_answer=pa, prior_conf=round(pconf, 4), prior_knows_truth=prior_known,
                   isolated=iso, iso_conf=iso_conf, iso_abst=iso_abst, madam=madam_rec, pred=pred, res=res, time=timing)
        f.write(json.dumps(out) + "\n"); f.flush()
        el = time.time() - t0
        print(f"[{n+1}/{len(rows)}] {r['id']} m={r['m']} {el/60:.1f}min "
              + " ".join(f"{k}:{v['acc']}" for k, v in res.items() if not k.startswith("pvote_g") or k == "pvote_g1.5"),
              flush=True)
