"""Keep facts whose closed-book answer (target model, same prompt as run.py) is already correct, so that
Part C tests the regime where the model's prior is right but the retrieved evidence may be poisoned.
Also drops facts where the 'wrong' answer is a near-synonym of the true one (would make poison arguably correct)."""
import json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from met.methods import LM
from met.text import contains, same_answer

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/met_known_all.jsonl")
p.add_argument("--out", default="data/met_known.jsonl")
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--dtype", default="float32")
p.add_argument("--max_facts", type=int, default=100)
p.add_argument("--configs", default="0,2,3,5")
a = p.parse_args()
rows = [json.loads(l) for l in open(a.data)]
tok = AutoTokenizer.from_pretrained(a.model)
lm = LM(AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).eval(), tok)
qs = {}
for r in rows:
    qs.setdefault(r["qid"], r)
known, stats = set(), {"known": 0, "unknown": 0, "synonym": 0}
for qid, r in qs.items():
    if same_answer(r["gold"], r["wrong"]) or any(same_answer(g, r["wrong"]) for g in r.get("gold_aliases", [])):
        stats["synonym"] += 1; continue
    ans = lm.greedy(lm.prompt_noctx(r["question"]))
    ok = any(contains(ans, g) for g in [r["gold"]] + r.get("gold_aliases", [])) and not contains(ans, r["wrong"])
    stats["known" if ok else "unknown"] += 1
    if ok and len(known) < a.max_facts:
        known.add(qid)
    print(f"{qid}: {ans!r} -> {'KNOWN' if ok else 'unknown'}", flush=True)
conf = {int(x) for x in a.configs.split(",")}
out = [r for r in rows if r["qid"] in known and r["m"] in conf]
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(stats, f"kept facts {len(known)}, instances {len(out)} -> {a.out}")
