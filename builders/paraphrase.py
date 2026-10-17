"""Create 5 style-varied misleading passages per TriState fact (original + LLM rewordings),
keeping only rewordings that still state the wrong answer and never mention the true one."""
import json, glob, re, sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p = argparse.ArgumentParser()
p.add_argument("--tristate", required=True); p.add_argument("--out", required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); p.add_argument("--limit", type=int, default=0)
p.add_argument("--dtype", default="float32")
a = p.parse_args()
facts = {}
for f in sorted(glob.glob(f"{a.tristate}/*/C_wrong_P_right.jsonl") + glob.glob(f"{a.tristate}/*/data/C_wrong_P_right.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("assigned_process") != 0: continue
        ctx = r["context_string"].split("\nUsing only the references")[0].strip()
        q = re.search(r"Question: (.*?)\s*\?*\s*\nAnswer:", r["context_string"])
        if q and r["fact_id"] not in facts:
            facts[r["fact_id"]] = dict(fact_id=r["fact_id"], type=r["answer_type"], question=q.group(1).strip().rstrip("?") + "?",
                                       gold=r["gold_answers"], wrong=r["wrong_answer"], ctx=ctx)
import os, random
F = list(facts.values())
random.Random(7).shuffle(F)
first = [json.loads(l)["fact_id"] for l in open(a.out)] if os.path.exists(a.out) else []
F = [x for x in F if x["fact_id"] in first] + [x for x in F if x["fact_id"] not in first]  # keep facts already done
F = F[: a.limit or None]
done = set()
try: done = {json.loads(l)["fact_id"] for l in open(a.out)}
except FileNotFoundError: pass
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).eval()
def has(t, s): return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t, re.I) is not None
torch.manual_seed(0)
with open(a.out, "a") as f:
    for i, x in enumerate(F):
        if x["fact_id"] in done: continue
        msgs = [{"role": "system", "content": "You rewrite encyclopedia passages. Keep every claim exactly as stated, even if you believe it is wrong. Change wording and sentence order. Output only the rewritten passage."},
                {"role": "user", "content": x["ctx"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        keep = []
        for attempt in range(2):
            with torch.no_grad():
                out = m.generate(ids, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=130, num_return_sequences=6,
                                 pad_token_id=tok.eos_token_id)
            for o in out:
                t = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip().split("\n\n")[0].strip()
                if len(t.split()) < 20 or not has(t, x["wrong"]) or any(has(t, g) for g in x["gold"]): continue
                if all(t != k for k in keep): keep.append(t)
            if len(keep) >= 4: break
        x["passages"] = [x["ctx"]] + keep[:4]
        f.write(json.dumps(x) + "\n"); f.flush()
        print(f"[{i+1}/{len(F)}] {x['fact_id']} kept {len(keep)}", flush=True)
