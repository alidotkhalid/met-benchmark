"""Adaptive attack A1: prior-aligned target selection.

Threat model. The attacker knows (i) the generator, which is public, and (ii) that the defence
weights the generator's closed-book answer. It queries the generator closed-book once per target
question --- a black-box query any user can make --- and, whenever that answer is wrong, adopts it
as the attack target instead of an arbitrary incorrect answer. Every prior-leaning defence then
votes for the attacker.

Construction is otherwise identical to the PoisonedRAG mixed-evidence component: the poisoned
passage is a style-matched paraphrase asserting the target, produced by substituting the target
into the same template that produces the benign passage. No new text generation is involved, so
the attack costs the attacker one closed-book query per question and nothing else.

Writes one file per model, since the attack depends on that model's prior.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _equiv import equivalent

p = argparse.ArgumentParser()
p.add_argument("--met", default="data/met.jsonl")
p.add_argument("--results", default="results/v3")
p.add_argument("--outdir", default="data")
p.add_argument("--configs", default="0,1,2,3,4")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
CONF = [int(x) for x in a.configs.split(",")]
MODELS = {"q7": "Qwen2.5-7B-Instruct", "l8": "Llama-3.1-8B-Instruct", "m7": "Mistral-7B-Instruct-v0.3"}
BAD = re.compile(r"^(i |no |not |none|unknown|unclear|n/?a\b|there is no|cannot)", re.I)


def wb(s):
    return re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.I)


def tidy(target, text, gold):
    """Substitute the target for the gold answer, dropping any tail of the target that the
    following context already supplies (prior answers are free text: '22 episodes' into
    '...total of 23 episodes' must not yield '22 episodes episodes')."""
    out, pos = [], 0
    for mt in wb(gold).finditer(text):
        tail = text[mt.end():]
        t = target
        tw = t.split()
        fw = tail.split()
        for n in range(min(len(tw), len(fw)), 0, -1):
            if [w.lower().strip(".,;:") for w in tw[-n:]] == [w.lower().strip(".,;:") for w in fw[:n]]:
                t = " ".join(tw[:-n]) or target
                break
        out.append(text[pos:mt.start()] + t)
        pos = mt.end()
    return "".join(out) + text[pos:]


rows_by_q = {}
for l in open(a.met):
    r = json.loads(l)
    if r["part"] == "mixed":
        rows_by_q.setdefault(r["qid"], {})[r["m"]] = r

for tag, full in MODELS.items():
    rng = random.Random(a.seed)
    prior = {}
    for l in open(f"{a.results}/{tag}_met.jsonl"):
        r = json.loads(l)
        if r["part"] == "mixed":
            prior[r["qid"]] = (r["prior_answer"], r["prior_knows_truth"])
    out, stats = [], {"kept": 0, "prior_correct": 0, "prior_degenerate": 0, "collides": 0, "no_subst": 0}
    for qid, bym in sorted(rows_by_q.items()):
        src = bym[0]                      # m=0: five benign, style-matched passages asserting the gold
        gold, wrong = src["gold"], src["wrong"]
        pa, knows = prior.get(qid, (None, 1))
        if pa is None:
            continue
        P = pa.strip().rstrip(".").strip()
        if knows:
            stats["prior_correct"] += 1; continue          # attack not available: prior already right
        if not P or len(P.split()) > 6 or BAD.match(P):
            stats["prior_degenerate"] += 1; continue        # refusal or over-long free text
        if equivalent(P, gold) or equivalent(P, wrong) or wb(gold).search(P) or wb(P).search(gold) \
                or wb(P).search(wrong) or wb(wrong).search(P):
            stats["collides"] += 1; continue                # not a genuinely new target
        benign = [d["text"] for d in src["docs"]]
        poison = [tidy(P, t, gold) for t in benign]
        if any(wb(gold).search(t) or not wb(P.split()[0]).search(t) for t in poison):
            stats["no_subst"] += 1; continue
        stats["kept"] += 1
        for m in CONF:
            docs = [{"text": poison[i], "poison": True} for i in range(m)] + \
                   [{"text": benign[i], "poison": False} for i in range(m, len(benign))]
            rng.shuffle(docs)
            out.append({"id": f"adv-{qid}-m{m}", "qid": f"adv-{qid}", "source": src["source"],
                        "part": "mixed", "attack": "prior_aligned", "m": m, "k": src["k"],
                        "question": src["question"], "gold": gold, "wrong": P,
                        "prior_target": P, "orig_wrong": wrong,
                        "gold_aliases": src.get("gold_aliases"), "docs": docs})
    path = os.path.join(a.outdir, f"met_adaptive_{tag}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{full}: {stats} -> {len(out)} instances ({stats['kept']} questions) -> {path}")
