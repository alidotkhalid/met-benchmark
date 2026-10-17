"""Multi-hop component of MET: HotpotQA bridge questions, poisoning or editing ONE hop.

Each question has two supporting paragraphs: the BRIDGE paragraph (resolves the entity the question refers to
only indirectly) and the ANSWER paragraph (contains the answer string). No single paragraph answers the question
on its own, which is the regime in which isolated per-document reading should fail hardest.

Conditions per question (k = 5 documents, order shuffled with a fixed seed):
  orig      bridge + answer paragraph + 3 distractors                          legitimate   gold = answer
  edit      bridge + answer paragraph with the answer substituted + 3 distr.  legitimate   gold = substitute
  poison1   bridge + answer + poisoned copy of the answer paragraph + 2 distr. poisoned    gold = answer
  poison2   as poison1 + a poisoned excerpt (the substituted sentences) + 1 distr.          gold = answer
The substitute (same answer shape: number / year / date / name) is the attacker's target in poison* and the
legitimately edited answer in `edit`, so poisoned and legitimate contexts differ only in what is true.
Distractors are HotpotQA distractor paragraphs of the same question (retrieved by bigram TF-IDF for the question
by the dataset authors) that do not mention the answer. The substitute is an entity of the answer's spaCy NER type
drawn from the question's remaining (unshown) distractor paragraphs, so it is topically plausible.

Usage: python build_multihop.py <hotpot_dev_distractor.parquet>  ->  data/met_multihop.jsonl
"""
import json, os, random, re, sys
import pandas as pd
from met.text import norm

SEED, N = 0, 100
rng = random.Random(SEED)
MONTHS = "january february march april may june july august september october november december".split()


def shape(a):
    n = norm(a)
    if re.fullmatch(r"\d{4}", n):
        return "year"
    if any(m in n.split() for m in MONTHS):
        return "date"
    if re.fullmatch(r"[\d,.\s]+", a.strip()) or re.match(r"^[\d,.]+\s", a.strip()):
        return "number"
    if a[:1].isupper():
        return "name"
    return None


def para(ctx, title):
    i = list(ctx["title"]).index(title)
    return title, "".join(ctx["sentences"][i]).strip()


def contains(text, a):
    return re.search(r"(?<!\w)" + re.escape(norm(a)) + r"(?!\w)", norm(text)) is not None


excl = set()
for f in ["data/met.jsonl", "data/met_retrieval.jsonl"]:
    for l in open(f):
        r = json.loads(l)
        if "hotpot" in r["source"]:
            excl.add(r["qid"].split("-")[-1])

# questions removed after manual review of the substitutes: the NER type of the answer or of the drawn substitute
# was wrong (e.g. a college replaced by a moor), which would make the edit or the poison incoherent
REVIEW_EXCL = set(json.load(open("data/multihop_excluded_ids.json"))) if os.path.exists("data/multihop_excluded_ids.json") else set()
d = pd.read_parquet(sys.argv[1])
d = d[(d["type"] == "bridge") & (~d["id"].isin(excl))]
cands = []
for _, r in d.iterrows():
    a = r["answer"].strip()
    if norm(a) in ("yes", "no") or len(a) > 40 or len(a.split()) > 6 or shape(a) is None:
        continue
    if contains(r["question"], a) or re.search(r"\bor\b", r["question"]):     # " or ": choice questions are comparisons
        continue
    titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
    if len(titles) != 2:
        continue
    p = [para(r["context"], t) for t in titles]
    has = [a in f"{t} {txt}" for t, txt in p]           # exact, case-sensitive: needed for substitution
    has_n = [contains(f"{t} {txt}", a) for t, txt in p]
    if sum(has) != 1 or sum(has_n) != 1:
        continue
    ans_p, br_p = (p[0], p[1]) if has[0] else (p[1], p[0])
    others = [para(r["context"], t) for t in r["context"]["title"] if t not in titles]
    cands.append(dict(id=r["id"], question=r["question"], answer=a, shape=shape(a), ans=ans_p, br=br_p,
                      others=others, level=r["level"]))
import spacy
nlp = spacy.load("en_core_web_sm")


def ner_type(c):
    """entity label of the answer as it occurs in its paragraph; dates/numbers further split by surface form"""
    lab = None
    for e in nlp(c["ans"][1]).ents:
        if norm(e.text) == norm(c["answer"]):
            lab = e.label_
            break
    if lab is None:
        return None
    a = c["answer"].strip()
    if lab == "DATE":
        lab += ":year" if re.fullmatch(r"\d{4}", a) else (":full" if re.search(r"\d{4}", a) and re.search(r"\d{1,2}\D", a) else ":other")
    return lab


for c in cands:
    c["shape"] = ner_type(c)
cands = [c for c in cands if c["shape"]]
print("eligible bridge questions:", len(cands))


HEADS = {"county", "university", "college", "academy", "school", "river", "lake", "site", "award", "awards", "church",
         "hotel", "records", "party", "conference", "league", "club", "fc", "stadium", "arena", "comics", "prize"}


def compatible(a, w, lab):
    """surface-form agreement beyond the NER label"""
    da, dw = bool(re.search(r"\d", a)), bool(re.search(r"\d", w))
    if da != dw:
        return False
    if w.lower().startswith("the ") and not a.lower().startswith("the "):
        return False
    if w.endswith(("'s", "\u2019s")) or w.count('"') % 2 or not w[:1].isupper() and a[:1].isupper():
        return False
    ha, hw = norm(a).split()[-1], norm(w).split()[-1]
    if (ha in HEADS or hw in HEADS) and ha != hw:      # a county for a county, a university for a university
        return False
    if ("." in a) != ("." in w):
        return False
    if lab in ("CARDINAL", "QUANTITY", "MONEY", "PERCENT"):
        la, lw = re.sub(r"[\d,.\s]", "", a), re.sub(r"[\d,.\s]", "", w)
        return la == lw                                    # same unit, or both bare numbers
    if lab.startswith("DATE"):
        return len(a.split()) == len(w.split())
    if lab == "PERSON":
        return (len(a.split()) > 1) == (len(w.split()) > 1)
    return True


rng.shuffle(cands)
rows, used = [], 0
for c in cands:
    if used == N:
        break
    if c["id"] in REVIEW_EXCL:
        continue
    qrng = random.Random(f"{SEED}-{c['id']}")      # per-question randomness: excluding one question changes no other
    a = c["answer"]
    # 1) three distractors that do not mention the answer
    distr = [o for o in c["others"] if not contains(f"{o[0]} {o[1]}", a)]
    if len(distr) < 4:
        continue
    qrng.shuffle(distr)
    chosen, spare = distr[:3], distr[3:]
    used_txt = " ".join([c["question"], *c["br"], *c["ans"]] + [f"{o[0]} {o[1]}" for o in chosen])
    # 2) substitute: an entity of the same NER type from this question's OWN unused distractor paragraphs, which
    #    HotpotQA retrieved for the question, so the substitute is topically plausible (a university for a
    #    university, a county for a county); it must not occur anywhere in the documents shown
    subs = []
    for _, txt in spare:
        for e in nlp(txt).ents:
            if not e.text[:1].isalnum():
                continue
            lab = e.label_
            if lab == "DATE":
                t = e.text.strip()
                lab += ":year" if re.fullmatch(r"\d{4}", t) else (":full" if re.search(r"\d{4}", t) and re.search(r"\d{1,2}\D", t) else ":other")
            if lab == c["shape"] and compatible(a, e.text.strip(), lab) and norm(e.text) != norm(a) \
                    and not contains(e.text, a) and not contains(a, e.text) and not contains(used_txt, e.text):
                subs.append(e.text.strip())
    subs = sorted(set(subs))
    if not subs:
        continue
    w = qrng.choice(subs)
    distr = chosen
    fmt = lambda t, x: f"{t}: {x}"
    bridge = dict(text=fmt(*c["br"]), role="bridge", poison=False)
    answer = dict(text=fmt(*c["ans"]), role="answer", poison=False)
    sub_title, sub_txt = c["ans"][0].replace(a, w), c["ans"][1].replace(a, w)
    edited = dict(text=fmt(sub_title, sub_txt), role="edit", poison=False)
    pois1 = dict(text=fmt(sub_title, sub_txt), role="poison", poison=True)
    # excerpt: the original sentences that state the answer, then substituted (splitting the original text keeps
    # abbreviations inside the answer, e.g. "Don S. Davis", from breaking the sentence split)
    sents = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", c["ans"][1]) if a in x]
    if not sents:
        i = c["ans"][1].find(a)
        sents = [c["ans"][1][max(0, i - 150): i + len(a) + 150]] if i >= 0 else [c["ans"][1]]
    pois2 = dict(text=fmt(sub_title, " ".join(x.replace(a, w) for x in sents)), role="poison", poison=True)
    D = [dict(text=fmt(*o), role="distractor", poison=False) for o in distr]
    conds = {"orig": ([bridge, answer] + D, a, w, 0),
             "edit": ([bridge, edited] + D, w, a, 0),
             "poison1": ([bridge, answer, pois1] + D[:2], a, w, 1),
             "poison2": ([bridge, answer, pois1, pois2] + D[:1], a, w, 2)}
    qid = f"mh-{c['id']}"
    for cond, (docs, g, wr, m) in conds.items():
        docs = docs[:]
        qrng.shuffle(docs)
        rows.append(dict(id=f"{qid}-{cond}", qid=qid, source="mh-hotpotqa", part="multihop", cond=cond, m=m, k=5,
                         question=c["question"], gold=g, wrong=wr, docs=docs, level=c["level"]))
    used += 1

with open("data/met_multihop.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print("wrote", len(rows), "instances,", used, "questions;", Counter(r["cond"] for r in rows),
      Counter(c["shape"] for c in cands[:0]) or "")
