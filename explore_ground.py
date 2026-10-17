"""Offline: count an isolated answer as a vote only if it is supported by (appears in) its own document."""
import json, re, sys
from explore_agg import agg, alias, dev
from met.text import score, norm, is_nonanswer

STOP = set("a an the of in on at to for from by with and or is are was were be it its as that this his her their".split())
DOCS = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for l in open(df):
        d = json.loads(l); DOCS[d["id"]] = [x["text"] for x in d["docs"]]


def supported(ans, doc, thr=0.5):
    toks = [t for t in norm(ans).split() if t not in STOP]
    if not toks:
        return False
    dn = set(norm(doc).split())
    return sum(t in dn for t in toks) / len(toks) >= thr


FILES = {"Qwen-7B": ["q7_known_all", "q7_met"], "Llama-8B": ["l8_known_all", "l8_met", "l8_ret_all", "l8_edits"],
         "Mistral-7B": ["m7_known_all", "m7_met", "m7_ret_all"]}
rows = []
for m, fs in FILES.items():
    for f in fs:
        for l in open(f"results/v3/{f}.jsonl"):
            r = json.loads(l)
            if r.get("isolated") and r["part"] != "counterfactual":
                r["model"] = m; r["set"] = "known" if "known" in f else {"mixed": "prnd", "retrieval": "retr", "edits": "edit"}[r["part"]]
                rows.append(r)
for split in ["dev", "test"]:
    cells = {}
    for r in rows:
        if dev(r["qid"]) != (split == "dev"): continue
        docs = DOCS[r["id"]]
        lvl = r.get("cond") or r.get("n_inj", r["m"])
        from met.text import same_answer
        for thr in [0.5, 1.0, "agree"]:
            if thr == "agree":
                iso = r["isolated"]
                iso_g = [a if (supported(a, d, 0.5) or any(j != i and not is_nonanswer(b) and same_answer(a, b) for j, b in enumerate(iso))) else ""
                         for i, (a, d) in enumerate(zip(iso, docs))]
            else:
                iso_g = [a if supported(a, d, thr) else "" for a, d in zip(r["isolated"], docs)]
            for g in [2, 2.5, 3]:
                p = agg(iso_g, r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"], "rag", "grounded")
                cells.setdefault((f"sup{thr}/g{g}", r["set"], lvl, r["model"]), []).append(score(p, r["gold"], r["wrong"], alias.get(r["id"]))["acc"])
        for g in [2.5]:
            p = agg(r["isolated"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"], "rag", "grounded")
            cells.setdefault((f"plain/g{g}", r["set"], lvl, r["model"]), []).append(score(p, r["gold"], r["wrong"], alias.get(r["id"]))["acc"])
        for b in ["rag", "arr"]:
            cells.setdefault((b, r["set"], lvl, r["model"]), []).append(r["res"][b]["acc"])
    print("\nsplit", split)
    for n in sorted({k[0] for k in cells}):
        line, means = f"{n:<12}", []
        for s in ["known", "prnd", "retr", "edit"]:
            v = [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == s]
            means.append(100 * sum(v) / len(v)); line += f" {s}:{means[-1]:5.1f}"
        e = {lv: [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == "edit" and k[2] == lv] for lv in ["orig", "edit", "update"]}
        print(line + f"  mean:{sum(means)/4:5.1f}  | edits " + " ".join(f"{lv}:{100*sum(v)/len(v):5.1f}" for lv, v in e.items()))
