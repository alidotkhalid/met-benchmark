"""Offline: prior-leak filter. An isolated answer that equals the closed-book answer but is not found in its own
document is parametric knowledge leaking through, not evidence; drop it before voting (the prior already votes)."""
import json
from explore_gate import load_rows, dev, alias, ours
from met.text import score, norm, same_answer, is_nonanswer

STOP = set("a an the of in on at to for from by with and or is are was were be it its as that this his her their".split())
DOCS = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for l in open(df):
        d = json.loads(l); DOCS[d["id"]] = [x["text"] for x in d["docs"]]


def supported(ans, doc, thr=0.5):
    t = [w for w in norm(ans).split() if w not in STOP]
    dn = set(norm(doc).split())
    return bool(t) and sum(w in dn for w in t) / len(t) >= thr


def deleak(r, thr=0.5):
    return [a if not (same_answer(a, r["prior_answer"]) and not supported(a, d, thr)) else ""
            for a, d in zip(r["isolated"], DOCS[r["id"]])]


rows = load_rows()
for split in ["dev", "test"]:
    cells = {}
    for r in rows:
        if dev(r["qid"]) != (split == "dev"):
            continue
        lvl = r.get("cond") or r.get("n_inj", r["m"])
        for g in [2.5, 3]:
            for name, iso in [("plain", r["isolated"]), ("deleak", deleak(r)), ("deleak1", deleak(r, 1.0))]:
                p = ours(iso, r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"], gate=False)
                cells.setdefault((f"{name}/g{g}", r["set"], lvl, r["model"]), []).append(score(p, r["gold"], r["wrong"], alias.get(r["id"]))["acc"])
        for b in ["rag", "arr"]:
            cells.setdefault((b, r["set"], lvl, r["model"]), []).append(r["res"][b]["acc"])
    print("\nsplit", split)
    for n in sorted({k[0] for k in cells}):
        line, means = f"{n:<12}", []
        for s in ["known", "prnd", "retr", "edit"]:
            v = [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == s]
            means.append(100 * sum(v) / len(v)); line += f" {s}:{means[-1]:5.1f}"
        e = {lv: [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == "edit" and k[2] == lv] for lv in ["orig", "edit", "update"]}
        rr = {lv: [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == "retr" and k[2] == lv] for lv in [0, 1, 3, 5]}
        print(line + f" mean:{sum(means)/4:5.1f} | edits " + " ".join(f"{lv}:{100*sum(v)/len(v):5.1f}" for lv, v in e.items())
              + " | retr " + " ".join(f"{lv}:{100*sum(v)/len(v):5.1f}" for lv, v in rr.items()))
