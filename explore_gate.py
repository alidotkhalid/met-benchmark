"""Offline: corroboration gate. Vote as before, but a winning answer backed by a single document and not by the
prior is not trusted over the joint reading of all documents (plain RAG answer)."""
import json
import hashlib
alias = {}
for _df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for _l in open(_df):
        _d = json.loads(_l); alias[_d["id"]] = _d.get("gold_aliases")


def dev(q):
    return int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 3
from met.methods import cluster_votes
from met.text import same_answer, norm, score, is_nonanswer


def ours(iso, prior, conf, gamma, rag, gate=True):
    cl = cluster_votes(iso)
    if not cl:
        return prior if norm(prior) else rag
    if len(cl) == 1 and not gate:
        return cl[0][0]
    sc = [c[1] for c in cl]
    hit = [i for i, c in enumerate(cl) if same_answer(prior, c[0])]
    if hit:
        sc[hit[0]] += gamma * conf
    best = max(sc)
    tied = [i for i, s in enumerate(sc) if abs(s - best) < 1e-9]
    r = [i for i in tied if same_answer(rag, cl[i][0])]
    w = (r or tied)[0]
    if gate and cl[w][1] == 1 and not (hit and hit[0] == w) and norm(rag) and not is_nonanswer(rag):
        return rag  # uncorroborated winner: defer to the joint reading
    return cl[w][0]


FILES = {"Qwen-7B": ["q7_known_all", "q7_met", "q7_ret_all", "q7_edits"], "Llama-8B": ["l8_known_all", "l8_met", "l8_ret_all", "l8_edits"],
         "Mistral-7B": ["m7_known_all", "m7_met", "m7_ret_all"]}


def load_rows():
    rows = []
    for m, fs in FILES.items():
        for f in fs:
            for l in open(f"results/v3/{f}.jsonl"):
                r = json.loads(l)
                if r.get("isolated") and r["part"] != "counterfactual":
                    r["model"] = m
                    r["set"] = "known" if "known" in f else {"mixed": "prnd", "retrieval": "retr", "edits": "edit"}[r["part"]]
                    rows.append(r)
    return rows


if __name__ == "__main__":
    rows = load_rows()
    for split in ["dev", "test"]:
        cells = {}
        for r in rows:
            if dev(r["qid"]) != (split == "dev"):
                continue
            lvl = r.get("cond") or r.get("n_inj", r["m"])
            for g in [2, 2.5, 3]:
                for gate in [False, True]:
                    p = ours(r["isolated"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"], gate)
                    cells.setdefault((f"{'gate' if gate else 'nogate'}/g{g}", r["set"], lvl, r["model"]), []).append(
                        score(p, r["gold"], r["wrong"], alias.get(r["id"]))["acc"])
            for b in ["rag", "arr"]:
                cells.setdefault((b, r["set"], lvl, r["model"]), []).append(r["res"][b]["acc"])
        print("\nsplit", split)
        for n in sorted({k[0] for k in cells}):
            line, means = f"{n:<11}", []
            for s in ["known", "prnd", "retr", "edit"]:
                v = [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == s]
                means.append(100 * sum(v) / len(v)); line += f" {s}:{means[-1]:5.1f}"
            e = {lv: [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == "edit" and k[2] == lv] for lv in ["orig", "edit", "update"]}
            rr = {lv: [sum(x) / len(x) for k, x in cells.items() if k[0] == n and k[1] == "retr" and k[2] == lv] for lv in [0, 1, 3, 5]}
            print(line + f" mean:{sum(means)/4:5.1f} | edits " + " ".join(f"{lv}:{100*sum(v)/len(v):5.1f}" for lv, v in e.items())
                  + " | retr " + " ".join(f"{lv}:{100*sum(v)/len(v):5.1f}" for lv, v in rr.items()))
