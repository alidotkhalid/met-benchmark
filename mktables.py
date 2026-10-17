"""Generate every LaTeX table in the paper from results/v3. Single source of truth for all reported numbers.

usage: python mktables.py paper/tables
"""
import json, os, sys
from met.methods import prior_vote
from met.text import score, is_nonanswer, same_answer, norm
from variants import mcnemar

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/tables"
os.makedirs(OUT, exist_ok=True)
D = "results/v3"
MODELS = [("q7", "Qwen2.5-7B"), ("l8", "Llama-3.1-8B"), ("m7", "Mistral-7B")]
OURS, FREE = "pvote_g2.5", "pvote_free_g4"
NAME = {"closed_book": "Closed-book (no retrieval)", "rag": "Standard RAG", "cad": "CAD", "adacad": "AdaCAD",
        "arr": "ARR", "sdag": "SDAG", "vote": "Isolate-then-vote", "rrag_kw": "RobustRAG-Keyword",
        "rrag_dec": "RobustRAG-Decoding", OURS: r"\textbf{MET-Vote} ($\gamma{=}2.5$)", FREE: r"\textbf{MET-Vote} (free, $\gamma{=}4$)"}
ORDER = ["closed_book", "rag", "cad", "adacad", "arr", "sdag", "vote", "rrag_dec", "rrag_kw", OURS, FREE]
ALIAS = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for l in open(df):
        d = json.loads(l); ALIAS[d["id"]] = d.get("gold_aliases")
DOCS = {}
for df in ["data/met_edits.jsonl", "data/met_known.jsonl", "data/met_retrieval.jsonl"]:
    for l in open(df):
        d = json.loads(l); DOCS[d["id"]] = d["docs"]


def load(tag, part):
    p = f"{D}/{tag}_{part}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def pct(rows, me, metric="acc"):
    v = [r["res"][me][metric] for r in rows if me in r["res"]]
    return 100 * sum(v) / len(v) if v else None


def cell(x, bold=False, star=""):
    if x is None:
        return "--"
    s = f"{x:.1f}{star}"
    return r"\textbf{" + s + "}" if bold else s


def sig(rows, a, b):
    """significance marker for ours (a) vs baseline (b): $^\dagger$ if ours better at p<0.01"""
    if a not in rows[0]["res"] or b not in rows[0]["res"]:
        return ""
    x, y, p = mcnemar(rows, a, b)
    return "" if p >= 0.01 else (r"$^{\dagger}$" if x > y else r"$^{\ddagger}$")


def write(name, s):
    open(f"{OUT}/{name}.tex", "w").write(s)
    print("wrote", name)


def block(caption, label, colspec, header, body, note=None, ncols=None):
    import re as _re
    nc = ncols or len(_re.findall(r"[lcr]|p\{[^}]*\}", colspec.replace("@{}", "")))
    n = ""
    return ("\\begin{table*}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n\\small\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule%s\n\\end{tabular}\n\\end{adjustbox}\n%s\\end{table*}\n"
            % (caption, label, colspec, header, body, n,
               ("\\par\\vspace{3pt}{\\footnotesize\\raggedright %s\\par}\n" % note) if note else ""))


# ---------------------------------------------------------------- Table: benchmark composition
def t_benchmark():
    rows = []
    for f, part, desc in [("data/met_known.jsonl", None, "Known facts (TriState); $m\\in\\{0,2,3,5\\}$ of 5 poisoned"),
                          ("data/met.jsonl", "mixed", "PoisonedRAG mixed evidence; $m\\in\\{0,1,2,3,5\\}$ of 5 poisoned"),
                          ("data/met_retrieval.jsonl", None, "Realistic retrieval (Contriever top-5, BEIR); inject $\\in\\{0,1,3,5\\}$"),
                          ("data/met_edits.jsonl", None, "Multi-document legitimate edits; orig / edit / update"),
                          ("data/met.jsonl", "counterfactual", "Single-document counterfactual (Faithfulness-QA)")]:
        R = [json.loads(l) for l in open(f)]
        if part:
            R = [r for r in R if r["part"] == part]
        if part == "mixed":                       # evaluated subset: 30 questions per source corpus (run with --limit_q 30)
            evq = {json.loads(l)["qid"] for l in open(f"{D}/l8_met.jsonl")}
            R = [r for r in R if r["qid"] in evq]
        rows.append((desc, len({r["qid"] for r in R}), len(R), R[0]["k"]))
    # distinct underlying questions: the PoisonedRAG mixed set is a subset of the
    # retrieval set, and the multi-document edit set a subset of the counterfactual set
    uniq = set()
    for f, part in [("data/met_known.jsonl", None), ("data/met.jsonl", "mixed"),
                    ("data/met_retrieval.jsonl", None), ("data/met_edits.jsonl", None),
                    ("data/met.jsonl", "counterfactual")]:
        for l in open(f):
            d = json.loads(l)
            if part and d["part"] != part:
                continue
            q = d["qid"]
            for p in ("edit-", "fqa-", "ret-"):
                if q.startswith(p):
                    q = q[len(p):]
            uniq.add(q)
    body = "\n".join(f"{d} & {q} & {i} & {k} \\\\" for d, q, i, k in rows)
    body += "\n\\midrule\nTotal (distinct questions) & %d & %d & \\\\" % (len(uniq), sum(r[2] for r in rows))
    return block("Composition of the Mixed-Evidence Trust (MET) benchmark. Every instance is the same question under a different evidence configuration, so all comparisons are paired.",
                 "tab:benchmark", "@{}p{0.62\\linewidth}rrr@{}",
                 "Component & Questions & Instances & $k$ \\\\", body,
                 note="The components share questions: the 90 PoisonedRAG questions are a subset of the 259 retrieval questions, and the 200 multi-document edit questions a subset of the 300 counterfactual questions. The component counts therefore sum to more than the %d distinct underlying questions." % len(uniq))


# ---------------------------------------------------------------- main result tables
def t_levels(part, key, levels, caption, label, note=None, models=MODELS):
    data = {tag: load(tag, part) for tag, _ in models}
    data = {t: r for t, r in data.items() if r}
    head1 = " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(levels), n) for t, n in models if t in data)
    cmid = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + i * len(levels), 1 + (i + 1) * len(levels)) for i in range(len(data)))
    head2 = " & ".join(str(l) for _ in data for l in levels)
    best = {}
    for t, R in data.items():
        for lv in levels:
            sub = [r for r in R if r[key] == lv]
            vals = {me: pct(sub, me) for me in ORDER if me in R[0]["res"] and me != "closed_book"}
            best[(t, lv)] = max((v for v in vals.values() if v is not None), default=None)
    lines = []
    for me in ORDER:
        cells = []
        for t, R in data.items():
            for lv in levels:
                sub = [r for r in R if r[key] == lv]
                v = pct(sub, me)
                st = sig(sub, OURS, me) if me not in (OURS, FREE, "closed_book") else ""
                cells.append(cell(v, bold=(v is not None and me != "closed_book" and abs(v - best[(t, lv)]) < 1e-9), star=st))
        lines.append(f"{NAME[me]} & " + " & ".join(cells) + r" \\")
        if me == "sdag":
            lines.append(r"\midrule")
        if me == "rrag_kw":
            lines.append(r"\midrule")
    klab = {"m": "$m$", "n_inj": "injected", "cond": "condition"}.get(key, key)
    return block(caption, label, "@{}l" + "r" * (len(levels) * len(data)) + "@{}",
                 f" & {head1} \\\\ {cmid}\n{klab} & {head2} \\\\", "\n".join(lines), note)


# ---------------------------------------------------------------- trust frontier
POIS = {("known", 2), ("known", 3), ("known", 5), ("retr", 1), ("retr", 3), ("retr", 5)}
LEGIT = {("known", 0), ("retr", 0), ("edit", "orig"), ("edit", "edit"), ("cf", 0), ("cf", 1)}


def frontier_rows(tag):
    rows = []
    for s, f in {"known": "known_all", "retr": "ret_all", "edit": "edits", "cf": "cf"}.items():
        for r in load(tag, f) or []:
            r["set"] = s
            r["lv"] = r.get("cond") or (r["n_inj"] if s == "retr" else r["m"])
            rows.append(r)
    return rows


def RU(rows, fn):
    c = {}
    for r in rows:
        k = (r["set"], r["lv"])
        if k in POIS or k in LEGIT:
            p = fn(r)
            if p is not None:
                c.setdefault(k, []).append(score(p, r["gold"], r["wrong"], ALIAS.get(r["id"]))["acc"])
    R = [sum(v) / len(v) for k, v in c.items() if k in POIS]
    U = [sum(v) / len(v) for k, v in c.items() if k in LEGIT]
    return 100 * sum(R) / len(R), 100 * sum(U) / len(U)


def t_frontier():
    data = {t: frontier_rows(t) for t, _ in MODELS}
    lines = []
    for me in ORDER:
        cells = []
        for t, _ in MODELS:
            rows = [r for r in data[t] if me in r["res"]]
            if len(rows) < len(data[t]):
                cells += ["--", "--"]; continue
            r_, u_ = RU(rows, lambda r: r["pred"][me])
            cells += [f"{r_:.1f}", f"{u_:.1f}"]
        lines.append(f"{NAME[me]} & " + " & ".join(cells) + r" \\")
        if me in ("sdag", "rrag_kw"):
            lines.append(r"\midrule")
    head1 = " & ".join(r"\multicolumn{2}{c}{%s}" % n for _, n in MODELS)
    cmid = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(3))
    return block("The trust frontier. \\textbf{R} (robustness) is mean accuracy over the six poisoned evidence configurations; \\textbf{U} (utility) is mean accuracy over the six legitimate-evidence configurations. A method dominates another if it is at least as high on both axes and higher on one (Section~\\ref{sec:problem}).",
                 "tab:frontier", "@{}l" + "rr" * 3 + "@{}",
                 f" & {head1} \\\\ {cmid}\n & " + " & ".join(["R", "U"] * 3) + r" \\", "\n".join(lines))


# ---------------------------------------------------------------- gamma ablation
def t_gamma():
    G = [0, 1, 1.5, 2, 2.5, 3, 4, 5]
    lines = []
    for t, n in MODELS:
        rows = frontier_rows(t)
        cells = []
        for g in G:
            r_, u_ = RU(rows, lambda r, g=g: prior_vote(r["isolated"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"]) if r.get("isolated") else r["pred"]["rag"])
            cells.append(f"{r_:.1f}/{u_:.1f}")
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
    return block("Effect of the prior weight $\\gamma$ on the trust frontier (R/U, as in Table~\\ref{tab:frontier}). $\\gamma=0$ gives the prior no vote (it then differs from isolate-then-vote only in tie-breaking and the all-abstain fallback). $\\gamma=2.5$ is a policy choice fixed before any test-split evaluation (Section~\\ref{sec:devsplit}), not the value that maximises either axis.",
                 "tab:gamma", "@{}l" + "c" * len(G) + "@{}",
                 "Model & " + " & ".join(f"$\\gamma{{=}}{g:g}$" for g in G) + r" \\", "\n".join(lines))


# ---------------------------------------------------------------- diagnostics: why isolation costs utility
def t_diag():
    lines = []
    for t, n in MODELS:
        E = load(t, "edits")
        tot = na = leak = ansok = ansn = 0
        for r in E:
            for d, a in zip(DOCS[r["id"]], r["isolated"]):
                if d["role"] == "distractor":
                    tot += 1
                    if not is_nonanswer(a):
                        na += 1
                        leak += same_answer(a, r["prior_answer"])
                elif d["role"] in ("answer", "edit"):   # the stale paragraph of the update
                    ansn += 1                               # condition cannot contain the gold
                    ansok += bool(same_answer(a, r["gold"]) or norm(r["gold"]) in norm(a))
        ret = load(t, "ret_all")
        cl = [r for r in ret if r["n_inj"] == 0]
        gold_in = 100 * sum(r.get("gold_in_benign_topk", False) for r in cl) / len(cl)
        lines.append(f"{n} & {100*na/tot:.1f} & {100*leak/max(na,1):.1f} & {100*ansok/ansn:.1f} & {gold_in:.1f} \\\\")
    return block("Why isolated reading costs utility. Columns 1--2: of the irrelevant paragraphs in each multi-document edit instance (four, or three in \\emph{update}), how often the model returns a substantive answer instead of abstaining, and how often that answer coincides with its own closed-book answer (parametric leakage). Column 3: how often the one relevant paragraph (the answer paragraph, or the edited paragraph in the \\emph{edit} and \\emph{update} conditions) is read correctly in isolation. Column 4: fraction of clean retrieval instances whose top-5 benign passages actually contain the gold answer.",
                 "tab:diag", "@{}lrrrr@{}",
                 "Model & Irrelevant doc answers (\\%) & \\quad of which $=$ prior (\\%) & Relevant doc read (\\%) & Gold in clean top-5 (\\%) \\\\", "\n".join(lines))


# ---------------------------------------------------------------- significance appendix
def t_sig():
    lines = []
    for part, key, levels, sname in [("known_all", "m", [2, 3, 5], "Known"), ("met", "m", [2, 3, 5], "PoisonedRAG"),
                                     ("ret_all", "n_inj", [1, 3, 5], "Retrieval"), ("edits", "cond", ["orig", "edit"], "Edits"),
                                     ("cf", "m", [1], "Counterf.")]:
        for t, n in MODELS:
            R = load(t, part)
            if not R:
                continue
            for b in ["rag", "arr", "vote", "rrag_kw", "rrag_dec"]:
                if b not in R[0]["res"]:
                    continue
                cs = []
                for lv in levels:
                    sub = [r for r in R if r[key] == lv]
                    x, y, p = mcnemar(sub, OURS, b)
                    cs.append(f"{x}/{y}, {p:.1g}" if p < 0.05 else f"{x}/{y}, n.s.")
                lines.append(f"{sname} & {n} & {NAME[b]} & " + " & ".join(cs) + r" \\")
        lines.append(r"\midrule")
    cap = ("Paired McNemar exact tests for the main comparisons, MET-Vote ($\\gamma{=}2.5$) against standard RAG, ARR, isolate-then-vote and both RobustRAG variants. "
           "Each cell gives (instances MET-Vote alone answers correctly) / (instances the baseline alone answers correctly) and the two-sided $p$-value (n.s.: $p \\ge 0.05$). "
           "Levels: known facts and PoisonedRAG $m = 2, 3, 5$; realistic retrieval 1, 3, 5 injected passages; edits \\emph{orig}, \\emph{edit}; counterfactual \\emph{edit}. "
           "Significance marks for the remaining baselines are given in the result tables.")
    return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{@{}lllccc@{}}\n\\caption{" + cap + "}\\label{tab:sig}\\\\\n\\toprule\n"
            "Setting & Model & Baseline & level 1 & level 2 & level 3 \\\\\n\\midrule\n\\endfirsthead\n"
            "\\multicolumn{6}{@{}l}{\\emph{Table~\\ref{tab:sig} continued}}\\\\\n\\toprule\nSetting & Model & Baseline & level 1 & level 2 & level 3 \\\\\n\\midrule\n\\endhead\n"
            "\\bottomrule\n\\endlastfoot\n" + "\n".join(lines[:-1]) + "\n\\end{longtable}\n\\endgroup\n")



# ---------------------------------------------------------------- adaptive (prior-aligned) attack
AMODELS = [(t, n) for t, n in MODELS if os.path.exists(f"{D}/{t}_adaptive.jsonl")]


def t_adaptive():
    cols = [("adaptive", "m", 1), ("adaptive", "m", 2), ("adaptive", "m", 3),
            ("adaptive_ret", "n_inj", 1), ("adaptive_ret", "n_inj", 3)]
    data = {t: {p: load(t, p) for p in ("adaptive", "adaptive_ret")} for t, _ in AMODELS}
    head1 = " & ".join(r"\multicolumn{5}{c}{%s}" % n for _, n in AMODELS)
    cmid = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 5 * i, 6 + 5 * i) for i in range(len(AMODELS)))
    head2 = " & ".join(["$m{=}1$", "$m{=}2$", "$m{=}3$", "inj 1", "inj 3"] * len(AMODELS))
    lines = []
    for me in ORDER:
        if me == "closed_book":
            continue
        cells = []
        for t, _ in AMODELS:
            for part, key, lv in cols:
                sub = [r for r in data[t][part] if r[key] == lv]
                v = pct(sub, me)
                st = sig(sub, OURS, me) if me not in (OURS, FREE) else ""
                cells.append(cell(v, star=st))
        lines.append(f"{NAME[me]} & " + " & ".join(cells) + r" \\")
        if me in ("sdag", "rrag_kw"):
            lines.append(r"\midrule")
    nq = ", ".join(f"{n}: {len({r['qid'] for r in data[t]['adaptive']})} / {len({r['qid'] for r in data[t]['adaptive_ret']})}"
                   for t, n in AMODELS)
    return block("Accuracy (\\%) under the \\emph{prior-aligned adaptive attack}: the attacker queries the model closed-book once per question and, where that answer is wrong, makes it the poisoning target. Left block of each model: controlled mixed evidence with $m$ of 5 documents poisoned. Right block: the same attack injected into the BEIR corpus and re-retrieved with Contriever. Closed-book accuracy is 0 by construction (every attacked question is one the model gets wrong) and is omitted. $\\dagger$/$\\ddagger$: MET-Vote significantly better/worse than that baseline ($p<0.01$, paired McNemar).",
                 "tab:adaptive", "@{}l" + "r" * (5 * len(AMODELS)) + "@{}",
                 f" & {head1} \\\\ {cmid}\n & {head2} \\\\", "\n".join(lines),
                 note=f"Questions (mixed / retrieval): {nq}. Only questions on which the model's closed-book answer is wrong and not a surface variant of the gold answer can be attacked this way.")


def t_adaptgamma():
    G = [0, 1, 1.5, 2.5, 4]
    lines = []
    for t, n in AMODELS:
        K = [r for r in load(t, "known_all") if r["m"] == 3]
        A = [r for r in load(t, "adaptive") if r["m"] == 2]
        f = lambda rows, g: 100 * sum(score(prior_vote(r["isolated"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"]),
                                             r["gold"], r["wrong"], ALIAS.get(r["id"]))["acc"] for r in rows) / len(rows)
        lines.append(f"{n} & " + " & ".join(f"{f(K, g):.0f} / {f(A, g):.0f}" for g in G) + r" \\")
    return block("The same parameter buys robustness and sells it. Each cell gives MET-Vote accuracy (\\%) on known facts with 3 of 5 documents poisoned by a standard attacker (the model's prior is right) / on prior-aligned adaptive poisoning with 2 of 5 documents poisoned (the model's prior is wrong and the attacker chose it). $\\gamma=0$ gives the prior no vote. Both columns turn at $\\gamma c = 1$, the point at which a confident prior can overturn a 3--2 split.",
                 "tab:adaptgamma", "@{}l" + "c" * len(G) + "@{}",
                 "Model & " + " & ".join(f"$\\gamma{{=}}{g:g}$" for g in G) + r" \\", "\n".join(lines))


# ---------------------------------------------------------------- MADAM-RAG (paired subset)
MMODELS = [(t, n) for t, n in MODELS if os.path.exists(f"{D}/{t}_madam.jsonl")]


def t_madam():
    import madam_report as MR
    meths = ["rag", "arr", "vote", "rrag_dec", "rrag_kw", OURS, "madam"]
    nm = dict(NAME, madam=r"MADAM-RAG")
    data, notes = {}, []
    for t, n in MMODELS:
        S = MR.join(MR.load(f"{D}/{t}_madam.jsonl"), [f"{D}/{t}_{f}.jsonl" for f in ["known_all", "ret_all", "edits", "cf"]])
        A = MR.join(MR.load(f"{D}/{t}_madam_adaptive.jsonl"), [f"{D}/{t}_adaptive.jsonl", f"{D}/{t}_adaptive_ret.jsonl"]) \
            if os.path.exists(f"{D}/{t}_madam_adaptive.jsonl") else []
        data[t] = (S, A)
        notes.append(f"{n}: {len(S)} standard / {len(A)} adaptive instances")
    lines = []
    for me in meths:
        cells = []
        for t, _ in MMODELS:
            S, A = data[t]
            R = sum(MR.pct([r for r in S if f(r)], me) for _, f in MR.CELLS_R) / len(MR.CELLS_R)
            U = sum(MR.pct([r for r in S if f(r)], me) for _, f in MR.CELLS_U) / len(MR.CELLS_U)
            E = MR.pct([r for r in S if (r["part"] == "edits" and r["cond"] == "edit") or (r["part"] == "counterfactual" and r["m"] == 1)], me)
            cells += [f"{R:.1f}", f"{U:.1f}", f"{E:.1f}"]
            cells += [f"{MR.pct(A, me):.1f}", f"{MR.pct(A, me, 'asr'):.1f}"] if A else ["--", "--"]
        lines.append(f"{nm[me]} & " + " & ".join(cells) + r" \\")
        if me == OURS:
            lines.append(r"\midrule")
    k = 5
    head1 = " & ".join(r"\multicolumn{%d}{c}{%s}" % (k, n) for _, n in MMODELS)
    cmid = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + k * i, 1 + k * (i + 1)) for i in range(len(MMODELS)))
    head2 = " & ".join(["R", "U", "Edit", "Adv.\\ acc", "Adv.\\ ASR"] * len(MMODELS))
    return block("MADAM-RAG against the main baselines on a fixed random subset of \\bench{} (every method scored on the same instances). \\textbf{R} and \\textbf{U} are cell-averaged as in Table~\\ref{tab:frontier}. \\textbf{Edit}: accuracy on the two legitimate-edit conditions (multi-document \\emph{edit} and single-document counterfactual), pooled. \\textbf{Adv.}: accuracy and attack success rate under the prior-aligned adaptive attack of Section~\\ref{sec:adaptive} (mixed $m \\in \\{1,2,3\\}$ and retrieval with 1 or 3 injected passages). MADAM-RAG is scored strictly: its answer list must contain the gold answer and not the attacker's.",
                 "tab:madam", "@{}l" + "r" * (k * len(MMODELS)) + "@{}",
                 f" & {head1} \\\\ {cmid}\n & {head2} \\\\", "\n".join(lines),
                 note="Instances: " + "; ".join(notes) + ". Subset: 50 known facts, 20 retrieval questions per corpus, 100 multi-document edit questions (\\emph{orig}, \\emph{edit}) and 100 counterfactual pairs, drawn with a fixed seed before any MADAM-RAG output existed.")


write("benchmark", t_benchmark())
write("known", t_levels("known_all", "m", [0, 2, 3, 5],
                        "Accuracy (\\%) on known facts (TriState-Bench) by number $m$ of poisoned documents out of five. The facts were chosen to be known closed-book; the closed-book answer is scored correct on 97, 97 and 94 of the 100 (most misses are surface variants), so for nearly every fact $m{=}5$ is indistinguishable from a unanimous legitimate update. Best non-closed-book result per column in bold. $\\dagger$/$\\ddagger$: MET-Vote significantly better/worse than that baseline ($p<0.01$, paired McNemar).",
                        "tab:known", note="$m$ is the number of poisoned documents among the five retrieved."))
write("poisonedrag", t_levels("met", "m", [0, 1, 2, 3, 5],
                              "Accuracy (\\%) on PoisonedRAG mixed evidence by number $m$ of poisoned documents out of five. RobustRAG baselines were not run on this component.",
                              "tab:poisonedrag"))
write("retrieval", t_levels("ret_all", "n_inj", [0, 1, 3, 5],
                            "Accuracy (\\%) under realistic retrieval: adversarial passages are injected into the BEIR corpus and the Contriever top-5 is re-retrieved, so how many poisoned passages reach the context is decided by the retriever, not by us. Column headings give the number of injected passages.",
                            "tab:retrieval", note="Mean poisoned passages actually retrieved: 0.99, 2.94 and 4.78 for 1, 3 and 5 injected. Qwen2.5-7B uses a 105-question subset."))
write("edits", t_levels("edits", "cond", ["orig", "edit", "update"],
                        "Accuracy (\\%) on multi-document legitimate edits: one relevant paragraph among four same-article distractors. \\emph{orig}: unmodified paragraph. \\emph{edit}: the answer entity is substituted and the edited answer is correct. \\emph{update}: the edited and the stale paragraph are both present.",
                        "tab:edits"))
write("counterfactual", t_levels("cf", "m", [0, 1],
                                 "Accuracy (\\%) on single-document counterfactual edits (Faithfulness-QA). $m{=}0$ is the original passage, $m{=}1$ the edited passage whose substituted answer the model must follow.",
                                 "tab:cf", note="With a single document MET-Vote reduces to reading that document, so it coincides with standard RAG by construction."))
write("frontier", t_frontier())
write("gamma", t_gamma())
write("diagnostics", t_diag())
write("significance", t_sig())

if AMODELS:
    write("adaptive", t_adaptive())
    write("adaptgamma", t_adaptgamma())

if MMODELS:
    write("madam", t_madam())

if any(os.path.exists(f"{D}/{t}_multihop.jsonl") for t, _ in MODELS):
    _mh = t_levels("multihop", "cond", ["orig", "edit", "poison1", "poison2"],
                   "Accuracy (\\%) on the multi-hop component: HotpotQA bridge questions whose answer needs two paragraphs, with three same-question distractors. \\emph{orig}: unmodified. \\emph{edit}: the answer in the answer paragraph is legitimately replaced and the replacement is correct. \\emph{p1}, \\emph{p2}: one or two poisoned passages asserting the replacement are added beside the true paragraphs. $\\dagger$/$\\ddagger$: MET-Vote significantly better/worse than that baseline ($p<0.01$, paired McNemar).",
                   "tab:multihop",
                   note="100 questions, 400 instances per model. Utility $U$ is the mean of \\emph{orig} and \\emph{edit}; robustness $R$ the mean of \\emph{p1} and \\emph{p2}.")
    _mh = _mh.replace("& poison1 & poison2", "& p1 & p2")
    write("multihop", _mh)

