"""Figure 1: the trust frontier. One panel per model; x = utility, y = robustness.

Palette: categorical slots 1 and 2 of the validated reference palette
(#2a78d6 blue, #eb6834 orange) -- passes lightness band, chroma floor, CVD
separation and normal-vision floor. Identity is additionally carried by marker
shape and by direct labels on every point, so the figure survives greyscale
printing and colour-vision deficiency without relying on hue.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from met.methods import prior_vote
from met.text import score

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f2328", "#6b7280"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/frontier.pdf"
D = "results/v3"
MODELS = [("q7", "Qwen2.5-7B"), ("l8", "Llama-3.1-8B"), ("m7", "Mistral-7B")]
ALIAS = {}
for df in ["data/met_known.jsonl", "data/met_retrieval.jsonl", "data/met.jsonl", "data/met_edits.jsonl"]:
    for l in open(df):
        d = json.loads(l); ALIAS[d["id"]] = d.get("gold_aliases")
POIS = {("known", 2), ("known", 3), ("known", 5), ("retr", 1), ("retr", 3), ("retr", 5)}
LEGIT = {("known", 0), ("retr", 0), ("edit", "orig"), ("edit", "edit"), ("cf", 0), ("cf", 1)}
BASE = [("closed_book", "Closed-book"), ("rag", "RAG"), ("cad", "CAD"), ("adacad", "AdaCAD"),
        ("arr", "ARR"), ("sdag", "SDAG"), ("vote", "Vote"), ("rrag_kw", "RobustRAG-KW"),
        ("rrag_dec", "RobustRAG-Dec")]
# label offsets (points), tuned per panel: the method positions differ by model,
# so a single offset table would collide. Direct labels are the primary identity
# encoding here, so their placement matters as much as the marks.
OFF = {
 "q7": {"Closed-book": (7, -3), "RobustRAG-KW": (-66, 3), "ARR": (7, 2), "Vote": (-30, -11),
        "RobustRAG-Dec": (7, 3), "RAG": (7, 3), "AdaCAD": (7, -9), "SDAG": (-37, -10), "CAD": (7, -3)},
 "l8": {"Closed-book": (7, -3), "RobustRAG-KW": (-66, 3), "ARR": (-32, -10), "Vote": (-33, -12),
        "RobustRAG-Dec": (-78, 3), "RAG": (7, -9), "AdaCAD": (7, -3), "SDAG": (-37, -10), "CAD": (7, -3)},
 "m7": {"Closed-book": (7, -3), "RobustRAG-KW": (-66, 3), "ARR": (-30, 5), "Vote": (-31, 5),
        "RobustRAG-Dec": (-78, -5), "RAG": (7, 3), "AdaCAD": (7, -4), "SDAG": (7, -3), "CAD": (7, -9)},
}
GAMMAS = [0, 1.5, 2.5, 3, 4]
DEFAULT_G = 2.5
# offsets for the two labelled endpoints of the gamma sweep, per panel
GOFF = {"q7": [(8, -3), (8, -3)], "l8": [(8, -13), (8, -3)], "m7": [(8, -3), (8, -3)]}


def rows(tag):
    out = []
    for s, f in {"known": "known_all", "retr": "ret_all", "edit": "edits", "cf": "cf"}.items():
        for l in open(f"{D}/{tag}_{f}.jsonl"):
            r = json.loads(l); r["set"] = s
            r["lv"] = r.get("cond") or (r["n_inj"] if s == "retr" else r["m"])
            out.append(r)
    return out


def RU(rs, fn):
    c = {}
    for r in rs:
        k = (r["set"], r["lv"])
        if k in POIS or k in LEGIT:
            c.setdefault(k, []).append(score(fn(r), r["gold"], r["wrong"], ALIAS.get(r["id"]))["acc"])
    R = [sum(v) / len(v) for k, v in c.items() if k in POIS]
    U = [sum(v) / len(v) for k, v in c.items() if k in LEGIT]
    return 100 * sum(U) / len(U), 100 * sum(R) / len(R)


fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharex=True, sharey=True)
for ax, (tag, name) in zip(axes, MODELS):
    rs = rows(tag)
    for me, lab in BASE:
        u, r = RU(rs, lambda x, me=me: x["pred"][me])
        ax.scatter(u, r, s=34, color=BLUE, marker="o", zorder=3, linewidths=0)
        ax.annotate(lab, (u, r), textcoords="offset points", xytext=OFF[tag][lab],
                    fontsize=7, color=INK)
    path = [RU(rs, lambda x, g=g: prior_vote(x["isolated"], x["prior_answer"], x["prior_conf"], g, x["pred"]["rag"]) if x.get("isolated") else x["pred"]["rag"])
            for g in GAMMAS]
    ax.plot([p[0] for p in path], [p[1] for p in path], color=ORANGE, lw=2, zorder=4,
            solid_capstyle="round")
    ax.scatter([p[0] for p in path], [p[1] for p in path], s=30, color=ORANGE, marker="D",
               zorder=5, linewidths=1.0, edgecolors="white")
    du, dr = path[GAMMAS.index(DEFAULT_G)]
    ax.scatter([du], [dr], s=78, color=ORANGE, marker="D", zorder=6, linewidths=1.6,
               edgecolors="white")
    # label only the two endpoints of the sweep; the default sits between them
    for (u, r), g, off in [(path[0], GAMMAS[0], GOFF[tag][0]), (path[-1], GAMMAS[-1], GOFF[tag][1])]:
        ax.annotate(rf"$\gamma$={g:g}", (u, r), textcoords="offset points", xytext=off,
                    fontsize=7, color=ORANGE)
    uf, rf_ = RU(rs, lambda x: prior_vote(x["isolated"], x["prior_answer"], x["prior_conf"], 4, x["pred"]["rag"], grounded=False) if x.get("isolated") else x["pred"]["rag"])
    ax.scatter(uf, rf_, s=52, facecolors="none", edgecolors=ORANGE, marker="D", zorder=5, linewidths=1.6)
    ax.annotate("free", (uf, rf_), textcoords="offset points", xytext=(7, -2), fontsize=7, color=ORANGE)
    ax.set_title(name, fontsize=10, color=INK)
    ax.grid(True, color="#e5e7eb", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlabel("Utility $U$ (%)", fontsize=9, color=INK)
axes[0].set_ylabel("Robustness $R$ (%)", fontsize=9, color=INK)
axes[0].set_xlim(22, 80); axes[0].set_ylim(2, 85)
h = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=5.5, label="Baselines"),
     plt.Line2D([], [], color=ORANGE, marker="D", ls="-", lw=2, ms=4.5, label=r"MET-Vote, $\gamma$ sweep"),
     plt.Line2D([], [], color=ORANGE, marker="D", ls="", ms=7.5, label=r"MET-Vote, default $\gamma$=2.5"),
     plt.Line2D([], [], color=ORANGE, marker="D", ls="", ms=6, mfc="none", mew=1.6, label="MET-Vote (free)")]
axes[1].legend(handles=h, loc="lower left", fontsize=8, frameon=False, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
