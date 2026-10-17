"""Figure 2: the prior weight buys robustness against a standard attacker and sells it to an adaptive one.

One panel per model with adaptive-attack results. x = gamma. Two series, same validated two-slot
palette as Figure 1 (#2a78d6 blue, #eb6834 orange); identity also carried by marker shape and a
direct label at the right end of each line, so the figure survives greyscale printing.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from met.methods import prior_vote
from met.text import score

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f2328", "#6b7280"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/adaptive_gamma.pdf"
D = "results/v3"
MODELS = [(t, n) for t, n in [("q7", "Qwen2.5-7B"), ("l8", "Llama-3.1-8B"), ("m7", "Mistral-7B")]
          if os.path.exists(f"{D}/{t}_adaptive.jsonl")]
G = [0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def load(t, p):
    return [json.loads(l) for l in open(f"{D}/{t}_{p}.jsonl")]


def acc(rows, g):
    return 100 * sum(score(prior_vote(r["isolated"], r["prior_answer"], r["prior_conf"], g, r["pred"]["rag"]),
                           r["gold"], r["wrong"])["acc"] for r in rows) / len(rows)


fig, axes = plt.subplots(1, len(MODELS), figsize=(3.9 * len(MODELS), 3.5), sharey=True, squeeze=False)
for ax, (t, name) in zip(axes[0], MODELS):
    K = [r for r in load(t, "known_all") if r["m"] == 3]
    A = [r for r in load(t, "adaptive") if r["m"] == 2]
    yk, ya = [acc(K, g) for g in G], [acc(A, g) for g in G]
    ax.axvspan(0, 1, color="#f3f4f6", zorder=0, lw=0)
    ax.plot(G, yk, color=BLUE, lw=2, marker="o", ms=4, zorder=3)
    ax.plot(G, ya, color=ORANGE, lw=2, marker="D", ms=4, zorder=3)
    ax.axvline(2.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.annotate("default", (2.5, 101), fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.annotate("standard attacker,\nprior right", (G[-1], yk[-1]), textcoords="offset points",
                xytext=(-4, -24), fontsize=7, color=BLUE, ha="right")
    ax.annotate("adaptive attacker,\nprior wrong", (G[-1], ya[-1]), textcoords="offset points",
                xytext=(-4, 8), fontsize=7, color=ORANGE, ha="right")
    ax.set_title(name, fontsize=10, color=INK)
    ax.set_xlabel(r"prior weight $\gamma$", fontsize=9, color=INK)
    ax.set_xlim(0, 5); ax.set_ylim(0, 108)
    ax.grid(True, color="#e5e7eb", lw=0.7, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
axes[0][0].set_ylabel("MET-Vote accuracy (%)", fontsize=9, color=INK)
axes[0][0].annotate(r"$\gamma c<1$:" "\n" "prior only" "\n" "breaks ties", (0.5, 50), fontsize=7, color=MUTED, ha="center", va="center")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT, "models:", [n for _, n in MODELS])
