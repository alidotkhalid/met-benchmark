"""Prior-weight sweep on the development / test split (30% / 70% of questions by hash of qid).
Reports R and U of MET-Vote (grounded) for every gamma, per model and pooled, and (R+U)/2.
gamma = 2.5 is fixed as a policy, not selected by this sweep (Section 4.7 of the paper)."""
import json, hashlib, sys, io, contextlib
sys.argv = [sys.argv[0], "/tmp/_mktables_scratch"]
with contextlib.redirect_stdout(io.StringIO()):
    import mktables as M
from met.methods import prior_vote

G = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
dev = lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 3
for split in ["dev", "test"]:
    pooled = {g: [0.0, 0.0] for g in G}
    for t, name in M.MODELS:
        rows = [r for r in M.frontier_rows(t) if dev(r["qid"]) == (split == "dev") and r.get("isolated")]
        cells = []
        for g in G:
            R, U = M.RU(rows, lambda r: prior_vote(r["isolated"], r["prior_answer"], r["prior_conf"], g,
                                                   r["pred"].get("rag", r["isolated"][0])))
            pooled[g][0] += R / len(M.MODELS); pooled[g][1] += U / len(M.MODELS)
            cells.append(f"{R:5.1f}/{U:5.1f}")
        print(f"{split:<5}{name:<14}" + "  ".join(cells))
    print(f"{split:<5}{'POOLED (R+U)/2':<14}" + "  ".join(f"g{g}:{(a+b)/2:.1f}" for g, (a, b) in pooled.items()))
