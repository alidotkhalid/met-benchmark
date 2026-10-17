"""Merge extra-method result files (e.g. RobustRAG runs) into a main results file, matching by instance id."""
import json, sys
main, extra, out = sys.argv[1], sys.argv[2], sys.argv[3]
E = {}
for l in open(extra):
    r = json.loads(l); E[r["id"]] = r
n = 0
with open(out, "w") as f:
    for l in open(main):
        r = json.loads(l); e = E.get(r["id"])
        if e:
            for k in ("pred", "res", "time"):
                r[k].update({m: v for m, v in e[k].items() if m not in ("closed_book",)})
            if e.get("iso_abst"): r["iso_abst"] = e["iso_abst"]
            n += 1
        f.write(json.dumps(r) + "\n")
print(f"merged {n} of {len(E)} extra rows into {out}")
