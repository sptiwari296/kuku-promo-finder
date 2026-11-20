"""Compare two runs that judged the SAME candidate pool with different models.

    python compare_judges.py outputs/groq_full/promos.json outputs/gemini_judge/promos.json

Prints how much the two judges agree (rank correlation, top-10 overlap) and lists
promos they disagree on most. Agreement between two different model families is a
cheap quality signal; big disagreements are the ones an editor should look at.
Writes judge_agreement.csv next to the first file.
"""
import csv
import json
import sys
from pathlib import Path


def key(p):
    return tuple((s["episode"], s["start_s"], s["end_s"]) for s in p["segments"])


def spearman(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ra = {x: i for i, x in enumerate(sorted(a, key=lambda k: -a[k]))}
    rb = {x: i for i, x in enumerate(sorted(b, key=lambda k: -b[k]))}
    d2 = sum((ra[k] - rb[k]) ** 2 for k in a)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main(pa, pb):
    A = {key(p): p for p in json.load(open(pa, encoding="utf-8"))}
    B = {key(p): p for p in json.load(open(pb, encoding="utf-8"))}
    common = [k for k in A if k in B]
    sa = {k: A[k]["score"] for k in common}
    sb = {k: B[k]["score"] for k in common}
    rho = spearman(sa, sb)
    topA = {k for k in common if A[k]["rank"] <= 10}
    topB = {k for k in common if B[k]["rank"] <= 10}
    print(f"Same promos judged by both: {len(common)} (A had {len(A)}, B had {len(B)})")
    print(f"Rank correlation (Spearman): {rho:.2f}   (1 = same order, 0 = unrelated)")
    print(f"Top-10 overlap: {len(topA & topB)}/10")
    mean_abs = sum(abs(sa[k] - sb[k]) for k in common) / max(1, len(common))
    print(f"Mean score difference: {mean_abs:.2f} points (0-10 scale)")

    rows = []
    for k in common:
        a, b = A[k], B[k]
        segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in a["segments"])
        both = (a["score"] + b["score"]) / 2
        gap = abs(a["score"] - b["score"])
        rows.append({"segments": segs, "duration_s": a["duration_s"], "kind": a["kind"],
                     "rank_A": a["rank"], "rank_B": b["rank"], "score_A": a["score"],
                     "score_B": b["score"], "avg": round(both, 2), "gap": round(gap, 2),
                     "editor_check": "yes" if gap >= 1.5 or a["uncertain"] or b["uncertain"] else "",
                     "hook": a["hook"], "cliffhanger": a["cliffhanger"], "why_A": a["why"], "why_B": b["why"]})
    rows.sort(key=lambda r: -r["avg"])
    for i, r in enumerate(rows, 1):
        r["consensus_rank"] = i
    print("\nBoth judges like (top 10 by average):")
    for r in rows[:10]:
        print(f"  {r['segments']:<40} A#{r['rank_A']:<3} B#{r['rank_B']:<3} avg {r['avg']}")
    print("\nBiggest disagreements:")
    for r in sorted(rows, key=lambda r: -r["gap"])[:5]:
        print(f"  {r['segments']:<40} A {r['score_A']} vs B {r['score_B']}  | A: {r['why_A'][:60]} | B: {r['why_B'][:60]}")
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["gap"])
    for k, g in by_kind.items():
        print(f"Mean judge gap for {k} promos: {sum(g) / len(g):.2f} (n={len(g)})")
    out = Path(pa).parent.parent / "consensus_promos.csv"
    cols = ["consensus_rank"] + [c for c in rows[0] if c != "consensus_rank"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])
