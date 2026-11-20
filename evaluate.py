"""Compare the tool's promos with promos picked by a human editor.

1. Fill in human_picks.csv (episode,start,end  - times as MM:SS.cc or seconds), BEFORE looking at the output.
2. python evaluate.py outputs/<run>/promos.json human_picks.csv

A tool promo "matches" a human pick if they overlap in the same episode by at
least 50% of the shorter one. Reports recall (how many of your picks the tool
found) and precision@10 (how many of the tool's top 10 match one of your picks).
"""
import csv
import json
import sys


def secs(x: str) -> float:
    x = x.strip()
    if ":" in x:
        m, s = x.split(":")
        return int(m) * 60 + float(s)
    return float(x)


def overlap(a, b) -> float:
    if a[0] != b[0]:
        return 0.0
    inter = max(0.0, min(a[2], b[2]) - max(a[1], b[1]))
    return inter / max(1e-6, min(a[2] - a[1], b[2] - b[1]))


def main(promos_path, picks_path, k=10):
    promos = json.load(open(promos_path, encoding="utf-8"))
    picks = [(int(r["episode"]), secs(r["start"]), secs(r["end"]))
             for r in csv.DictReader(open(picks_path, encoding="utf-8"))]
    tool = [(p["rank"], [(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]) for p in promos]

    def matches(segs, pick):
        return any(overlap(s, pick) >= 0.5 for s in segs)

    found = [pk for pk in picks if any(matches(segs, pk) for _, segs in tool)]
    top = [(r, segs) for r, segs in tool if r <= k]
    hits = [(r, segs) for r, segs in top if any(matches(segs, pk) for pk in picks)]
    print(f"Human picks: {len(picks)}   Tool promos: {len(tool)}")
    print(f"Recall (your picks the tool found anywhere): {len(found)}/{len(picks)}")
    print(f"Precision@{k} (tool top-{k} that match a pick): {len(hits)}/{len(top)}")
    for pk in picks:
        r = [rank for rank, segs in tool if matches(segs, pk)]
        print(f"  Ep{pk[0]} {pk[1]:.1f}-{pk[2]:.1f}s -> {'tool rank ' + str(r) if r else 'MISSED'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
