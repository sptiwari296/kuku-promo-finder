"""Editor review: approve / reject / unsure each promo, optionally nudge the cut points,
and measure how often the tool's picks are good enough to post.

This is the human check the AI-only checks cannot give. Reviews are saved as a small
CSV so they can be reloaded, shared, or fed back into the ranking later.
"""
from __future__ import annotations

import csv
import io

from .candidates import MAX_LEN, MIN_LEN
from .parse import fmt_time

DECISIONS = ("approve", "reject", "unsure")
FIELDS = ["promo_id", "rank", "segments", "decision", "start_nudge_s", "end_nudge_s", "reason"]
REASONS = ["", "weak hook", "weak cliffhanger", "needs backstory", "spoiler", "joined clips don't connect",
           "bad cut point", "other"]


def seg_label(p: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])


def apply_nudges(p: dict, start_nudge: float, end_nudge: float) -> list[dict]:
    """Move the first segment's start and the last segment's end. Returns new segments."""
    segs = [dict(s) for s in p["segments"]]
    segs[0]["start_s"] = round(max(0.0, segs[0]["start_s"] + start_nudge), 2)
    segs[-1]["end_s"] = round(segs[-1]["end_s"] + end_nudge, 2)
    for s in segs:
        s["start"], s["end"] = fmt_time(s["start_s"]), fmt_time(s["end_s"])
        s["length_s"] = round(s["end_s"] - s["start_s"], 1)
    return segs


def nudge_problem(segs: list[dict]) -> str:
    total = sum(s["end_s"] - s["start_s"] for s in segs)
    if any(s["end_s"] <= s["start_s"] for s in segs):
        return "a segment would end before it starts"
    if not (MIN_LEN <= total <= MAX_LEN):
        return f"length would be {total:.1f}s (must be 20-60s)"
    return ""


def summarize(promos: list[dict], reviews: dict[str, dict]) -> dict:
    """Approval numbers. 'Decided' = approve or reject (unsure is left out)."""
    by_id = {p["id"]: p for p in promos}
    rows = [(by_id[i], r) for i, r in reviews.items() if i in by_id and r.get("decision") in DECISIONS]
    decided = [(p, r) for p, r in rows if r["decision"] in ("approve", "reject")]
    appr = lambda xs: sum(r["decision"] == "approve" for _, r in xs)

    def rate(xs):
        return round(100 * appr(xs) / len(xs)) if xs else None

    out = {"reviewed": len(rows), "decided": len(decided), "approved": appr(decided),
           "approval_rate": rate(decided)}
    for k in (5, 10):
        top = [(p, r) for p, r in decided if p["rank"] <= k]
        out[f"approved_in_top{k}"] = f"{appr(top)}/{len(top)}" if top else "-"
    flagged = [(p, r) for p, r in decided if p.get("uncertain")]
    clean = [(p, r) for p, r in decided if not p.get("uncertain")]
    out["approval_rate_flagged"] = rate(flagged)       # is the editor-check flag useful?
    out["approval_rate_not_flagged"] = rate(clean)
    for kind in ("single", "stitched"):
        out[f"approval_rate_{kind}"] = rate([(p, r) for p, r in decided if p["kind"] == kind])
    reasons: dict[str, int] = {}
    for _, r in decided:
        if r["decision"] == "reject" and r.get("reason"):
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["reject_reasons"] = dict(sorted(reasons.items(), key=lambda x: -x[1]))
    out["nudged"] = sum(1 for _, r in rows if r.get("start_nudge_s") or r.get("end_nudge_s"))
    return out


def to_csv(promos: list[dict], reviews: dict[str, dict]) -> str:
    by_id = {p["id"]: p for p in promos}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    for pid, r in sorted(reviews.items(), key=lambda x: by_id.get(x[0], {}).get("rank", 999)):
        if pid in by_id and r.get("decision"):
            w.writerow({"promo_id": pid, "rank": by_id[pid]["rank"], "segments": seg_label(by_id[pid]),
                        "decision": r["decision"], "start_nudge_s": r.get("start_nudge_s", 0),
                        "end_nudge_s": r.get("end_nudge_s", 0), "reason": r.get("reason", "")})
    return buf.getvalue()


def from_csv(text: str, promos: list[dict]) -> dict[str, dict]:
    """Load saved reviews. Matches on the clip's segments too, so reviews survive a re-run."""
    by_seg = {seg_label(p): p["id"] for p in promos}
    ids = {p["id"] for p in promos}
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        pid = by_seg.get(row.get("segments", "")) or (row.get("promo_id") if row.get("promo_id") in ids else None)
        if not pid or row.get("decision") not in DECISIONS:
            continue
        out[pid] = {"decision": row["decision"], "reason": row.get("reason", ""),
                    "start_nudge_s": float(row.get("start_nudge_s") or 0),
                    "end_nudge_s": float(row.get("end_nudge_s") or 0)}
    return out
