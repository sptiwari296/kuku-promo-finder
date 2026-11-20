"""Stage 3 - Build candidate promos. Pure code: durations, cut points, overlap.

A promo = 1 or 2 segments. Every cut lands on a camera-shot boundary that is not
inside a line of dialogue, so an editor can cut it cleanly.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, asdict

from .parse import Episode, fmt_time
from .tag import Moment

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0          # hook must land in the first 3 seconds


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt_time(self.start), "end": fmt_time(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2), "length_s": round(self.length, 1)}


@dataclass
class Candidate:
    id: str
    kind: str                       # single | stitched
    segments: list[Segment]
    hook: Moment
    cliff: Moment
    prescore: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.length for s in self.segments)


def _snap_down(points: list[float], t: float) -> float | None:
    i = bisect.bisect_right(points, t + 0.01) - 1
    return points[i] if i >= 0 else None


def _snap_up(points: list[float], t: float) -> float | None:
    i = bisect.bisect_left(points, t - 0.01)
    return points[i] if i < len(points) else None


def _overlap(a: Candidate, b: Candidate) -> float:
    """How much two promos reuse the same footage (0..1). Takes the larger of:
    shared time / shorter promo, and shared time / shorter segment for any segment
    pair (so two stitched promos with the same 5-second hook count as duplicates)."""
    inter, seg_max = 0.0, 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                x = max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
                inter += x
                seg_max = max(seg_max, x / max(1e-6, min(sa.length, sb.length)))
    return max(inter / max(1e-6, min(a.duration, b.duration)), seg_max)


def build(episodes: list[Episode], moments: list[Moment], pool_size: int = 30,
          allow_stitched: bool = True, max_per_episode: int = 3) -> tuple[list[Candidate], dict]:
    eps = {e.number: e for e in episodes}
    blocks = {b.id: b for e in episodes for b in e.blocks}
    cuts = {e.number: e.cut_points() for e in episodes}
    hooks = [m for m in moments if m.hook_score >= 5 and m.role in ("hook", "both")]
    cliffs = [m for m in moments if m.cliff_score >= 5 and m.role in ("cliffhanger", "both")]
    by_ep_cliffs: dict[int, list[Moment]] = {}
    for c in cliffs:
        by_ep_cliffs.setdefault(c.episode, []).append(c)

    def score(h: Moment, c: Moment, segs: list[Segment], kind: str) -> tuple[float, list[str]]:
        notes = []
        inside = [m for m in moments if any(
            m.episode == s.episode and s.start - 0.01 <= blocks[m.id].start < s.end for s in segs)]
        avg_int = sum(m.intensity for m in inside) / len(inside) if inside else 0
        s = 0.40 * h.hook_score + 0.40 * c.cliff_score + 0.10 * avg_int + 0.10 * (10 - h.needs_context)
        if c.intensity >= h.intensity:
            s += 0.3
            notes.append("rising intensity")
        dur = sum(x.length for x in segs)
        if 25 <= dur <= 45:
            s += 0.2
        if kind == "stitched":
            s -= 0.6                       # joins are riskier: prefer one clean stretch
        s -= 0.01 * (h.episode - 1)        # mild preference for early episodes (fewer spoilers)
        return round(s, 3), notes

    raw: list[Candidate] = []
    rejected = {"too_short": 0, "too_long": 0, "hook_late": 0, "no_cut": 0}
    for h in hooks:
        ep = eps[h.episode]
        hb = blocks[h.id]
        start = _snap_down(cuts[h.episode], hb.start)
        if start is None:
            rejected["no_cut"] += 1
            continue
        if hb.start - start > HOOK_WINDOW:
            rejected["hook_late"] += 1
            continue
        # --- single continuous stretch in the same episode
        for c in by_ep_cliffs.get(h.episode, []):
            cb = blocks[c.id]
            if cb.end <= hb.start:
                continue
            end = _snap_up(cuts[h.episode], cb.end)
            if end is None:
                rejected["no_cut"] += 1
                continue
            dur = end - start
            if dur < MIN_LEN:
                rejected["too_short"] += 1
                continue
            if dur > MAX_LEN:
                rejected["too_long"] += 1
                continue
            segs = [Segment(h.episode, start, end)]
            sc, notes = score(h, c, segs, "single")
            raw.append(Candidate("", "single", segs, h, c, sc, notes))
        # --- stitched: hook moment + a later cliffhanger (same or next 3 episodes)
        if not allow_stitched:
            continue
        seg_a_end = _snap_up(cuts[h.episode], hb.end + 2.0)
        if seg_a_end is None or seg_a_end - start > 15:
            continue
        seg_a = Segment(h.episode, start, seg_a_end)
        for e2 in range(h.episode, h.episode + 4):
            for c in by_ep_cliffs.get(e2, []):
                cb = blocks[c.id]
                if e2 == h.episode and cb.start - seg_a.end < MAX_LEN:
                    continue                   # close enough for a single stretch; skip
                end = _snap_up(cuts[e2], cb.end)
                if end is None:
                    continue
                lead = max(12.0, 30.0 - seg_a.length)
                b_start = _snap_down(cuts[e2], end - lead)
                if b_start is None or (e2 == h.episode and b_start < seg_a.end):
                    continue
                segs = [seg_a, Segment(e2, b_start, end)]
                dur = sum(s.length for s in segs)
                if not (MIN_LEN <= dur <= MAX_LEN):
                    continue
                sc, notes = score(h, c, segs, "stitched")
                raw.append(Candidate("", "stitched", segs, h, c, sc, notes + ["stitched"]))

    # --- pick a diverse pool: best first, skip near-duplicates, cap per episode
    raw.sort(key=lambda c: -c.prescore)
    pool: list[Candidate] = []
    per_ep: dict[int, int] = {}
    used_moments: set[str] = set()

    def try_add(c: Candidate) -> bool:
        if len(pool) >= pool_size or c in pool:
            return False
        if c.hook.id in used_moments or c.cliff.id in used_moments:
            return False                   # each hook / cliffhanger used once
        if any(_overlap(c, p) > 0.3 for p in pool):
            return False
        e0 = c.segments[0].episode
        if per_ep.get(e0, 0) >= max_per_episode:
            return False
        used_moments.update({c.hook.id, c.cliff.id})
        per_ep[e0] = per_ep.get(e0, 0) + 1
        pool.append(c)
        return True

    # pass 1 - coverage: the best candidate that starts in each episode, so no episode is skipped
    covered: set[int] = set()
    for c in raw:
        e0 = c.segments[0].episode
        if e0 not in covered and try_add(c):
            covered.add(e0)
    # pass 2 - fill the rest of the pool with the best remaining candidates
    for c in raw:
        if len(pool) >= pool_size:
            break
        try_add(c)
    pool.sort(key=lambda c: -c.prescore)
    for i, c in enumerate(pool, 1):
        c.id = f"C{i:03d}"
    all_eps = {e.number for e in episodes}
    stats = {"hooks": len(hooks), "cliffs": len(cliffs), "raw_candidates": len(raw),
             "pool": len(pool), "episodes_covered": len(covered), "episodes_total": len(all_eps),
             "episodes_without_candidate": sorted(all_eps - covered), "rejected": rejected}
    return pool, stats


def candidate_to_dict(c: Candidate) -> dict:
    return {"id": c.id, "kind": c.kind, "duration_s": round(c.duration, 1), "prescore": c.prescore,
            "segments": [s.as_dict() for s in c.segments], "hook_moment": asdict(c.hook),
            "cliff_moment": asdict(c.cliff), "notes": c.notes}
