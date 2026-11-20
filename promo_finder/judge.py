"""Stage 4 - Judge each candidate like a promo editor, then rank.

The judge sees ONLY the clip's own text (no logline, no backstory), because a
new viewer only sees the clip. It scores each promo on its own against a rubric
(pointwise, not head-to-head), and it runs twice with the rubric order and the
candidate order shuffled. Big disagreement between the two runs = "uncertain".
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

from .candidates import Candidate, candidate_to_dict
from .llm import LLM
from .parse import Episode, fmt_time

PROMPT_VERSION = "judge-v1"

CRITERIA = {
    "hook": "Do the first 3 seconds stop a scrolling viewer? (slap, reveal, shocking line, humiliation)",
    "cliffhanger": "Does the last moment leave an urgent unanswered question, so the viewer must know what happens next?",
    "standalone": "Would a viewer who has never seen the show understand what is going on from this clip alone?",
    "payoff_link": "Does the clip tell one mini-story, where the ending is the open version of the question the opening raises?",
    "spoiler_risk": "How much does it give away a major twist or the show's ending? (10 = big spoiler)",
}
# Kuku's brief names three things a promo needs: a hook, a cliffhanger, and making sense on its own.
# They get equal weight; the hook-to-ending link is a smaller bonus.
WEIGHTS = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.30, "payoff_link": 0.10}
SPOILER_PENALTY = 0.05
UNCERTAIN_CRITERION_GAP = 3.0
UNCERTAIN_TOTAL_GAP = 1.5

JUDGE_SYSTEM = """You are a demanding promo editor at a vertical microdrama app. You decide which
20-60 second clips get posted on TikTok, Instagram Reels and YouTube Shorts to pull in NEW viewers.
You are shown each clip as screenplay text (camera, action, dialogue). You see only what the viewer sees:
no backstory. Judge each clip on its own; do not compare clips to each other. Be strict: a 9 or 10 is rare.
Output JSON only."""


def clip_text(c: Candidate, eps: dict[int, Episode]) -> str:
    out = []
    for k, s in enumerate(c.segments, 1):
        out.append(f"-- segment {k}: {s.length:.1f}s --")
        for b in eps[s.episode].blocks_between(s.start, s.end):
            t = f"[+{max(0.0, b.start - s.start):04.1f}s]"
            if b.kind == "dialogue":
                d = f" ({b.delivery})" if b.delivery else ""
                out.append(f"{t} {b.speaker}{d}: \"{b.text}\"")
            else:
                out.append(f"{t} {b.kind.upper()}: {b.text}")
    return "\n".join(out)


def _clean(s) -> str:
    """Models sometimes emit invisible zero-width or odd hyphen characters."""
    s = str(s or "")
    for bad, good in (("​", ""), ("‌", ""), ("‍", ""), ("﻿", ""), ("‑", "-")):
        s = s.replace(bad, good)
    return s.strip()


def _rubric(order: list[str]) -> str:
    return "\n".join(f"- {k} (0-10): {CRITERIA[k]}" for k in order)


def _judge_batch(batch: list[Candidate], eps, llm: LLM, order: list[str], run: int) -> dict[str, dict]:
    schema = ('{"clips": [{"id": "<clip id>", "scores": {' + ", ".join(f'"{k}": 0-10' for k in order) +
              '}, "hook": "<what happens in the first 3 seconds, max 15 words>", '
              '"cliffhanger": "<the unanswered question at the cut, max 15 words>", '
              '"why": "<one line on why it works or fails, max 20 words>"}]}')
    clips = "\n\n".join(f"=== CLIP {c.id} ({c.duration:.0f}s) ===\n{clip_text(c, eps)}" for c in batch)
    user = f"Score every clip below on this rubric:\n{_rubric(order)}\n\nReturn JSON exactly like:\n{schema}\n\n{clips}"
    out = llm.json(JUDGE_SYSTEM, user, temperature=0.3, tag=f"{PROMPT_VERSION}_r{run}_{batch[0].id}")
    res = {}
    for item in out.get("clips", []):
        cid = str(item.get("id", "")).strip()
        sc = item.get("scores", {}) or {}
        scores = {k: _to_score(sc.get(k)) for k in CRITERIA}
        if any(v is None for v in scores.values()):
            # e.g. the model wrote a sentence where a number belongs: drop this run
            res[cid] = {"malformed": {k: str(sc.get(k))[:60] for k, v in scores.items() if v is None}}
            continue
        res[cid] = {"scores": scores,
                    "hook": _clean(item.get("hook", "")), "cliffhanger": _clean(item.get("cliffhanger", "")),
                    "why": _clean(item.get("why", ""))}
    return res


def _to_score(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 10 else None


def _heuristic_judgement(c: Candidate, eps) -> dict:
    """Baseline judge: reuse tag scores. Same output shape as the AI judge."""
    first = eps[c.segments[0].episode].blocks_between(c.segments[0].start, c.segments[0].start + 3.5)
    last = eps[c.segments[-1].episode].blocks_between(c.segments[-1].end - 4, c.segments[-1].end)
    words = lambda bl: " / ".join(b.text for b in bl if b.kind != "camera")[:120]
    return {"scores": {"hook": c.hook.hook_score, "cliffhanger": c.cliff.cliff_score,
                       "standalone": 10 - c.hook.needs_context, "payoff_link": 5.0,
                       "spoiler_risk": min(10.0, c.segments[-1].episode / 5)},
            "hook": words(first), "cliffhanger": words(last), "why": "baseline: keyword scores"}


def final_score(scores: dict) -> float:
    s = sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS)
    return round(s - SPOILER_PENALTY * scores.get("spoiler_risk", 0), 3)


def judge(cands: list[Candidate], episodes: list[Episode], llm: LLM, batch_size: int = 5,
          runs: int = 2, seed: int = 7, log=print, stats: dict | None = None,
          workers: int = 1) -> list[dict]:
    stats = stats if stats is not None else {}
    stats.update({"malformed": 0, "malformed_examples": [], "missing": 0, "failed_batches": 0, "single_run": 0})
    eps = {e.number: e for e in episodes}
    rng = random.Random(seed)
    results: dict[str, list[dict]] = {c.id: [] for c in cands}
    if llm.is_heuristic:
        for c in cands:
            j = _heuristic_judgement(c, eps)
            results[c.id] = [j, j]
    else:
        jobs = []                                # build all batches first so shuffles stay reproducible
        for run in range(runs):
            order = list(CRITERIA)
            items = list(cands)
            if run > 0:                          # shuffle rubric and clip order on later runs
                rng.shuffle(order)
                rng.shuffle(items)
            jobs += [(run, order, items[i:i + batch_size]) for i in range(0, len(items), batch_size)]

        def one(job):
            run, order, b = job
            log(f"  judging run {run + 1}/{runs}: clips {b[0].id}..{b[-1].id}")
            try:
                return _judge_batch(b, eps, llm, order, run)
            except Exception as e:               # one bad batch must not sink the whole run
                stats["failed_batches"] += 1
                log(f"  ! batch failed, skipped: {str(e)[:150]}")
                return {}

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outs = list(pool.map(one, jobs))
        else:
            outs = [one(j) for j in jobs]
        for out in outs:
            for cid, j in out.items():
                    if cid not in results:
                        continue
                    if "malformed" in j:
                        stats["malformed"] += 1
                        if len(stats["malformed_examples"]) < 5:
                            stats["malformed_examples"].append({cid: j["malformed"]})
                        continue
                    results[cid].append(j)

    judged = []
    for c in cands:
        js = results.get(c.id) or []
        if not js:
            stats["missing"] += 1
            continue
        if len(js) < runs:
            stats["single_run"] += 1
        avg = {k: round(sum(j["scores"][k] for j in js) / len(js), 2) for k in CRITERIA}
        totals = [final_score(j["scores"]) for j in js]
        # only swings on the three things Kuku asked for count; spoiler_risk swings a lot but barely moves the score
        crit_gap = max((max(j["scores"][k] for j in js) - min(j["scores"][k] for j in js))
                       for k in ("hook", "cliffhanger", "standalone"))
        total_gap = max(totals) - min(totals)
        uncertain = len(js) > 1 and (crit_gap >= UNCERTAIN_CRITERION_GAP or total_gap >= UNCERTAIN_TOTAL_GAP)
        d = candidate_to_dict(c)
        d.update({"scores": avg, "score": final_score(avg), "run_totals": totals,
                  "judge_gap": round(total_gap, 2), "uncertain": uncertain,
                  "hook": js[0]["hook"], "cliffhanger": js[0]["cliffhanger"], "why": js[0]["why"],
                  "judge_runs": js, "clip_text": clip_text(c, eps)})
        judged.append(d)
    return judged


def early_bonus(ep: int) -> float:
    """Promos send viewers to Episode 1. Early-episode clips need less backstory and spoil less."""
    return 0.6 if ep <= 5 else 0.4 if ep <= 10 else 0.2 if ep <= 20 else 0.0


def merge_judges(per_judge: dict[str, list[dict]]) -> list[dict]:
    """Combine several judges (e.g. Groq + Gemini) on the same candidates: average their
    scores; mark 'uncertain' if any judge was unsure or the judges disagree by >= 1.5 points."""
    specs = list(per_judge)
    if len(specs) == 1:
        out = per_judge[specs[0]]
        for d in out:
            d["judges"] = {specs[0]: {"score": d["score"], "scores": d["scores"], "why": d["why"]}}
        return out
    by_id = [{d["id"]: d for d in per_judge[s]} for s in specs]
    common = [cid for cid in by_id[0] if all(cid in m for m in by_id)]
    # Judges differ in generosity (Gemini scored ~0.8 higher on average in our run). Compare each
    # judge against its own average, so "disagree" means a different opinion, not a stricter scale.
    means = [sum(m[c]["score"] for c in common) / max(1, len(common)) for m in by_id]
    overall = sum(means) / len(means)
    merged = []
    for cid in common:
        base = by_id[0][cid]
        ds = [m[cid] for m in by_id]
        d = dict(base)
        d["scores"] = {k: round(sum(x["scores"][k] for x in ds) / len(ds), 2) for k in CRITERIA}
        d["score"] = final_score(d["scores"])
        adj = [x["score"] - (mu - overall) for x, mu in zip(ds, means)]
        gap = max(adj) - min(adj)
        d["judge_gap"] = round(gap, 2)
        d["judge_offsets"] = [round(mu - overall, 2) for mu in means]
        d["uncertain"] = any(x["uncertain"] for x in ds) or gap >= UNCERTAIN_TOTAL_GAP
        d["judges"] = {s: {"score": x["score"], "scores": x["scores"], "why": x["why"],
                           "hook": x["hook"], "cliffhanger": x["cliffhanger"], "uncertain": x["uncertain"]}
                       for s, x in zip(specs, ds)}
        merged.append(d)
    return merged


def agreement(per_judge: dict[str, list[dict]]) -> dict:
    """How much the first two judges agree: rank correlation and top-10 overlap."""
    specs = list(per_judge)
    if len(specs) < 2:
        return {}
    a = {d["id"]: d["score"] for d in per_judge[specs[0]]}
    b = {d["id"]: d["score"] for d in per_judge[specs[1]]}
    ids = [i for i in a if i in b]
    n = len(ids)
    if n < 2:
        return {}
    ra = {x: r for r, x in enumerate(sorted(ids, key=lambda i: -a[i]))}
    rb = {x: r for r, x in enumerate(sorted(ids, key=lambda i: -b[i]))}
    rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in ids) / (n * (n * n - 1))
    top = lambda r: {i for i in ids if r[i] < 10}
    return {"judges": specs[:2], "n": n, "spearman": round(rho, 2),
            "top10_overlap": len(top(ra) & top(rb)),
            "mean_abs_gap": round(sum(abs(a[i] - b[i]) for i in ids) / n, 2)}


def rank(judged: list[dict], max_per_episode_top10: int = 2) -> list[dict]:
    """Final order: judge score, minus a penalty for uncertain picks, plus a bonus for
    early episodes, then spread the top 10 across episodes."""
    for d in judged:
        ep0 = d["segments"][0]["episode"]
        d["rank_score"] = round(d["score"] - (0.5 if d["uncertain"] else 0) + early_bonus(ep0), 3)
    ordered = sorted(judged, key=lambda d: -d["rank_score"])
    top, rest, count = [], [], {}
    for d in ordered:
        ep0 = d["segments"][0]["episode"]
        if len(top) < 10 and count.get(ep0, 0) >= max_per_episode_top10:
            rest.append(d)
            continue
        if len(top) < 10:
            count[ep0] = count.get(ep0, 0) + 1
            top.append(d)
        else:
            rest.append(d)
    final = top + sorted(rest, key=lambda d: -d["rank_score"])
    for i, d in enumerate(final, 1):
        d["rank"] = i
    return final
