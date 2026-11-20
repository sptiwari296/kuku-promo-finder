"""Stage 6 - Posting plan: which promos to post first, and why, in plain words.

The AI may only pick from the ranked promos it is shown (by ID); anything else is
dropped. If no AI is available, a rule-based plan is written instead.
"""
from __future__ import annotations

from .llm import LLM

PROMPT_VERSION = "plan-v1"

PLAN_SYSTEM = """You are the social media lead for a vertical microdrama app. Promos exist to make
strangers start the show at Episode 1. You get a shortlist of ranked promo clips with editor scores.
Choose the order to post the first ones and say why, in plain words a busy editor understands.
Prefer: clips a stranger understands with no backstory, clips from early episodes (they lead straight
into Episode 1 and spoil less), strong hooks, and variety (do not post two similar scenes back to back).
Avoid clips that spoil big twists. Use ONLY the clip IDs given. Output JSON only."""


def _line(p: dict) -> str:
    s = p["scores"]
    segs = " + ".join(f"Ep{x['episode']} {x['start']}-{x['end']}" for x in p["segments"])
    return (f"{p['id']} | rank {p['rank']} | {segs} | {p['duration_s']}s | {p['kind']} | "
            f"hook {s['hook']} cliff {s['cliffhanger']} standalone {s['standalone']} spoiler {s['spoiler_risk']} | "
            f"{'UNCERTAIN ' if p['uncertain'] else ''}hook type {p['hook_moment']['hook_type']} | "
            f"hook: {p['hook']} | cliffhanger: {p['cliffhanger']}")


def rule_plan(ranked: list[dict], k: int = 5) -> list[dict]:
    """No-AI fallback: post in rank order, skipping uncertain ones, with a templated reason."""
    out = []
    for p in [p for p in ranked if not p["uncertain"]][:k]:
        s = p["scores"]
        ep = p["segments"][0]["episode"]
        bits = []
        if s["standalone"] >= 7:
            bits.append("understandable with no backstory")
        if ep <= 5:
            bits.append(f"from Episode {ep}, so it leads straight into the start")
        if s["hook"] >= 8:
            bits.append("very strong opening")
        if s["cliffhanger"] >= 8:
            bits.append("ends on an urgent question")
        out.append({"slot": len(out) + 1, "id": p["id"], "rank": p["rank"],
                    "reason": "; ".join(bits) or "highest remaining overall score", "source": "rules"})
    return out


def posting_plan(ranked: list[dict], llm: LLM, shortlist: int = 12, k: int = 5, log=print) -> dict:
    top = ranked[:shortlist]
    if not top:
        return {"plan": [], "hold": []}
    if llm.is_heuristic:
        return {"plan": rule_plan(ranked, k), "hold": []}
    ids = {p["id"]: p for p in top}
    user = (f"Shortlist (best first by score):\n" + "\n".join(_line(p) for p in top) +
            f"\n\nPick the first {k} to post, in posting order. Then list up to 3 clips from the shortlist "
            "to hold back, with the reason. Return JSON exactly like:\n"
            '{"plan": [{"id": "<clip id>", "reason": "<why post it in this slot, max 25 words>"}], '
            '"hold": [{"id": "<clip id>", "reason": "<max 20 words>"}]}')
    log(f"Stage 6: writing posting plan with {llm.active_spec}")
    try:
        out = llm.json(PLAN_SYSTEM, user, temperature=0.2, tag=PROMPT_VERSION)
    except Exception as e:  # never lose the run over the plan
        log(f"  plan failed ({e}); using rule-based plan")
        return {"plan": rule_plan(ranked, k), "hold": [], "error": str(e)[:300]}
    plan, seen = [], set()
    for item in out.get("plan", []):
        cid = str(item.get("id", "")).strip()
        if cid in ids and cid not in seen:
            seen.add(cid)
            plan.append({"slot": len(plan) + 1, "id": cid, "rank": ids[cid]["rank"],
                         "reason": str(item.get("reason", "")).strip(), "source": llm.active_spec})
    dropped = [str(i.get("id")) for i in out.get("plan", []) if str(i.get("id", "")).strip() not in ids]
    for p in rule_plan(ranked, k + len(plan)):           # top up if the model returned too few
        if len(plan) >= k:
            break
        if p["id"] not in seen:
            seen.add(p["id"])
            plan.append({**p, "slot": len(plan) + 1})
    hold = [{"id": str(h.get("id")).strip(), "rank": ids[str(h.get("id")).strip()]["rank"],
             "reason": str(h.get("reason", "")).strip()}
            for h in out.get("hold", []) if str(h.get("id", "")).strip() in ids and str(h.get("id")).strip() not in seen]
    return {"plan": plan, "hold": hold, "invalid_ids_dropped": dropped}
