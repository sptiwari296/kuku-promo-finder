"""Stage 2 - Tag moments. The AI reads episodes and marks hooks and cliffhangers.

The AI only returns block IDs (e.g. E01B012). It never writes timestamps.
Any ID that does not exist in the parsed screenplay is dropped and counted,
so we can report how often the model invents things.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

from .llm import LLM
from .parse import Episode

PROMPT_VERSION = "tag-v1"

HOOK_TYPES = ["slap_or_violence", "public_humiliation", "identity_reveal", "betrayal_exposed",
              "rejection", "shocking_line", "power_reversal", "threat_or_danger",
              "romance_tension", "secret_or_scheme", "pregnancy_or_child", "other"]

TAG_SYSTEM = f"""You are a senior promo editor for vertical microdrama apps (like ReelShort and DramaBox).
You read timestamped screenplays and mark the moments that make the best short social-media promos.

Definitions:
- HOOK: a moment that would stop someone scrolling within 3 seconds if a promo OPENED on it.
  Examples: a slap, a public humiliation, a shocking reveal, a cruel or shocking line, a sudden threat.
  The hook must make sense with zero context. Mark the block where the promo should START.
- CLIFFHANGER: a moment where cutting away leaves an urgent unanswered question
  ("what will she do?", "who is he really?", "will he die?"). Mark the block where the promo should END.
- Hook types: {", ".join(HOOK_TYPES)}.

Scoring (integers 0-10):
- hook_score: how hard it stops a scroll when used as the very first seconds.
- cliff_score: how badly a viewer needs to know what happens next if we cut right after it.
- intensity: emotional intensity of the moment.
- needs_context: how much backstory a brand-new viewer needs to understand it (0 = none, 10 = lost without it).

Rules:
- Refer to moments ONLY by their block ID exactly as written (e.g. E03B045). Never invent IDs. Never write timestamps.
- Visual moments (CAMERA / ACTION) count: a slap described in an ACTION line is a hook.
- Return 6 to 15 moments per episode, the strongest ones only. Include weak cliffhangers only if nothing better exists.
- Also list 1-3 turning_points per episode: block IDs where the story changes direction.
- Output JSON only."""

TAG_SCHEMA = """{
  "episodes": [
    {"episode": <int>,
     "summary": "<one sentence>",
     "turning_points": ["<block id>", ...],
     "moments": [
       {"id": "<block id>", "role": "hook|cliffhanger|both",
        "hook_type": "<one hook type>", "hook_score": 0-10, "cliff_score": 0-10,
        "intensity": 0-10, "needs_context": 0-10, "why": "<max 15 words>"}
     ]}
  ]
}"""


@dataclass
class Moment:
    id: str
    episode: int
    role: str
    hook_type: str
    hook_score: float
    cliff_score: float
    intensity: float
    needs_context: float
    why: str
    source: str  # which model produced it


def episode_prompt(ep: Episode) -> str:
    head = f"=== EPISODE {ep.number} (duration {ep.duration:.1f}s) ===\nLogline: {ep.logline}\n"
    return head + "\n".join(b.line() for b in ep.blocks)


def _num(x, default=0.0) -> float:
    try:
        return max(0.0, min(10.0, float(x)))
    except (TypeError, ValueError):
        return default


def tag_with_llm(episodes: list[Episode], llm: LLM, batch_size: int = 3, log=print,
                 workers: int = 1) -> tuple[list[Moment], dict, dict]:
    """Returns (moments, per-episode info, quality stats). workers > 1 sends batches in parallel
    (episodes are independent), which is what makes a live demo fast on a paid API."""
    valid = {b.id: b for e in episodes for b in e.blocks}
    moments: list[Moment] = []
    info: dict[int, dict] = {}
    stats = {"returned": 0, "invalid_ids": 0, "invalid_examples": [], "failed_episodes": []}
    batches = [episodes[i:i + batch_size] for i in range(0, len(episodes), batch_size)]

    def one(batch):
        log(f"  tagging episodes {batch[0].number}-{batch[-1].number} with {llm.active_spec}")
        user = ("Tag these episodes. Return JSON in exactly this shape:\n" + TAG_SCHEMA + "\n\n"
                + "\n\n".join(episode_prompt(e) for e in batch))
        try:
            return llm.json(TAG_SYSTEM, user, temperature=0.2, tag=f"{PROMPT_VERSION}_ep{batch[0].number:02d}")
        except Exception as e:                   # one bad episode must not sink the show
            log(f"  ! tagging failed for episodes {batch[0].number}-{batch[-1].number}: {str(e)[:120]}")
            stats["failed_episodes"] += [b.number for b in batch]
            return {}

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outs = list(pool.map(one, batches))
    else:
        outs = [one(b) for b in batches]
    for out in outs:
        for ep_out in out.get("episodes", []):
            n = int(ep_out.get("episode", 0) or 0)
            tps = [t for t in ep_out.get("turning_points", []) if t in valid]
            info[n] = {"summary": ep_out.get("summary", ""), "turning_points": tps}
            for m in ep_out.get("moments", []):
                stats["returned"] += 1
                mid = str(m.get("id", "")).strip()
                if mid not in valid:
                    stats["invalid_ids"] += 1
                    if len(stats["invalid_examples"]) < 10:
                        stats["invalid_examples"].append(mid)
                    continue
                moments.append(Moment(id=mid, episode=valid[mid].episode, role=str(m.get("role", "hook")),
                                      hook_type=str(m.get("hook_type", "other")),
                                      hook_score=_num(m.get("hook_score")), cliff_score=_num(m.get("cliff_score")),
                                      intensity=_num(m.get("intensity")), needs_context=_num(m.get("needs_context"), 5),
                                      why=str(m.get("why", ""))[:200], source=llm.active_spec))
    return moments, info, stats


# ---------------------------------------------------------------- baseline
# A deliberately simple keyword tagger. It needs no API key, so the tool always
# runs, and it gives a baseline to measure the AI against.
HOOK_WORDS = {
    "slap_or_violence": r"\bslap|strik|hit |punch|grab|seiz|drag|shov|stab|blade|sword|whip|crop|blood|choke|throw",
    "public_humiliation": r"humiliat|laugh|mock|kneel|beg|disgrace|shame|scrub|mud|tolerate",
    "identity_reveal": r"reveal|true identity|who are you|duke|heir|crest|signet|unmask|real name",
    "betrayal_exposed": r"betray|lied|knew|marrying me|affair|frame|poison",
    "threat_or_danger": r"execut|hang|gallows|burn|fire|kill|die|death|witch|arrest|dungeon|guards",
    "shocking_line": r"!$",
}
CLIFF_WORDS = r"\?$|cliffhanger|ominous|freez|stun|gasp|door (swings|bursts)|cut to black|who|what\b"
EMO_HIGH = r"outrag|furious|desperat|devastat|scream|terrif|horrif|shock|cruel|menac|venom|vicious|rage"


def tag_with_heuristic(episodes: list[Episode]) -> tuple[list[Moment], dict, dict]:
    moments: list[Moment] = []
    info = {}
    for ep in episodes:
        scored = []
        for b in ep.blocks:
            if b.kind == "camera":
                continue
            low = (b.text + " " + b.delivery).lower()
            htype, hs = "other", 0.0
            for t, pat in HOOK_WORDS.items():
                if re.search(pat, low if t != "shocking_line" else b.text.strip()):
                    htype, hs = t, hs + 3
            if re.search(EMO_HIGH, low):
                hs += 2
            cs = 3.0 if re.search(CLIFF_WORDS, low) else 0.0
            if b.end >= ep.duration - 8:      # writers end episodes on cliffhangers
                cs += 5
            if b.kind == "sound" and "cliffhanger" in low:
                cs += 3
            hs, cs = min(hs, 10), min(cs, 10)
            if hs >= 3 or cs >= 3:
                scored.append((max(hs, cs), Moment(
                    id=b.id, episode=ep.number, role="both" if hs >= 3 and cs >= 3 else ("hook" if hs >= cs else "cliffhanger"),
                    hook_type=htype, hook_score=hs, cliff_score=cs, intensity=min(10, hs + 2),
                    needs_context=5, why="keyword match", source="heuristic")))
        scored.sort(key=lambda x: -x[0])
        moments += [m for _, m in scored[:15]]
        info[ep.number] = {"summary": ep.logline, "turning_points": []}
    return moments, info, {"returned": len(moments), "invalid_ids": 0, "invalid_examples": []}


LAST_BEAT_SCORE = 7.0   # a floor, not a verdict: the judges still score the clip


def add_last_beats(episodes: list[Episode], moments: list[Moment]) -> int:
    """Rule in code: every episode's last story beat counts as a cliffhanger.
    Microdrama writers end each episode on the reason to watch the next one, so we don't
    leave that to the AI noticing it (the no-hint test showed the AI partly relied on the
    script saying "cliffhanger"). Returns how many moments were added or upgraded."""
    changed = 0
    for ep in episodes:
        beats = [b for b in ep.blocks if b.kind in ("dialogue", "action")]
        if not beats:
            continue
        last = beats[-1]
        mine = [m for m in moments if m.id == last.id]
        if mine:
            for m in mine:
                if m.cliff_score < LAST_BEAT_SCORE or m.role == "hook":
                    m.cliff_score = max(m.cliff_score, LAST_BEAT_SCORE)
                    m.role = "both" if m.role == "hook" else m.role
                    changed += 1
            continue
        moments.append(Moment(id=last.id, episode=ep.number, role="cliffhanger", hook_type="other",
                              hook_score=0.0, cliff_score=LAST_BEAT_SCORE, intensity=5.0, needs_context=5.0,
                              why="episode's last beat (code rule)", source="rule:last-beat"))
        changed += 1
    return changed


def tag(episodes: list[Episode], llm: LLM, batch_size: int = 3, log=print, workers: int = 1,
        last_beat_rule: bool = True):
    if llm.is_heuristic:
        return tag_with_heuristic(episodes)
    moments, info, stats = tag_with_llm(episodes, llm, batch_size=batch_size, log=log, workers=workers)
    if last_beat_rule:
        stats["last_beat_rule_added"] = add_last_beats(episodes, moments)
    return moments, info, stats


def moments_to_dicts(ms: list[Moment]) -> list[dict]:
    return [asdict(m) for m in ms]
