"""Runs the whole thing: parse -> tag -> build -> judge -> rank -> check -> write outputs."""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import candidates as cand_mod
from . import judge as judge_mod
from . import plan as plan_mod
from . import tag as tag_mod
from .llm import LLM
from .parse import Episode, fmt_time


@dataclass
class Config:
    tag_model: str = "heuristic"
    judge_model: str = "heuristic"
    pool_size: int = 30
    tag_batch: int = 1                  # episodes per tagging call (raise for big-context paid models)
    judge_batch: int = 5
    judge_runs: int = 2
    allow_stitched: bool = True
    cache_dir: str = "cache"
    api_keys: dict | None = None        # optional {provider: key} from the web app


def _make_llm(spec: str, cfg: Config) -> LLM:
    llm = LLM(spec, cache_dir=Path(cfg.cache_dir))
    if cfg.api_keys and llm.provider in cfg.api_keys:
        llm.api_key = cfg.api_keys[llm.provider]
    return llm


def check_promos(promos: list[dict], episodes: list[Episode]) -> dict:
    """Hard rules, checked in code. Anything that fails is reported, not hidden."""
    eps = {e.number: e for e in episodes}
    cuts = {n: set(e.cut_points()) for n, e in eps.items()}
    problems = []
    for p in promos:
        if not (cand_mod.MIN_LEN - 0.01 <= p["duration_s"] <= cand_mod.MAX_LEN + 0.01):
            problems.append(f"{p['id']}: duration {p['duration_s']}s outside 20-60s")
        for s in p["segments"]:
            e = eps.get(s["episode"])
            if e is None:
                problems.append(f"{p['id']}: episode {s['episode']} does not exist")
                continue
            if s["start_s"] < 0 or s["end_s"] > e.duration + 0.01 or s["end_s"] <= s["start_s"]:
                problems.append(f"{p['id']}: bad segment times {s}")
            for t in (s["start_s"], s["end_s"]):
                if round(t, 2) not in cuts[s["episode"]]:
                    problems.append(f"{p['id']}: cut at {t} is not a clean shot boundary")
        hb = p["hook_moment"]
        first = p["segments"][0]
        hblock = next(b for b in eps[hb["episode"]].blocks if b.id == hb["id"])
        if hblock.start - first["start_s"] > cand_mod.HOOK_WINDOW + 0.01:
            problems.append(f"{p['id']}: hook starts {hblock.start - first['start_s']:.1f}s in (>3s)")
    return {"checked": len(promos), "passed": len(promos) - len({x.split(':')[0] for x in problems}),
            "problems": problems}


def run(episodes: list[Episode], cfg: Config, log=print) -> dict:
    t0 = time.time()
    tag_llm = _make_llm(cfg.tag_model, cfg)
    # several independent judges separated by ";"  (each may have ","-separated fallbacks)
    judge_llms = [_make_llm(s.strip(), cfg) for s in cfg.judge_model.split(";") if s.strip()]
    log(f"Parsed {len(episodes)} episodes, {sum(len(e.blocks) for e in episodes)} blocks")

    log(f"Stage 2: tagging with {cfg.tag_model}")
    moments, ep_info, tag_stats = tag_mod.tag(episodes, tag_llm, batch_size=cfg.tag_batch, log=log)
    log(f"  {len(moments)} moments kept, {tag_stats['invalid_ids']} invented IDs dropped")

    log("Stage 3: building candidate promos")
    pool, cand_stats = cand_mod.build(episodes, moments, pool_size=cfg.pool_size,
                                      allow_stitched=cfg.allow_stitched)
    log(f"  {cand_stats['raw_candidates']} possible clips -> {len(pool)} in the judging pool "
        f"(covers {cand_stats['episodes_covered']}/{cand_stats['episodes_total']} episodes)")

    per_judge, judge_stats = {}, {}
    for jl in judge_llms:
        log(f"Stage 4: judging with {jl.spec} ({cfg.judge_runs} runs)")
        judge_stats[jl.spec] = {}
        per_judge[jl.spec] = judge_mod.judge(pool, episodes, jl, batch_size=cfg.judge_batch,
                                             runs=cfg.judge_runs, log=log, stats=judge_stats[jl.spec])
        s = judge_stats[jl.spec]
        log(f"  malformed answers dropped: {s['malformed']}, failed batches: {s['failed_batches']}, "
            f"clips with no valid score: {s['missing']}")
    agree = judge_mod.agreement(per_judge)
    if agree:
        log(f"  judges agree: rank correlation {agree['spearman']}, top-10 overlap {agree['top10_overlap']}/10")
    ranked = judge_mod.rank(judge_mod.merge_judges(per_judge))
    checks = check_promos(ranked, episodes)
    log(f"Stage 5: {len(ranked)} promos ranked; rule checks passed {checks['passed']}/{checks['checked']}")

    plan = plan_mod.posting_plan(ranked, judge_llms[0], log=log)

    return {
        "promos": ranked,
        "plan": plan,
        "moments": tag_mod.moments_to_dicts(moments),
        "episode_info": ep_info,
        "log": {"config": {k: v for k, v in cfg.__dict__.items() if k != "api_keys"},
                "episodes": len(episodes), "tag_stats": tag_stats, "candidate_stats": cand_stats,
                "judge_agreement": agree, "judge_stats": judge_stats, "checks": checks, "tag_llm_usage": tag_llm.stats,
                "judge_llm_usage": {jl.spec: jl.stats for jl in judge_llms},
                "seconds": round(time.time() - t0, 1)},
    }


def _seg_str(p: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])


def write_outputs(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    promos = result["promos"]
    (out / "promos.json").write_text(json.dumps(promos, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "moments.json").write_text(json.dumps({"episode_info": result["episode_info"],
                                                  "moments": result["moments"]}, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    (out / "run_log.json").write_text(json.dumps(result["log"], indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        judges = list(promos[0].get("judges", {})) if promos else []
        w.writerow(["rank", "segments", "duration_s", "kind", "hook", "cliffhanger", "why", "score",
                    "hook_score", "cliffhanger_score", "standalone", "payoff_link", "spoiler_risk",
                    "uncertain", "judge_gap", "hook_type"] + [f"score_{safe_name(j)}" for j in judges])
        for p in promos:
            s = p["scores"]
            w.writerow([p["rank"], _seg_str(p), p["duration_s"], p["kind"], p["hook"], p["cliffhanger"], p["why"],
                        p["score"], s["hook"], s["cliffhanger"], s["standalone"], s["payoff_link"],
                        s["spoiler_risk"], p["uncertain"], p.get("judge_gap", ""), p["hook_moment"]["hook_type"]]
                       + [p["judges"][j]["score"] for j in judges])
    by_id = {p["id"]: p for p in promos}
    plan = result.get("plan", {"plan": [], "hold": []})
    (out / "posting_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    pl = ["# Posting plan: what to post first, and why", ""]
    for it in plan["plan"]:
        p = by_id[it["id"]]
        pl += [f"{it['slot']}. **{_seg_str(p)}** ({p['duration_s']}s, rank #{p['rank']})  ",
               f"   Hook: {p['hook']} · Cliffhanger: {p['cliffhanger']}  ", f"   **Why now:** {it['reason']}", ""]
    if plan.get("hold"):
        pl += ["## Hold back", ""] + [f"- **{_seg_str(by_id[h['id']])}** (rank #{h['rank']}): {h['reason']}"
                                       for h in plan["hold"]] + [""]
    (out / "posting_plan.md").write_text("\n".join(pl), encoding="utf-8")
    ag = result["log"].get("judge_agreement") or {}
    lines = ["# Ranked promos", "",
             f"Tag model: `{result['log']['config']['tag_model']}` · Judge model(s): `{result['log']['config']['judge_model']}`",
             f"Rule checks passed: {result['log']['checks']['passed']}/{result['log']['checks']['checked']}",
             (f"Judge agreement: rank correlation {ag['spearman']}, top-10 overlap {ag['top10_overlap']}/10" if ag else ""),
             ""] + pl
    for p in promos:
        flag = " ⚠ uncertain" if p["uncertain"] else ""
        lines += [f"## #{p['rank']} · {_seg_str(p)} · {p['duration_s']}s · score {p['score']}{flag}",
                  f"- **Hook:** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}", f"- **Why:** {p['why']}",
                  f"- Scores: " + ", ".join(f"{k} {v}" for k, v in p["scores"].items()),
                  "", "<details><summary>Clip text</summary>", "", "```", p["clip_text"], "```", "</details>", ""]
    (out / "promos.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def safe_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec).strip("-")
