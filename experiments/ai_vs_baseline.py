"""What does the AI add over simple keyword rules?

Runs the no-AI keyword baseline with the same settings as the AI run, then compares:
1. Structure: overlap of the top 10, joined vs single clips, how early in the show, hook variety.
2. Blind judging: the top 10 of each tool are mixed, shuffled and given neutral IDs, and an AI judge
   (not the tagger) scores them without knowing which tool picked which. The judge is still an AI,
   so this is supporting evidence, not proof; the editor review in the app is the human check.

    python experiments/ai_vs_baseline.py --input "<folder>" [--ai outputs/final/promos.json] [--judge-model ...]

Writes outputs/experiments/ai_vs_baseline/report.md and report.json.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder import judge as judge_mod                 # noqa: E402
from promo_finder.candidates import Candidate, Segment      # noqa: E402
from promo_finder.llm import LLM, load_env                  # noqa: E402
from promo_finder.parse import load_show                    # noqa: E402
from promo_finder.pipeline import Config, run               # noqa: E402
from promo_finder.tag import Moment                         # noqa: E402


def overlap(a: dict, b: dict) -> float:
    inter = 0.0
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] == sb["episode"]:
                inter += max(0.0, min(sa["end_s"], sb["end_s"]) - max(sa["start_s"], sb["start_s"]))
    return inter / max(1e-6, min(a["duration_s"], b["duration_s"]))


def to_candidate(p: dict, new_id: str) -> Candidate:
    segs = [Segment(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]
    return Candidate(new_id, p["kind"], segs, Moment(**p["hook_moment"]), Moment(**p["cliff_moment"]))


def describe(top: list[dict]) -> dict:
    eps = [p["segments"][0]["episode"] for p in top]
    return {"joined_clips": sum(p["kind"] == "stitched" for p in top),
            "mean_start_episode": round(sum(eps) / len(eps), 1),
            "from_episodes_1_to_5": sum(e <= 5 for e in eps),
            "distinct_episodes": len(set(eps)),
            "hook_types": dict(Counter(p["hook_moment"]["hook_type"] for p in top).most_common())}


def main():
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ai", default="outputs/final/promos.json")
    ap.add_argument("--judge-model", default="gemini:gemini-3.5-flash,gemini:gemini-3.1-flash-lite")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="outputs/experiments/ai_vs_baseline")
    a = ap.parse_args()

    show = load_show(a.input)
    ai = json.load(open(a.ai, encoding="utf-8"))
    print("Running the keyword baseline (no AI) with the same settings...")
    base = run(show, Config(tag_model="heuristic", judge_model="heuristic", pool_size=60, variants_for_top=0),
               log=lambda m: None)["promos"]
    top_ai, top_base = ai[:a.k], base[:a.k]

    # ---- blind judging: mix, shuffle, neutral IDs
    pairs = [("ai", p) for p in top_ai] + [("baseline", p) for p in top_base]
    random.Random(11).shuffle(pairs)
    cands, source = [], {}
    for i, (src, p) in enumerate(pairs, 1):
        cid = f"X{i:02d}"
        cands.append(to_candidate(p, cid))
        source[cid] = (src, p["rank"])
    llm = LLM(a.judge_model)
    print(f"Blind-judging {len(cands)} clips with {a.judge_model} ...")
    judged = judge_mod.judge(cands, show, llm, batch_size=5, runs=1, log=print)
    by_src = {"ai": [], "baseline": []}
    for d in judged:
        by_src[source[d["id"]][0]].append(d)
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 2)
    blind = {src: {"clips_scored": len(ds), "mean_score": mean([d["score"] for d in ds]),
                   **{f"mean_{k}": mean([d["scores"][k] for d in ds])
                      for k in ("hook", "cliffhanger", "standalone", "payoff_link")}}
             for src, ds in by_src.items()}
    ranked = sorted(judged, key=lambda d: -d["score"])
    blind["ai_clips_in_blind_top_half"] = f"{sum(source[d['id']][0] == 'ai' for d in ranked[:len(ranked) // 2])}/{len(ranked) // 2}"

    same = sum(any(overlap(p, q) >= 0.5 for q in top_base) for p in top_ai)
    rep = {"k": a.k, "top_k_same_footage": f"{same}/{a.k}",
           "ai": describe(top_ai), "baseline": describe(top_base), "blind_judging": blind,
           "judge_model": a.judge_model, "judge_calls": llm.stats["calls"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    lines = [f"# AI vs keyword baseline (top {a.k} of each)", "",
             f"Same footage in both top {a.k}s: {rep['top_k_same_footage']}", "",
             "| | AI tool | Keyword baseline |", "|---|---|---|"]
    for key in ("joined_clips", "mean_start_episode", "from_episodes_1_to_5", "distinct_episodes"):
        lines.append(f"| {key.replace('_', ' ')} | {rep['ai'][key]} | {rep['baseline'][key]} |")
    for key in ("mean_score", "mean_hook", "mean_cliffhanger", "mean_standalone", "mean_payoff_link"):
        lines.append(f"| blind judge {key.replace('mean_', '').replace('_', ' ')} | {blind['ai'][key]} | {blind['baseline'][key]} |")
    lines += ["", f"AI picks in the blind judge's top half: {blind['ai_clips_in_blind_top_half']}",
              f"Hook types (AI): {rep['ai']['hook_types']}", f"Hook types (baseline): {rep['baseline']['hook_types']}",
              "", "The blind judge is an AI from a different family than the tagger and does not know which tool "
              "picked each clip. It is supporting evidence; the human check is the editor review in the app."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
