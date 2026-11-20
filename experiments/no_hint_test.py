"""Does the tool find cliffhangers from the story, or from the script literally saying "cliffhanger"?

Some scripts' camera/sound notes contain the word "cliffhanger" (e.g. "strings rise to a tense
cliffhanger beat"). That is a free hint the AI gets. This experiment removes the word from those
episodes, runs the same pipeline twice (with and without the hint) and compares the results.

    python experiments/no_hint_test.py --input "<folder>" [--tag-model ...] [--judge-model ...]

Writes outputs/experiments/no_hint/report.md and report.json.
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promo_finder.llm import load_env                      # noqa: E402
from promo_finder.parse import load_show                    # noqa: E402
from promo_finder.pipeline import Config, run               # noqa: E402

HINT = re.compile(r"cliff-?hang", re.I)
AT_END = re.compile(r"\bcliff-?hang(?:er|ing)s?\b(?=\s*[.,;!?]|\s*$)", re.I)   # "...the scene ends on a cliffhanger."
INNER = re.compile(r"\s*\bcliff-?hang(?:er|ing)s?\b", re.I)                    # "...a tense cliffhanger beat"


def strip_hint(text: str) -> str:
    return INNER.sub("", AT_END.sub("moment", text))


def overlap(a: dict, b: dict) -> float:
    """Share of the shorter promo that is the same footage."""
    inter = 0.0
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] == sb["episode"]:
                inter += max(0.0, min(sa["end_s"], sb["end_s"]) - max(sa["start_s"], sb["start_s"]))
    return inter / max(1e-6, min(a["duration_s"], b["duration_s"]))


def main():
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag-model", default="groq:openai/gpt-oss-20b,groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-120b")
    ap.add_argument("--judge-model", default="groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b")
    ap.add_argument("--out", default="outputs/experiments/no_hint")
    a = ap.parse_args()

    show = load_show(a.input)
    hinted = [e for e in show if any(HINT.search(b.text) for b in e.blocks)]
    hint_blocks = {b.id for e in hinted for b in e.blocks if HINT.search(b.text)}
    stripped = copy.deepcopy(hinted)
    for e in stripped:
        for b in e.blocks:
            if HINT.search(b.text):
                b.text = strip_hint(b.text)
    assert not any(HINT.search(b.text) for e in stripped for b in e.blocks)
    print(f"{len(hinted)} episodes contain the word; {len(hint_blocks)} blocks changed")

    cfg = lambda: Config(tag_model=a.tag_model, judge_model=a.judge_model, pool_size=len(hinted) + 6,
                         judge_runs=1, variants_for_top=0)
    print("\n--- WITH the hint ---")
    with_h = run(hinted, cfg())
    print("\n--- WITHOUT the hint ---")
    without = run(stripped, cfg())

    def tag_view(res):
        ms = res["moments"]
        cliffs = {m["id"]: m for m in ms if m["cliff_score"] >= 5 and m["role"] in ("cliffhanger", "both")}
        hooks = {m["id"] for m in ms if m["hook_score"] >= 5 and m["role"] in ("hook", "both")}
        # best cliffhanger per episode: is it in the episode's last 8 seconds?
        ends = {e.number: e.duration for e in hinted}
        best = {}
        for m in ms:
            if m["cliff_score"] > best.get(m["episode"], {"cliff_score": -1})["cliff_score"]:
                best[m["episode"]] = m
        return cliffs, hooks, best, ends

    c1, h1, best1, ends = tag_view(with_h)
    c2, h2, best2, _ = tag_view(without)
    blocks = {b.id: b for e in hinted for b in e.blocks}
    in_last8 = lambda m: m and blocks[m["id"]].end >= ends[m["episode"]] - 8
    jac = lambda x, y: round(len(x & y) / max(1, len(x | y)), 2)
    near = lambda bid, ids: any(i[:4] == bid[:4] and abs(int(i[4:]) - int(bid[4:])) <= 3 for i in ids)

    top_a, top_b = with_h["promos"][:10], without["promos"][:10]
    same_top = sum(any(overlap(p, q) >= 0.5 for q in top_b) for p in top_a)
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 2)
    rep = {
        "episodes": [e.number for e in hinted],
        "blocks_changed": len(hint_blocks),
        "hint_block_tagged_as_cliffhanger": {
            "with_hint": sum(1 for b in hint_blocks if near(b, c1)),
            "without_hint": sum(1 for b in hint_blocks if near(b, c2)),
            "out_of": len(hint_blocks),
            "note": "counts the hint block or a block within 3 lines of it"},
        "episodes_whose_best_cliffhanger_is_in_last_8s": {
            "with_hint": sum(1 for n in ends if in_last8(best1.get(n))),
            "without_hint": sum(1 for n in ends if in_last8(best2.get(n))), "out_of": len(ends)},
        "mean_cliff_score_at_episode_end": {
            "with_hint": mean([m["cliff_score"] for m in c1.values() if in_last8(m)]),
            "without_hint": mean([m["cliff_score"] for m in c2.values() if in_last8(m)])},
        "cliffhanger_moments_jaccard": jac(set(c1), set(c2)),
        "hook_moments_jaccard_control": jac(h1, h2),
        "top10_promos_same_footage": f"{same_top}/{len(top_a)}",
        "mean_judge_cliffhanger_score_top10": {
            "with_hint": mean([p["scores"]["cliffhanger"] for p in top_a]),
            "without_hint": mean([p["scores"]["cliffhanger"] for p in top_b])},
        "models": {"tag": a.tag_model, "judge": a.judge_model},
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    lines = ["# No-hint test: does the tool rely on the word 'cliffhanger'?", "",
             f"Episodes with the word: {rep['episodes']} ({rep['blocks_changed']} blocks changed).", "",
             "| Measure | With hint | Without hint |", "|---|---|---|"]
    for k in ("hint_block_tagged_as_cliffhanger", "episodes_whose_best_cliffhanger_is_in_last_8s",
              "mean_cliff_score_at_episode_end", "mean_judge_cliffhanger_score_top10"):
        v = rep[k]
        suffix = f" / {v['out_of']}" if "out_of" in v else ""
        lines.append(f"| {k.replace('_', ' ')} | {v['with_hint']}{suffix} | {v['without_hint']}{suffix} |")
    lines += ["", f"- Cliffhanger moments that stayed the same (Jaccard): {rep['cliffhanger_moments_jaccard']}",
              f"- Hook moments that stayed the same (control, Jaccard): {rep['hook_moments_jaccard_control']}",
              f"- Top-10 promos using the same footage: {rep['top10_promos_same_footage']}", "",
              "Note: the AI is not fully repeatable even on identical input, so the hook Jaccard is the "
              "baseline for 'normal' run-to-run change. Cliffhanger change well beyond it would mean the hint mattered."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
