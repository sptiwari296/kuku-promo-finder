"""Command line entry point.

    python run.py --input "<folder of episode .txt files>"
    python run.py --input <folder> --tag-model gemini:gemini-2.5-flash-lite --judge-model gemini:gemini-2.5-flash

Models can also be set in .env as PROMO_TAG_MODEL / PROMO_JUDGE_MODEL.
"""
import argparse
import os
from pathlib import Path

from promo_finder.llm import load_env
from promo_finder.parse import load_show
from promo_finder.pipeline import Config, fast, run, write_outputs, safe_name


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    ap = argparse.ArgumentParser(description="Find ranked promo clips in a microdrama screenplay.")
    ap.add_argument("--input", required=True, help="folder with episode .txt files")
    ap.add_argument("--tag-model", default=os.environ.get("PROMO_TAG_MODEL", "heuristic"))
    ap.add_argument("--judge-model", default=os.environ.get("PROMO_JUDGE_MODEL", "heuristic"))
    ap.add_argument("--pool", type=int, default=30, help="how many candidates the judge scores")
    ap.add_argument("--episodes", default="", help="optional range, e.g. 1-10")
    ap.add_argument("--tag-batch", type=int, default=int(os.environ.get("PROMO_TAG_BATCH", 1)),
                    help="episodes per tagging call (1 fits free-tier limits)")
    ap.add_argument("--fast", action="store_true",
                    help="demo mode: 20-clip shortlist, one judge, one pass, 4 parallel calls")
    ap.add_argument("--workers", type=int, default=0, help="parallel AI calls (default 1, or 4 with --fast)")
    ap.add_argument("--no-stitched", action="store_true", help="only single continuous clips")
    ap.add_argument("--no-last-beat", action="store_true", help="turn off the rule that every episode's last beat is a cliffhanger")
    ap.add_argument("--out", default="", help="output folder (default outputs/<tag>__<judge>)")
    a = ap.parse_args()

    eps = load_show(a.input)
    if a.episodes:
        lo, hi = (int(x) for x in a.episodes.split("-"))
        eps = [e for e in eps if lo <= e.number <= hi]
    if not eps:
        raise SystemExit(f"No episodes found in {a.input}")
    cfg = Config(tag_model=a.tag_model, judge_model=a.judge_model, pool_size=a.pool, tag_batch=a.tag_batch,
                 allow_stitched=not a.no_stitched, last_beat_rule=not a.no_last_beat)
    if a.fast:
        cfg = fast(cfg)
    if a.workers:
        cfg.workers = a.workers
    result = run(eps, cfg)
    out = a.out or f"outputs/{safe_name(a.tag_model)}__{safe_name(a.judge_model)}"
    path = write_outputs(result, out)
    print(f"\nWrote {path}: posting_plan.md, promos.csv/json/md, cut_list.csv, edl/, result.json, "
          f"moments.json, run_log.json")
    for p in result["promos"][:5]:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
        print(f"#{p['rank']} {segs} ({p['duration_s']}s) score {p['score']}  hook: {p['hook'][:70]}")


if __name__ == "__main__":
    main()
