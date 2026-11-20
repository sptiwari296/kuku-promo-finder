# Promo Finder

Reads a microdrama's timestamped screenplays and returns ranked 20–60 second promo clips.
For each clip it gives the episode, the exact start and end of every segment, the hook,
the cliffhanger and one line on why it works, plus a short plan of which to post first and why.
An editor approves the final picks.

**Live app:** https://kuku-promo-finder-sp.streamlit.app (no keys stored: paste your own Groq/Gemini key in the sidebar, or pick "Keyword baseline").

## For the editor (in the app)

- **Post these first:** the posting plan, with reasons and what to hold back.
- **Editor review:** approve / reject / unsure each promo, pick a reason, nudge the start or end by up to ±10s.
  Shows the approval rate, approvals in the top 5 and top 10, and whether flagged ("editor check") and joined
  promos get rejected more. Save as `reviews.csv` and load it again later ([review.py](promo_finder/review.py)).
- **Hook variants:** for the top 10 promos, 2 alternative openings that end on the same cliffhanger, for A/B
  tests. Built in code from the AI's tags, so they are marked untested ([candidates.py](promo_finder/candidates.py)).
- **Export for editors:** a cut list (`cut_list.csv`, timecodes plus variants) and one CMX3600 EDL per promo
  for Premiere Pro / DaVinci Resolve, using the editor's nudged cuts and optionally only approved promos
  ([export.py](promo_finder/export.py)). Assumes one video file per episode (`episode-001.mp4`) and 30 fps (selectable).
- **Reopen a run without AI:** upload `result.json` from an earlier run.

## Run it

```
pip install -r requirements.txt
copy .env.example .env        # then paste a free Groq and/or Gemini key into .env
python run.py --input "<folder with episode .txt files>"
streamlit run app.py          # web app: upload .txt files or a .zip
```

No key? `python run.py --input <folder>` runs the keyword baseline (`heuristic`), so the tool always works.

The exact command used for the submitted results:

```
python run.py --input "<folder>" --pool 60 --out outputs/final ^
  --tag-model "groq:openai/gpt-oss-20b,groq:qwen/qwen3.8-27b,groq:openai/gpt-oss-120b" ^
  --judge-model "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b;gemini:gemini-3.5-flash,gemini:gemini-3.1-flash-lite"
```

## Swap the model (one setting)

Models are written `provider:model` and set in `.env`, on the command line, or in the app.
A comma gives fallbacks (used if a model hits its daily limit); a semicolon adds an independent second judge.

```
python run.py --input <folder> --tag-model gemini:gemini-3.1-flash-lite --judge-model "anthropic:claude-sonnet-5-5;openai:gpt-4.1-mini"
```

Providers: `gemini`, `groq`, `openrouter`, `openai`, `anthropic`, `ollama` (local), `custom`
(any OpenAI-compatible endpoint via `PROMO_BASE_URL` + `PROMO_API_KEY`), and `heuristic` (no AI).
All AI calls go through [promo_finder/llm.py](promo_finder/llm.py).

## How it works

| Step | Who | File |
|---|---|---|
| 1. Parse screenplays into timestamped blocks; find clean cut points (shot boundaries not inside dialogue) | Code | [parse.py](promo_finder/parse.py) |
| 2. Tag hooks and cliffhangers. The AI returns block IDs only, never timestamps. Invented IDs are dropped and counted | AI (cheap model, cached) | [tag.py](promo_finder/tag.py) |
| 3. Build 20–60s candidates (hook in the first 3s, cut right after a cliffhanger, one stretch or two joined moments). Shortlist the best clip from every episode first, then the best of the rest; remove duplicates | Code | [candidates.py](promo_finder/candidates.py) |
| 4. Two judges from different companies score each clip on its own (hook, cliffhanger, stands alone, hook↔ending link, spoiler risk), seeing only the clip's text. Each judge runs twice with shuffled rubric and clip order | AI | [judge.py](promo_finder/judge.py) |
| 5. Average the judges; flag "editor check" when a judge disagrees with itself on hook/cliffhanger/stands-alone or the judges disagree by ≥1.5 points. Rank (hook, cliffhanger and stands-alone weighted equally, bonus for early episodes, penalty for flagged). Check hard rules in code | Code | [judge.py](promo_finder/judge.py), [pipeline.py](promo_finder/pipeline.py) |
| 6. Posting plan: which 5 to post first and why, and which to hold back. AI picks only from ranked IDs; rule-based fallback | AI + code | [plan.py](promo_finder/plan.py) |
| 7. Editor reviews the plan and flagged promos | Human | app |

## Outputs (`outputs/final/`)

- `posting_plan.md` / `.json`: post these first, in order, with reasons; and what to hold back
- `cut_list.csv` and `edl/promo_NN.edl`: cut points as timecodes, ready to import into an editor
- `result.json`: the whole run in one file, to reopen in the app without calling any AI
- `promos.csv` / `promos.json`: all ranked promos with segments, hook, cliffhanger, why, scores per judge, editor-check flag
- `promos.md`: readable report with each clip's script text
- `moments.json`: every tagged moment
- `run_log.json`: models, calls, tokens, invented IDs, malformed answers, coverage, judge agreement, rule checks

## Results on the 50-episode pack (free tier)

| | |
|---|---|
| Tagging | Groq `gpt-oss-20b`: 47 calls, ~160k input / 25k output tokens, 359 moments, **0 invented IDs** |
| Candidates | 2,125 possible clips → 60 judged, covering **49/50 episodes** (Ep50 had no hook+cliffhanger pair that fits 20–60s) |
| Judges | Groq `gpt-oss-120b` and Gemini `gemini-3.5-flash`, 24 calls each, ~45k input tokens each |
| Judge agreement | rank correlation **0.53**, top-10 overlap **5/10**, mean gap 0.81 points |
| Rule checks | **60/60** passed (length, real timestamps, clean cuts, hook within 3s) |
| Editor check | 22/60 flagged, 2 of them in the top 10 |
| Model errors caught | 1 malformed judge answer (text where a score belonged), dropped and counted |
| Wall time | ~27 min tagging + ~15 min judging on free tiers, almost all waiting on rate limits |

`compare_judges.py` compares two separate runs judged by different models (used for the first experiment).

## Check quality against your own picks

Fill `human_picks.csv` (episode,start,end) **before** looking at the output, then:

```
python evaluate.py outputs/final/promos.json human_picks.csv
```

## Confidentiality

The screenplays are Kuku's. They are never committed: `.gitignore` excludes `*.txt`, `data/`, `cache/` and `.env`.
Output files quote dialogue from the clips, so share the outputs folder only through a private link.
