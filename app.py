"""Streamlit app: upload screenplays, pick a model, get ranked promos.

    streamlit run app.py
"""
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder import export as ex
from promo_finder import review as rv
from promo_finder.llm import load_env
from promo_finder.parse import load_texts
from promo_finder.pipeline import Config, run

load_env()
st.set_page_config(page_title="Promo Finder", layout="wide")

st.title("Promo Finder")
st.caption("Reads microdrama screenplays and returns ranked 20–60s promo clips with exact cut points, "
           "the hook, the cliffhanger and why each one works. An editor approves the final picks.")

PRESETS = {
    "Groq gpt-oss-20b → qwen fallback (free)": "groq:openai/gpt-oss-20b,groq:qwen/qwen3.8-27b",
    "Groq gpt-oss-120b (free)": "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b",
    "Gemini 3.1 Flash-Lite (free)": "gemini:gemini-3.1-flash-lite",
    "Gemini 3.5 Flash (free, low daily limit)": "gemini:gemini-3.5-flash",
    "Claude Sonnet (paid)": "anthropic:claude-sonnet-5-5",
    "OpenAI GPT-4.1 mini (paid)": "openai:gpt-4.1-mini",
    "Keyword baseline (no AI, no key)": "heuristic",
}

with st.sidebar:
    st.header("Settings")
    tag_choice = st.selectbox("Tagging model (reads every line)", list(PRESETS), index=0, key="tag_model")
    judge_choice = st.selectbox("Judging model (scores the shortlist)", list(PRESETS), index=1, key="judge_model")
    judge2_choice = st.selectbox("Second judge (optional, from another company)", ["None"] + list(PRESETS),
                                 index=0, key="judge2_model",
                                 help="Two judges from different companies: scores are averaged and "
                                      "disagreements are flagged for the editor.")
    with st.expander("Use any other model"):
        st.caption("Format provider:model, e.g. openrouter:meta-llama/llama-3.3-70b-instruct. "
                   "Leave empty to use the choices above.")
        tag_custom = st.text_input("Custom tagging model", "", key="tag_custom")
        judge_custom = st.text_input("Custom judging model", "", key="judge_custom")
    tag_spec = tag_custom.strip() or PRESETS[tag_choice]
    judge_spec = judge_custom.strip() or PRESETS[judge_choice]
    if judge2_choice != "None" and not judge_custom.strip():
        judge_spec += ";" + PRESETS[judge2_choice]
    pool = st.slider("Candidates to judge", 10, 80, 60, key="pool")
    key = st.text_input("API key (not stored)", type="password", key="api_key",
                        help="Leave empty if the key is already in the server's .env file.")
    stitched = st.checkbox("Allow stitched promos (two moments joined)", True, key="stitched")

if not any(os.environ.get(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                                        "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")):
    st.info("This server has no API key stored. Paste your own Groq or Gemini key in the sidebar "
            "(free at console.groq.com or aistudio.google.com), or choose **Keyword baseline** as both "
            "models to try it with no key.")

uploads = st.file_uploader("Episode screenplays (.txt files or one .zip)", type=["txt", "zip"],
                           accept_multiple_files=True, key="uploads")
local = st.text_input("…or a folder path on this machine", "", key="local_path")

texts: dict[str, str] = {}
for f in uploads or []:
    if f.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(f.read())) as z:
            for n in z.namelist():
                if n.lower().endswith(".txt"):
                    texts[Path(n).name] = z.read(n).decode("utf-8", errors="replace")
    else:
        texts[f.name] = f.read().decode("utf-8", errors="replace")
if local and Path(local).is_dir():
    for p in sorted(Path(local).rglob("*.txt")):
        texts[p.name] = p.read_text(encoding="utf-8", errors="replace")

episodes = load_texts(texts) if texts else []
if episodes:
    st.success(f"Loaded {len(episodes)} episodes "
               f"({min(e.number for e in episodes)}–{max(e.number for e in episodes)}), "
               f"{sum(len(e.blocks) for e in episodes)} timestamped blocks.")

if st.button("Find promos", type="primary", disabled=not episodes, key="go"):
    keys = {}
    if key:   # one pasted key applies to the first model's provider
        keys[tag_spec.split(":")[0]] = key
        keys[judge_spec.split(":")[0]] = key
    cfg = Config(tag_model=tag_spec, judge_model=judge_spec, pool_size=pool, allow_stitched=stitched,
                 api_keys=keys or None)
    log_box = st.empty()
    lines: list[str] = []

    def log(msg):
        lines.append(msg)
        log_box.code("\n".join(lines[-12:]))

    try:
        with st.spinner("Working… tagging is the slow part"):
            st.session_state["result"] = run(episodes, cfg, log=log)
            st.session_state["reviews_store"] = {}   # new run = new promo IDs
            for _k in [k for k in st.session_state if str(k).startswith(("dec_", "why_", "ns_", "ne_"))]:
                del st.session_state[_k]
    except Exception as e:  # show the real reason, e.g. missing key or rate limit
        st.error(f"Run failed: {e}")

saved = st.file_uploader("…or open saved results (result.json from an earlier run, no AI needed)",
                         type=["json"], key="saved_result")
if saved is not None and st.session_state.get("_saved_name") != saved.name + str(saved.size):
    try:
        data = json.loads(saved.read().decode("utf-8"))
        if "promos" not in data:
            raise ValueError("this file has no 'promos' list; use result.json from the outputs folder")
        data.setdefault("plan", {"plan": [], "hold": []})
        data.setdefault("log", {})
        st.session_state["result"] = data
        st.session_state["reviews_store"] = {}
        for _k in [k for k in st.session_state if str(k).startswith(("dec_", "why_", "ns_", "ne_"))]:
            del st.session_state[_k]
        st.session_state["_saved_name"] = saved.name + str(saved.size)
    except Exception as e:
        st.error(f"Could not open that file: {e}")

res = st.session_state.get("result")


def seg_text(segments) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in segments)


def review_store() -> dict:
    """Reviews live here, not only in the widgets: Streamlit forgets the value of any widget
    that is not on screen, so switching views or 'Top 10' would otherwise lose decisions."""
    return st.session_state.setdefault("reviews_store", {})


def save_widgets_to_store(promos) -> None:
    store = review_store()
    for p in promos:
        pid = p["id"]
        d = st.session_state.get(f"dec_{pid}")
        if d:
            store[pid] = {"decision": d, "reason": st.session_state.get(f"why_{pid}", ""),
                          "start_nudge_s": float(st.session_state.get(f"ns_{pid}", 0.0)),
                          "end_nudge_s": float(st.session_state.get(f"ne_{pid}", 0.0))}


def seed_widgets_from_store(pid: str) -> None:
    r = review_store().get(pid)
    if r and f"dec_{pid}" not in st.session_state:
        st.session_state[f"dec_{pid}"] = r["decision"]
        st.session_state[f"why_{pid}"] = r["reason"] if r["reason"] in rv.REASONS else "other"
        st.session_state[f"ns_{pid}"] = max(-10.0, min(10.0, r["start_nudge_s"]))
        st.session_state[f"ne_{pid}"] = max(-10.0, min(10.0, r["end_nudge_s"]))


if res:
    promos = res["promos"]
    lg = res.get("log", {})
    cs = lg.get("candidate_stats", {})
    by_id = {p["id"]: p for p in promos}
    checks = lg.get("checks", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Promos ranked", len(promos))
    c2.metric("Episodes covered", f"{cs.get('episodes_covered', '?')}/{cs.get('episodes_total', '?')}")
    c3.metric("Rule checks passed", f"{checks.get('passed', '?')}/{checks.get('checked', '?')}")
    c4.metric("Editor check", sum(p["uncertain"] for p in promos),
              help="Judges disagreed or were unsure: an editor should look at these.")
    c5.metric("Invented IDs dropped", lg.get("tag_stats", {}).get("invalid_ids", "?"))
    ag = lg.get("judge_agreement") or {}
    if ag:
        st.caption(f"Two judges agree: rank correlation {ag['spearman']}, top-10 overlap "
                   f"{ag['top10_overlap']}/10, mean score gap {ag['mean_abs_gap']} points.")

    # a radio, not st.tabs: tabs jump back to the first one whenever a review button reruns the app
    view = st.radio("View", ["Post these first", "Editor review", "All promos", "Export for editors", "Run log"],
                    horizontal=True, key="view", label_visibility="collapsed")

    # ------------------------------------------------------------ posting plan
    if view == "Post these first":
        for it in res.get("plan", {}).get("plan", []):
            p = by_id.get(it["id"])
            if not p:
                continue
            st.markdown(f"**{it['slot']}. {seg_text(p['segments'])}** · {p['duration_s']}s · rank #{p['rank']}  \n"
                        f"Hook: {p['hook']} · Cliffhanger: {p['cliffhanger']}  \n*Why now:* {it['reason']}")
        for h in res.get("plan", {}).get("hold", []):
            p = by_id.get(h["id"])
            if p:
                st.caption(f"Hold back rank #{p['rank']} ({seg_text(p['segments'])}): {h['reason']}")

    # ------------------------------------------------------------ editor review
    if view == "Editor review":
        st.markdown("Approve or reject each promo, say why, and nudge the cut points if needed. "
                    "This is the human check: the approval rate shows how often the tool's picks are good "
                    "enough to post. Save your reviews as a file to continue later.")
        rv_file = st.file_uploader("Load saved reviews (reviews.csv)", type=["csv"], key="reviews_upload")
        if rv_file is not None and st.session_state.get("_rv_name") != rv_file.name + str(rv_file.size):
            loaded = rv.from_csv(rv_file.read().decode("utf-8"), promos)
            review_store().update(loaded)
            for pid in loaded:                      # drop stale widget values so the loaded ones show
                for k in ("dec_", "why_", "ns_", "ne_"):
                    st.session_state.pop(k + pid, None)
            st.session_state["_rv_name"] = rv_file.name + str(rv_file.size)
            st.success(f"Loaded {len(loaded)} reviews.")

        summary_box = st.container()
        show = st.radio("Show", ["Top 10", "Top 20", "All"], horizontal=True, key="rv_show")
        limit = {"Top 10": 10, "Top 20": 20, "All": len(promos)}[show]
        for p in promos[:limit]:
            pid = p["id"]
            seed_widgets_from_store(pid)
            flag = " · ⚠ editor check" if p["uncertain"] else ""
            # the title must not change when a decision is made, or Streamlit closes the panel
            with st.expander(f"#{p['rank']} · {seg_text(p['segments'])} · {p['duration_s']}s · "
                             f"score {p['score']}{flag}"):
                st.markdown(f"**Hook:** {p['hook']}  \n**Cliffhanger:** {p['cliffhanger']}  \n**Why:** {p['why']}")
                st.caption(" · ".join(f"{k} {v}" for k, v in p["scores"].items()))
                if len(p.get("judges", {})) > 1:
                    for spec, j in p["judges"].items():
                        st.caption(f"{spec.split(',')[0]}: {j['score']} · {j['why']}")
                if p.get("hook_variants"):
                    st.markdown("**Alternative openings to A/B test** (same cliffhanger, untested):")
                    for k, v in enumerate(p["hook_variants"]):
                        st.markdown(f"- **Variant {chr(66 + k)}** · {seg_text(v['segments'])} · {v['duration_s']}s · "
                                    f"{v['hook_type'].replace('_', ' ')} (hook {v['hook_score']}/10): {v['hook']}")
                st.code(p.get("clip_text", ""), language=None)
                for k in (f"ns_{pid}", f"ne_{pid}"):
                    st.session_state.setdefault(k, 0.0)
                st.session_state.setdefault(f"why_{pid}", "")
                a, b = st.columns([2, 2])
                extra = {} if f"dec_{pid}" in st.session_state else {"index": None}
                a.radio("Decision", list(rv.DECISIONS), horizontal=True, key=f"dec_{pid}", **extra)
                b.selectbox("Reason (if rejected)", rv.REASONS, key=f"why_{pid}",
                            format_func=lambda x: x or "—")
                n1, n2 = st.columns(2)
                n1.number_input("Move start (seconds, + = later)", min_value=-10.0, max_value=10.0,
                                step=0.5, key=f"ns_{pid}")
                n2.number_input("Move end (seconds, + = later)", min_value=-10.0, max_value=10.0,
                                step=0.5, key=f"ne_{pid}")
                ns, ne = st.session_state.get(f"ns_{pid}", 0.0), st.session_state.get(f"ne_{pid}", 0.0)
                if ns or ne:
                    new = rv.apply_nudges(p, ns, ne)
                    problem = rv.nudge_problem(new)
                    if problem:
                        st.warning(f"Nudge not applied: {problem}")
                    else:
                        st.caption(f"New cut: {seg_text(new)}")

        save_widgets_to_store(promos[:limit])
        reviews = review_store()
        s = rv.summarize(promos, reviews)
        if reviews:
            icon = {"approve": "✅", "reject": "❌", "unsure": "❔"}
            st.caption("Your decisions: " + "  ".join(
                f"#{by_id[pid]['rank']} {icon.get(r['decision'], '')}"
                for pid, r in sorted(reviews.items(), key=lambda x: by_id[x[0]]["rank"]) if pid in by_id))
        with summary_box:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Reviewed", f"{s['reviewed']}/{len(promos)}")
            m2.metric("Approval rate", f"{s['approval_rate']}%" if s["approval_rate"] is not None else "–")
            m3.metric("Approved in top 5", s["approved_in_top5"])
            m4.metric("Approved in top 10", s["approved_in_top10"])
            if s["decided"]:
                st.caption(
                    f"Flagged for editor check: {s['approval_rate_flagged'] if s['approval_rate_flagged'] is not None else '–'}% approved · "
                    f"not flagged: {s['approval_rate_not_flagged'] if s['approval_rate_not_flagged'] is not None else '–'}% · "
                    f"single clips: {s['approval_rate_single'] if s['approval_rate_single'] is not None else '–'}% · "
                    f"joined clips: {s['approval_rate_stitched'] if s['approval_rate_stitched'] is not None else '–'}%"
                    + (f" · top reject reasons: {', '.join(f'{k} ({v})' for k, v in s['reject_reasons'].items())}"
                       if s["reject_reasons"] else ""))
            st.download_button("Save reviews (reviews.csv)", rv.to_csv(promos, reviews), "reviews.csv",
                               "text/csv", key="dl_reviews", disabled=not reviews)

    # ------------------------------------------------------------ all promos table
    if view == "All promos":
        rows = [{"rank": p["rank"], "segments": seg_text(p["segments"]), "sec": p["duration_s"],
                 "kind": p["kind"], "score": p["score"], "editor check": p["uncertain"],
                 "hook": p["hook"], "cliffhanger": p["cliffhanger"], "why": p["why"],
                 "variants": len(p.get("hook_variants", []))} for p in promos]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------ export
    if view == "Export for editors":
        reviews = review_store()
        only_ok = st.checkbox("Only promos I approved", value=bool(reviews), key="exp_only_ok")
        fps = st.selectbox("Frame rate of the episode videos", [30, 25, 24, 60], key="exp_fps")
        export_list = []
        for p in promos:
            r = reviews.get(p["id"], {})
            if only_ok and r.get("decision") != "approve":
                continue
            q = dict(p)
            q["review"] = r
            if r.get("start_nudge_s") or r.get("end_nudge_s"):
                new = rv.apply_nudges(p, r["start_nudge_s"], r["end_nudge_s"])
                if not rv.nudge_problem(new):
                    q["segments_final"] = new
            export_list.append(q)
        st.caption(f"{len(export_list)} promos in the export. Assumes each episode is its own video file "
                   f"(episode-001.mp4, …) starting at 00:00:00:00; relink in your editor if names differ.")
        if export_list:
            e1, e2 = st.columns(2)
            e1.download_button("Cut list (.csv, with timecodes and hook variants)",
                               ex.cutlist_csv(export_list, fps), "cut_list.csv", "text/csv", key="dl_cut")
            e2.download_button("EDL files for Premiere / DaVinci (.zip)", ex.edl_zip(export_list, fps),
                               "promo_edls.zip", "application/zip", key="dl_edl")
            st.markdown("Preview of the first EDL:")
            st.code(ex.edl(export_list[0], fps), language=None)
        st.download_button("Full results (result.json, reopen later without AI)",
                           json.dumps(res, ensure_ascii=False), "result.json", "application/json",
                           key="dl_result")

    if view == "Run log":
        st.json(lg)
