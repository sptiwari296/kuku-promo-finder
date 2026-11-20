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
    except Exception as e:  # show the real reason, e.g. missing key or rate limit
        st.error(f"Run failed: {e}")

res = st.session_state.get("result")
if res:
    promos = res["promos"]
    lg = res["log"]
    cs = lg["candidate_stats"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Promos ranked", len(promos))
    c2.metric("Episodes covered", f"{cs.get('episodes_covered', '?')}/{cs.get('episodes_total', '?')}")
    c3.metric("Rule checks passed", f"{lg['checks']['passed']}/{lg['checks']['checked']}")
    c4.metric("Editor check", sum(p["uncertain"] for p in promos),
              help="Judges disagreed or were unsure: an editor should look at these.")
    c5.metric("Invented IDs dropped", lg["tag_stats"]["invalid_ids"])
    ag = lg.get("judge_agreement") or {}
    if ag:
        st.caption(f"Two judges agree: rank correlation {ag['spearman']}, top-10 overlap "
                   f"{ag['top10_overlap']}/10, mean score gap {ag['mean_abs_gap']} points.")

    by_id = {p["id"]: p for p in promos}
    st.subheader("Post these first")
    for it in res.get("plan", {}).get("plan", []):
        p = by_id[it["id"]]
        segs = " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
        st.markdown(f"**{it['slot']}. {segs}** · {p['duration_s']}s · rank #{p['rank']}  \n"
                    f"Hook: {p['hook']} · Cliffhanger: {p['cliffhanger']}  \n*Why now:* {it['reason']}")
    for h in res.get("plan", {}).get("hold", []):
        p = by_id[h["id"]]
        st.caption(f"Hold back rank #{p['rank']} ({' + '.join('Ep%d' % s['episode'] for s in p['segments'])}): {h['reason']}")

    st.subheader("All ranked promos")

    rows = [{"rank": p["rank"],
             "segments": " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in p["segments"]),
             "sec": p["duration_s"], "score": p["score"], "uncertain": p["uncertain"],
             "hook": p["hook"], "cliffhanger": p["cliffhanger"], "why": p["why"]} for p in promos]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Review each promo")
    for p in promos:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
        flag = " · ⚠ editor check" if p["uncertain"] else ""
        with st.expander(f"#{p['rank']} · {segs} · {p['duration_s']}s · score {p['score']}{flag}"):
            st.markdown(f"**Hook:** {p['hook']}  \n**Cliffhanger:** {p['cliffhanger']}  \n**Why:** {p['why']}")
            st.write({k: v for k, v in p["scores"].items()})
            if len(p.get("judges", {})) > 1:
                for spec, j in p["judges"].items():
                    st.caption(f"{spec.split(',')[0]}: {j['score']} · {j['why']}")
            st.code(p["clip_text"], language=None)

    st.download_button("Download promos.json", json.dumps(promos, indent=1, ensure_ascii=False),
                       "promos.json", "application/json", key="dl_json")
    st.download_button("Download promos.csv", pd.DataFrame(rows).to_csv(index=False), "promos.csv",
                       "text/csv", key="dl_csv")
    with st.expander("Run log"):
        st.json(lg)
