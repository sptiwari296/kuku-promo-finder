"""Cut lists editors can use directly: a timecode CSV and one EDL per promo.

EDL = CMX3600 edit decision list, the plain-text format Premiere Pro, DaVinci Resolve
and Final Cut can import. Each promo becomes a short timeline built from its segments.

Assumptions (stated in the files too):
- each episode is its own video file named like episode-001.mp4, starting at timecode 00:00:00:00
- 30 frames per second (change FPS if Kuku's masters differ)
"""
from __future__ import annotations

import csv
import io
import zipfile

FPS = 30
CLIP_NAME = "episode-{ep:03d}.mp4"


def timecode(seconds: float, fps: int = FPS) -> str:
    frames = int(round(max(seconds, 0.0) * fps))
    ff = frames % fps
    s = frames // fps
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{ff:02d}"


def _segments(p: dict) -> list[dict]:
    """Use the editor's nudged times when present."""
    return p.get("segments_final") or p["segments"]


def edl(p: dict, fps: int = FPS, title: str | None = None) -> str:
    title = title or f"PROMO_{p['rank']:02d}"
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    rec = 3600.0                                     # timelines conventionally start at 01:00:00:00
    for i, s in enumerate(_segments(p), 1):
        src_in, src_out = s["start_s"], s["end_s"]
        length = src_out - src_in
        reel = f"EP{s['episode']:02d}"[:8]
        lines.append(f"{i:03d}  {reel:<8} AA/V  C        {timecode(src_in, fps)} {timecode(src_out, fps)} "
                     f"{timecode(rec, fps)} {timecode(rec + length, fps)}")
        lines.append(f"* FROM CLIP NAME: {CLIP_NAME.format(ep=s['episode'])}")
        rec += length
    lines += ["", f"* HOOK: {p.get('hook', '')}", f"* CLIFFHANGER: {p.get('cliffhanger', '')}"]
    return "\n".join(lines) + "\n"


def cutlist_rows(promos: list[dict], fps: int = FPS, include_variants: bool = True) -> list[dict]:
    rows = []
    for p in promos:
        versions = [("main", p)]
        if include_variants:
            versions += [(f"hook variant {chr(66 + k)}", v) for k, v in enumerate(p.get("hook_variants", []))]
        for label, v in versions:
            for i, s in enumerate(_segments(v) if label == "main" else v["segments"], 1):
                rows.append({"rank": p["rank"], "version": label, "segment": i, "episode": s["episode"],
                             "clip_name": CLIP_NAME.format(ep=s["episode"]),
                             "start_tc": timecode(s["start_s"], fps), "end_tc": timecode(s["end_s"], fps),
                             "start_s": s["start_s"], "end_s": s["end_s"],
                             "length_s": round(s["end_s"] - s["start_s"], 2),
                             "hook": v.get("hook", ""), "editor_decision": p.get("review", {}).get("decision", "")})
    return rows


def cutlist_csv(promos: list[dict], fps: int = FPS) -> str:
    rows = cutlist_rows(promos, fps)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["rank"])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def edl_zip(promos: list[dict], fps: int = FPS) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in promos:
            z.writestr(f"promo_{p['rank']:02d}.edl", edl(p, fps))
        z.writestr("README.txt", f"One EDL (CMX3600) per promo, {fps} fps, non-drop frame.\n"
                                 "Assumes each episode is a separate file named episode-001.mp4 etc. "
                                 "starting at 00:00:00:00. Relink clips in your editor if names differ.\n")
    return buf.getvalue()
