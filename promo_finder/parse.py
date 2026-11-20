"""Stage 1 - Parse screenplay .txt files into structured blocks. Pure code, no AI.

Each episode file looks like:
    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. OFFICE - NIGHT
    **CAMERA:**
    Start: (00:00.00) | End: (00:03.20)
    <description>
    **MAYA (Shocked):**
    Start: (00:01.50) | End: (00:02.90)
    "You lied to me."

Timestamps are MM:SS.cc relative to the start of the episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
EP_NUM_RE = re.compile(r"EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
FILE_EP_RE = re.compile(r"episode-(\d+)", re.I)
SCENE_HEADING_RE = re.compile(r"^scene heading:\s*(.+)$", re.I)


def fmt_time(seconds: float) -> str:
    """Seconds -> MM:SS.cc (the screenplay's own format)."""
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Block:
    id: str            # e.g. E01B012 - stable ID the AI refers to
    episode: int
    idx: int
    kind: str          # camera | action | dialogue | sound
    speaker: str       # for dialogue
    delivery: str      # e.g. "Cold, Commanding"
    start: float
    end: float
    text: str
    scene: str

    def line(self) -> str:
        """Compact one-line form used in prompts."""
        t = f"[{fmt_time(self.start)}-{fmt_time(self.end)}]"
        if self.kind == "dialogue":
            d = f" ({self.delivery})" if self.delivery else ""
            return f"{self.id} {t} {self.speaker}{d}: {self.text}"
        return f"{self.id} {t} {self.kind.upper()}: {self.text}"


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    blocks: list[Block] = field(default_factory=list)
    file: str = ""

    @property
    def duration(self) -> float:
        return max((b.end for b in self.blocks), default=0.0)

    def cut_points(self) -> list[float]:
        """Times where an editor can cut cleanly: camera shot boundaries that do
        not fall inside a line of dialogue. Episode start/end are always valid."""
        pts = {0.0, round(self.duration, 2)}
        for b in self.blocks:
            if b.kind == "camera":
                pts.add(round(b.start, 2))
                pts.add(round(b.end, 2))
        speech = [(b.start, b.end) for b in self.blocks if b.kind == "dialogue"]
        ok = [p for p in pts if not any(s + 0.05 < p < e - 0.05 for s, e in speech)]
        return sorted(ok)

    def blocks_between(self, start: float, end: float) -> list[Block]:
        return [b for b in self.blocks if b.end > start + 0.01 and b.start < end - 0.01]

    def to_dict(self) -> dict:
        return {"number": self.number, "logline": self.logline, "continuity": self.continuity,
                "file": self.file, "blocks": [asdict(b) for b in self.blocks]}


def _kind_and_speaker(label: str) -> tuple[str, str, str]:
    up = label.strip()
    base = up.split("(")[0].strip().upper()
    if base == "CAMERA":
        return "camera", "", ""
    if base == "ACTION":
        return "action", "", ""
    if base in ("SOUND FX", "SFX", "SOUND", "MUSIC"):
        return "sound", "", ""
    parens = re.findall(r"\(([^)]*)\)", up)
    # drop technical tags like O.S. / V.O. / CONT'D, keep the acting direction
    delivery = [p for p in parens if p.upper().replace(".", "").replace("'", "") not in ("OS", "VO", "CONTD", "OC")]
    return "dialogue", base.title(), (delivery[-1] if delivery else "")


def parse_text(text: str, fallback_number: int = 0, file: str = "") -> Episode:
    lines = text.splitlines()
    m = EP_NUM_RE.search(text)
    number = int(m.group(1)) if m else fallback_number
    logline = continuity = ""
    for ln in lines[:10]:
        if ln.upper().startswith("EPISODE LOGLINE:"):
            logline = ln.split(":", 1)[1].strip()
        elif ln.upper().startswith("CONTINUITY NOTES:"):
            continuity = ln.split(":", 1)[1].strip()

    ep = Episode(number=number, logline=logline, continuity=continuity, file=file)
    scene = ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        sh = SCENE_HEADING_RE.match(ln)
        if sh:
            scene = sh.group(1).strip()
        lab = LABEL_RE.match(ln)
        if lab and i + 1 < len(lines):
            tm = TIME_RE.search(lines[i + 1])
            if tm:
                start = int(tm.group(1)) * 60 + float(tm.group(2))
                end = int(tm.group(3)) * 60 + float(tm.group(4))
                body = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not LABEL_RE.match(lines[j].strip()):
                    body.append(lines[j].strip())
                    j += 1
                kind, speaker, delivery = _kind_and_speaker(lab.group(1))
                txt = " ".join(body).strip()
                if kind == "dialogue":
                    txt = txt.strip('"').strip("“”")
                idx = len(ep.blocks)
                ep.blocks.append(Block(id=f"E{number:02d}B{idx:03d}", episode=number, idx=idx, kind=kind,
                                       speaker=speaker, delivery=delivery, start=start, end=end,
                                       text=txt, scene=scene))
                i = j
                continue
        i += 1
    return ep


def parse_file(path: Path) -> Episode:
    fm = FILE_EP_RE.search(path.name)
    return parse_text(path.read_text(encoding="utf-8", errors="replace"),
                      fallback_number=int(fm.group(1)) if fm else 0, file=path.name)


def load_show(folder: str | Path) -> list[Episode]:
    """Load every episode .txt under a folder (searched recursively), sorted by number."""
    paths = sorted(Path(folder).rglob("*.txt"))
    eps = [parse_file(p) for p in paths]
    eps = [e for e in eps if e.blocks]
    return sorted(eps, key=lambda e: e.number)


def load_texts(named_texts: dict[str, str]) -> list[Episode]:
    """Same as load_show but from in-memory files (used by the web app)."""
    eps = []
    for name, txt in named_texts.items():
        fm = FILE_EP_RE.search(name)
        e = parse_text(txt, fallback_number=int(fm.group(1)) if fm else 0, file=name)
        if e.blocks:
            eps.append(e)
    return sorted(eps, key=lambda e: e.number)
