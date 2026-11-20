"""The ONLY place that talks to an AI model. Swap models with one setting.

A model is written as  "provider:model", for example
    gemini:gemini-2.5-flash-lite
    groq:llama-3.3-70b-versatile
    openrouter:meta-llama/llama-3.3-70b-instruct:free
    openai:gpt-4.1-mini
    anthropic:claude-sonnet-5-5
    ollama:llama3.1            (local, free, no key)
    heuristic                  (no AI at all - keyword baseline, used for testing/comparison)

Any OpenAI-compatible endpoint also works:  custom:<model>  with PROMO_BASE_URL and PROMO_API_KEY.

Every response is cached on disk (keyed by model + prompt), so re-runs are free
and the raw model output is kept for the "raw outputs" folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# provider -> (base_url, env var holding the key)
OPENAI_COMPATIBLE = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", ""),
    "custom": (os.environ.get("PROMO_BASE_URL", ""), "PROMO_API_KEY"),
}


def load_env(path: str | Path = ".env") -> None:
    """Tiny .env loader so keys never live in code."""
    p = Path(path)
    if not p.exists():
        return
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]   # accept either name


class LLMError(RuntimeError):
    pass


class FatalLLMError(Exception):
    """Bad key, unknown model, request too large: stop instead of retrying."""


def extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add chatter. Pull out the object."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


@dataclass
class LLM:
    spec: str
    cache_dir: Path = Path("cache")
    api_key: str | None = None          # optional override (web app passes it in)
    max_retries: int = 10
    stats: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "in_tokens": 0,
                                                 "out_tokens": 0, "seconds": 0.0, "errors": 0})

    def __post_init__(self):
        # "a,b,c" = try model a; if it hits a daily limit or keeps failing, fall back to b, then c.
        self.specs = [s.strip() for s in self.spec.split(",") if s.strip()]
        for s in self.specs:
            prov, _, model = s.partition(":")
            prov = prov.strip().lower()
            if prov != "heuristic" and not model:
                raise LLMError(f"Model spec '{s}' must look like provider:model")
            if prov not in OPENAI_COMPATIBLE and prov not in ("anthropic", "heuristic"):
                raise LLMError(f"Unknown provider '{prov}'")
        self.stats["used"] = {}
        self._lock = threading.Lock()   # several threads may share one model
        self._activate(0)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _activate(self, i: int) -> None:
        self.active = i
        self.active_spec = self.specs[i]
        p, _, m = self.active_spec.partition(":")
        self.provider, self.model = p.strip().lower(), m.strip()

    @property
    def is_heuristic(self) -> bool:
        return self.specs[0].lower() == "heuristic"

    def _cache_path(self, spec: str, system: str, user: str, temperature: float, tag: str) -> Path:
        key = hashlib.sha256(json.dumps([spec, system, user, temperature]).encode()).hexdigest()[:24]
        return self.cache_dir / f"{tag or 'call'}_{key}.json"

    # ------------------------------------------------------------------ public
    def json(self, system: str, user: str, temperature: float = 0.2, tag: str = "") -> dict:
        """Send a prompt, get a parsed JSON object back. Cached; falls back to the next model if needed."""
        if self.is_heuristic:
            raise LLMError("heuristic provider has no model; callers handle it directly")
        for s in self.specs:                      # resume from any model's cached answer
            cp = self._cache_path(s, system, user, temperature, tag)
            if cp.exists():
                self.stats["cache_hits"] += 1
                return json.loads(cp.read_text(encoding="utf-8"))["parsed"]

        errors = []
        while True:
            failed = self.active_spec
            try:
                return self._json_one(system, user, temperature, tag)
            except (LLMError, FatalLLMError) as e:
                errors.append(f"{failed}: {e}")
                with self._lock:
                    if self.active_spec != failed:          # another thread already switched
                        continue
                    if self.active + 1 >= len(self.specs):
                        raise LLMError(" | ".join(errors)[:1500])
                    self._activate(self.active + 1)
                    print(f"  ! switching to fallback model {self.active_spec}")

    def _json_one(self, system: str, user: str, temperature: float, tag: str) -> dict:
        cpath = self._cache_path(self.active_spec, system, user, temperature, tag)
        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                raw, usage = self._call(system, user, temperature)
                parsed = extract_json(raw)
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["used"][self.active_spec] = self.stats["used"].get(self.active_spec, 0) + 1
                    self.stats["seconds"] += time.time() - t0
                    self.stats["in_tokens"] += usage.get("in", 0)
                    self.stats["out_tokens"] += usage.get("out", 0)
                cpath.write_text(json.dumps({"spec": self.active_spec, "tag": tag, "raw": raw, "parsed": parsed,
                                             "usage": usage}, ensure_ascii=False, indent=1), encoding="utf-8")
                return parsed
            except (requests.RequestException, json.JSONDecodeError, LLMError, KeyError, IndexError) as e:
                last_err = e
                self.stats["errors"] += 1
                msg = str(e)
                if re.search(r"per day|\bTPD\b|\bRPD\b|daily", msg, re.I):
                    raise FatalLLMError(f"daily limit reached: {msg[:300]}")
                wait = min(60, 4 * (2 ** attempt))
                m = re.search(r"retry in ([\d.]+)s|try again in (?:(\d+)m)?([\d.]+)s", msg, re.I)
                if m:
                    secs = float(m.group(1)) if m.group(1) else float(m.group(3)) + 60 * float(m.group(2) or 0)
                    wait = min(90, secs + 1)
                time.sleep(wait)
        raise LLMError(f"failed after {self.max_retries} tries: {last_err}")

    # ----------------------------------------------------------------- private
    def _call(self, system: str, user: str, temperature: float) -> tuple[str, dict]:
        if self.provider == "anthropic":
            return self._anthropic(system, user, temperature)
        base, env = OPENAI_COMPATIBLE[self.provider]
        key = self.api_key or (os.environ.get(env, "") if env else "")
        if env and not key:
            raise LLMError(f"No API key: set {env} in .env")
        body = {"model": self.model, "temperature": temperature, "max_tokens": 6000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        if "gpt-oss" in self.model:
            # reasoning models: keep thinking short so we fit free-tier token limits
            body["reasoning_effort"] = os.environ.get("PROMO_REASONING_EFFORT", "low")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=180)
        if r.status_code == 400 and "response_format" in r.text:
            body.pop("response_format")
            r = requests.post(f"{base}/chat/completions", json=body, headers=headers, timeout=180)
        if r.status_code in (429, 500, 502, 503, 504):
            raise requests.RequestException(f"HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code in (401, 403, 404, 413):
            raise FatalLLMError(f"HTTP {r.status_code}: {r.text[:400]}")   # retrying will not help
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        u = data.get("usage") or {}
        return data["choices"][0]["message"]["content"], {"in": u.get("prompt_tokens", 0),
                                                          "out": u.get("completion_tokens", 0)}

    def _anthropic(self, system: str, user: str, temperature: float) -> tuple[str, dict]:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("No API key: set ANTHROPIC_API_KEY in .env")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=300, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 8000, "temperature": temperature, "system": system,
                  "messages": [{"role": "user", "content": user + "\n\nReply with JSON only."}]})
        if r.status_code in (429, 500, 529, 503):
            raise requests.RequestException(f"HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        text = "".join(c.get("text", "") for c in data["content"] if c.get("type") == "text")
        u = data.get("usage") or {}
        return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
