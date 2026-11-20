"""Optional Gemini second opinion on the FINAL recommendations only (never on all candidates).

The LLM sees exactly what a new viewer would hear (the promo's lines in playback order, with timestamps)
and answers four yes/no questions plus a one-line pitch. Its verdict is stored next to the deterministic
scores and shown to the editor; it does NOT change the ranking. If no key is configured or a call fails,
the review is recorded as unavailable and the pipeline continues.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.models import PromoCandidate, fmt_ts

logger = logging.getLogger(__name__)

# Tried in order; Google often answers 503 "high demand" on one model while another is fine.
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash")
TIMEOUT_MS = 30_000
RETRY_ROUNDS = 3          # full passes over MODELS
RETRY_PAUSE_S = (0, 4, 10)  # pause before each pass (transient 429/503/504 errors)

PROMPT = """You review short promotional clips for a vertical micro-drama app. The clip below is cut from the show's
transcript and will be posted on Instagram/TikTok/YouTube to attract viewers who have NEVER watched the show.

Clip lines in playback order (timestamps are positions in the source episodes):
{lines}

Answer strictly as JSON with keys:
  "hook_works": true/false  - do the first ~3 seconds grab attention?
  "cliffhanger_works": true/false - does it end on an unresolved moment that makes you want to know what happens next?
  "standalone": true/false - would a new viewer understand who is involved and why it matters?
  "one_line": string - one sentence on why this promo works (or why it does not)
  "concerns": string - the biggest weakness, or "" if none
Judge only from the lines given. Do not invent events."""


def resolve_api_key(explicit: str = "", env_file: Optional[Path] = None) -> str:
    """Explicit key (e.g. typed in the UI) wins; then GEMINI_API_KEY env var; then the .env file.
    The key is never written back into os.environ."""
    if explicit and explicit.strip():
        return explicit.strip()
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return os.environ["GEMINI_API_KEY"].strip()
    if env_file is not None and Path(env_file).exists():
        try:
            from dotenv import dotenv_values
            return (dotenv_values(env_file).get("GEMINI_API_KEY") or "").strip()
        except Exception:
            return ""
    return ""


def _client(api_key: str):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def _render(c: PromoCandidate) -> str:
    rows = []
    for k, s in enumerate(c.segments, 1):
        rows.append(f"[segment {k}: episode {s.episode_number}, {fmt_ts(s.start)}-{fmt_ts(s.end)}]")
        for u in s.lines:
            who = "(action)" if u.kind == "action" else u.character
            rows.append(f"  {fmt_ts(u.start)} {who}: {u.text}")
    return "\n".join(rows)


def review_one(client, c: PromoCandidate) -> dict:
    from google.genai import types
    last_err = ""
    for rnd in range(RETRY_ROUNDS):
        time.sleep(RETRY_PAUSE_S[min(rnd, len(RETRY_PAUSE_S) - 1)])
        for model in MODELS:
            try:
                resp = client.models.generate_content(
                    model=model, contents=PROMPT.format(lines=_render(c)),
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
                )
                data = json.loads(resp.text)
                data = {k: data.get(k) for k in ("hook_works", "cliffhanger_works", "standalone", "one_line", "concerns")}
                data["model"] = model
                data["status"] = "ok"
                return data
            except Exception as exc:  # network, quota, overload, bad JSON
                last_err = f"{model}: {type(exc).__name__}: {str(exc)[:200]}"
                logger.warning("Gemini review failed (%s)", last_err)
    return {"status": "failed", "error": last_err}


def review_promos(promos: list, api_key: str, max_workers: int = 3) -> dict:
    """Attach ``llm_review`` to each promo. Returns a summary dict."""
    if not promos:
        return {"status": "skipped", "reason": "no promos"}
    if not api_key:
        for c in promos:
            c.llm_review = {"status": "unavailable", "reason": "no GEMINI_API_KEY configured"}
        return {"status": "skipped", "reason": "no GEMINI_API_KEY configured"}
    try:
        client = _client(api_key)
    except Exception as exc:
        for c in promos:
            c.llm_review = {"status": "unavailable", "reason": f"client error: {exc}"}
        return {"status": "failed", "reason": str(exc)}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda c: review_one(client, c), promos))
    for c, r in zip(promos, results):
        c.llm_review = r
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "status": "ok" if ok else "failed",
        "reviewed": len(ok),
        "failed": len(results) - len(ok),
        "agree_hook": sum(1 for r in ok if r.get("hook_works") is True),
        "agree_cliffhanger": sum(1 for r in ok if r.get("cliffhanger_works") is True),
        "agree_standalone": sum(1 for r in ok if r.get("standalone") is True),
    }
