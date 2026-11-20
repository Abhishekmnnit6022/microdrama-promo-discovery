"""Pretrained Hugging Face models, loaded once per process, with a persistent on-disk cache.

* Emotion: ``j-hartmann/emotion-english-distilroberta-base`` (7 Ekman labels + neutral).
  Run once per unique line of the whole show (batched), never per window.
* Zero-shot NLI: ``MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33``. Expensive, so it is only
  applied in stage 2 to the opening/closing lines of shortlisted promo candidates.

If a model cannot be loaded (offline, not installed) the functions return ``None`` per text and the
pipeline continues with the transparent lexical features only; the run manifest records this.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ZEROSHOT_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
ZS_TEMPLATE = "This line from a TV drama contains {}."
ZS_LABELS = {
    "threat": "a threat or danger",
    "revelation": "a shocking revelation or secret",
    "betrayal": "a betrayal or accusation",
    "violence": "physical violence or a fight",
    "suspense": "an unresolved, suspenseful moment",
    "calm": "an ordinary calm conversation",
}
HIGH_AROUSAL = ("anger", "fear", "surprise", "disgust")

_lock = threading.Lock()
_models: dict = {}
_status: dict = {"emotion": "not loaded", "zeroshot": "not loaded"}


def _key(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:20]


class DiskCache:
    """Tiny JSON cache: {text_hash: value}. Saved explicitly by the caller."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.data: dict = {}
        self.dirty = False
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Cache %s unreadable; starting fresh.", self.path)

    def get(self, text: str):
        return self.data.get(_key(text))

    def put(self, text: str, value) -> None:
        self.data[_key(text)] = value
        self.dirty = True

    def save(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data), encoding="utf-8")
            self.dirty = False


def model_status() -> dict:
    return dict(_status)


def _load(kind: str):
    with _lock:
        if kind in _models:
            return _models[kind]
        try:
            from transformers import pipeline
            if kind == "emotion":
                pipe = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, truncation=True, device=-1)
            else:
                pipe = pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=-1)
            _models[kind] = pipe
            _status[kind] = "loaded: " + (EMOTION_MODEL if kind == "emotion" else ZEROSHOT_MODEL)
        except Exception as exc:  # offline / missing dependency
            logger.warning("Could not load %s model: %s", kind, exc)
            _models[kind] = None
            _status[kind] = f"unavailable ({type(exc).__name__}: {str(exc)[:120]})"
        return _models[kind]


def classify_emotions(texts: list, cache: Optional[DiskCache] = None, batch_size: int = 32,
                      enabled: bool = True) -> list:
    """Return per text a dict label->prob, or None if the model is unavailable/disabled."""
    results: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        hit = cache.get(t) if cache else None
        if hit is not None:
            results[i] = hit
        elif t.strip():
            todo.append(i)
    if not todo or not enabled:
        if enabled and texts and _status["emotion"] == "not loaded":
            _status["emotion"] = f"not loaded: outputs served from cache ({EMOTION_MODEL})"
        return results
    pipe = _load("emotion")
    if pipe is None:
        return results
    for b in range(0, len(todo), batch_size):
        chunk = todo[b:b + batch_size]
        try:
            out = pipe([texts[i][:1000] for i in chunk], batch_size=batch_size)
        except Exception as exc:
            logger.warning("Emotion batch failed: %s", exc)
            continue
        for i, item in zip(chunk, out):
            scores = {d["label"]: round(float(d["score"]), 5) for d in item}
            results[i] = scores
            if cache:
                cache.put(texts[i], scores)
    return results


def classify_zeroshot(texts: list, cache: Optional[DiskCache] = None, batch_size: int = 16,
                      enabled: bool = True) -> list:
    """Return per text a dict short_label->prob (multi-label), or None if unavailable/disabled."""
    results: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        hit = cache.get(t) if cache else None
        if hit is not None:
            results[i] = hit
        elif t.strip():
            todo.append(i)
    if not todo or not enabled:
        if enabled and texts and _status["zeroshot"] == "not loaded":
            _status["zeroshot"] = f"not loaded: outputs served from cache ({ZEROSHOT_MODEL})"
        return results
    pipe = _load("zeroshot")
    if pipe is None:
        return results
    inv = {v: k for k, v in ZS_LABELS.items()}
    uniq = list(dict.fromkeys(texts[i] for i in todo))
    by_text = {}
    for b in range(0, len(uniq), 64):  # chunked so the cache can be flushed and an interrupted run resumes
        chunk = uniq[b:b + 64]
        try:
            out = pipe([t[:600] for t in chunk], candidate_labels=list(ZS_LABELS.values()), multi_label=True,
                       hypothesis_template=ZS_TEMPLATE, batch_size=batch_size)
            if isinstance(out, dict):
                out = [out]
        except Exception as exc:
            logger.warning("Zero-shot batch failed: %s", exc)
            continue
        for t, o in zip(chunk, out):
            by_text[t] = {inv[l]: round(float(s), 5) for l, s in zip(o["labels"], o["scores"])}
            if cache:
                cache.put(t, by_text[t])
        if cache:
            cache.save()
    for i in todo:
        results[i] = by_text.get(texts[i])
    return results


def arousal(emotions: Optional[dict]) -> Optional[float]:
    """Probability mass on high-arousal emotions (anger, fear, surprise, disgust)."""
    if not emotions:
        return None
    return min(1.0, sum(emotions.get(k, 0.0) for k in HIGH_AROUSAL))


def intensity(emotions: Optional[dict]) -> Optional[float]:
    """1 - P(neutral)."""
    if not emotions:
        return None
    return max(0.0, min(1.0, 1.0 - emotions.get("neutral", 0.0)))


def dominant(emotions: Optional[dict]) -> str:
    if not emotions:
        return "unknown"
    return max(emotions, key=emotions.get)
