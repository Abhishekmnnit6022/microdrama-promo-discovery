"""Per-line features: transparent lexical cues + script emotion tags + pretrained-model outputs.

Features are computed once per line of the show and shared by every candidate that contains the line,
so candidate scoring is cheap even with tens of thousands of candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src import lexicon as lx
from src import ml
from src.models import Utterance


@dataclass
class LineFeatures:
    cues: dict = field(default_factory=dict)
    cue_strength: float = 0.0
    tag_intensity: float = 0.0
    emotions: Optional[dict] = None
    arousal: Optional[float] = None
    intensity: Optional[float] = None
    question: bool = False
    interrupted: bool = False
    exclamation: bool = False
    resolution: list = field(default_factory=list)
    context_refs: list = field(default_factory=list)
    dangling_start: bool = False
    words: int = 0
    zeroshot: Optional[dict] = None

    def ml_arousal(self) -> float:
        """Model arousal if available, otherwise fall back to the script's emotion tag."""
        return self.arousal if self.arousal is not None else self.tag_intensity

    def ml_intensity(self) -> float:
        return self.intensity if self.intensity is not None else self.tag_intensity

    def to_dict(self) -> dict:
        return {
            "cues": self.cues, "cue_strength": self.cue_strength, "tag_intensity": round(self.tag_intensity, 3),
            "emotions": self.emotions, "arousal": self.arousal, "intensity": self.intensity,
            "question": self.question, "interrupted": self.interrupted, "resolution": self.resolution,
            "context_refs": self.context_refs, "dangling_start": self.dangling_start, "zeroshot": self.zeroshot,
        }


def line_key(u: Utterance) -> tuple:
    return (u.episode_id, u.index)


def lexical_features(u: Utterance) -> LineFeatures:
    text = u.text
    hits = lx.cue_hits(text)
    f = LineFeatures(
        cues=hits,
        cue_strength=lx.cue_strength(hits, kind=u.kind),
        tag_intensity=lx.tag_intensity(u.emotion_tag),
        question=u.kind == "dialogue" and lx.is_question(text),
        interrupted=u.kind == "dialogue" and lx.is_interrupted(text),
        exclamation="!" in text,
        resolution=lx.find_phrases(text, lx.RESOLUTION if u.kind == "dialogue" else lx.ACTION_RESOLUTION),
        context_refs=lx.find_phrases(text, lx.CONTEXT_DEPENDENT) if u.kind == "dialogue" else [],
        dangling_start=u.kind == "dialogue" and lx.starts_dangling(text),
        words=len(text.split()),
    )
    return f


def build_features(episodes: list, emotion_cache=None, use_emotion: bool = True) -> dict:
    """Return {(episode_id, index): LineFeatures} for every content line of every episode.

    Emotion inference is batched across the whole show in one pass.
    """
    feats: dict = {}
    lines = []
    for ep in episodes:
        for u in ep.utterances:
            if u.is_content:
                feats[line_key(u)] = lexical_features(u)
                if u.kind == "dialogue":
                    lines.append(u)
    emos = ml.classify_emotions([u.text for u in lines], cache=emotion_cache, enabled=use_emotion)
    for u, e in zip(lines, emos):
        f = feats[line_key(u)]
        f.emotions = e
        f.arousal = ml.arousal(e)
        f.intensity = ml.intensity(e)
    return feats


def add_zeroshot(feats: dict, utterances: list, cache=None, enabled: bool = True) -> int:
    """Stage-2 enrichment for a subset of lines. Returns number of lines that received scores."""
    todo = [u for u in utterances if feats.get(line_key(u)) is not None and feats[line_key(u)].zeroshot is None]
    if not todo:
        return 0
    res = ml.classify_zeroshot([u.text for u in todo], cache=cache, enabled=enabled)
    n = 0
    for u, r in zip(todo, res):
        if r is not None:
            feats[line_key(u)].zeroshot = r
            n += 1
    return n
