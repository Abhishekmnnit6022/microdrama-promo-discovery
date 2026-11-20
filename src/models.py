"""Core data structures shared by every stage of the promo discovery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

CONTENT_KINDS = ("dialogue", "action")  # lines that carry story content a viewer hears/sees


def fmt_ts(seconds: Optional[float]) -> str:
    """Format seconds as MM:SS.ff (the source script's own precision). None -> 'n/a'."""
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


@dataclass
class Utterance:
    """One timed line of a transcript (internal representation)."""
    episode_id: str
    index: int                     # position in the episode (0-based, includes camera lines)
    start: Optional[float]         # seconds; None when the source has no usable timestamp
    end: Optional[float]
    character: str                 # speaker name, or ACTION / SOUND FX / CAMERA / UNKNOWN
    text: str
    kind: str = "dialogue"         # dialogue | action | sound | camera
    emotion_tag: str = ""          # e.g. "Cold, Commanding" from the script header
    scene: int = 0
    start_raw: str = ""            # timestamp exactly as written in the source
    end_raw: str = ""

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None and self.end >= self.start

    @property
    def duration(self) -> float:
        return (self.end - self.start) if self.timed else 0.0

    @property
    def is_content(self) -> bool:
        return self.kind in CONTENT_KINDS and bool(self.text.strip())


@dataclass
class Episode:
    show: str
    episode_id: str                # exact file stem, e.g. "episode-001-Episode 1671703"
    number: Optional[int]
    source_path: str
    source_format: str             # txt | srt | json
    title: str = ""
    logline: str = ""
    continuity: str = ""
    utterances: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    file_hash: str = ""

    @property
    def content(self) -> list:
        """Timed dialogue/action lines sorted by start time — the material promos are cut from."""
        return sorted((u for u in self.utterances if u.is_content and u.timed), key=lambda u: (u.start, u.index))

    @property
    def duration(self) -> float:
        timed = [u.end for u in self.utterances if u.timed]
        return max(timed) if timed else 0.0

    @property
    def characters(self) -> set:
        return {u.character for u in self.utterances if u.kind == "dialogue" and u.character}


@dataclass
class Segment:
    """A contiguous stretch of one episode, defined by first/last content line."""
    episode_id: str
    episode_number: Optional[int]
    start: float
    end: float
    lines: list                    # list[Utterance] (content lines only, in order)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.lines)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "start_seconds": round(self.start, 2),
            "end_seconds": round(self.end, 2),
            "start": fmt_ts(self.start),
            "end": fmt_ts(self.end),
            "duration_seconds": self.duration,
            "dialogue": [
                {"start": fmt_ts(u.start), "end": fmt_ts(u.end), "character": u.character,
                 "kind": u.kind, "text": u.text}
                for u in self.lines
            ],
        }


@dataclass
class ComponentResult:
    score: float
    evidence: list = field(default_factory=list)
    timestamp: Optional[float] = None
    episode_id: str = ""
    text: str = ""


@dataclass
class PromoCandidate:
    promo_id: str
    show: str
    structure: str                 # continuous | multi_segment | cross_episode
    segments: list                 # list[Segment], in playback order
    hook: Optional[ComponentResult] = None
    cliffhanger: Optional[ComponentResult] = None
    coherence: Optional[ComponentResult] = None
    emotion: Optional[ComponentResult] = None
    continuity: Optional[ComponentResult] = None
    dialogue_quality: Optional[ComponentResult] = None
    score: float = 0.0
    passed: bool = False
    rejection_reasons: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    explanation: str = ""
    rank: Optional[int] = None
    llm_review: Optional[dict] = None

    @property
    def total_duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list:
        out = []
        for s in self.segments:
            if s.episode_id not in out:
                out.append(s.episode_id)
        return out

    def components(self) -> dict:
        return {
            "hook": self.hook.score if self.hook else 0.0,
            "cliffhanger": self.cliffhanger.score if self.cliffhanger else 0.0,
            "coherence": self.coherence.score if self.coherence else 0.0,
            "emotion": self.emotion.score if self.emotion else 0.0,
            "continuity": self.continuity.score if self.continuity else 0.0,
            "dialogue_quality": self.dialogue_quality.score if self.dialogue_quality else 0.0,
        }

    def to_dict(self) -> dict:
        def comp(c: Optional[ComponentResult]):
            if c is None:
                return None
            d = asdict(c)
            d["score"] = round(c.score, 4)
            d["timestamp_fmt"] = fmt_ts(c.timestamp) if c.timestamp is not None else None
            return d
        return {
            "rank": self.rank,
            "promo_id": self.promo_id,
            "show": self.show,
            "structure": self.structure,
            "priority_score": round(self.score, 4),
            "total_duration_seconds": self.total_duration,
            "episodes": self.episodes,
            "segments": [dict(order=i + 1, **s.to_dict()) for i, s in enumerate(self.segments)],
            "opening_hook": comp(self.hook),
            "ending_cliffhanger": comp(self.cliffhanger),
            "standalone_coherence": comp(self.coherence),
            "emotional_intensity": comp(self.emotion),
            "narrative_continuity": comp(self.continuity),
            "dialogue_quality": comp(self.dialogue_quality),
            "score_breakdown": {k: round(v, 4) for k, v in self.components().items()},
            "passed_quality_gates": self.passed,
            "rejection_reasons": self.rejection_reasons,
            "risk_flags": self.risk_flags,
            "why_it_works": self.explanation,
            "llm_review": self.llm_review,
        }
