"""Standalone-coherence analysis: would a viewer who never saw the show follow and care?

Four sub-scores (each 0–1, equal weight) minus context-dependency penalties:
  who       — at least two speaking characters, and at least one character named on screen
  what      — enough spoken content to establish a situation (lines, words)
  why       — stakes are explicit (relationship/status/danger words, conflict cues) and the promo overlaps
              the episode logline (i.e. it shows the episode's central situation, not a side beat)
  structure — few scene switches; for multi-segment promos the segments share characters and vocabulary;
              no repeated lines
Penalties: opens on a dangling pronoun/conjunction, opens with an answer to a question the viewer never heard,
references to earlier events ("again", "last night", "after what happened"), cross-episode jumps.
Contradiction detection is NOT implemented (documented limitation).
"""

from __future__ import annotations

import re

from src.context import ShowContext, content_words
from src.models import ComponentResult


def _norm(t: str) -> str:
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def analyse_coherence(segments: list, ctx: ShowContext, structure: str) -> ComponentResult:
    lines = [u for s in segments for u in s.lines]
    dialogue = [u for u in lines if u.kind == "dialogue"]
    text = " ".join(u.text for u in lines)
    ev, pen_notes = [], []

    speakers = {u.character for u in dialogue}
    named = set()
    for u in lines:
        named |= ctx.mentions_names(u.text)
    who = min(1.0, 0.4 + (0.3 if len(speakers) >= 2 else 0.0) + (0.3 if named else 0.0))
    ev.append(f"who: {len(speakers)} speaking character(s)" + (f", names on screen: {', '.join(sorted(named)[:4])}" if named else ", no names said aloud"))

    words = sum(len(u.text.split()) for u in dialogue)
    what = min(1.0, len(dialogue) / 4.0) * 0.6 + min(1.0, words / 30.0) * 0.4
    ev.append(f"what: {len(dialogue)} dialogue lines, {words} words")

    stakes_hits = set()
    conflict = False
    for u in lines:
        f = ctx.f(u)
        stakes_hits |= set(f.cues.get("stakes", []))
        conflict = conflict or any(c in f.cues for c in ("betrayal", "threat", "revelation", "physical", "secret_setup"))
    log_overlap = 0.0
    eps = {s.episode_id for s in segments}
    lw = set().union(*(ctx.logline_words.get(e, set()) for e in eps))
    if lw:
        log_overlap = len(content_words(text) & lw) / max(1, min(len(lw), 12))
    why = min(1.0, 0.35 * bool(stakes_hits) + 0.35 * conflict + 0.30 * min(1.0, log_overlap))
    ev.append("why: " + (f"stakes ({', '.join(sorted(stakes_hits)[:4])})" if stakes_hits else "no explicit stakes words")
              + ("; conflict cue present" if conflict else "") + f"; logline overlap {min(1.0, log_overlap):.2f}")

    scenes = {(u.episode_id, u.scene) for u in lines}
    scene_switches = max(0, len(scenes) - len(segments))
    struct = 1.0 - min(0.5, 0.15 * scene_switches)
    if len(segments) > 1:
        chars = [{u.character for u in s.lines if u.kind == "dialogue"} | set().union(*(ctx.mentions_names(u.text) for u in s.lines)) for s in segments]
        chars_l = [{c.lower() for c in cs} | {t.lower() for c in cs for t in c.split()} for cs in chars]
        shared = set.intersection(*chars_l) if chars_l else set()
        vocab = [content_words(s.text) for s in segments]
        jacc = len(vocab[0] & vocab[-1]) / max(1, len(vocab[0] | vocab[-1]))
        if not shared:
            struct -= 0.5
            pen_notes.append("segments share no character")
        struct += min(0.15, jacc)
        ev.append(f"structure: {len(segments)} segments, shared characters {sorted(shared)[:3] or 'none'}, vocab overlap {jacc:.2f}")
    else:
        ev.append(f"structure: continuous, {scene_switches} scene switch(es)")
    norm = [_norm(u.text) for u in dialogue if len(u.text.split()) > 2]
    repeats = len(norm) - len(set(norm))
    if repeats:
        struct -= 0.15 * repeats
        pen_notes.append(f"{repeats} repeated line(s)")
    struct = max(0.0, min(1.0, struct))

    penalty = 0.0
    first = dialogue[0] if dialogue else None
    if first is not None:
        ff = ctx.f(first)
        if ff.dangling_start and first.start - segments[0].start < 3.0:
            penalty += 0.12
            pen_notes.append(f"opens on a dangling reference (“{first.text[:40]}”)")
        prev = ctx.line_before(first)
        if (prev is not None and prev.kind == "dialogue" and ctx.f(prev).question and prev.character != first.character
                and prev.start < segments[0].start):
            penalty += 0.08
            pen_notes.append("opens with an answer to a question the viewer does not hear")
    refs = sorted({r for u in dialogue for r in ctx.f(u).context_refs})
    if refs:
        penalty += min(0.2, 0.07 * len(refs))
        pen_notes.append(f"refers to earlier events ({', '.join(repr(r) for r in refs[:3])})")
    if structure == "cross_episode":
        penalty += 0.08
        pen_notes.append("jumps between episodes")

    score = max(0.0, min(1.0, 0.25 * (who + what + why + struct) - penalty))
    ev += [f"penalty: {p}" for p in pen_notes]
    summary = ("clear who/what/why" if score >= 0.7 else "partly self-explanatory" if score >= 0.55 else "needs prior context")
    return ComponentResult(score=round(score, 4), evidence=[summary] + ev)
