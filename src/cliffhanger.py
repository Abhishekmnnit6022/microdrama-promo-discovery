"""Ending-cliffhanger detector: does the promo cut at a genuinely unresolved moment?

A ``?`` or ``...`` alone is NOT enough. The detector combines:
  1. open-loop signal at the cut (secret about to come out, threat, revelation in progress, arrival/discovery,
     interrupted line, charged question)
  2. tension of the closing lines (emotion model arousal / script tags)
  3. context AFTER the cut in the source transcript: if the withheld lines carry a real payoff (high intensity,
     reveal/answer), the cut creates anticipation; if the very next line trivially answers it or the same speaker
     simply continues the sentence, the "cliffhanger" is weak
  4. penalties for resolving language ("thank you", "it's over") and calm zero-shot readings
  5. a small bonus when the cut coincides with the episode's own final beat (the writers' cliffhanger)
"""

from __future__ import annotations

from src.context import ShowContext
from src.hook import _short
from src.models import ComponentResult, Segment, fmt_ts

CLOSING_WINDOW = 4.0
OPEN_CUES = {"secret_setup": 0.9, "threat": 0.85, "revelation": 0.75, "arrival": 0.75, "discovery": 0.7,
             "physical": 0.7, "shock": 0.65, "betrayal": 0.6}
QUESTION_WORDS = ("who", "what", "why", "how", "where", "when", "is it", "are you", "do you", "did you", "will you")


def is_split_line(u, ctx: ShowContext) -> bool:
    """True if the script continues the same sentence in the speaker's next line ("I said..." / "...let her go.")."""
    nxt = next((a for a in ctx.lines_after(u, n=3, cross_episode=False) if a.kind == "dialogue"), None)
    return (nxt is not None and nxt.character == u.character and nxt.start - u.end < 1.5
            and nxt.text.lstrip().startswith(("...", "…")))


def open_signal(u, f, ctx: ShowContext = None) -> tuple:
    sig, why = 0.0, []
    split = ctx is not None and f.interrupted and is_split_line(u, ctx)
    for cat, w in OPEN_CUES.items():
        if cat == "physical" and u.kind == "dialogue":
            w = 0.4  # spoken physical verbs are rarely an on-screen cliffhanger
        if cat in f.cues and w > sig:
            sig = w
            why = [f"{cat.replace('_', ' ')} ({', '.join(repr(h) for h in f.cues[cat][:2])})"]
    if f.interrupted and not split and sig < 0.6:
        sig, why = 0.6, ["line breaks off (—/...)"]
    if f.question:
        low = u.text.lower()
        q = 0.5 + (0.05 if any(low.lstrip("\"' ").startswith(w) or f" {w} " in low for w in QUESTION_WORDS) else 0.0)
        if f.ml_arousal() > 0.5 or f.tag_intensity >= 0.66:
            q += 0.15
        if q > sig:
            sig, why = q, ["unanswered question" + (" (charged)" if q >= 0.65 else "")]
    if f.zeroshot and not f.resolution:
        # stage 2: the NLI model catches declarative threats/reveals that carry no lexicon cue
        lab, z = max(((k, f.zeroshot.get(k, 0.0)) for k in ("threat", "revelation", "suspense")), key=lambda kv: kv[1])
        if 0.8 * z > sig:
            sig, why = 0.8 * z, [f"zero-shot reads an unresolved {lab} ({z:.2f})"]
    if (sig < 0.5 and u.kind == "dialogue" and not f.resolution and not split
            and (f.ml_arousal() >= 0.6 or f.tag_intensity >= 0.66)):
        sig, why = 0.5, ["charged statement left hanging"]
    return sig, why


def detect_cliffhanger(seg: Segment, ctx: ShowContext) -> ComponentResult:
    final = seg.lines[-1]
    ff = ctx.f(final)
    closing = [u for u in seg.lines if u.end > seg.end - CLOSING_WINDOW][-3:] or [final]

    sig, why = open_signal(final, ff, ctx)
    for u in ([] if ff.resolution else closing[:-1]):  # the beat just before the cut can carry the open loop (e.g. a threat + a stare)
        s2, w2 = open_signal(u, ctx.f(u), ctx)
        if 0.85 * s2 > sig:
            sig, why = 0.85 * s2, [w + " (just before the cut)" for w in w2]
    tension = max(ctx.f(u).ml_arousal() if u.kind == "dialogue" else ctx.f(u).cue_strength * 0.8 for u in closing)
    tension = max(tension, max(ctx.f(u).tag_intensity for u in closing))

    after = ctx.lines_after(final, n=3)
    evidence, penalties = [], []
    if after:
        payoff = max(0.5 * ctx.f(a).ml_intensity() + 0.5 * ctx.f(a).cue_strength for a in after)
        nxt = after[0]
        nf = ctx.f(nxt)
        if (ff.question and nxt.kind == "dialogue" and nxt.character != final.character and nf.words <= 3
                and nf.cue_strength == 0 and nf.ml_intensity() < 0.5):
            payoff *= 0.5
            penalties.append(f"next line answers it at once (“{_short(nxt.text, 30)}”)")
        nxt_dialogue = next((a for a in after if a.kind == "dialogue"), None)
        if (nxt_dialogue is not None and nxt_dialogue.episode_id == final.episode_id
                and nxt_dialogue.character == final.character and nxt_dialogue.start - final.end < 1.0
                and nxt_dialogue.text.lstrip().startswith(("...", "…"))):
            # the script splits one sentence across two lines: cutting here is mid-sentence, not a cliffhanger
            sig = min(sig, 0.3)
            payoff *= 0.5
            penalties.append(f"cuts mid-sentence; sentence continues “{_short(nxt_dialogue.text, 30)}”")
        elif (nxt.kind == "dialogue" and nxt.character == final.character and nxt.episode_id == final.episode_id
                and nxt.start - final.end < 1.0 and (ff.interrupted or ff.question)):
            payoff *= 0.7
            sig *= 0.8
            penalties.append("same speaker continues immediately (split line / rhetorical)")
        evidence.append(f"withheld next beat: {nxt.character} “{_short(nxt.text, 50)}” (payoff {payoff:.2f})")
    else:
        payoff = 0.5
        evidence.append("cut is the last line of the show; nothing after it")

    episode_end = ctx.is_episode_final(final, tail=2)
    bonus = 0.12 if episode_end and sig >= 0.5 else 0.0

    zs = None
    if ff.zeroshot:
        zs = max(ff.zeroshot.get(k, 0.0) for k in ("suspense", "threat", "revelation"))
        if ff.zeroshot.get("calm", 0.0) > 0.7:
            penalties.append(f"zero-shot reads closing line as calm ({ff.zeroshot['calm']:.2f})")
    if zs is None:
        score = 0.45 * sig + 0.30 * tension + 0.25 * payoff
    else:
        score = 0.40 * sig + 0.25 * tension + 0.20 * payoff + 0.15 * zs
    if ff.resolution:
        score -= 0.3
        penalties.append(f"resolving language ({', '.join(repr(r) for r in ff.resolution[:2])})")
    if any(p.startswith("zero-shot reads") for p in penalties):
        score -= 0.1
    score = max(0.0, min(1.0, score + bonus))

    label = why[0] if why else "no unresolved signal at the cut"
    ev = [label, f"{fmt_ts(final.start)}–{fmt_ts(final.end)} {final.character}: “{_short(final.text)}”",
          f"closing tension {tension:.2f}"]
    if zs is not None:
        ev.append(f"zero-shot suspense/threat/reveal {zs:.2f}")
    if bonus:
        ev.append("cut matches the episode's own final beat (+0.12)")
    ev += evidence + [f"penalty: {p}" for p in penalties]
    res = ComponentResult(score=round(score, 4), evidence=ev, timestamp=final.end,
                          episode_id=final.episode_id, text=final.text)
    res.open_signal = sig  # type: ignore[attr-defined]  (used by the quality gate)
    return res
