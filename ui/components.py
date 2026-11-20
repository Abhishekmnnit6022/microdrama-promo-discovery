"""Reusable render helpers for the promo discovery UI. All inputs are the JSON dicts written by src.batch."""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from ui.theme import ACCENT, INK, MUTED, RULE

COMPONENT_LABELS = {
    "hook": "Opening hook", "cliffhanger": "Ending cliffhanger", "coherence": "Standalone coherence",
    "emotion": "Emotional intensity", "continuity": "Narrative continuity", "dialogue_quality": "Dialogue quality",
}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def ep_label(episode_id: str, number=None) -> str:
    return f"Ep {number}" if number is not None else episode_id


def kpis(items: list):
    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>',
                     unsafe_allow_html=True)


def score_breakdown(p: dict, weights: dict) -> str:
    rows = []
    for k, v in p["score_breakdown"].items():
        w = weights.get(k)
        lab = COMPONENT_LABELS.get(k, k) + (f" · {w:.0%}" if w is not None else "")
        rows.append(f'<div class="brk"><span>{esc(lab)}</span><div class="bar"><span style="width:{v*100:.0f}%"></span></div><span>{v:.2f}</span></div>')
    return "".join(rows)


def promo_card(p: dict, weights: dict, decision: str = ""):
    segs = p["segments"]
    eps = ", ".join(ep_label(s["episode_id"], s["episode_number"]) for s in segs)
    eps = ", ".join(dict.fromkeys(eps.split(", ")))
    h, c, coh = p["opening_hook"], p["ending_cliffhanger"], p["standalone_coherence"]
    h_ep = next((s["episode_number"] for s in segs if s["episode_id"] == h["episode_id"]), "?")
    c_ep = next((s["episode_number"] for s in segs if s["episode_id"] == c["episode_id"]), "?")
    struct = p["structure"].replace("_", " ")
    dec_pill = ""
    if decision:
        cls = {"approved": "ok", "partially_correct": "warn", "rejected": "bad"}.get(decision, "")
        dec_pill = f'<span class="pill {cls}">{esc(decision.replace("_", " "))}</span>'
    flags = "".join(f'<div class="flag">▲ {esc(f)}</div>' for f in p.get("risk_flags", []))
    llm = p.get("llm_review") or {}
    llm_html = ""
    if llm.get("status") == "ok":
        def yn(v):
            return "yes" if v is True else "no" if v is False else "?"
        llm_html = (f'<div class="meta" style="margin-top:8px"><b>Gemini second opinion</b> ({esc(llm.get("model"))}): '
                    f'hook {yn(llm.get("hook_works"))} · cliffhanger {yn(llm.get("cliffhanger_works"))} · standalone {yn(llm.get("standalone"))} — '
                    f'{esc(llm.get("one_line"))}' + (f' <i>Concern: {esc(llm.get("concerns"))}</i>' if llm.get("concerns") else "") + "</div>")
    st.markdown(f"""
<div class="card">
  <div class="head">
    <span class="rank">#{esc(p.get('rank') or '–')}</span>
    <span class="score">priority {p['priority_score']:.3f}</span>
    <span class="meta">{p['total_duration_seconds']:.1f}s · {len(segs)} segment{'s' if len(segs) > 1 else ''} · {esc(struct)}</span>
    <span class="meta">{esc(p['show'])} · {esc(eps)}</span> {dec_pill}
  </div>
  <div class="row2">
    <div><div class="lbl">Opening hook · Ep {esc(h_ep)} {esc(h['timestamp_fmt'])}</div>
      <div class="quote">“{esc(h['text'])}”</div><div class="meta">{esc(h['evidence'][0])} · strength {h['score']:.2f}</div></div>
    <div><div class="lbl">Ending cliffhanger · Ep {esc(c_ep)} cut at {esc(c['timestamp_fmt'])}</div>
      <div class="quote">“{esc(c['text'])}”</div><div class="meta">{esc(c['evidence'][0])} · strength {c['score']:.2f}</div></div>
  </div>
  <div class="row2">
    <div><div class="lbl">Standalone coherence · {coh['score']:.2f}</div><div class="meta">{esc(coh['evidence'][0])}; {esc('; '.join(coh['evidence'][1:4]))}</div></div>
    <div><div class="lbl">Score breakdown</div>{score_breakdown(p, weights)}</div>
  </div>
  <div class="why"><b>Why selected:</b> {esc(p['why_it_works'])}</div>
  {flags}{llm_html}
</div>""", unsafe_allow_html=True)


def segment_table(p: dict) -> pd.DataFrame:
    rows = []
    n = len(p["segments"])
    for s in p["segments"]:
        role = []
        if s["order"] == 1:
            role.append("HOOK")
        if s["order"] == n:
            role.append("CLIFFHANGER")
        rows.append({
            "Segment": s["order"], "Role": " + ".join(role) or "body", "Episode": s["episode_number"],
            "Episode ID": s["episode_id"], "Start": s["start"], "End": s["end"],
            "Duration (s)": s["duration_seconds"],
            "Dialogue": " / ".join(f"{d['character']}: {d['text']}" for d in s["dialogue"]),
        })
    return pd.DataFrame(rows)


def promo_timeline(p: dict):
    """Horizontal bars: where each segment sits inside its episode; hook window and cut marked."""
    rows = []
    for s in p["segments"]:
        rows.append({"episode": f"Ep {s['episode_number']}", "start": s["start_seconds"], "end": s["end_seconds"],
                     "segment": f"Segment {s['order']}"})
    df = pd.DataFrame(rows)
    h, c = p["opening_hook"], p["ending_cliffhanger"]
    first, last = p["segments"][0], p["segments"][-1]
    marks = pd.DataFrame([
        {"episode": f"Ep {first['episode_number']}", "t": first["start_seconds"], "t2": first["start_seconds"] + 3.0, "what": "hook window (first 3 s)"},
        {"episode": f"Ep {last['episode_number']}", "t": c["timestamp"] - 0.4, "t2": c["timestamp"], "what": "cliffhanger cut"},
    ])
    base = alt.Chart(df).mark_bar(height=18, color=INK, opacity=0.85).encode(
        x=alt.X("start:Q", title="seconds into episode"), x2="end:Q",
        y=alt.Y("episode:N", title=None, sort=None), tooltip=["segment", "start", "end"])
    mk = alt.Chart(marks).mark_bar(height=26, color=ACCENT, opacity=0.9).encode(
        x="t:Q", x2="t2:Q", y=alt.Y("episode:N", sort=None), tooltip=["what"])
    st.altair_chart((base + mk).properties(height=40 + 34 * df["episode"].nunique()).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor=RULE, domainColor=RULE).configure_view(stroke=None),
        use_container_width=True)


def funnel_chart(summary: dict):
    steps = [
        ("Candidates generated", summary["candidates_total"]),
        ("20–60 s duration", summary["passed_duration"]),
        ("Valid opening hook", summary["valid_hook"]),
        ("Valid cliffhanger", summary["valid_cliffhanger"]),
        ("Coherent (not rejected)", summary["candidates_total"] - summary["rejected_poor_coherence"]),
        ("Passed all gates", summary["passed_all_gates"]),
        ("Recommended (diverse)", summary["recommended"]),
    ]
    df = pd.DataFrame(steps, columns=["step", "count"])
    ch = alt.Chart(df).mark_bar(color=INK, height=16).encode(
        x=alt.X("count:Q", title=None, scale=alt.Scale(type="symlog")),
        y=alt.Y("step:N", sort=None, title=None), tooltip=["step", "count"])
    txt = ch.mark_text(align="left", dx=4, color=MUTED, fontSize=11).encode(text="count:Q")
    st.altair_chart((ch + txt).properties(height=230).configure_axis(labelColor=INK, gridColor=RULE, domainColor=RULE,
                                                                     labelLimit=220).configure_view(stroke=None),
                    use_container_width=True)
