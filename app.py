"""Promo Discovery — Streamlit front end.

Screens: Overview · Dataset & processing · Promo results · Segment details · Quality review · Export.
Every number shown comes from a results folder written by ``src.batch`` (``raw_outputs/`` for the full
reference run, a private per-visitor workspace for runs on uploaded files). Nothing is hard-coded.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A long-running Streamlit server keeps project modules in sys.modules across reruns. If a file under ui/ or src/
# was edited (or replaced) after it was imported, drop the stale copy so this rerun imports the current code.
import os  # noqa: E402
import time  # noqa: E402

for _name, _mod in list(sys.modules.items()):
    _file = getattr(_mod, "__file__", None)
    if _name.split(".")[0] in ("ui", "src") and _file and str(ROOT) in os.path.abspath(_file):
        try:
            if os.path.getmtime(_file) > getattr(_mod, "_loaded_at", 0.0):
                del sys.modules[_name]
        except OSError:
            del sys.modules[_name]

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import feedback as fb  # noqa: E402
from src.batch import RunConfig, load_outputs, output_path, reference_results_dir, run, write_outputs  # noqa: E402
from src.ingest import discover_shows, parse_episode_file  # noqa: E402
from src.llm_review import resolve_api_key  # noqa: E402
from src.models import fmt_ts  # noqa: E402
from ui import components as ui  # noqa: E402
from ui.theme import header, inject  # noqa: E402

for _name, _mod in list(sys.modules.items()):  # stamp load time for the stale-module check above
    if _name.split(".")[0] in ("ui", "src") and not hasattr(_mod, "_loaded_at"):
        _mod._loaded_at = time.time()

FEEDBACK_FILE = ROOT / "feedback" / "editor_feedback.jsonl"
# Each visitor gets a private workspace (uploads + results). The app never processes the bundled example
# dataset in data/; that is only used by run_pipeline.py and the tests to reproduce raw_outputs/.
WORKSPACE_ROOT = ROOT / "workspace"
WORKSPACE_TTL_S = 24 * 3600
REFERENCE_LABEL = "Reference full-dataset run (raw_outputs/)"
LATEST_LABEL = "Your latest run (from your uploads)"


def _workspace() -> Path:
    """This visitor's private folder: <workspace>/<session id>/{uploads,results}."""
    if "workspace_id" not in st.session_state:
        st.session_state["workspace_id"] = uuid.uuid4().hex[:16]
        _cleanup_old_workspaces()
    ws = WORKSPACE_ROOT / st.session_state["workspace_id"]
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _cleanup_old_workspaces() -> None:
    """Best effort: drop other visitors' workspaces untouched for a day."""
    if not WORKSPACE_ROOT.exists():
        return
    now = time.time()
    for d in WORKSPACE_ROOT.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > WORKSPACE_TTL_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def upload_dir() -> Path:
    return _workspace() / "uploads"


def result_dirs() -> dict:
    return {REFERENCE_LABEL: reference_results_dir(ROOT), LATEST_LABEL: _workspace() / "results"}


# ── state helpers ─────────────────────────────────────────────────────────
def results_dir() -> Path:
    return Path(st.session_state.get("results_dir", str(reference_results_dir(ROOT))))


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    return load_outputs(Path(path))


def results() -> dict:
    d = results_dir()
    man = output_path(d, "run_manifest.json")
    if not man.exists():
        return {}
    return _load(str(d), man.stat().st_mtime)


def promos() -> list:
    return results().get("recommendations", {}).get("promos", [])


def all_promos_by_id() -> dict:
    r = results()
    out = {}
    for key in ("recommendations", "near_misses", "episode_best"):
        for p in r.get(key, {}).get("promos", []):
            out.setdefault(p["promo_id"], p)
    for lst in r.get("alternatives_by_structure", {}).values():
        for p in lst:
            out.setdefault(p["promo_id"], p)
    return out


def decisions() -> dict:
    return fb.latest_by_promo(fb.load(FEEDBACK_FILE))


def weights() -> dict:
    return results().get("run_manifest", {}).get("config", {}).get("weights", {})


def _secret_gemini_key() -> str:
    """Gemini key from the host's secrets store (e.g. Streamlit Community Cloud); empty if none.

    Accepts GEMINI_API_KEY (preferred), GOOGLE_API_KEY, or a [gemini] section with api_key.
    """
    try:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key"):
            if st.secrets.get(name):
                return str(st.secrets[name]).strip()
        section = st.secrets.get("gemini")
        if section and section.get("api_key"):
            return str(section["api_key"]).strip()
    except Exception:  # no secrets file configured
        pass
    return ""


def no_results_note():
    st.info("No results yet for your uploads. Open **Dataset & processing** to upload a show and process it, "
            "or pick the reference full-dataset run in the sidebar.")


def promo_title(p: dict) -> str:
    eps = "+".join(str(s["episode_number"]) for s in p["segments"])
    return f"#{p.get('rank') or '–'} · ep {eps} · {p['segments'][0]['start']}→{p['segments'][-1]['end']} · {p['total_duration_seconds']:.0f}s · {p['priority_score']:.3f}"


# ── screens ───────────────────────────────────────────────────────────────
def page_overview():
    header("Overview", "Promo discovery across whole shows",
           "Finds 20–60 second promo cuts that open on a hook within the first 3 seconds, end on an unresolved "
           "moment, and make sense to someone who has never seen the show. Scores rank editorial priority; "
           "they do not predict views, CTR or virality.")
    r = results()
    if not r:
        no_results_note()
        return
    man = r["run_manifest"]
    s = man["stats"]["summary"]
    ui.kpis([(s["shows"], "shows discovered"), (s["episodes_processed_ok"], f"episodes processed (of {s['episodes_discovered']})"),
             (f"{s['candidates_total']:,}", "candidate promos generated"), (s["recommended"], "recommended promos")])
    st.write("")
    status = "completed" if s["episodes_failed"] == 0 else f"completed with {s['episodes_failed']} failed episode(s)"
    st.markdown(f'<div class="meta">Run status: <b>{status}</b> · generated {man["generated_at"]} · pipeline v{man["pipeline_version"]} · '
                f'{s["episodes_without_valid_promo"]} episode(s) without a valid single-episode promo</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Candidate funnel")
        ui.funnel_chart(s)
    with c2:
        st.markdown("### Why candidates were rejected")
        rc = man["stats"].get("rejection_reason_counts", {})
        if rc:
            st.dataframe(pd.DataFrame(list(rc.items()), columns=["reason", "candidates"]), hide_index=True, use_container_width=True)
        st.markdown("### Models")
        for k, v in man.get("models", {}).items():
            st.markdown(f'<div class="meta"><b>{k}</b>: {ui.esc(v)}</div>', unsafe_allow_html=True)
        llm = man.get("llm_review", {})
        st.markdown(f'<div class="meta"><b>Gemini review</b>: {ui.esc(llm.get("status"))}'
                    + (f' — {llm.get("reviewed")} reviewed' if llm.get("reviewed") is not None else "")
                    + (f', {llm.get("failed")} failed' if llm.get("failed") else "")
                    + (f' ({ui.esc(llm.get("reason"))})' if llm.get("reason") else "") + "</div>", unsafe_allow_html=True)
    st.markdown("### Top recommendations")
    top = promos()[:5]
    dec = decisions()
    for p in top:
        ui.promo_card(p, weights(), dec.get(p["promo_id"], {}).get("decision", ""))


def page_dataset():
    header("Dataset & processing", "Upload a show and find its promos",
           "Upload the episode transcripts of one show (.txt scripts, .srt subtitles or .json). Only your files are "
           "processed, and they are private to this browser session. The saved full-dataset run stays available "
           "under “Reference full-dataset run” in the sidebar.")
    up = st.file_uploader("Episode transcripts", type=["txt", "srt", "json"], accept_multiple_files=True)
    show_name = st.text_input("Show name", value="My show")
    if up and st.button("Use these files", type="primary"):
        udir = upload_dir()
        shutil.rmtree(udir, ignore_errors=True)  # a new upload replaces the previous one
        target = udir / (show_name.strip().replace("/", "_").replace("\\", "_") or "My show")
        target.mkdir(parents=True, exist_ok=True)
        for f in up:
            (target / Path(f.name).name).write_bytes(f.getvalue())
        st.success(f"Loaded {len(up)} file(s) as “{target.name}”. Previous uploads in this session were replaced.")

    ds = upload_dir()
    shows = discover_shows(ds)
    if not shows:
        st.info("No episodes uploaded yet. Choose transcript files above and click “Use these files”.")
        _episode_status_table()
        return
    rows, warns = [], []
    for show, files in shows.items():
        for f in files:
            try:
                ep = parse_episode_file(f, show)
                rows.append({"Show": show, "Episode": ep.number, "Episode ID": ep.episode_id, "Format": ep.source_format,
                             "Duration": fmt_ts(ep.duration), "Dialogue lines": sum(1 for u in ep.utterances if u.kind == "dialogue"),
                             "Untimed lines": sum(1 for u in ep.utterances if not u.timed), "Logline": ep.logline[:140]})
                warns += [{"Episode ID": ep.episode_id, "Warning": w} for w in ep.warnings]
            except Exception as exc:
                warns.append({"Episode ID": f.name, "Warning": f"PARSE ERROR: {exc}"})
    ui.kpis([(len(shows), "shows found"), (len(rows), "episodes parsed"), (len(warns), "parse warnings / errors")])
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=320)
    if warns:
        with st.expander(f"Parse warnings and errors ({len(warns)})", expanded=False):
            st.dataframe(pd.DataFrame(warns), hide_index=True, use_container_width=True)

    st.markdown(f"### Process “{', '.join(shows)}”")
    c1, c2, c3 = st.columns(3)
    top_k = c1.number_input("Promos to recommend", 3, 50, 15)
    zs = c2.toggle("Zero-shot NLI re-scoring (stage 2)", value=True, help="Slower on first run; cached afterwards.")
    llm = c3.toggle("Gemini second opinion on final list", value=False)
    key = st.text_input("Gemini API key (optional; otherwise the app secret / environment / .env is used)", type="password") if llm else ""
    gemini_key = resolve_api_key(key or _secret_gemini_key(), ROOT / ".env") if llm else ""
    if llm:
        if gemini_key:
            src = "the key you typed" if key.strip() else "the app's configured key"
            st.caption(f"Gemini key found: using {src}.")
        else:
            st.warning("No Gemini key found. Type one above, or add `GEMINI_API_KEY = \"...\"` to the app's secrets "
                       "(Streamlit Cloud: app Settings → Secrets). Without a key the Gemini review is skipped.")
    if st.button("Process uploaded episodes", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        cfg = RunConfig(dataset_dir=str(ds), cache_dir=str(ROOT / ".cache"), top_k=int(top_k), use_zeroshot=zs,
                        use_llm=llm, api_key=gemini_key)
        res = run(cfg, progress=lambda p, m: bar.progress(min(1.0, p), text=m))
        out = result_dirs()[LATEST_LABEL]
        if res.shows:
            shutil.rmtree(out, ignore_errors=True)
            write_outputs(res, out)
            st.cache_data.clear()
            s = res.stats["summary"]
            st.session_state["pending_results_choice"] = LATEST_LABEL
            st.session_state["flash"] = (f"Processed {s['episodes_processed_ok']} episode(s) in {res.timings.get('total', 0):.0f}s — "
                       f"{s['candidates_total']:,} candidates, {s['recommended']} recommended.")
            st.rerun()
        else:
            st.error(res.stats.get("error", "Nothing processed"))
    _episode_status_table()


def _episode_status_table():
    r = results()
    if r:
        label = "your latest run" if results_dir() == result_dirs()[LATEST_LABEL] else "reference full-dataset run"
        st.markdown(f"### Episode status — {label}")
        eps = r["run_manifest"]["episode_status"]
        df = pd.DataFrame([{ "Episode": e["episode_number"], "Status": e["status"], "Candidates": e["candidates"],
                             "Passed gates": e["candidates_passed"], "Best score": e["best_score"],
                             "Why no promo": e["no_promo_reason"], "Error": e["error"], "Warnings": len(e["warnings"])} for e in eps])
        st.dataframe(df, hide_index=True, use_container_width=True, height=300)


def page_results():
    header("Promo discovery results", "Ranked promos", "Strongest distinct promos across all processed episodes.")
    r = results()
    if not r:
        no_results_note()
        return
    view = st.radio("List", ["Recommended", "Best per structure", "Best per episode", "Near misses (failed a gate)"], horizontal=True)
    if view == "Recommended":
        lst = promos()
    elif view == "Best per structure":
        alt = r.get("alternatives_by_structure", {})
        stc = st.selectbox("Structure", list(alt), format_func=lambda s: s.replace("_", " "))
        lst = alt.get(stc, [])
    elif view == "Best per episode":
        lst = r.get("episode_best", {}).get("promos", [])
    else:
        lst = r.get("near_misses", {}).get("promos", [])
        st.caption("These scored highly but failed a quality gate; the rejection reason is listed on each card.")
    c1, c2 = st.columns(2)
    eps = sorted({s["episode_number"] for p in lst for s in p["segments"] if s["episode_number"] is not None})
    ep_f = c1.multiselect("Episodes", eps)
    min_s = c2.slider("Minimum priority score", 0.0, 1.0, 0.0, 0.05)
    dec = decisions()
    shown = 0
    for p in lst:
        if ep_f and not any(s["episode_number"] in ep_f for s in p["segments"]):
            continue
        if p["priority_score"] < min_s:
            continue
        ui.promo_card(p, weights(), dec.get(p["promo_id"], {}).get("decision", ""))
        if p.get("rejection_reasons"):
            st.markdown(f'<div class="flag">Rejected: {ui.esc("; ".join(p["rejection_reasons"]))}</div>', unsafe_allow_html=True)
        shown += 1
    if not shown:
        st.info("No promos match the filters.")


def _pick_promo(label: str = "Promo") -> dict | None:
    pool = all_promos_by_id()
    if not pool:
        no_results_note()
        return None
    ordered = sorted(pool.values(), key=lambda p: (p.get("rank") is None, p.get("rank") or 0, -p["priority_score"]))
    ids = [p["promo_id"] for p in ordered]
    default = st.session_state.get("selected_promo")
    idx = ids.index(default) if default in ids else 0
    pid = st.selectbox(label, ids, index=idx, format_func=lambda i: promo_title(pool[i]))
    st.session_state["selected_promo"] = pid
    return pool[pid]


def page_segments():
    header("Segment-level details", "Exact cut list", "Ordered segments with source timestamps. The first segment carries the hook, the last the cliffhanger.")
    p = _pick_promo()
    if not p:
        return
    ui.kpis([(f"{p['total_duration_seconds']:.1f}s", "total duration"), (len(p["segments"]), "segments"),
             (p["opening_hook"]["timestamp_fmt"], "hook timestamp"), (p["ending_cliffhanger"]["timestamp_fmt"], "cliffhanger cut")])
    st.write("")
    st.dataframe(ui.segment_table(p), hide_index=True, use_container_width=True,
                 column_config={"Dialogue": st.column_config.TextColumn(width="large")})
    ui.promo_timeline(p)
    st.markdown("### Line by line")
    first = p["segments"][0]
    for s in p["segments"]:
        st.markdown(f"**Segment {s['order']} — Ep {s['episode_number']} · {s['start']}–{s['end']} ({s['duration_seconds']:.1f}s)**")
        for d in s["dialogue"]:
            t = d["start"]
            mark = ""
            if s is first and d["start"] == p["opening_hook"]["timestamp_fmt"]:
                mark = " ◀ hook"
            if s["order"] == len(p["segments"]) and d is s["dialogue"][-1]:
                mark += " ◀ cut (cliffhanger)"
            who = "*action*" if d["kind"] == "action" else f"**{d['character']}**"
            st.markdown(f"`{t}` {who}: {d['text']}{mark}")
    with st.expander("Detection evidence"):
        for k in ("opening_hook", "ending_cliffhanger", "standalone_coherence", "emotional_intensity", "narrative_continuity", "dialogue_quality"):
            c = p.get(k)
            if c:
                st.markdown(f"**{k.replace('_', ' ').title()}** — {c['score']:.2f}")
                for e in c["evidence"]:
                    st.markdown(f"- {e}")


def page_review():
    header("Quality review", "Editor decisions", "Approve, reject or mark partially correct. Decisions are appended to feedback/editor_feedback.jsonl and used by eval/evaluate.py.")
    p = _pick_promo("Promo to review")
    if not p:
        return
    dec = decisions()
    ui.promo_card(p, weights(), dec.get(p["promo_id"], {}).get("decision", ""))
    with st.form("review"):
        c1, c2 = st.columns(2)
        decision = c1.radio("Decision", fb.DECISIONS, format_func=lambda d: d.replace("_", " ").capitalize(), horizontal=True)
        reason = c2.selectbox("Reason (for reject / partial)", ("",) + fb.REJECTION_REASONS)
        note = st.text_input("Note (optional)")
        editor = st.text_input("Editor initials (optional)")
        if st.form_submit_button("Save decision", type="primary"):
            if decision != "approved" and not reason:
                st.warning("Pick a reason for a rejection or partial approval.")
            else:
                fb.record(FEEDBACK_FILE, p, decision, reason, note, editor)
                st.success("Saved.")
                st.rerun()

    st.markdown("### Compare alternatives")
    pool = all_promos_by_id()
    eps = set(p["episodes"])
    alts = [q for q in pool.values() if q["promo_id"] != p["promo_id"] and eps & set(q["episodes"])]
    alts.sort(key=lambda q: -q["priority_score"])
    if not alts:
        st.caption("No alternative candidates touching the same episodes in the saved lists.")
    else:
        q_id = st.selectbox("Alternative", [q["promo_id"] for q in alts[:20]], format_func=lambda i: promo_title(pool[i]))
        q = pool[q_id]
        rows = []
        for lab, x in (("Selected", p), ("Alternative", q)):
            rows.append({"": lab, "priority": x["priority_score"], "duration": x["total_duration_seconds"], "structure": x["structure"],
                         **{k: v for k, v in x["score_breakdown"].items()},
                         "passed gates": x["passed_quality_gates"], "rejection": "; ".join(x.get("rejection_reasons", []))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Selected**")
            st.dataframe(ui.segment_table(p)[["Segment", "Episode", "Start", "End", "Duration (s)"]], hide_index=True)
        with c2:
            st.markdown("**Alternative**")
            st.dataframe(ui.segment_table(q)[["Segment", "Episode", "Start", "End", "Duration (s)"]], hide_index=True)

    st.markdown("### Decisions so far")
    entries = fb.load(FEEDBACK_FILE)
    if entries:
        s = fb.summary(entries)
        ui.kpis([(s["reviewed_promos"], "promos reviewed"), (s["approved"], "approved"),
                 (s["partially_correct"], "partially correct"), (s["rejected"], "rejected")])
        st.dataframe(pd.DataFrame(entries)[["time", "promo_id", "rank", "decision", "reason", "note", "editor"]],
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No editor decisions recorded yet.")


def page_export():
    header("Export", "Download results", "Every export contains segment order, exact timestamps, hook and cliffhanger timestamps and the explanation.")
    r = results()
    if not r:
        no_results_note()
        return
    d = results_dir()
    dec = decisions()
    rec = r.get("recommendations", {})
    for p in rec.get("promos", []):
        e = dec.get(p["promo_id"])
        p["editor_decision"] = {k: e[k] for k in ("decision", "reason", "note", "time")} if e else None
    st.download_button("Recommendations — JSON", json.dumps(rec, indent=2, ensure_ascii=False), "promo_recommendations.json", "application/json")
    for name, label, mime in (("recommendations.csv", "Recommendations — CSV (one row per segment)", "text/csv"),
                              ("recommendations_report.md", "Readable report — Markdown", "text/markdown"),
                              ("candidates_all.csv", "All scored candidates — CSV", "text/csv")):
        f = output_path(d, name)
        if f.exists():
            st.download_button(label, f.read_bytes(), name, mime)
    if FEEDBACK_FILE.exists():
        st.download_button("Editor feedback — JSONL", FEEDBACK_FILE.read_bytes(), "editor_feedback.jsonl", "application/json")
    st.markdown("### Report preview")
    f = output_path(d, "recommendations_report.md")
    if f.exists():
        st.markdown(f.read_text(encoding="utf-8")[:12000])


def main():
    st.set_page_config(page_title="Promo Discovery", page_icon="🎬", layout="wide")
    inject()
    with st.sidebar:
        st.markdown('<div class="eyebrow">Microdrama</div>', unsafe_allow_html=True)
        st.markdown("### Promo Discovery")
        dirs = result_dirs()
        labels = list(dirs)
        if "pending_results_choice" in st.session_state:
            st.session_state["results_choice"] = st.session_state.pop("pending_results_choice")
        if "results_choice" not in st.session_state:  # default to the first folder that has results
            st.session_state["results_choice"] = next(
                (k for k in labels if output_path(dirs[k], "run_manifest.json").exists()), labels[0])
        choice = st.radio("Results shown", labels, key="results_choice")
        st.session_state["results_dir"] = str(dirs[choice])
        if st.session_state.get("flash"):
            st.success(st.session_state.pop("flash"))
    pages = [
        st.Page(page_overview, title="Overview", url_path="overview", default=True),
        st.Page(page_dataset, title="Dataset & processing", url_path="dataset"),
        st.Page(page_results, title="Promo results", url_path="results"),
        st.Page(page_segments, title="Segment details", url_path="segments"),
        st.Page(page_review, title="Quality review", url_path="review"),
        st.Page(page_export, title="Export", url_path="export"),
    ]
    st.navigation(pages).run()


main()
