"""Transcript ingestion: TXT screenplay scripts, SRT subtitles and JSON transcripts.

Every parser returns an :class:`Episode` whose utterances keep the exact source timestamps.
Lines whose timestamps are missing or malformed are kept with ``start=end=None`` and a warning
is recorded on the episode — timestamps are never invented.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from src.models import Episode, Utterance

SUPPORTED_EXTENSIONS = (".txt", ".srt", ".json")
PARSER_VERSION = "3"  # bump when parsing changes so cached parses are invalidated

# ── timestamp helpers ─────────────────────────────────────────────────────
_TS_ANY = re.compile(
    r"^\s*(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<f>\d{1,3}))?\s*$"
)


def parse_timestamp(value) -> Optional[float]:
    """Parse 'MM:SS.ff', 'HH:MM:SS,mmm', 'HH:MM:SS.mmm' or a number of seconds. None if invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    m = _TS_ANY.match(str(value))
    if not m:
        try:
            v = float(str(value).strip())
            return v if v >= 0 else None
        except ValueError:
            return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m"))
    s = int(m.group("s"))
    if s >= 60 or (m.group("h") is not None and mi >= 60):
        return None
    frac = m.group("f") or "0"
    return h * 3600 + mi * 60 + s + int(frac) / (10 ** len(frac))


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip wrapping quotes; keep meaningful punctuation (?, !, ..., —)."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"<[^>]+>", "", text)  # SRT formatting tags like <i>
    if len(text) >= 2 and text[0] in "\"“'" and text[-1] in "\"”'":
        text = text[1:-1].strip()
    return text.strip("\"“”").strip()


def _kind_for(speaker: str) -> str:
    up = speaker.upper()
    if "CAMERA" in up:
        return "camera"
    if up.startswith("ACTION"):
        return "action"
    if "SOUND" in up:
        return "sound"
    return "dialogue"


def _episode_number(stem: str, header_value: str = "") -> Optional[int]:
    for src in (header_value, stem):
        m = re.search(r"\b(?:episode|ep)[\s\-_.]*0*(\d+)", src, re.I)
        if m:
            return int(m.group(1))
    m = re.match(r"^0*(\d+)", stem)
    return int(m.group(1)) if m else None


def file_hash(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


# ── TXT screenplay format ─────────────────────────────────────────────────
_HEADER = re.compile(r"^\*\*([^*]+?)\s*:\s*\*\*\s*$")
_DELIVERY = {"O.S.", "OS", "V.O.", "VO", "CONT'D", "CONTD", "OFF-SCREEN", "OFF SCREEN", "O.C."}


def split_header(inner: str) -> tuple:
    """'THE BEARDED STRANGER (O.S.) (Affirming, Sincere)' -> ('THE BEARDED STRANGER', 'Affirming, Sincere')."""
    groups = re.findall(r"\(([^)]*)\)", inner)
    speaker = re.sub(r"\s*\([^)]*\)", "", inner).strip()
    tags = [g.strip() for g in groups if g.strip().upper() not in _DELIVERY]
    return speaker, ", ".join(tags)


_TXT_TS = re.compile(
    r"Start:\s*\(\s*([^)]*?)\s*\)\s*\|\s*End:\s*\(\s*([^)]*?)\s*\)", re.I
)
_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)


def parse_txt(content: str, episode_id: str, show: str = "", source_path: str = "") -> Episode:
    lines = content.replace("\r\n", "\n").split("\n")
    meta = {}
    for raw in lines[:12]:
        m = re.match(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES|EPISODE TITLE)\s*:\s*(.*)$", raw.strip(), re.I)
        if m:
            meta[m.group(1).upper()] = m.group(2).strip()
    ep = Episode(
        show=show, episode_id=episode_id,
        number=_episode_number(episode_id, meta.get("EPISODE NUMBER", "")),
        source_path=source_path, source_format="txt",
        title=meta.get("EPISODE TITLE", meta.get("EPISODE NUMBER", "")),
        logline=meta.get("EPISODE LOGLINE", ""), continuity=meta.get("CONTINUITY NOTES", ""),
    )
    if ep.number is None:
        ep.warnings.append("Episode number not found in header or filename; ordered by filename.")

    scene = 0
    i = 0
    idx = 0
    while i < len(lines):
        line = lines[i].strip()
        sm = _SCENE.match(line)
        if sm:
            scene = int(sm.group(1))
            i += 1
            continue
        hm = _HEADER.match(line)
        if not hm:
            i += 1
            continue
        speaker, tag = split_header(hm.group(1))
        j = i + 1
        start = end = None
        start_raw = end_raw = ""
        if j < len(lines):
            tm = _TXT_TS.search(lines[j])
            if tm:
                start_raw, end_raw = tm.group(1), tm.group(2)
                start, end = parse_timestamp(start_raw), parse_timestamp(end_raw)
                j += 1
                if start is None or end is None:
                    ep.warnings.append(f"Line {i + 1} ({speaker}): malformed timestamp '{start_raw}' | '{end_raw}'.")
                    start = end = None
                elif end < start:
                    ep.warnings.append(f"Line {i + 1} ({speaker}): end {end_raw} before start {start_raw}; timestamp marked unavailable.")
                    start = end = None
            elif "start" in lines[j].lower():
                ep.warnings.append(f"Line {j + 1}: unparseable timestamp line '{lines[j].strip()[:60]}'.")
                j += 1
            else:
                ep.warnings.append(f"Line {i + 1} ({speaker}): no timestamp; marked unavailable.")
        text_lines = []
        while j < len(lines):
            tl = lines[j].strip()
            if not tl or _HEADER.match(tl) or _TXT_TS.search(tl) or _SCENE.match(tl) or tl.startswith("---"):
                break
            text_lines.append(tl)
            j += 1
        ep.utterances.append(Utterance(
            episode_id=episode_id, index=idx, start=start, end=end, character=speaker,
            text=_clean_text(" ".join(text_lines)), kind=_kind_for(speaker), emotion_tag=tag,
            scene=scene, start_raw=start_raw, end_raw=end_raw,
        ))
        idx += 1
        i = j
    if not ep.utterances:
        ep.warnings.append("No script lines recognised in TXT file.")
    return ep


# ── SRT ───────────────────────────────────────────────────────────────────
_SRT_TS = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")
_SPEAKER_PREFIX = re.compile(r"^([A-Z][A-Za-z'\. ]{0,40}):\s*(.+)$")


def parse_srt(content: str, episode_id: str, show: str = "", source_path: str = "") -> Episode:
    ep = Episode(show=show, episode_id=episode_id, number=_episode_number(episode_id),
                 source_path=source_path, source_format="srt")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    idx = 0
    for b_no, block in enumerate(blocks, 1):
        rows = [r for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        ts_row = next((k for k, r in enumerate(rows[:3]) if "-->" in r), None)
        start = end = None
        start_raw = end_raw = ""
        if ts_row is None:
            text_rows = rows[1:] if rows[0].strip().isdigit() else rows
            ep.warnings.append(f"SRT block {b_no}: no timing line; timestamp marked unavailable.")
        else:
            tm = _SRT_TS.search(rows[ts_row])
            text_rows = rows[ts_row + 1:]
            if tm:
                start_raw, end_raw = tm.group(1), tm.group(2)
                start, end = parse_timestamp(start_raw), parse_timestamp(end_raw)
                if start is None or end is None or end < start:
                    ep.warnings.append(f"SRT block {b_no}: invalid timing '{rows[ts_row].strip()}'; marked unavailable.")
                    start = end = None
            else:
                ep.warnings.append(f"SRT block {b_no}: malformed timing '{rows[ts_row].strip()}'; marked unavailable.")
        text = _clean_text(" ".join(r.strip() for r in text_rows))  # multiline subtitles are joined
        if not text:
            continue
        speaker = "UNKNOWN"
        sm = _SPEAKER_PREFIX.match(text)
        if sm:
            speaker, text = sm.group(1).strip().upper(), _clean_text(sm.group(2))
        kind = "action" if re.match(r"^\[.*\]$", text) else "dialogue"
        ep.utterances.append(Utterance(
            episode_id=episode_id, index=idx, start=start, end=end, character=speaker,
            text=text.strip("[]") if kind == "action" else text, kind=kind,
            start_raw=start_raw, end_raw=end_raw,
        ))
        idx += 1
    if not ep.utterances:
        ep.warnings.append("No subtitle blocks recognised in SRT file.")
    return ep


# ── JSON ──────────────────────────────────────────────────────────────────
def parse_json(content: str, episode_id: str, show: str = "", source_path: str = "") -> Episode:
    """Accepts either a list of lines or {"episode_id", "logline", "utterances"|"lines"|"segments": [...]}.

    Each line may use start/start_time, end/end_time, character/speaker, dialogue/text, kind/type.
    """
    data = json.loads(content)
    meta = {}
    if isinstance(data, dict):
        meta = data
        rows = data.get("utterances") or data.get("lines") or data.get("segments") or []
        episode_id = str(data.get("episode_id") or episode_id)
    else:
        rows = data
    ep = Episode(show=show or str(meta.get("show", "")), episode_id=episode_id,
                 number=_episode_number(episode_id, str(meta.get("episode_number", ""))),
                 source_path=source_path, source_format="json",
                 logline=str(meta.get("logline", "")), title=str(meta.get("title", "")))
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            ep.warnings.append(f"JSON row {idx}: not an object; skipped.")
            continue
        s_raw = r.get("start", r.get("start_time"))
        e_raw = r.get("end", r.get("end_time"))
        start, end = parse_timestamp(s_raw), parse_timestamp(e_raw)
        if start is None or end is None or end < start:
            if s_raw is not None or e_raw is not None:
                ep.warnings.append(f"JSON row {idx}: invalid timestamps {s_raw!r}–{e_raw!r}; marked unavailable.")
            else:
                ep.warnings.append(f"JSON row {idx}: no timestamps; marked unavailable.")
            start = end = None
        speaker = str(r.get("character") or r.get("speaker") or "UNKNOWN")
        kind = str(r.get("kind") or r.get("type") or _kind_for(speaker))
        ep.utterances.append(Utterance(
            episode_id=episode_id, index=idx, start=start, end=end, character=speaker,
            text=_clean_text(str(r.get("dialogue") or r.get("text") or "")), kind=kind,
            emotion_tag=str(r.get("emotion", "")), scene=int(r.get("scene", 0) or 0),
            start_raw="" if s_raw is None else str(s_raw), end_raw="" if e_raw is None else str(e_raw),
        ))
    return ep


# ── dispatch & discovery ──────────────────────────────────────────────────
def parse_episode_text(content: str, filename: str, show: str = "", source_path: str = "") -> Episode:
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    if ext == ".srt" or (ext != ".json" and "-->" in content[:2000] and "**" not in content[:2000]):
        ep = parse_srt(content, stem, show, source_path)
    elif ext == ".json":
        ep = parse_json(content, stem, show, source_path)
    else:
        ep = parse_txt(content, stem, show, source_path)
    return ep


def parse_episode_file(path: Path, show: str = "") -> Episode:
    path = Path(path)
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")
    ep = parse_episode_text(content, path.name, show or path.parent.name, str(path))
    ep.file_hash = hashlib.sha1(raw).hexdigest()[:16]
    return ep


def discover_shows(dataset_dir: Path) -> dict:
    """Map show name -> ordered list of episode files.

    Every directory that directly contains transcript files is one show (folder name = show name).
    Episode order: parsed episode number, then filename.
    """
    dataset_dir = Path(dataset_dir)
    shows: dict = {}
    if not dataset_dir.exists():
        return shows
    for d in sorted([dataset_dir] + [p for p in dataset_dir.rglob("*") if p.is_dir()]):
        if any(part.startswith((".", "__")) or part in ("raw_outputs", "outputs", "tests", "sample_data", "src", "ui", "eval")
               for part in d.relative_to(dataset_dir).parts):
            continue
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                 and not f.name.startswith(".")]
        files = [f for f in files if f.suffix.lower() != ".json" or not f.name.startswith(("output", "run_", "recommend"))]
        if files:
            files.sort(key=lambda f: (_episode_number(f.stem) is None, _episode_number(f.stem) or 0, f.name))
            shows[d.name] = files
    return shows
