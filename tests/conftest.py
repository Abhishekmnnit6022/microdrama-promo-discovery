"""Shared helpers: build synthetic scripts in the dataset's TXT format and a ShowContext without loading models."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import ShowContext  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingest import parse_txt  # noqa: E402

REAL_SHOW = ROOT / "data" / "I Married a Broken Soldier. He was a Duke"


def ts(sec: float) -> str:
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:05.2f}"


def script(lines, number=1, logline="A bride is betrayed by her family at her wedding.") -> str:
    """lines: (start, end, speaker, text[, tag]). speaker ACTION/CAMERA allowed."""
    out = [f"EPISODE NUMBER: Episode {number}", f"EPISODE LOGLINE: {logline}", "CONTINUITY NOTES: test", "",
           "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. HALL – DAY", ""]
    for row in lines:
        s, e, spk, text = row[:4]
        tag = row[4] if len(row) > 4 else ""
        out.append(f"**{spk}{f' ({tag})' if tag else ''}:**")
        out.append(f"Start: ({ts(s)}) | End: ({ts(e)})")
        out.append(f'"{text}"' if spk not in ("ACTION", "CAMERA") else text)
        out.append("")
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def episode(lines, number=1, show="Test Show", **kw):
    return parse_txt(script(lines, number, **kw), f"episode-{number:03d}", show)


def context(episodes):
    feats = build_features(episodes, use_emotion=False)
    return ShowContext(episodes[0].show if episodes else "Test Show", episodes, feats)


def dialogue_run(start, n, speakers=("JULIANA", "GERALD"), step=3.0, text="We need to talk about the wedding plans today."):
    return [(start + i * step, start + i * step + step - 0.3, speakers[i % len(speakers)], text) for i in range(n)]


@pytest.fixture
def betrayal_episode():
    lines = [
        (0.0, 2.5, "ACTION", "Gerald slaps Juliana across the face in front of the guests."),
        (2.6, 5.0, "JULIANA", "How dare you! You swore you loved me!", "Furious, Betrayed"),
        (5.2, 8.0, "GERALD", "I am marrying your sister, Juliana. Accept it.", "Cold"),
        (8.2, 11.0, "CLARISSA", "He was never going to marry you, sister.", "Gloating"),
        (11.2, 14.0, "JULIANA", "Father, tell them this is a mistake!", "Desperate"),
        (14.2, 17.0, "FATHER", "You have shamed this family long enough.", "Cold"),
        (17.2, 20.0, "JULIANA", "Then I have nothing left here.", "Broken"),
        (20.2, 23.0, "STRANGER", "Take your hands off her.", "Menacing"),
        (23.2, 26.0, "GERALD", "And who are you to give orders here?", "Angry"),
        (26.2, 29.5, "STRANGER", "There is something you need to know about who I really am...", "Menacing"),
        (29.7, 33.0, "STRANGER", "I am the Duke of Ashbury, and this woman is under my protection.", "Commanding"),
        (33.2, 36.0, "GERALD", "Impossible. The Duke is dead!", "Shocked"),
        (36.2, 39.0, "JULIANA", "Thank you. It's all right now, everything is fine.", "Calm"),
    ]
    return episode(lines, 1)
