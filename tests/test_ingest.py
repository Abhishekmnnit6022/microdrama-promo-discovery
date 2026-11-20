import json

import pytest

from src.ingest import (discover_shows, parse_episode_file, parse_episode_text, parse_json, parse_srt,
                        parse_timestamp, parse_txt, split_header)
from conftest import REAL_SHOW, script


@pytest.mark.parametrize("raw,expected", [
    ("00:13.50", 13.5), ("02:18.00", 138.0), ("1:05.7", 65.7), ("00:00:31,000", 31.0),
    ("01:02:03,450", 3723.45), ("00:01:02.5", 62.5), (12.25, 12.25), ("7.5", 7.5),
])
def test_timestamp_conversion(raw, expected):
    assert parse_timestamp(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "abc", "00:75.00", "-3", None, "12:xx.10", -1])
def test_malformed_timestamps_are_none(raw):
    assert parse_timestamp(raw) is None


def test_split_header_handles_delivery_marks():
    assert split_header("THE BEARDED STRANGER (O.S.) (Affirming, Sincere)") == ("THE BEARDED STRANGER", "Affirming, Sincere")
    assert split_header("LADY JULIANA (Confused, Breathless)") == ("LADY JULIANA", "Confused, Breathless")
    assert split_header("CAMERA") == ("CAMERA", "")


def test_txt_parse_preserves_speaker_tag_kind_and_raw_timestamps():
    ep = parse_txt(script([(11.3, 13.1, "MATRIARCH", "You should turn around, Juliana.", "Cold, Commanding"),
                           (13.5, 14.3, "CAMERA", "Close-up."),
                           (14.5, 16.0, "ACTION", "Juliana freezes.")]), "episode-001-Episode 1", "Show")
    assert ep.episode_id == "episode-001-Episode 1"
    assert ep.number == 1
    assert ep.logline.startswith("A bride")
    u = ep.utterances[0]
    assert (u.character, u.emotion_tag, u.kind) == ("MATRIARCH", "Cold, Commanding", "dialogue")
    assert (u.start, u.end, u.start_raw, u.end_raw) == (11.3, 13.1, "00:11.30", "00:13.10")
    assert u.text == "You should turn around, Juliana."      # quotes stripped, punctuation kept
    assert [x.kind for x in ep.utterances] == ["dialogue", "camera", "action"]
    assert [u.text for u in ep.content] == ["You should turn around, Juliana.", "Juliana freezes."]  # camera excluded


def test_txt_missing_and_inverted_timestamps_marked_unavailable():
    txt = ("EPISODE NUMBER: Episode 3\n\n**JULIANA:**\n\"No timestamp here.\"\n\n"
           "**GERALD:**\nStart: (00:20.00) | End: (00:18.00)\n\"Backwards.\"\n\n"
           "**GERALD:**\nStart: (00:2x.00) | End: (00:30.00)\n\"Garbled.\"\n")
    ep = parse_txt(txt, "episode-003", "S")
    assert len(ep.utterances) == 3
    assert all(not u.timed for u in ep.utterances)
    assert len(ep.warnings) == 3
    assert ep.content == []                                  # untimed lines are never used for cuts


def test_txt_uppercase_scene_keys_and_number_from_header():
    txt = script([(0, 2, "A", "Hello there my friend.")], number=20).replace("Scene heading", "SCENE HEADING")
    ep = parse_txt(txt, "weird-name", "S")
    assert ep.number == 20 and len(ep.content) == 1


def test_srt_multiline_speaker_and_malformed():
    srt = ("1\n00:00:01,000 --> 00:00:04,500\nMARCUS: I never told you\nabout my first wife.\n\n"
           "2\n00:00:05,000 --> 00:00:07,000\n<i>She is alive.</i>\n\n"
           "3\n00:00:09 --> broken\nWho are you?\n\n"
           "4\n00:00:12,000 --> 00:00:10,000\nBackwards\n")
    ep = parse_srt(srt, "ep-07", "S")
    assert ep.utterances[0].character == "MARCUS"
    assert ep.utterances[0].text == "I never told you about my first wife."
    assert ep.utterances[0].start == 1.0 and ep.utterances[0].end == 4.5
    assert ep.utterances[1].text == "She is alive." and ep.utterances[1].character == "UNKNOWN"
    assert not ep.utterances[2].timed and not ep.utterances[3].timed
    assert len(ep.warnings) == 2
    assert ep.number == 7


def test_json_list_and_dict_formats():
    rows = [{"start": "00:01.00", "end": "00:03.00", "speaker": "A", "text": "Hi?"},
            {"start_time": 4, "end_time": 6, "character": "B", "dialogue": "Who are you?"},
            {"text": "no time"}]
    ep = parse_json(json.dumps(rows), "episode-002", "S")
    assert [u.timed for u in ep.utterances] == [True, True, False]
    ep2 = parse_json(json.dumps({"episode_id": "ep-9", "logline": "L", "utterances": rows}), "x", "S")
    assert ep2.episode_id == "ep-9" and ep2.logline == "L" and ep2.number == 9


def test_dispatch_by_extension_and_empty_content():
    assert parse_episode_text("", "a.srt").utterances == []
    assert parse_episode_text("", "a.txt").warnings
    assert parse_episode_text("[]", "a.json").utterances == []


def test_discover_orders_episodes_numerically(tmp_path):
    show = tmp_path / "My Show"
    show.mkdir()
    for n in (10, 2, 1):
        (show / f"episode-{n}.txt").write_text(script([(0, 2, "A", "Hi there friend.")], number=n), encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore", encoding="utf-8")
    shows = discover_shows(tmp_path)
    assert list(shows) == ["My Show"]
    assert [f.stem for f in shows["My Show"]] == ["episode-1", "episode-2", "episode-10"]


def test_discover_empty_dataset(tmp_path):
    assert discover_shows(tmp_path) == {}
    assert discover_shows(tmp_path / "missing") == {}


def test_missing_episode_identifier_warns(tmp_path):
    f = tmp_path / "pilot.txt"
    f.write_text("**A:**\nStart: (00:01.00) | End: (00:02.00)\n\"Hello.\"\n", encoding="utf-8")
    ep = parse_episode_file(f, "S")
    assert ep.number is None and any("Episode number" in w for w in ep.warnings)
    assert ep.episode_id == "pilot"


def test_hindi_devanagari_text_is_preserved(tmp_path):
    f = tmp_path / "episode-1.txt"
    f.write_text(script([(0, 3, "MEERA", "तुमने मुझे धोखा दिया? You lied to me!")]), encoding="utf-8")
    ep = parse_episode_file(f, "S")
    assert "धोखा" in ep.utterances[0].text


@pytest.mark.skipif(not REAL_SHOW.exists(), reason="real dataset not present")
def test_real_dataset_all_episodes_parse_with_exact_source_timestamps():
    shows = discover_shows(REAL_SHOW.parent)
    files = shows[REAL_SHOW.name]
    assert len(files) == 50
    for f in files:
        ep = parse_episode_file(f, REAL_SHOW.name)
        assert ep.number is not None and ep.content, f.name
        raw = f.read_text(encoding="utf-8")
        for u in ep.content[:5]:
            assert f"Start: ({u.start_raw}) | End: ({u.end_raw})" in raw   # timestamps come from the file verbatim


def test_sample_fixtures_parse():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "sample_data"
    srt = parse_episode_file(root / "episode_sample.srt", "Sample")
    txt = parse_episode_file(root / "episode_sample.txt", "Sample")
    assert srt.source_format == "srt" and len(srt.content) >= 5
    assert txt.content and all(u.timed for u in txt.content)
