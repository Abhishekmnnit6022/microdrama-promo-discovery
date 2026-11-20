"""End-to-end: dataset folder -> parsing -> candidates -> hook/cliffhanger/coherence -> ranking -> export."""

import json

import pytest

from src import feedback as fb
from src import llm_review, ml
from src.batch import RunConfig, load_outputs, run, write_outputs
from conftest import REAL_SHOW, script
from test_candidates_ranking import drama_episode


def _write_show(root, name="Duke Show", n=3, broken=True):
    d = root / name
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        ep = drama_episode(k)
        lines = [(u.start, u.end, u.character, u.text, u.emotion_tag) for u in ep.utterances]
        (d / f"episode-{k:03d}.txt").write_text(script(lines, number=k), encoding="utf-8")
    if broken:
        (d / f"episode-{n + 1:03d}.txt").write_text("this file is not a script at all", encoding="utf-8")
    return d


def _cfg(tmp_path, **kw):
    return RunConfig(dataset_dir=str(tmp_path / "data"), cache_dir=str(tmp_path / "cache"), use_emotion=False,
                     use_zeroshot=False, **kw)


def test_full_pipeline_with_failure_isolation_and_exports(tmp_path):
    _write_show(tmp_path / "data")
    res = run(_cfg(tmp_path))
    s = res.stats["summary"]
    assert s["episodes_discovered"] == 4 and s["episodes_failed"] == 1 and s["episodes_processed_ok"] == 3
    failed = [e for e in res.episodes if e.status == "failed"]
    assert failed and "no timed dialogue" in failed[0].error
    assert s["candidates_total"] > 0 and s["recommended"] > 0
    assert {c.structure for c in res.candidates} >= {"continuous", "multi_segment", "cross_episode"}
    v = res.stats["validation"]
    assert v["promos_valid"] == v["promos_checked"] and v["duplicate_or_overlapping_pairs"] == 0
    for c in res.recommended:
        assert 20 <= c.total_duration <= 60 and c.passed
    out = tmp_path / "out"
    files = write_outputs(res, out)
    assert (out / "1_recommendations" / "recommendations.json").exists()
    assert (out / "5_run_logs" / "errors.jsonl").read_text(encoding="utf-8").strip()
    assert len(list((out / "3_per_episode").glob("*.json"))) == 4
    loaded = load_outputs(out)
    assert loaded["recommendations"]["promos"][0]["rank"] == 1
    assert loaded["run_manifest"]["stats"]["summary"]["episodes_failed"] == 1
    assert "api_key" not in json.dumps(loaded["run_manifest"]["config"])
    assert len(files) >= 10


def test_parse_cache_is_reused_on_resume(tmp_path, caplog):
    _write_show(tmp_path / "data", broken=False)
    run(_cfg(tmp_path))
    caplog.set_level("INFO")
    run(_cfg(tmp_path))
    assert "[cache]" in caplog.text


def test_empty_dataset_reports_error(tmp_path):
    (tmp_path / "data").mkdir()
    res = run(_cfg(tmp_path))
    assert res.recommended == [] and "No transcripts" in res.stats["error"]


def test_model_failure_falls_back_to_transparent_features(tmp_path, monkeypatch):
    _write_show(tmp_path / "data", broken=False)
    monkeypatch.setattr(ml, "_load", lambda kind: None)
    cfg = _cfg(tmp_path)
    cfg.use_emotion = True
    cfg.use_zeroshot = True
    res = run(cfg)
    assert res.stats["summary"]["episodes_failed"] == 0 and res.recommended


def test_llm_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    _write_show(tmp_path / "data", broken=False)

    class Boom:
        class models:
            @staticmethod
            def generate_content(**kw):
                raise RuntimeError("quota exceeded")
    monkeypatch.setattr(llm_review, "_client", lambda key: Boom())
    res = run(_cfg(tmp_path, use_llm=True, api_key="test-key"))
    assert res.llm_summary["status"] == "failed"
    assert all(c.llm_review["status"] == "failed" for c in res.recommended)


def test_llm_without_key_is_skipped(tmp_path):
    _write_show(tmp_path / "data", broken=False)
    res = run(_cfg(tmp_path, use_llm=True, api_key=""))
    assert res.llm_summary["status"] == "skipped"


def test_feedback_roundtrip(tmp_path):
    _write_show(tmp_path / "data", broken=False)
    res = run(_cfg(tmp_path))
    p = res.recommended[0].to_dict()
    f = tmp_path / "fb.jsonl"
    fb.record(f, p, "rejected", "weak opening hook")
    fb.record(f, p, "approved")
    s = fb.summary(fb.load(f))
    assert s["reviewed_promos"] == 1 and s["approved"] == 1 and s["acceptance_rate"] == 1.0
    with pytest.raises(ValueError):
        fb.record(f, p, "maybe")


def test_resolve_api_key_prefers_explicit(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm_review.resolve_api_key("typed", env) == "typed"
    assert llm_review.resolve_api_key("", env) == "from-file"
    import os
    assert "GEMINI_API_KEY" not in os.environ


@pytest.mark.skipif(not REAL_SHOW.exists(), reason="real dataset not present")
def test_real_dataset_first_episodes_end_to_end(tmp_path):
    cfg = RunConfig(dataset_dir=str(REAL_SHOW.parent), cache_dir=str(tmp_path / "c"), use_emotion=False,
                    use_zeroshot=False, max_episodes=4, shows=[REAL_SHOW.name])
    res = run(cfg)
    s = res.stats["summary"]
    assert s["episodes_processed_ok"] == 4 and s["episodes_failed"] == 0
    assert res.recommended
    v = res.stats["validation"]
    assert v["promos_valid"] == v["promos_checked"]
    for c in res.recommended:
        for seg in c.segments:
            raw = open(res.contexts[c.show].episode(seg.episode_id).source_path, encoding="utf-8").read()
            assert f"Start: ({seg.lines[0].start_raw})" in raw
