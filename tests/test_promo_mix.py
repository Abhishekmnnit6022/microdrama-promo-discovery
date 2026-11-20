"""Promo mix: which promos fill the final list. Scores never change; only the selection does."""

import pytest

from src.batch import RunConfig, run
from src.ranking import conflicts, select_with_mix
from test_batch_integration import _write_show


def _run(tmp_path, mix):
    return run(RunConfig(dataset_dir=str(tmp_path / "data"), cache_dir=str(tmp_path / "cache"), use_emotion=False,
                         use_zeroshot=False, mix=mix))


@pytest.fixture
def show(tmp_path):
    _write_show(tmp_path / "data", n=4, broken=False)
    return tmp_path


def test_best_is_unchanged_default(show):
    a = _run(show, "best")
    assert RunConfig().mix == "best"
    assert [c.rank for c in a.recommended] == list(range(1, len(a.recommended) + 1))


def test_include_multi_episode_adds_cross_episode_promos(show):
    best = _run(show, "best")
    mixed = _run(show, "include_multi_episode")
    n_best = sum(c.structure == "cross_episode" for c in best.recommended)
    n_mixed = sum(c.structure == "cross_episode" for c in mixed.recommended)
    assert n_mixed >= 1 and n_mixed >= n_best
    assert all(c.passed for c in mixed.recommended)
    for i, a in enumerate(mixed.recommended):          # diversity rules still hold
        for b in mixed.recommended[i + 1:]:
            assert not conflicts(a, b)
    scores = [c.score for c in mixed.recommended]       # still ranked by score
    assert scores == sorted(scores, reverse=True)


def test_multi_episode_only(show):
    res = _run(show, "multi_episode_only")
    assert res.recommended and all(c.structure == "cross_episode" for c in res.recommended)
    for c in res.recommended:
        assert len({s.episode_id for s in c.segments}) == 2 and 20 <= c.total_duration <= 60


def test_unknown_mix_rejected():
    with pytest.raises(ValueError):
        select_with_mix([], mix="random")
