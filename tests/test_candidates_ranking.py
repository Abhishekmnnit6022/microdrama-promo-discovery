import csv
import io
import json

from src.candidates import GenConfig, continuous_candidates, ending_lines, hook_beats, multi_candidates
from src.export import SEGMENT_COLUMNS, promos_to_csv, promos_to_json, promos_to_markdown
from src.ranking import Gates, Scorer, Weights, conflicts, select_diverse
from conftest import context, dialogue_run, episode


def drama_episode(number, offset_names=("JULIANA", "GERALD")):
    a, b = offset_names
    lines = [
        (0.0, 2.5, "ACTION", f"{b.title()} slaps {a.title()} across the face."),
        (2.6, 5.0, a, "How dare you! You swore you loved me!", "Furious"),
        (5.2, 8.0, b, f"I am marrying your sister, {a.title()}.", "Cold"),
        (8.2, 11.0, a, "You betrayed me in front of everyone.", "Devastated"),
        (11.2, 14.0, b, "Your father agreed to it. The wedding is over.", "Cold"),
        (14.2, 17.0, a, "Then I have no family left.", "Broken"),
        (17.2, 20.0, b, "Guards, take her away.", "Commanding"),
        (20.2, 23.0, a, "Let me go! Please!", "Desperate"),
        (23.2, 26.0, "STRANGER", "Take your hands off her.", "Menacing"),
        (26.2, 29.5, "STRANGER", f"There is something you need to know about who I really am, {b.title()}...", "Menacing"),
        (29.7, 33.0, b, "Who are you?", "Shocked"),
        (40.0, 43.0, a, "The carriage is ready, we can go home now.", "Calm"),
        (43.2, 46.0, b, "Thank you. Good night.", "Calm"),
        (55.0, 58.0, a, f"{b.title()}, I know your secret. You will pay for this.", "Furious"),
        (58.2, 61.0, b, "If you tell anyone, you will not live to see morning.", "Menacing"),
        (61.2, 64.0, a, "Then why are you shaking?", "Defiant"),
        (70.0, 73.0, b, "Who is at the door?", "Fearful"),
    ]
    return episode(lines, number)


def scored(ctx, cands):
    sc = Scorer(ctx)
    return [sc.score(c) for c in cands]


def test_continuous_candidates_respect_20_to_60_seconds():
    ctx = context([drama_episode(1)])
    cands = continuous_candidates(ctx, "episode-001", GenConfig())
    assert cands
    for c in cands:
        assert 20.0 <= c.total_duration <= 60.0
        s = c.segments[0]
        assert s.start == s.lines[0].start and s.end == max(u.end for u in s.lines)


def test_very_short_episode_yields_no_candidates():
    ctx = context([episode([(0, 3, "A", "You lied to me!"), (3.5, 8, "B", "Who told you?"), (8.5, 15, "A", "Your sister.")])])
    assert continuous_candidates(ctx, "episode-001", GenConfig()) == []


def test_multi_segment_and_cross_episode_order_gap_and_duration():
    eps = [drama_episode(1), drama_episode(2)]
    ctx = context(eps)
    cfg = GenConfig()
    beats = {e.episode_id: hook_beats(ctx, e.episode_id, cfg) for e in eps}
    ends = {e.episode_id: ending_lines(ctx, e.episode_id, cfg) for e in eps}
    stats = {}
    multi = multi_candidates(ctx, beats, ends, cfg, stats)
    kinds = {c.structure for c in multi}
    assert {"multi_segment", "cross_episode"} <= kinds
    for c in multi:
        a, b = c.segments
        assert cfg.min_total <= c.total_duration <= cfg.max_total
        assert abs(c.total_duration - (a.duration + b.duration)) < 0.02
        if c.structure == "multi_segment":
            assert a.episode_id == b.episode_id and b.start >= a.end + cfg.min_gap
        else:
            assert ctx.order[b.episode_id] - ctx.order[a.episode_id] in (1, 2)


def test_unrelated_segments_are_not_combined():
    e1 = drama_episode(1, ("JULIANA", "GERALD"))
    e2 = drama_episode(2, ("MARTHA", "THOMAS"))
    e2.utterances = [u for u in e2.utterances if u.character != "STRANGER"]
    ctx = context([e1, e2])
    cfg = GenConfig()
    beats = {e.episode_id: hook_beats(ctx, e.episode_id, cfg) for e in (e1, e2)}
    ends = {e.episode_id: ending_lines(ctx, e.episode_id, cfg) for e in (e1, e2)}
    stats = {}
    multi = multi_candidates(ctx, beats, ends, cfg, stats)
    assert not [c for c in multi if c.structure == "cross_episode"]
    assert stats.get("rejected_unrelated_pairs", 0) > 0


def test_score_is_weighted_sum_and_gates_apply():
    ctx = context([drama_episode(1)])
    cands = scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig()))
    w = Weights().normalised()
    for c in cands[:50]:
        b = c.components()
        expect = (w.hook * b["hook"] + w.cliffhanger * b["cliffhanger"] + w.coherence * b["coherence"] + w.emotion * b["emotion"]
                  + w.continuity * b["continuity"] + w.dialogue_quality * b["dialogue_quality"])
        assert abs(c.score - expect) < 1e-9 and 0.0 <= c.score <= 1.0
        assert c.passed == (not c.rejection_reasons)
    assert any(c.passed for c in cands) and any(not c.passed for c in cands)


def test_gate_rejects_short_promos_and_untimed_lines():
    ctx = context([drama_episode(1)])
    c = continuous_candidates(ctx, "episode-001", GenConfig(min_total=5, max_total=12))[0]
    Scorer(ctx).score(c)
    assert any("too short" in r for r in c.rejection_reasons)
    c.segments[0].lines[0].start = None
    Scorer(ctx, gates=Gates(min_duration=0)).score(c)
    assert any("without timestamps" in r for r in c.rejection_reasons)


def test_episode_with_no_dramatic_dialogue_has_no_valid_promo():
    ctx = context([episode(dialogue_run(0, 25))])
    cands = scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig()))
    assert cands and not any(c.passed for c in cands)


def test_emotional_episode_without_cliffhanger_fails_cliffhanger_gate():
    lines = [(i * 3.0, i * 3.0 + 2.8, ("JULIANA", "GERALD")[i % 2], t, "Furious") for i, t in enumerate(
        ["I hate you, Gerald!", "I hate you more, Juliana!", "You ruined my wedding!", "You ruined my life!",
         "I am so angry!", "So am I!", "Fine. We are done now.", "Yes. It is over. Goodbye.", "Goodbye.", "Thank you."])]
    ctx = context([episode(lines)])
    cands = scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig()))
    ending_late = [c for c in cands if c.segments[0].end >= 26]
    assert ending_late and all(any("cliffhanger" in r for r in c.rejection_reasons) for c in ending_late)


def test_diversity_removes_overlaps_and_limits_per_episode():
    ctx = context([drama_episode(1)])
    cands = scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig()))
    chosen = select_diverse(cands, top_k=10, max_per_episode=2)
    assert 1 <= len(chosen) <= 2
    assert [c.rank for c in chosen] == list(range(1, len(chosen) + 1))
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert not conflicts(a, b)
    assert chosen == sorted(chosen, key=lambda c: -c.score)


def test_repeated_dialogue_across_episodes_does_not_duplicate_recommendations():
    eps = [drama_episode(1), drama_episode(2)]
    ctx = context(eps)
    cands = []
    for e in eps:
        cands += scored(ctx, continuous_candidates(ctx, e.episode_id, GenConfig()))
    chosen = select_diverse(cands, top_k=10, max_per_episode=1)
    assert len({c.episodes[0] for c in chosen}) == len(chosen)


def test_explanations_never_claim_virality():
    ctx = context([drama_episode(1)])
    for c in scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig()))[:30]:
        low = c.explanation.lower()
        assert "viral" not in low and "ctr" not in low and "guarantee" not in low


def test_exports_contain_every_timestamp():
    ctx = context([drama_episode(1)])
    chosen = select_diverse(scored(ctx, continuous_candidates(ctx, "episode-001", GenConfig())), top_k=2)
    data = json.loads(promos_to_json(chosen, {"x": 1}))
    p = data["promos"][0]
    for k in ("rank", "episodes", "segments", "opening_hook", "ending_cliffhanger", "why_it_works", "score_breakdown"):
        assert k in p
    seg = p["segments"][0]
    assert {"order", "episode_id", "start", "end", "start_seconds", "end_seconds", "duration_seconds"} <= set(seg)
    assert p["opening_hook"]["timestamp_fmt"] and p["ending_cliffhanger"]["timestamp_fmt"]
    rows = list(csv.DictReader(io.StringIO(promos_to_csv(chosen))))
    assert list(rows[0]) == SEGMENT_COLUMNS
    assert len(rows) == sum(len(c.segments) for c in chosen)
    md = promos_to_markdown(chosen)
    assert "| # | Episode | Start | End | Duration | Dialogue |" in md
