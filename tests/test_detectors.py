"""Hook, cliffhanger and coherence detectors on controlled synthetic scenes (no ML models loaded)."""

from src.candidates import make_segment
from src.cliffhanger import detect_cliffhanger
from src.coherence import analyse_coherence
from src.hook import detect_hook
from conftest import context, episode


def seg_between(ctx, eid, t0, t1):
    lines = ctx.content[eid]
    i = next(k for k, u in enumerate(lines) if u.start >= t0 - 1e-6)
    j = max(k for k, u in enumerate(lines) if u.end <= t1 + 1e-6)
    return make_segment(ctx, eid, i, j)


# ── hook ──────────────────────────────────────────────────────────────────
def test_physical_confrontation_hook_beats_calm_opener(betrayal_episode):
    ctx = context([betrayal_episode])
    strong = detect_hook(seg_between(ctx, "episode-001", 0, 14), ctx)
    calm = detect_hook(seg_between(ctx, "episode-001", 36.2, 39), ctx)
    assert strong.score >= 0.45 > calm.score
    assert strong.evidence[0] in ("physical confrontation", "betrayal/accusation")
    assert 0.0 <= strong.timestamp <= 3.0


def test_hook_only_looks_at_first_three_seconds():
    lines = [(0.0, 2.0, "A", "The weather is nice today, isn't it.", "Calm"),
             (2.5, 4.5, "B", "Yes, very nice weather for a walk.", "Calm"),
             (5.0, 7.0, "A", "I will kill you! You betrayed me!", "Furious")]
    ctx = context([episode(lines)])
    h = detect_hook(seg_between(ctx, "episode-001", 0, 7), ctx)
    assert h.score < 0.45
    assert h.timestamp in (0.0, 2.5)


def test_hook_penalises_dangling_opening():
    a = context([episode([(0, 2.5, "A", "He betrayed us all.", "Angry"), (3, 5, "B", "When?")])])
    b = context([episode([(0, 2.5, "A", "Gerald betrayed us all.", "Angry"), (3, 5, "B", "When?")])])
    sa = detect_hook(seg_between(a, "episode-001", 0, 5), a).score
    sb = detect_hook(seg_between(b, "episode-001", 0, 5), b).score
    assert sa < sb


def test_plain_question_is_not_a_strong_hook():
    ctx = context([episode([(0, 2, "A", "Is dinner ready?"), (2.2, 4, "B", "Almost.")])])
    assert detect_hook(seg_between(ctx, "episode-001", 0, 4), ctx).score < 0.45


# ── cliffhanger ───────────────────────────────────────────────────────────
def test_secret_setup_cut_is_strong_and_resolution_cut_is_weak(betrayal_episode):
    ctx = context([betrayal_episode])
    secret = detect_cliffhanger(seg_between(ctx, "episode-001", 20.2, 29.5), ctx)
    resolved = detect_cliffhanger(seg_between(ctx, "episode-001", 26.2, 39.0), ctx)
    assert secret.score >= 0.5 > resolved.score
    assert secret.timestamp == 29.5
    assert any("resolving language" in e for e in resolved.evidence)


def test_trivially_answered_question_is_weaker_than_charged_question_with_payoff():
    trivial = context([episode([(0, 3, "A", "We should go now, the carriage is here."), (3.2, 5, "A", "Is it raining?"),
                                (5.2, 6, "B", "No."), (6.2, 9, "A", "Good, let us go.")])])
    charged = context([episode([(0, 3, "A", "You lied to me about everything."), (3.2, 5, "A", "Who is the real father of this child?", "Furious"),
                                (5.2, 8, "B", "The truth is... it was the Duke himself.", "Shocked")])])
    t = detect_cliffhanger(seg_between(trivial, "episode-001", 0, 5), trivial)
    c = detect_cliffhanger(seg_between(charged, "episode-001", 0, 5), charged)
    assert c.score > t.score
    assert t.score < 0.5
    assert any("answers it at once" in e for e in t.evidence)


def test_cut_mid_sentence_is_not_a_cliffhanger():
    ctx = context([episode([(0, 2, "A", "Listen carefully."), (2.2, 4, "B", "I said..."), (4.1, 6, "B", "...let her go."),
                            (6.5, 8, "A", "Fine.")])])
    c = detect_cliffhanger(seg_between(ctx, "episode-001", 0, 4), ctx)
    assert c.score < 0.5
    assert any("mid-sentence" in e for e in c.evidence)


def test_question_mark_alone_is_not_enough():
    ctx = context([episode([(0, 2, "A", "Shall we have tea?"), (2.5, 4, "B", "Lovely idea."), (4.5, 6, "A", "Sugar?"),
                            (6.5, 8, "B", "Yes please.")])])
    assert detect_cliffhanger(seg_between(ctx, "episode-001", 0, 6), ctx).score < 0.5


def test_multiple_possible_cliffhangers_prefers_the_stronger_cut(betrayal_episode):
    ctx = context([betrayal_episode])
    weak = detect_cliffhanger(seg_between(ctx, "episode-001", 0, 20), ctx)      # "Then I have nothing left here."
    strong = detect_cliffhanger(seg_between(ctx, "episode-001", 0, 29.5), ctx)  # secret about to come out
    assert strong.score > weak.score


# ── coherence ─────────────────────────────────────────────────────────────
def test_named_conflict_with_stakes_is_coherent(betrayal_episode):
    ctx = context([betrayal_episode])
    coh = analyse_coherence([seg_between(ctx, "episode-001", 0, 29.5)], ctx, "continuous")
    assert coh.score >= 0.55


def test_fragment_needing_prior_context_scores_lower():
    lines = [(0, 3, "A", "He did it again, just like last night."), (3.2, 6, "A", "After what happened, they will come back."),
             (6.2, 9, "A", "It is happening again.")]
    ctx = context([episode(lines, logline="Juliana confronts Gerald about the stolen inheritance.")])
    coh = analyse_coherence([seg_between(ctx, "episode-001", 0, 9)], ctx, "continuous")
    assert coh.score < 0.55
    assert any("earlier events" in e for e in coh.evidence)


def test_strong_hook_without_context_fails_coherence_gate():
    lines = [(0, 2.5, "ACTION", "He slaps her."), (2.6, 5, "A", "Why?!", "Shocked"), (5.2, 8, "A", "Them again?"),
             (8.2, 22, "ACTION", "She runs.")]
    ctx = context([episode(lines, logline="Juliana fights for her inheritance.")])
    seg = seg_between(ctx, "episode-001", 0, 22)
    assert detect_hook(seg, ctx).score >= 0.45
    assert analyse_coherence([seg], ctx, "continuous").score < 0.55


def test_segments_without_shared_characters_are_penalised():
    lines = ([(i * 3.0, i * 3.0 + 2.8, ("JULIANA", "GERALD")[i % 2], "Juliana, you betrayed this family!") for i in range(5)]
             + [(60 + i * 3.0, 60 + i * 3.0 + 2.8, ("MARTHA", "TOM")[i % 2], "The bread is burning, Tom!") for i in range(5)])
    ctx = context([episode(lines)])
    a = seg_between(ctx, "episode-001", 0, 14.8)
    b = seg_between(ctx, "episode-001", 60, 74.8)
    joined = analyse_coherence([a, b], ctx, "multi_segment")
    assert any("share no character" in e for e in joined.evidence)


def test_repeated_lines_penalised():
    rep = [(i * 3.0, i * 3.0 + 2.8, ("JULIANA", "GERALD")[i % 2], "You will regret this, Gerald.") for i in range(6)]
    var = [(i * 3.0, i * 3.0 + 2.8, ("JULIANA", "GERALD")[i % 2], t) for i, t in enumerate(
        ["You will regret this, Gerald.", "I regret nothing, Juliana.", "You married my sister!", "Your father agreed.",
         "Then I have no family.", "Then leave."])]
    c1 = context([episode(rep)])
    c2 = context([episode(var)])
    s1 = analyse_coherence([seg_between(c1, "episode-001", 0, 18)], c1, "continuous").score
    s2 = analyse_coherence([seg_between(c2, "episode-001", 0, 18)], c2, "continuous").score
    assert s1 < s2


def test_spoken_physical_verb_is_not_a_cliffhanger():
    """Regression from the full-dataset run: 'Let me grab my ledger. We go now.' was scored as a physical cliffhanger."""
    ctx = context([episode([(0, 3, "A", "Toby cannot breathe, please help him!", "Desperate"),
                            (3.2, 6, "B", "Stand up. Let me grab my ledger. We go now.", "Firm"),
                            (6.5, 9, "A", "Thank you, my lady.")])])
    c = detect_cliffhanger(seg_between(ctx, "episode-001", 0, 6), ctx)
    assert c.score < 0.5 and getattr(c, "open_signal", 1) < 0.5


def test_action_de_escalation_is_penalised():
    """Regression: 'The guards slowly lower their weapon points' was scored as a revelation cliffhanger."""
    ctx = context([episode([(0, 3, "DUKE", "Lower them, or answer to the Crown!", "Commanding"),
                            (3.2, 6, "ACTION", "The guards slowly lower their weapons away from the Duke."),
                            (6.5, 9, "A", "It is over.")])])
    c = detect_cliffhanger(seg_between(ctx, "episode-001", 0, 6), ctx)
    assert any("resolving language" in e for e in c.evidence)
