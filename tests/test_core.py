"""Tests: one block per milestone in README.md. LM tests use `sshleifer/tiny-gpt2` (~2 MB) to stay fast."""
import pytest

from app import core

TINY = "sshleifer/tiny-gpt2"


# Milestone 1: Represent HR events (hiring, training, promotion) as ordered sequences.
def test_transition_model_is_a_valid_answer_key():
    for event, nxt in core.TRANSITIONS.items():
        assert event in core.EVENTS and abs(sum(nxt.values()) - 1) < 1e-9
        assert all(n in core.EVENTS or n == core.END for n in nxt)


def test_careers_follow_the_rules_and_keep_their_order():
    careers = core.simulate(200, seed=1)
    assert careers == core.simulate(200, seed=1)
    for c in careers:
        assert c[0] == "hire"
        assert all(b in core.TRANSITIONS[a] for a, b in zip(c, c[1:]))  # only allowed steps
    assert core.to_text(["hire", "probation_review"]) == "Career: hire > probation review"


def test_careers_file_round_trip(tmp_path):
    careers = core.simulate(5)
    core.save_careers(careers, tmp_path / "c.jsonl")
    assert core.load_careers(tmp_path / "c.jsonl") == careers


# Milestone 2: Mine frequent HR event sequences using `thilimarketbasket`.
def test_sequence_items_keep_order():
    items = core.sequence_items(["hire", "onboarding", "training"])
    assert "hire > onboarding" in items and "onboarding > hire" not in items
    assert "hire > onboarding > training" in items and "hire" in items


@pytest.fixture(scope="module")
def patterns():
    return core.mine_patterns(core.simulate(800, seed=2), min_support=0.05)


def test_mined_supports_match_direct_counts(patterns):
    careers = core.simulate(800, seed=2)
    direct = sum("hire > onboarding" in core.sequence_items(c) for c in careers) / len(careers)
    assert patterns["sequences"]["hire > onboarding"] == pytest.approx(direct)
    assert all(sup >= 0.05 for sup in patterns["sequences"].values())


def test_next_step_stats_recover_the_true_transition(patterns):
    stats = {nxt: share for nxt, _, share in core.next_step_stats(patterns, "hire")}
    assert stats["onboarding"] == pytest.approx(0.95, abs=0.03)  # the planted 95%
    assert all(not core._overlapping(a, c) for a, c, *_ in patterns["rules"])


# Milestone 3: Train an LM to predict the next likely HR event in a sequence.
@pytest.fixture(scope="module")
def lm():
    model, tok = core.load_lm(TINY)
    careers = core.simulate(60, seed=3)
    history = core.pretrain(model, tok, careers, val=careers[:20], steps=20, batch_size=8, lr=5e-2,
                            warmup=2, eval_every=10, log=lambda *_: None)
    return model, tok, history


def test_continued_pretraining_lowers_validation_loss(lm):
    _, _, history = lm
    assert history[-1]["val_loss"] < history[0]["val_loss"]


def test_score_next_is_a_distribution_over_events(lm):
    model, tok, _ = lm
    probs = core.score_next(model, tok, ["hire", "onboarding"])
    assert set(probs) == set(core.EVENTS) | {core.END}
    assert sum(probs.values()) == pytest.approx(1.0)


# Milestone 4: Design a text-based interactive tutor that queries the LM for explanations or next steps.
def test_parse_history():
    assert core.parse_history("hire > onboarding, probation review") == ["hire", "onboarding", "probation_review"]
    with pytest.raises(ValueError):
        core.parse_history("hire > coffee break")


def test_explain_gives_steps_with_reasons(lm, patterns):
    model, tok, _ = lm
    ans = core.explain(model, tok, patterns, ["hire"], top=3)
    assert len(ans["next_steps"]) == 3
    assert all(s["what"] and s["why"] for s in ans["next_steps"])


from app.cli import tutor  # noqa: E402


def test_tutor_session_with_feedback(lm, patterns, tmp_path):
    model, tok, _ = lm
    lines = iter(["events", "explain promotion", "hire > onboarding", "rate 4", "hire > nap", ""])
    printed = []
    answers = tutor(model, tok, patterns, inp=lambda _: next(lines), out=printed.append,
                    feedback_file=tmp_path / "f.jsonl")
    assert len(answers) == 1 and answers[0]["history"] == ["hire", "onboarding"]
    assert any("unknown event" in p for p in printed)
    assert '"rating": 4' in (tmp_path / "f.jsonl").read_text()


# Milestone 5: Evaluate the tutor's ability to accurately guide users through HR processes.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

