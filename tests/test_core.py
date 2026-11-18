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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Train an LM to predict the next likely HR event in a sequence.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Design a text-based interactive tutor that queries the LM for explanations or next steps.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Evaluate the tutor's ability to accurately guide users through HR processes.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

