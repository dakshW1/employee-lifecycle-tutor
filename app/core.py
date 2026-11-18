"""Employee Lifecycle Tutor: mine frequent HR event sequences and teach a small LM what usually comes next.

Same plan as the market-basket drill, with one change: a career is an *ordered* sequence, so order is kept.

  1. sequences  - HR events as ordered careers, simulated from a known transition model (the answer key)
  2. mining     - frequent sequence patterns with thilimarketbasket's exact Apriori miner
  3. LM         - continued pretraining of a small causal LM on careers written as text
  4. tutor      - text interface (app/cli.py) using the LM for next steps and the mined patterns for "why"
  5. evaluation - MRR / hits@K of the next event vs popularity, mined-pattern and true-model baselines
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_hr"
CAREERS_FILE = DATA_DIR / "careers.jsonl"
PATTERNS_FILE = DATA_DIR / "patterns.json"
MODEL_DIR = ROOT / "models" / "hr-lm"
FEEDBACK_FILE = ROOT / "feedback.jsonl"


# --- Milestone 1: Represent HR events (hiring, training, promotion) as ordered sequences.

EVENTS = {
    "hire": "The candidate accepts the offer and is added to the payroll.",
    "onboarding": "Accounts, equipment, policies and a buddy: the first weeks in the company.",
    "probation_review": "End-of-probation check (usually 3-6 months) to confirm the employee.",
    "training": "A course or certification to build skills for the current or next role.",
    "performance_review": "The regular (often yearly) review of goals, results and feedback.",
    "promotion": "A move to a more senior role, usually after strong reviews.",
    "salary_raise": "A pay increase, often decided in the review cycle.",
    "internal_transfer": "A move to another team or location at the same level.",
    "improvement_plan": "A performance improvement plan (PIP) with clear goals and a deadline.",
    "warning": "A formal written warning about conduct or performance.",
    "leave_of_absence": "Extended leave: parental, medical or a sabbatical.",
    "return_from_leave": "Coming back from leave, often with a phased return.",
    "resignation": "The employee decides to leave and gives notice.",
    "termination": "The company ends the employment.",
    "exit_interview": "A final conversation to learn why the employee is leaving.",
    "retirement": "The employee retires at the end of their career.",
}
END = "end"

# The "true" process the careers are simulated from. Every number here is the answer key the miner and the
# LM are scored against: e.g. after a hire, onboarding comes next 95% of the time.
TRANSITIONS = {
    "hire": {"onboarding": 0.95, "resignation": 0.05},
    "onboarding": {"probation_review": 0.85, "training": 0.15},
    "probation_review": {"training": 0.45, "performance_review": 0.4, "termination": 0.08, "resignation": 0.07},
    "training": {"performance_review": 0.75, "internal_transfer": 0.1, "promotion": 0.15},
    "performance_review": {"salary_raise": 0.35, "promotion": 0.2, "training": 0.15, "improvement_plan": 0.12,
                           "leave_of_absence": 0.06, "resignation": 0.07, "retirement": 0.05},
    "salary_raise": {"performance_review": 0.7, "training": 0.15, "resignation": 0.1, "internal_transfer": 0.05},
    "promotion": {"training": 0.45, "performance_review": 0.4, "salary_raise": 0.1, "resignation": 0.05},
    "internal_transfer": {"onboarding": 0.5, "training": 0.3, "performance_review": 0.2},
    "improvement_plan": {"performance_review": 0.55, "warning": 0.3, "resignation": 0.15},
    "warning": {"termination": 0.55, "improvement_plan": 0.25, "resignation": 0.2},
    "leave_of_absence": {"return_from_leave": 0.85, "resignation": 0.15},
    "return_from_leave": {"performance_review": 0.6, "training": 0.4},
    "resignation": {"exit_interview": 0.9, END: 0.1},
    "termination": {"exit_interview": 0.6, END: 0.4},
    "exit_interview": {END: 1.0},
    "retirement": {"exit_interview": 0.5, END: 0.5},
}


def simulate_career(rng: random.Random, max_len: int = 14) -> list[str]:
    """One career, sampled step by step from TRANSITIONS, starting with a hire."""
    career = ["hire"]
    while len(career) < max_len:
        options = TRANSITIONS[career[-1]]
        nxt = rng.choices(list(options), weights=list(options.values()))[0]
        if nxt == END:
            break
        career.append(nxt)
    return career


def simulate(n: int = 3000, seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    return [simulate_career(rng) for _ in range(n)]


def to_text(career: list[str], upto: int | None = None) -> str:
    """A career as text the LM reads. Unlike a shopping basket, order matters, so it is never shuffled."""
    events = career if upto is None else career[:upto]
    return "Career: " + " > ".join(e.replace("_", " ") for e in events)


def save_careers(careers: list[list[str]], path: Path = CAREERS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in careers), encoding="utf-8")


def load_careers(path: Path = CAREERS_FILE) -> list[list[str]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split(careers: list[list[str]], test_share: float = 0.15) -> tuple[list, list]:
    cut = int(len(careers) * (1 - test_share))
    return careers[:cut], careers[cut:]


# --- Milestone 2: Mine frequent HR event sequences using `thilimarketbasket`.
# Hint: Apply `thilimarketbasket` for frequent sequence mining.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Train an LM to predict the next likely HR event in a sequence.
# Hint: Utilize continued pretraining for the language model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Design a text-based interactive tutor that queries the LM for explanations or next steps.
# Hint: Design a text-based interface for user interaction.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate the tutor's ability to accurately guide users through HR processes.
# Hint: Implement evaluation metrics for accuracy and guidance.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
