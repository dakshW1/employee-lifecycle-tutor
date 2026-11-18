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

def sequence_items(career: list[str], max_gap_free: int = 3) -> set[str]:
    """Turn an ordered career into a 'basket' of order-preserving items the Apriori miner can count:
    every contiguous run of 1..max_gap_free events, written 'a > b > c'. A frequent item is then a
    frequent sequence pattern, and a frequent itemset is a set of patterns that occur in the same career."""
    items = set()
    for size in range(1, max_gap_free + 1):
        for i in range(len(career) - size + 1):
            items.add(" > ".join(career[i:i + size]))
    return items


def _overlapping(a, c) -> bool:
    """'x > y > z' always implies 'y > z': such rules are true by construction, so they are dropped."""
    return any(x in y or y in x for x in a for y in c)


def mine_patterns(careers: list[list[str]], min_support: float = 0.05, max_size: int = 2) -> dict:
    """Exact frequent patterns via thilimarketbasket.mine (Apriori on bitsets)."""
    from thilimarketbasket.mine import frequent_itemsets, rules

    baskets = [sorted(sequence_items(c)) for c in careers]
    itemsets = frequent_itemsets(baskets, min_support=min_support, max_size=max_size)
    sequences = {next(iter(s)): sup for s, sup in itemsets.items() if len(s) == 1 and ">" in next(iter(s))}
    found_rules = [(sorted(a), sorted(c), round(sup, 4), round(conf, 3), round(lift, 2))
                   for a, c, sup, conf, lift in rules(itemsets, min_confidence=0.5)
                   if not _overlapping(a, c)]
    return {"n_careers": len(careers), "min_support": min_support,
            "sequences": dict(sorted(sequences.items(), key=lambda kv: -kv[1])),
            "event_support": {next(iter(s)): sup for s, sup in itemsets.items()
                              if len(s) == 1 and ">" not in next(iter(s))},
            "rules": found_rules[:200]}


def next_step_stats(patterns: dict, last: str) -> list[tuple[str, float, float]]:
    """From the mined 2-step patterns: (next event, support of 'last > next', share of careers with 'last'
    that continue to 'next'). This is the tutor's 'why' evidence."""
    base = patterns["event_support"].get(last)
    out = []
    for seq, sup in patterns["sequences"].items():
        parts = seq.split(" > ")
        if len(parts) == 2 and parts[0] == last and base:
            out.append((parts[1], sup, min(1.0, sup / base)))
    return sorted(out, key=lambda t: -t[1])


def save_patterns(patterns: dict, path: Path = PATTERNS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patterns, indent=1), encoding="utf-8")


def load_patterns(path: Path = PATTERNS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Milestone 3: Train an LM to predict the next likely HR event in a sequence.

def load_lm(name: str = "distilgpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def next_token_loss(model, ids, mask):
    """Average cross-entropy of predicting every token from the ones before it, ignoring padding
    (the drill's continued-pretraining objective, in PyTorch)."""
    import torch.nn.functional as F

    logits = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1]).logits
    loss = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
    m = mask[:, 1:].float()
    return (loss * m).sum() / m.sum()


def _batch(tok, careers):
    import torch

    enc = [tok(to_text(c))["input_ids"] + [tok.eos_token_id] for c in careers]
    width = max(len(e) for e in enc)
    ids = torch.tensor([e + [tok.eos_token_id] * (width - len(e)) for e in enc])
    mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc])
    return ids, mask


def validation_loss(model, tok, careers, batch_size: int = 64) -> float:
    import torch

    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for i in range(0, len(careers), batch_size):
            ids, mask = _batch(tok, careers[i:i + batch_size])
            n = float(mask[:, 1:].sum())
            total += next_token_loss(model, ids, mask).item() * n
            count += n
    return total / count


def pretrain(model, tok, careers, val=None, steps: int = 300, batch_size: int = 16, lr: float = 1e-4,
             warmup: int = 20, eval_every: int = 50, seed: int = 0, log=print) -> list[dict]:
    """Continued pretraining: the plain next-token objective on careers-as-text, warm-up then cosine decay."""
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    history, start, recent = [], time.time(), []
    if val:
        history.append({"step": 0, "loss": None, "val_loss": round(validation_loss(model, tok, val), 4)})
        log(f"step 0  val loss {history[-1]['val_loss']:.3f}")
    for step in range(1, steps + 1):
        model.train()
        ids, mask = _batch(tok, rng.sample(careers, min(batch_size, len(careers))))
        loss = next_token_loss(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent.append(loss.item())
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "loss": round(float(np.mean(recent)), 4), "seconds": round(time.time() - start, 1)}
            if val:
                row["val_loss"] = round(validation_loss(model, tok, val), 4)
            history.append(row)
            recent = []
            log(f"step {step}  train {row['loss']:.3f}" + (f"  val {row['val_loss']:.3f}" if val else ""))
    model.eval()
    return history


def score_next(model, tok, history: list[str]) -> dict[str, float]:
    """P(next event | history) over all HR events (plus 'end'), from the LM's token log-probabilities:
    log P(" event words" | "Career: a > b >"), summed over the candidate's tokens, then normalised."""
    import torch

    text_ids = tok(to_text(history))["input_ids"]
    sep = tok(" >")["input_ids"]
    prefix = text_ids + sep
    names = list(EVENTS)
    cands = [tok(" " + e.replace("_", " "))["input_ids"] for e in names]
    with torch.no_grad():
        lp_text = torch.log_softmax(model(torch.tensor([prefix])).logits[0], dim=-1)
        # log P(" >" | career so far): the career continues ...
        cont = sum(lp_text[len(text_ids) - 1 + i, t].item() for i, t in enumerate(sep))
        # ... versus log P(end-of-text | career so far): the career ends here
        end_lp = lp_text[len(text_ids) - 1, tok.eos_token_id].item()
        logp = []
        for c in cands:
            lp = torch.log_softmax(model(torch.tensor([prefix + c])).logits[0, len(prefix) - 1:-1], dim=-1)
            logp.append(cont + sum(lp[i, t].item() for i, t in enumerate(c)))
    names.append(END)
    logp.append(end_lp)
    p = np.exp(np.array(logp) - max(logp))
    p /= p.sum()
    return dict(sorted(zip(names, p.tolist()), key=lambda kv: -kv[1]))


def save_lm(model, tok, out_dir: Path = MODEL_DIR, history: list | None = None) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (Path(out_dir) / "train_log.json").write_text(json.dumps(history or [], indent=1))


# --- Milestone 4: Design a text-based interactive tutor that queries the LM for explanations or next steps.
# Hint: Design a text-based interface for user interaction.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate the tutor's ability to accurately guide users through HR processes.
# Hint: Implement evaluation metrics for accuracy and guidance.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
