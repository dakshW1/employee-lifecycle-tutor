# Employee Lifecycle Tutor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/4k5aya7vnh/badge.svg)](https://ask.thili.ai/b/4k5aya7vnh)
[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A text tutor for new HR staff. Type where an employee is in their career (`hire > onboarding > probation review`)
and it explains the likely next steps. The **probabilities** come from a small language model given continued
pretraining on HR careers. The **reasons** come from frequent event sequences mined exactly with
`thilimarketbasket`, the library from the *Teach a Language Model What Sells Together* drill.

```text
$ python -m app.cli ask "hire > onboarding > probation review"

Career so far: hire > onboarding > probation_review
  1. performance review  (48%)
     what: The regular (often yearly) review of goals, results and feedback.
     why:  In the mined HR records, 40% of careers that reach 'probation_review' continue with
           'performance_review' (seen in 33% of all careers).
  2. training  (41%)
     what: A course or certification to build skills for the current or next role.
     why:  In the mined HR records, 47% of careers that reach 'probation_review' continue with 'training' ...
  3. termination  (6%)
     ...
```

## Milestones

- [x] **1. Represent HR events (hiring, training, promotion) as ordered sequences.** 16 HR events; 3,000 careers
      simulated from a known transition model, which is the answer key.
- [x] **2. Mine frequent HR event sequences using `thilimarketbasket`.** Its exact Apriori miner
      (`thilimarketbasket.mine.frequent_itemsets` and `rules`) on order-preserving sequence items.
- [x] **3. Train an LM to predict the next likely HR event in a sequence.** Continued pretraining of `distilgpt2`
      on careers written as text, with the drill's next-token loss (padding masked).
- [x] **4. Design a text-based interactive tutor that queries the LM for explanations or next steps.** `tutor`
      and `ask`.
- [x] **5. Evaluate the tutor's ability to accurately guide users through HR processes.** MRR and hits@K
      (`thilimarketbasket.evaluate.summarise`, `rank_of`) against three baselines, including the true process.

Stretch ideas done: custom sequences, a "why" for every suggestion, and a 1-5 feedback rating saved to
`feedback.jsonl`.

## Run it

You need Python 3.12+ and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m app.cli simulate      # 3,000 careers -> data_hr/careers.jsonl (already included)
python -m app.cli mine          # frequent sequences -> data_hr/patterns.json (already included)
python -m app.cli train         # continued pretraining of distilgpt2 (about 5 minutes on a laptop CPU)
python -m app.cli evaluate      # MRR / hits@K vs the baselines
python -m app.cli tutor         # interactive: careers, `explain <event>`, `events`, `rate 1-5`
python -m app.cli ask "hire > onboarding > probation review"
pytest
```

The trained model (about 330 MB) is not committed; `train` rebuilds it into `models/hr-lm`.

## How it works (`app/core.py`)

**1. Careers as ordered sequences.** `EVENTS` lists 16 HR events with a one-line explanation each.
`TRANSITIONS` is the "true" process: after `hire`, `onboarding` follows 95% of the time, a `warning` leads to
`termination` 55% of the time, and so on. `simulate` samples 3,000 careers from it. This is the same idea as the
drill's *planted itemsets*: because we know the real probabilities, we can score every method against the truth
rather than against an impression. Unlike a shopping basket, a career has an order, so `to_text` never shuffles
it: `Career: hire > onboarding > probation review > training`.

**2. Mining with thilimarketbasket.** The Apriori miner counts *sets*, so `sequence_items` turns each career into a
set of order-preserving items: every contiguous run of 1-3 events, written `a > b > c`. Then a frequent item is
a frequent sequence pattern, and `frequent_itemsets` (bitset Apriori) finds all of them above 5% support. It
finds 75 patterns, e.g. `hire > onboarding` in 94.7% of careers and `performance_review > salary_raise` in 48.9%.
`rules` builds association rules. Rules like `x > y > z => y > z` are true by construction, so `_overlapping`
drops them. `next_step_stats` turns the 2-step patterns into "X% of careers that reach A continue with B", which is
the tutor's *why*. A test checks that the mined share for `hire > onboarding` matches the planted 95%.

**3. Continued pretraining.** `pretrain` is the drill's recipe in PyTorch, because the library's own `pretrain`
uses Apple's MLX: the plain next-token objective on careers-as-text, with padding masked out of the loss
(`next_token_loss`), AdamW, warm-up then cosine decay, and validation loss every 50 steps. `score_next` asks the
model for `P(next event | career)` over all 16 events plus "end". It sums the log-probabilities of each event's
tokens after `" >"`, compares that with the probability of the career ending there, and normalises.

**4. The tutor.** `parse_history` accepts `>` or commas, spaces or underscores, and rejects unknown events with
the list of valid ones. `explain` combines the LM's top 3 next steps with each event's meaning and the mined
evidence. The `tutor` loop also supports `events`, `explain <event>` and `rate <1-5>`.

**5. Evaluation.** `make_cases` cuts each held-out career at a random point and hides the next event. Each scorer
ranks all 17 possible answers, and `thilimarketbasket.evaluate.rank_of` / `summarise` give MRR and hits@K.

## Results

Continued pretraining: 300 steps, batch 16; validation loss 4.24 to 0.33.

| 300 held-out next-event cases | MRR | hits@1 | hits@3 |
|---|---:|---:|---:|
| popularity (ignores the career) | 0.380 | 19.0% | 44.3% |
| mined 2-step patterns (no LM) | 0.771 | 64.0% | 88.7% |
| **continued-pretrained LM** | **0.776** | 62.3% | **91.7%** |
| true process (ceiling) | 0.781 | 63.3% | 91.7% |

- The true process is the best any model can do, because careers are random: after a performance review there
  are 7 possible next events, and the most likely one has only 35%. So hits@1 cannot be 100%, and **63% is the
  ceiling**.
- The LM reaches 99% of the ceiling's MRR and matches its hits@3: it has learned the transition probabilities from
  text alone.
- The mined patterns are almost as good, because this process only depends on the last event. The LM is
  slightly better at hits@3 because it uses the whole career (e.g. how a career ends after a resignation), which a
  2-step pattern cannot see.
- Popularity, which ignores the career, is far behind (MRR 0.38): context matters.

## Limitations

- The careers are simulated, so this shows the method works and can be measured exactly. Real HR data would have
  dates, departments and much messier sequences, and would need anonymising.
- The simulated process only depends on the previous event. Real careers have longer memory (two warnings, then
  termination), which is where an LM should pull further ahead of 2-step patterns.

Built for the askThili drill *Teach a Language Model What Sells Together*. Written with help from an AI coding
assistant.
