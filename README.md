# Employee Lifecycle Tutor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/4k5aya7vnh/badge.svg)](https://ask.thili.ai/b/4k5aya7vnh)

This project develops an Employee Lifecycle Tutor, an interactive tool designed for new HR personnel. It leverages data mining and language modeling to explain typical employee lifecycle event sequences and predict likely next steps, streamlining the onboarding process for HR professionals.

## What you are building

**Goal.** You will build an interactive tutor for new HR personnel that explains common employee lifecycle event sequences and suggests likely next steps.

**Done when.** The interactive tutor accurately guides users through HR processes by predicting the next likely HR event in a sequence, using frequent HR event sequences mined from historical data.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Represent HR events (hiring, training, promotion) as ordered sequences.**  
      Hint: Use data structuring to represent events as sequences.
- [ ] **2. Mine frequent HR event sequences using `thilimarketbasket`.**  
      Hint: Apply `thilimarketbasket` for frequent sequence mining.
- [ ] **3. Train an LM to predict the next likely HR event in a sequence.**  
      Hint: Utilize continued pretraining for the language model.
- [ ] **4. Design a text-based interactive tutor that queries the LM for explanations or next steps.**  
      Hint: Design a text-based interface for user interaction.
- [ ] **5. Evaluate the tutor's ability to accurately guide users through HR processes.**  
      Hint: Implement evaluation metrics for accuracy and guidance.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to input a custom sequence of HR events and receive predictions for the next steps.
- Implement a 'why' feature where the tutor explains *why* a certain event is a likely next step based on the mined sequences.
- Add a feedback mechanism for users to rate the helpfulness of the tutor's explanations.
