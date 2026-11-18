"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import core


def _print_answer(ans: dict, out=print) -> None:
    out(f"\nCareer so far: {' > '.join(ans['history'])}")
    for i, s in enumerate(ans["next_steps"], 1):
        out(f"  {i}. {s['event'].replace('_', ' ')}  ({s['probability']:.0%})")
        out(f"     what: {s['what']}")
        out(f"     why:  {s['why']}")


def tutor(model, tok, patterns: dict, inp=input, out=print, feedback_file: Path | None = None) -> list[dict]:
    """Interactive tutor. Type a career (e.g. 'hire > onboarding'), get likely next steps with reasons.
    'explain <event>' describes an event, 'events' lists them, 'rate 1-5' rates the last answer, empty quits."""
    out("HR lifecycle tutor. Enter events separated by '>' (try: hire > onboarding > probation review).")
    out("Commands: events | explain <event> | rate <1-5> | empty line to quit")
    answers, last = [], None
    while True:
        try:
            line = inp("\ntutor> ").strip()
        except EOFError:
            line = ""
        if not line:
            return answers
        if line == "events":
            out(", ".join(core.EVENTS))
        elif line.startswith("explain "):
            name = line[8:].strip().replace(" ", "_")
            out(core.EVENTS.get(name, f"unknown event {name!r}"))
        elif line.startswith("rate "):
            if last is None:
                out("ask about a career first")
            else:
                core.save_feedback(last["history"], last["next_steps"][0]["event"], int(line[5:]),
                                   feedback_file or core.FEEDBACK_FILE)
                out("thanks, feedback saved")
        else:
            try:
                history = core.parse_history(line)
            except ValueError as e:
                out(str(e))
                continue
            last = core.explain(model, tok, patterns, history)
            answers.append(last)
            _print_answer(last, out)


def _load():
    model, tok = core.load_lm(str(core.MODEL_DIR))
    return model, tok, core.load_patterns()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="employee-lifecycle-tutor", description="New HR personnel use an interactive tutor that explains common employee lifecycle event sequences (e.g., hire -> onboarding -> performance review) and likely next steps.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate HR careers from the transition model")
    p.add_argument("--n", type=int, default=3000)
    p = sub.add_parser("mine", help="mine frequent sequence patterns with thilimarketbasket")
    p.add_argument("--min-support", type=float, default=0.05)
    p = sub.add_parser("train", help="continued pretraining of a small LM on careers-as-text")
    p.add_argument("--base", default="distilgpt2")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    sub.add_parser("tutor", help="start the interactive tutor")
    p = sub.add_parser("ask", help="one question: likely next steps after a career")
    p.add_argument("career", help='e.g. "hire > onboarding > probation review"')
    p = sub.add_parser("evaluate", help="next-event MRR / hits@K: LM vs baselines vs the true model")
    p.add_argument("--cases", type=int, default=300)

    args = parser.parse_args(argv)

    if args.command == "simulate":
        careers = core.simulate(args.n)
        core.save_careers(careers)
        lengths = [len(c) for c in careers]
        print(f"{len(careers)} careers, {sum(lengths) / len(lengths):.1f} events on average -> {core.CAREERS_FILE.relative_to(core.ROOT)}")
        for c in careers[:3]:
            print(" ", core.to_text(c))
    elif args.command == "mine":
        train, _ = core.split(core.load_careers())
        patterns = core.mine_patterns(train, args.min_support)
        core.save_patterns(patterns)
        print(f"{len(patterns['sequences'])} frequent sequence patterns (support >= {args.min_support:.0%}):")
        for seq, sup in list(patterns["sequences"].items())[:15]:
            print(f"  {sup:6.1%}  {seq}")
        print(f"{len(patterns['rules'])} rules, e.g.:")
        for a, c, sup, conf, lift in patterns["rules"][:5]:
            print(f"  {a} => {c}  confidence {conf:.0%}, lift {lift}")
    elif args.command == "train":
        train, test = core.split(core.load_careers())
        model, tok = core.load_lm(args.base)
        history = core.pretrain(model, tok, train, val=test[:200], steps=args.steps, lr=args.lr)
        core.save_lm(model, tok, core.MODEL_DIR, history)
        print(f"saved -> {core.MODEL_DIR.relative_to(core.ROOT)}")
    elif args.command == "tutor":
        tutor(*_load())
    elif args.command == "ask":
        model, tok, patterns = _load()
        _print_answer(core.explain(model, tok, patterns, core.parse_history(args.career)))
    elif args.command == "evaluate":
        train, test = core.split(core.load_careers())
        cases = core.make_cases(test, args.cases)
        model, tok, patterns = _load()
        results = {
            "popularity (ignores the career)": core.evaluate(cases, core.popularity_scorer(train)),
            "mined 2-step patterns": core.evaluate(cases, core.pattern_scorer(patterns)),
            "continued-pretrained LM": core.evaluate(cases, core.lm_scorer(model, tok)),
            "true process (ceiling)": core.evaluate(cases, core.oracle_scorer),
        }
        print(f"{len(cases)} held-out next-event cases")
        for name, r in results.items():
            print(f"  {name:<33} MRR {r['mrr']:.3f}  hits@1 {r['hits@1']:.1%}  hits@3 {r['hits@3']:.1%}")
        print(json.dumps(results))


if __name__ == "__main__":
    main()
