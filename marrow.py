#!/usr/bin/env python3
"""Marrow -- decide whether the person at the computer is the enrolled
user, from OS-level input events only (pointer + text as it arrives).

Usage:
  python3 marrow.py
      Full pipeline on shipped fake data: generate, enrol, swap test,
      report FAR/FRR. One command, no dependencies, works offline.

  python3 marrow.py --train enrol.jsonl --test test.jsonl
      Train on owner sessions in enrol.jsonl (headers with persona
      owner-fresh / owner-tired) and score every session in test.jsonl.
      Prints accept/reject per session plus FAR/FRR if the test file
      carries persona labels.

  python3 marrow.py --generate data/
      Write the fake dataset to data/ as JSONL recordings and stop.

  python3 marrow.py --record [--out FILE] [--style X]
      Private on-machine recording kit: opens a small window, records
      pointer moves/clicks and text (typed vs dictated/inserted) inside
      it, and saves a JSONL recording in the documented format. Local
      only, nothing leaves the machine. Needs tkinter (python3-tk).

  python3 marrow.py --seed 2000
      Same pipeline with a different deterministic seed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marrow import evaluate, generate
from marrow.format import Session, read_sessions, write_sessions
from marrow.model import MarrowModel


def _as_dicts(sessions):
    return [{"header": s.header, "events": s.events} for s in sessions]


def cmd_train_test(train_path, test_path):
    train = read_sessions(train_path)
    test = read_sessions(test_path)
    train_d = _as_dicts(train)
    test_d = _as_dicts(test)

    fresh = [s for s in train_d if s["header"]["persona"] == "owner-fresh"]
    tired = [s for s in train_d if s["header"]["persona"] == "owner-tired"]
    owner = [s for s in train_d if s["header"]["persona"] == "owner"]
    if not fresh and owner:
        fresh = owner
    if not fresh and not tired:
        print("error: no owner sessions (persona owner-fresh/owner-tired) "
              "in %s" % train_path)
        return 1
    model = MarrowModel().enrol({"fresh": fresh, "tired": tired})
    print("trained on %d fresh + %d tired enrolment sessions"
          % (len(fresh), len(tired)))
    print("threshold %.2f (LOO mean %.2f, std %.2f, max %.2f)"
          % (model.threshold, model.loo_mean, model.loo_std, model.loo_max))

    tally = {}
    for s in test_d:
        score = model.score(s)
        ok = score < model.threshold
        persona = s["header"]["persona"]
        print("%-28s %-12s score %8.2f  %s"
              % (s["header"]["id"], persona, score,
                 "ACCEPT" if ok else "reject"))
        a, n = tally.get(persona, (0, 0))
        tally[persona] = (a + (1 if ok else 0), n + 1)
    print("-" * 60)
    for persona in sorted(tally):
        a, n = tally[persona]
        print("%-12s accepted %d/%d" % (persona, a, n))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", metavar="FILE",
                    help="JSONL recording file to enrol from")
    ap.add_argument("--test", metavar="FILE",
                    help="JSONL recording file to score")
    ap.add_argument("--generate", metavar="DIR",
                    help="write the fake dataset to DIR and stop")
    ap.add_argument("--record", action="store_true",
                    help="open the private on-machine recorder window")
    ap.add_argument("--out", metavar="FILE", default="recording.jsonl",
                    help="output file for --record (default recording.jsonl)")
    ap.add_argument("--style", metavar="STYLE", default="local",
                    help="style label for the --record session")
    ap.add_argument("--seed", type=int, default=1000,
                    help="deterministic seed base (default 1000)")
    args = ap.parse_args(argv)

    if args.record:
        from marrow import record
        record.run_gui(args.out, style=args.style, persona="owner")
        return 0

    if args.generate:
        enrol, test = generate.generate_all(seed_base=args.seed)
        os.makedirs(args.generate, exist_ok=True)
        for style in ("A", "B"):
            for name, data in (("enrol", enrol), ("test", test)):
                ss = [s for s in data if s["header"]["style"] == style]
                path = os.path.join(
                    args.generate, "style-%s-%s.jsonl" % (style.lower(), name))
                write_sessions(path, [Session(s["header"], s["events"])
                                      for s in ss])
                print("wrote %s (%d sessions)" % (path, len(ss)))
        return 0

    if args.train or args.test:
        if not (args.train and args.test):
            print("error: --train and --test must be given together")
            return 2
        return cmd_train_test(args.train, args.test)

    # default: full pipeline on generated fake data
    root = os.path.dirname(os.path.abspath(__file__))
    evaluate.run_full_pipeline(seed_base=args.seed,
                               data_dir=os.path.join(root, "data"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
