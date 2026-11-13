"""Marrow recording format: plain, documented, OS-neutral JSONL.

A recording file is a sequence of JSON objects, one per line. A file may
contain several sessions; each session starts with a ``session`` header
event and is followed by its input events.

Header event::

    {"type": "session", "id": "A-owner-fresh-03", "style": "A",
     "persona": "owner-fresh", "seed": 123}

``persona`` is a free-form label (``owner-fresh``, ``owner-tired``,
``stranger``, ``mimic``, ...). ``style`` is a free-form label for the
input setup (``A``, ``B``, ...). Both are descriptive only; the verifier
never trusts them, it only groups sessions by them.

Input events (all timestamps ``t`` are integer milliseconds, relative to
an arbitrary session start; pointer positions ``x``/``y`` are fractions
of the screen, 0.0-1.0, so a recording does not depend on where it was
captured)::

    {"t": 1234, "type": "move", "x": 0.512, "y": 0.334}
    {"t": 1300, "type": "click", "x": 0.512, "y": 0.334, "button": "left"}
    {"t": 1500, "type": "text", "source": "typed", "kind": "char", "text": "a"}
    {"t": 1600, "type": "text", "source": "inserted", "kind": "word", "text": "hello"}
    {"t": 1700, "type": "correct", "source": "typed", "kind": "backspace", "n": 1}
    {"t": 1800, "type": "correct", "source": "inserted", "kind": "delete-word", "n": 1}

``text`` events carry how the text arrived in ``source``:
``typed`` (physical keyboard), ``onscreen`` (on-screen keyboard),
``inserted`` (inserted by software, e.g. dictation). ``correct`` events
are corrections: backspaces, deleted words, retypes. No audio, no camera,
no device identifiers anywhere in the format.
"""

import json


class Session(object):
    """One recorded session: a header dict plus an ordered event list."""

    def __init__(self, header, events):
        self.header = header
        self.events = events

    @property
    def id(self):
        return self.header.get("id", "?")

    @property
    def style(self):
        return self.header.get("style", "?")

    @property
    def persona(self):
        return self.header.get("persona", "?")


def read_sessions(path):
    """Read a JSONL recording file; return a list of Session objects."""
    sessions = []
    header = None
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "session":
                if header is not None:
                    sessions.append(Session(header, events))
                header = obj
                events = []
            else:
                events.append(obj)
    if header is not None:
        sessions.append(Session(header, events))
    return sessions


def write_sessions(path, sessions):
    """Write Session objects to a JSONL recording file."""
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(s.header) + "\n")
            for ev in s.events:
                f.write(json.dumps(ev) + "\n")
