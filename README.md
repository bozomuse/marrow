# Marrow

Decide whether the person at the computer is the enrolled user, from
OS-level input events only: pointer movement and clicks, plus text as
it arrives (physical keyboard, on-screen keyboard, or dictation
software, including the pattern of corrections). No audio, no camera,
no assumption about which input device is in use.

Built for 1F916 grant 1f9b4 / listing 65 ("I can't use a fingerprint.
Help me prove I'm me", sponsor whetstone).

## Run it

```
python3 marrow.py
```

One command, from a fresh copy. Pure Python 3 standard library, no
installs, works offline. It generates fake data, enrols, runs the swap
test, and prints the report (about 2 seconds).

Other modes:

```
python3 marrow.py --train enrol.jsonl --test test.jsonl
    Train on owner sessions in enrol.jsonl (session headers with
    persona owner-fresh / owner-tired) and score every session in
    test.jsonl. Prints accept/reject per session plus a tally.

python3 marrow.py --generate data/
    Write the fake dataset to data/ as JSONL recordings and stop.

python3 marrow.py --record [--out recording.jsonl] [--style local]
    Private on-machine recording kit: opens a small window, records
    pointer moves/clicks and text (typed vs dictated/inserted) inside
    it, and saves a JSONL recording in the format below. Local only,
    nothing leaves the machine. Needs tkinter (python3-tk on
    Debian/Ubuntu; stock on macOS and Windows). Then score your own
    recording with --train/--test.

python3 marrow.py --seed 2000
    Same pipeline with a different deterministic seed.
```

## Private recording kit

`python3 marrow.py --record` opens a window with a text area. Type,
dictate, or paste inside it; pointer moves and clicks in the window
are recorded too. Closing the window saves `recording.jsonl` (or
`--out FILE`). Everything stays on your machine: the kit makes no
network calls, has no account, and sends nothing anywhere.

How text arrival is distinguished: a keypress that produces a
character is `typed`; text that appears without a keypress (dictation
insert, on-screen keyboard, paste) is detected by diffing the widget
contents and recorded as `inserted`; BackSpace is a `typed`
correction and dictated deletions are `inserted` corrections. The
event-to-format logic lives in `marrow/record.py` (`Recorder`) and is
testable without a display; the tkinter layer is a thin binding on
top of it.

## The recording format

A recording file is JSONL, one event per line. A file may hold several
sessions; each session starts with a `session` header event.

Timestamps `t` are integer milliseconds, relative to an arbitrary
session start. Pointer positions `x`/`y` are fractions of the screen
(0.0-1.0), so a recording does not depend on where it was captured.
`text` events say how the text arrived in `source`: `typed` (physical
keyboard), `onscreen` (on-screen keyboard), `inserted` (inserted by
software such as dictation). `correct` events are corrections. No
audio, no camera, no device identifiers anywhere.

Small example (from `data/style-a-enrol.jsonl`):

```json
{"type": "session", "id": "A-owner-fresh-1000", "style": "A", "persona": "owner-fresh", "seed": 1000}
{"t": 37, "type": "move", "x": 0.5048, "y": 0.496}
{"t": 82, "type": "move", "x": 0.5041, "y": 0.4823}
{"t": 410, "type": "move", "x": 0.5311, "y": 0.2389}
{"t": 448, "type": "move", "x": 0.5474, "y": 0.2151}
{"t": 944, "type": "click", "x": 0.556, "y": 0.1773, "button": "left"}
{"t": 1759, "type": "text", "source": "typed", "kind": "char", "text": "l"}
{"t": 1853, "type": "text", "source": "typed", "kind": "char", "text": "t"}
{"t": 15450, "type": "correct", "source": "typed", "kind": "backspace", "n": 1}
```

Event types:

| type | fields | meaning |
|---|---|---|
| `session` | `id`, `style`, `persona`, `seed` | header starting a session |
| `move` | `t`, `x`, `y` | pointer position sample |
| `click` | `t`, `x`, `y`, `button` | pointer click |
| `text` | `t`, `source`, `kind`, `text` | text arrived (`kind`: `char`/`word`) |
| `correct` | `t`, `source`, `kind`, `n` | correction (`kind`: `backspace`/`delete-word`) |

## The method

**Ratio-invariant features.** Absolute timings drift with fatigue, but
ratios stay stable. Each session becomes a 13-feature vector:

Marginal (what the user does on average):
hover/move ratio, correction rate, pause ratio between text bursts,
direction-change rate per move step, click-to-next-move latency ratio,
move-speed variability, burst length, pause variability, hover
variability.

Joint (how the quantities hang together inside a session):
correlation of hover with preceding move duration; correlation of the
pause before a text burst with that burst's length; correlation of the
pause before a burst with corrections in it; correlation of move
duration with the pointer path's RMS deviation from a straight line.

**One-class model.** From enrolment sessions, estimate per-feature mean
and standard deviation; score a session as the sum of squared z-scores
(diagonal Mahalanobis). Enrolment is in two states, fresh and tired,
so evening fatigue is not mistaken for a stranger; a session is scored
against both states and takes the better (lower) score.

**Mimic defense.** The joint features are the point. A mimic who
copies the owner's average hover, average correction rate, and average
move time still fails, because copying means does not copy
correlations: in the owner's data a long move is followed by a long
hover, a long pause before a burst means a longer burst and more
corrections, and a long move wanders further off a straight line. The
mimic samples each quantity independently, so all four joint
correlations sit near zero while the owner's sit at 0.5-0.9.

**Threshold rule (exact).** The threshold is frozen from enrolment
alone and never touched after test data is seen:
1. For every enrolment session, compute its leave-one-out score:
   rebuild the profile of its own state from the OTHER enrolment
   sessions of that state, and score the session against it.
2. `threshold = mean(LOO scores) + 3 * std(LOO scores)`.
The report prints the threshold with the LOO mean, std, and max that
produced it. Per-feature standard deviations have a floor reflecting
the sampling noise of the statistic (0.15 for correlation features,
`0.05 * |mean| + 0.02` otherwise), so a feature that is nearly
constant in enrolment cannot manufacture a huge z-score.

## How to read the report

```
Style A: threshold 32.63  (LOO mean 11.27, std 7.12, max 37.44)
  FRR fresh owner : 6.2%  (2/32 rejected)
  FRR tired owner : 0.0%  (0/32 rejected)
  FAR stranger    : 0.0%  (0/32 accepted)
  FAR mimic       : 0.0%  (0/32 accepted)
```

- **FRR** (false reject rate): how often the real user is doubted.
  Reported separately for fresh and tired owner sessions.
- **FAR** (false accept rate): how often a stranger, or a mimic who
  copied the owner's habits, is let in. Reported separately.
- **SWAP TEST**: a profile trained on the stranger's sessions, scored
  against owner sessions. Opens must be 0: a profile trained on
  someone else must not open for the owner.
- The acceptance-criteria block at the end gives a PASS/FAIL per
  criterion from the listing.

## The fake data

`marrow/generate.py` ships two input styles, each with four personas,
all deterministic from a seed:

- **Style A**: standard mouse + keyboard. Smooth ballistic moves,
  short hovers, typed keystrokes, occasional backspaces.
- **Style B**: slow pointer (lip joystick / head mouse) + dictation.
  Slow stepwise moves, long hovers, bursts of inserted words with
  dictation-style corrections (delete word, re-insert).

Personas: `owner-fresh`, `owner-tired` (same user, slower, more
variable, more corrections, ratios intact), `stranger` (different
base parameters), `mimic` (copies the owner's marginal means, no
joint structure). The default run enrols on 24 fresh + 24 tired owner
sessions per style and tests on 32 sessions per persona per style.

## Layout

```
marrow.py            entry point + CLI
marrow/
  format.py          recording format: parse/write JSONL sessions
  generate.py        fake-data generators (styles A/B, 4 personas)
  features.py        session -> 13-feature vector
  model.py           one-class profile, scoring, threshold rule
  evaluate.py        enrol/test/swap-test pipeline + report
data/                sample recordings written by the default run
README.md            this file
BUILD_NOTES.md       measured numbers, limitations
```
