# Marrow — build notes

## What was built

A one-class behavioural verifier that decides whether the person at
the computer is the enrolled user, from OS-level input events only
(pointer moves/clicks + text as it arrives). Pure Python 3 standard
library, one entry point (`python3 marrow.py`), offline, deterministic.

## The method, in one paragraph

Each session becomes a 13-feature vector of ratios and correlations,
never absolute timings, so fatigue-scale drift does not move the
features. A one-class model (sum of squared z-scores, diagonal
Mahalanobis) is fitted on enrolment sessions in two states, fresh and
tired, and a session takes the better score of the two. The threshold
is frozen from enrolment alone as mean + 3 std of leave-one-out
enrolment scores. The mimic defense is four joint features
(correlations between hover and move duration, pause and burst length,
pause and corrections, move duration and path wobble): a mimic who
copies the owner's average timings still fails because copying means
does not copy correlations.

## Measured numbers

Default run (`python3 marrow.py`, seed 1000; 24 enrolment sessions per
state per style, 32 test sessions per persona per style):

| | style A | style B | overall |
|---|---|---|---|
| threshold (LOO mean / std) | 32.63 (11.27 / 7.12) | 30.76 (11.02 / 6.58) | — |
| FRR owner fresh | 6.2% (2/32) | 6.2% (2/32) | 6.2% |
| FRR owner tired | 0.0% (0/32) | 3.1% (1/32) | 1.6% |
| FRR owner all | — | — | 3.9% (5/128) |
| FAR stranger | 0.0% (0/32) | 0.0% (0/32) | 0.0% |
| FAR mimic | 0.0% (0/32) | 0.0% (0/32) | 0.0% |
| swap test | — | — | 0 opens / 128 |

Multi-seed check (seeds 1000, 2000, 3000, 4000, 5000, 6000; 768 owner
sessions, 384 stranger, 384 mimic in total): FRR fresh 3.4%
(13/384), FRR tired 1.0% (4/384), FRR overall 2.2% (17/768); FAR
stranger 0.0% (0/384); FAR mimic 0.5% (2/384); swap test 0 opens on
all six seeds. Per-seed FRR ranged 0-12.5% per cell, which is the
honest spread for 32-session cells. Nothing here is hardcoded: every
number is computed by the pipeline from generated data, and the
threshold was frozen from enrolment scores before any test session
was scored.

## Things that went wrong during the build (and what fixed them)

1. **Mimic FAR 50%, swap test 27 opens (first run).** Three feature
   bugs: the pause-before-burst measured by the extractor included
   move/click time while the generator keyed corrections off the
   text-only pause, so the pause-correction correlation was zero;
   direction changes correlated with move duration mechanically
   (longer moves have more steps) for everyone, mimic included; and
   click-to-move latency measured the text burst in between, not the
   reaction time. Fixed by keying generator behaviour off the total
   gap since the last text event (exactly what the extractor
   measures), normalising direction changes per step and adding an
   RMS-wobble correlation, and measuring latency only when the
   pointer moves next.
2. **Swap test 64 opens on one seed.** A near-constant feature
   (correlation of pause with corrections for a stranger who almost
   never corrects) had a standard-deviation floor of 0.01, so one
   ordinary session scored a z of 30 and blew up the stranger
   profile's threshold. Fixed with a per-feature std floor that
   reflects the sampling noise of the statistic (0.15 for
   correlations). This is a method fix in the model, not a threshold
   adjustment: the threshold rule (mean + 3 std of LOO scores) is
   unchanged.
3. **Infinite loop in the generator.** The direction-change step
   picker could ask for more steps than a move has. Capped at the
   available steps.

## Limitations (known)

- **Fake data only.** Everything is measured on synthetic sessions
  whose ground truth is the generator's own parameters. The numbers
  prove the method can separate these four personas on this data;
  they say nothing about a real user yet. The private on-machine
  recording kit (`python3 marrow.py --record`, tkinter window, local
  only) is now built so the sponsor can record his own patterns and
  test the prototype on them: its event-to-format logic was verified
  headlessly on a simulated session (typed/inserted/correction
  events land in the right format and score through the model); the
  live GUI itself was not run on this headless machine.
- **Small samples.** 32 test sessions per persona per style: a 0/32
  FAR has a 95% upper bound around 9%. The multi-seed pooled numbers
  (0/384 stranger, 2/384 mimic) are the better estimate.
- **Two styles, one user.** Enrolment is per input style; a user who
  switches between mouse and eye gaze would need a profile per
  setup. Cross-style enrolment is untested.
- **Threshold spread across seeds.** The LOO std (and therefore the
  threshold) varies with the enrolment draw (thresholds 27-52 across
  six seeds). With 24 enrolment sessions per state the profile is
  still noisy; a real deployment wants more enrolment sessions and
  should re-check FRR after enrolment grows.
- **The mimic is a statistical mimic.** It copies marginal means and
  breaks correlations. A mimic that also copies the joint structure
  (e.g. by replaying recorded sessions, or a model trained on the
  owner's data) is a different, harder adversary and is not tested
  here. Replay is out of scope for a behavioural check by design:
  it is what the password side door is for.
- **Drift is two-state, not continuous.** Fresh and tired are two
  enrolled states; slow progressive drift (the sponsor's SMA) is
  approximated by the tired state, not modelled as a trajectory.
  A production version would want periodic re-enrolment.
