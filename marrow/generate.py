"""Fake-data generators for Marrow. Deterministic: same seed, same data.

Two input styles, each with four personas:

  style A  standard mouse + keyboard
           pointer: smooth ballistic moves, short hovers, clicks
           text: typed keystrokes, occasional backspaces
  style B  slow pointer (lip joystick / head mouse) + dictation
           pointer: slow stepwise moves, long hovers
           text: bursts of inserted words, dictation-style corrections

Personas per style:
  owner-fresh   the enrolled user, rested
  owner-tired   the same user, fatigued (slower, more variable, more
                corrections). Ratios stay near fresh: that is the point
                of ratio-invariant features.
  stranger      a different person: different base parameters
  mimic         copies the owner's MARGINAL means (same average hover,
                same average correction rate, same average move time)
                but samples every quantity INDEPENDENTLY: no joint
                structure. This is the caregiver who watched thousands
                of sessions and copied the habits, but not the way the
                habits hang together.

The owner (fresh and tired) is generated with real joint structure:
  * hover before a click grows with the move that preceded it
  * bursts preceded by a long pause carry more corrections
  * longer moves wander more (more direction changes)
The mimic matches the marginal means of those quantities and breaks
all three correlations. That is exactly what the joint features in
``features.py`` are built to catch.
"""

import math
import random

WORDS = ("the quick brown fox jumps over lazy dog and then runs home "
         "again while sun sets over quiet hills").split()


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _gauss(rng, mean, std, lo=None, hi=None):
    v = rng.gauss(mean, std)
    if lo is not None or hi is not None:
        v = _clip(v, lo if lo is not None else -1e18,
                  hi if hi is not None else 1e18)
    return v


class PersonaParams(object):
    """Timing parameters for one persona in one style."""

    def __init__(self, **kw):
        self.move_dur_mean = kw["move_dur_mean"]
        self.move_dur_std = kw["move_dur_std"]
        self.hover_base = kw["hover_base"]
        self.hover_per_ms = kw["hover_per_ms"]
        self.hover_noise = kw["hover_noise"]
        self.text_interval_mean = kw["text_interval_mean"]
        self.text_interval_std = kw["text_interval_std"]
        self.burst_len = kw["burst_len"]          # (min, max) text events
        self.pause_mean = kw["pause_mean"]
        self.pause_std = kw["pause_std"]
        self.corr_base = kw["corr_base"]          # P(correction) after a burst
        self.corr_long_pause = kw["corr_long_pause"]
        self.dir_base = kw["dir_base"]            # mean direction changes
        self.dir_per_ms = kw["dir_per_ms"]
        self.latency_mean = kw["latency_mean"]    # click -> next move, ms
        self.latency_std = kw["latency_std"]
        self.wobble_base = kw["wobble_base"]      # perpendicular wobble, screen frac
        self.wobble_per_ms = kw["wobble_per_ms"]
        self.gap_mean = kw["gap_mean"]            # mean total gap since last
        self.gap_std = kw["gap_std"]              #   text event, ms
        self.gap_long = kw["gap_long"]            # "long gap" threshold, ms
        self.burst_gap_slope = kw["burst_gap_slope"]  # extra text events
        self.correlated = kw.get("correlated", True)  #   per +1 sd of gap


# ---------------------------------------------------------------- style A
_A_OWNER = PersonaParams(
    move_dur_mean=480, move_dur_std=110,
    hover_base=60, hover_per_ms=0.55, hover_noise=30,
    text_interval_mean=150, text_interval_std=45,
    burst_len=(5, 15), pause_mean=900, pause_std=350,
    corr_base=0.08, corr_long_pause=0.30,
    dir_base=1.0, dir_per_ms=0.004,
    latency_mean=180, latency_std=60,
    wobble_base=0.003, wobble_per_ms=0.000008,
    gap_mean=1790, gap_std=380, gap_long=2000, burst_gap_slope=3.0,
    correlated=True)

_A_STRANGER = PersonaParams(
    move_dur_mean=300, move_dur_std=80,
    hover_base=30, hover_per_ms=0.15, hover_noise=25,
    text_interval_mean=95, text_interval_std=30,
    burst_len=(10, 25), pause_mean=550, pause_std=200,
    corr_base=0.01, corr_long_pause=0.02,
    dir_base=0.5, dir_per_ms=0.001,
    latency_mean=90, latency_std=40,
    wobble_base=0.0016, wobble_per_ms=0.000002,
    gap_mean=1100, gap_std=300, gap_long=1300, burst_gap_slope=0.0,
    correlated=False)

# ---------------------------------------------------------------- style B
_B_OWNER = PersonaParams(
    move_dur_mean=2200, move_dur_std=500,
    hover_base=400, hover_per_ms=0.60, hover_noise=100,
    text_interval_mean=55, text_interval_std=20,
    burst_len=(3, 8), pause_mean=2400, pause_std=800,
    corr_base=0.15, corr_long_pause=0.40,
    dir_base=2.0, dir_per_ms=0.002,
    latency_mean=600, latency_std=250,
    wobble_base=0.006, wobble_per_ms=0.000004,
    gap_mean=6480, gap_std=1030, gap_long=7000, burst_gap_slope=1.5,
    correlated=True)

_B_STRANGER = PersonaParams(
    move_dur_mean=1400, move_dur_std=350,
    hover_base=200, hover_per_ms=0.20, hover_noise=100,
    text_interval_mean=40, text_interval_std=15,
    burst_len=(6, 14), pause_mean=1500, pause_std=500,
    corr_base=0.02, corr_long_pause=0.03,
    dir_base=1.0, dir_per_ms=0.001,
    latency_mean=300, latency_std=120,
    wobble_base=0.003, wobble_per_ms=0.000001,
    gap_mean=4200, gap_std=800, gap_long=4800, burst_gap_slope=0.0,
    correlated=False)


def _mimic_params(owner):
    """Marginal-matched copy of the owner: same means, no correlations.

    hover mean/std from the owner's linear rule, direction-change mean
    from the owner's rule at the mean move duration, correction rate
    from the owner's average long-pause fraction (~0.3).
    """
    hover_mean = owner.hover_base + owner.hover_per_ms * owner.move_dur_mean
    hover_std = math.sqrt((owner.hover_per_ms * owner.move_dur_std) ** 2
                          + owner.hover_noise ** 2)
    return PersonaParams(
        move_dur_mean=owner.move_dur_mean, move_dur_std=owner.move_dur_std,
        hover_base=hover_mean, hover_per_ms=0.0, hover_noise=hover_std,
        text_interval_mean=owner.text_interval_mean,
        text_interval_std=owner.text_interval_std,
        burst_len=owner.burst_len,
        pause_mean=owner.pause_mean, pause_std=owner.pause_std,
        corr_base=owner.corr_base + 0.5 * owner.corr_long_pause,
        corr_long_pause=0.0,
        dir_base=owner.dir_base + owner.dir_per_ms * owner.move_dur_mean,
        dir_per_ms=0.0,
        latency_mean=owner.latency_mean, latency_std=owner.latency_std,
        wobble_base=owner.wobble_base + owner.wobble_per_ms * owner.move_dur_mean,
        wobble_per_ms=0.0,
        gap_mean=owner.gap_mean, gap_std=owner.gap_std,
        gap_long=owner.gap_long, burst_gap_slope=0.0,
        correlated=False)


def _tired_params(owner):
    """Same user, fatigued: everything slower and noisier, ratios intact."""
    return PersonaParams(
        move_dur_mean=owner.move_dur_mean * 1.5,
        move_dur_std=owner.move_dur_std * 1.4,
        hover_base=owner.hover_base * 1.35,
        hover_per_ms=owner.hover_per_ms,
        hover_noise=owner.hover_noise * 1.5,
        text_interval_mean=owner.text_interval_mean * 1.6,
        text_interval_std=owner.text_interval_std * 1.5,
        burst_len=owner.burst_len,
        pause_mean=owner.pause_mean * 1.3,
        pause_std=owner.pause_std * 1.4,
        corr_base=min(0.9, owner.corr_base * 2.0),
        corr_long_pause=owner.corr_long_pause,
        dir_base=owner.dir_base, dir_per_ms=owner.dir_per_ms,
        latency_mean=owner.latency_mean * 1.5,
        latency_std=owner.latency_std * 1.5,
        wobble_base=owner.wobble_base * 1.2,
        wobble_per_ms=owner.wobble_per_ms,
        gap_mean=owner.gap_mean * 1.4, gap_std=owner.gap_std * 1.4,
        gap_long=owner.gap_long * 1.4,
        burst_gap_slope=owner.burst_gap_slope,
        correlated=True)


def params_for(style, persona):
    if style == "A":
        owner, stranger = _A_OWNER, _A_STRANGER
    elif style == "B":
        owner, stranger = _B_OWNER, _B_STRANGER
    else:
        raise ValueError("unknown style %r" % style)
    if persona == "owner-fresh":
        return owner
    if persona == "owner-tired":
        return _tired_params(owner)
    if persona == "stranger":
        return stranger
    if persona == "mimic":
        return _mimic_params(owner)
    raise ValueError("unknown persona %r" % persona)


# ---------------------------------------------------------------- session

def generate_session(style, persona, seed, n_actions=20):
    """Generate one fake session in the documented recording format.

    Returns a dict {"header": ..., "events": [...]}.
    """
    rng = random.Random(seed)
    p = params_for(style, persona)
    t = 0
    x, y = 0.5, 0.5
    last_text_t = None
    events = []

    text_source = "typed" if style == "A" else "inserted"
    text_kind = "char" if style == "A" else "word"
    step_ms = 40 if style == "A" else 150

    for _ in range(n_actions):
        # --- move to a new target -------------------------------------
        tx = _clip(rng.gauss(0.5, 0.22), 0.05, 0.95)
        ty = _clip(rng.gauss(0.5, 0.22), 0.05, 0.95)
        dur = max(60.0, _gauss(rng, p.move_dur_mean, p.move_dur_std, 60, 8000))
        n_steps = max(4, int(dur / step_ms))
        wobble = p.wobble_base + p.wobble_per_ms * dur
        # direction-change target: owner wanders more on long moves,
        # mimic wanders the same amount on average, independent of dur
        dir_mean = p.dir_base + p.dir_per_ms * dur
        n_dir = rng.randint(0, max(0, int(dir_mean * 2))) if dir_mean > 0 else 0
        n_dir = min(n_dir, max(0, n_steps - 1))  # cannot exceed step slots
        dir_steps = set()
        while len(dir_steps) < n_dir and n_steps > 4:
            dir_steps.add(rng.randint(1, n_steps - 1))
        px, py = x, y
        perp_x, perp_y = -(ty - y), (tx - x)
        plen = math.hypot(perp_x, perp_y) or 1.0
        perp_x, perp_y = perp_x / plen, perp_y / plen
        sign = 1.0
        for i in range(1, n_steps + 1):
            frac = i / float(n_steps)
            # smoothstep easing: accelerate, then decelerate
            e = frac * frac * (3 - 2 * frac)
            if i in dir_steps:
                sign = -sign
            off = sign * rng.gauss(0, wobble)
            nx = x + (tx - x) * e + perp_x * off
            ny = y + (ty - y) * e + perp_y * off
            t += max(5, int(dur / n_steps) + rng.randint(-5, 5))
            events.append({"t": t, "type": "move",
                           "x": round(_clip(nx, 0.0, 1.0), 4),
                           "y": round(_clip(ny, 0.0, 1.0), 4)})
            px, py = nx, ny
        x, y = tx, ty

        # --- hover, then click ----------------------------------------
        if p.correlated:
            hover = p.hover_base + p.hover_per_ms * dur + rng.gauss(0, p.hover_noise)
        else:
            hover = rng.gauss(p.hover_base, p.hover_noise)
        hover = max(20.0, hover)
        t += int(hover)
        events.append({"t": t, "type": "click", "x": round(x, 4),
                       "y": round(y, 4), "button": "left"})

        # reaction latency after every click, before whatever comes next
        t += max(20, int(_gauss(rng, p.latency_mean, p.latency_std, 20, 5000)))

        # --- text burst, sometimes ------------------------------------
        if rng.random() < 0.7:
            pause = max(150.0, _gauss(rng, p.pause_mean, p.pause_std, 150, 9000))
            t += int(pause)
            # total gap since the last text event: exactly what the
            # feature extractor measures as the pause before a burst.
            # The owner's behaviour keys off this gap; the mimic's
            # does not.
            if last_text_t is None:
                total_gap = p.gap_mean
            else:
                total_gap = t - last_text_t
            long_gap = total_gap > p.gap_long
            if p.correlated:
                base_len = rng.randint(*p.burst_len)
                n_text = base_len + int(round(
                    p.burst_gap_slope * (total_gap - p.gap_mean) / p.gap_std))
                n_text = max(2, min(30, n_text))
            else:
                n_text = rng.randint(*p.burst_len)
            for j in range(n_text):
                t += max(5, int(_gauss(rng, p.text_interval_mean,
                                       p.text_interval_std, 5, 2000)))
                word = WORDS[rng.randrange(len(WORDS))]
                events.append({"t": t, "type": "text", "source": text_source,
                               "kind": text_kind,
                               "text": word if text_kind == "word" else word[0]})
            last_text_t = t
            # corrections: owner corrects more after a long gap;
            # mimic corrects at the marginal rate, regardless of gap
            if p.correlated:
                p_corr = p.corr_base + (p.corr_long_pause if long_gap else 0.0)
            else:
                p_corr = p.corr_base
            if rng.random() < p_corr:
                n_corr = rng.randint(1, 3)
                kind = "backspace" if style == "A" else "delete-word"
                for _ in range(n_corr):
                    t += rng.randint(30, 150)
                    events.append({"t": t, "type": "correct",
                                   "source": text_source, "kind": kind, "n": 1})
                # retype / re-insert the corrected material
                for _ in range(n_corr):
                    t += max(5, int(_gauss(rng, p.text_interval_mean,
                                           p.text_interval_std, 5, 2000)))
                    word = WORDS[rng.randrange(len(WORDS))]
                    events.append({"t": t, "type": "text", "source": text_source,
                                   "kind": text_kind,
                                   "text": word if text_kind == "word" else word[0]})
                last_text_t = t

    header = {"type": "session",
              "id": "%s-%s-%04d" % (style, persona, seed),
              "style": style, "persona": persona, "seed": seed}
    return {"header": header, "events": events}


def generate_all(seed_base=1000, n_enrol=16, n_test=24):
    """Generate the full fake dataset.

    Returns (enrol, test): lists of session dicts. Enrolment sessions
    are owner-fresh and owner-tired only; test sessions cover all four
    personas. Seeds are fixed, so the dataset is reproducible.
    """
    enrol, test = [], []
    seed = seed_base
    for style in ("A", "B"):
        for persona in ("owner-fresh", "owner-tired"):
            for _ in range(n_enrol):
                enrol.append(generate_session(style, persona, seed))
                seed += 1
        for persona in ("owner-fresh", "owner-tired", "stranger", "mimic"):
            for _ in range(n_test):
                test.append(generate_session(style, persona, seed))
                seed += 1
    return enrol, test
