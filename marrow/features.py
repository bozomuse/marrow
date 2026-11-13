"""Feature extraction: one session -> one feature vector.

All features are RATIOS or correlations, never absolute timings, so a
user who slows down with fatigue keeps nearly the same features. The
vector has 13 entries, in a fixed order (see FEATURE_NAMES):

Marginal features (what the user does on average):
  0  hover_move_ratio        median hover-before-click / median move time
  1  correction_rate         corrections per text event
  2  pause_ratio             median pause between text bursts /
                             median interval inside a burst
  3  dir_change_rate         mean direction changes per move step
  4  click_move_latency_ratio median click->next-pointer-move latency /
                             median move time (only actions where the
                             pointer moves next; a text burst after a
                             click would bury the latency under its pause)
  5  move_speed_cv           std/mean of move speeds
  6  burst_len               mean text events per burst
  7  pause_cv                std/mean of pauses between bursts
  8  hover_cv                std/mean of hovers before clicks

Joint features (how the quantities hang together inside a session).
A mimic who copies the marginal means still fails these, because
copying means does not copy correlations:
  9  corr_hover_move         Pearson(hover, move duration) over clicks
  10 corr_pause_burstlen     Pearson(pause before burst, burst length)
                             over bursts
  11 corr_pause_correct      Pearson(pause before burst, corrections
                             in that burst) over bursts
  12 corr_dur_wobble         Pearson(move duration, RMS perpendicular
                             deviation of the pointer path) over moves
"""

import math

FEATURE_NAMES = [
    "hover_move_ratio",
    "correction_rate",
    "pause_ratio",
    "dir_change_rate",
    "click_move_latency_ratio",
    "move_speed_cv",
    "burst_len",
    "pause_cv",
    "hover_cv",
    "corr_hover_move",
    "corr_pause_burstlen",
    "corr_pause_correct",
    "corr_dur_wobble",
]

BURST_GAP_MS = 600        # gap between text events that starts a new burst
DIR_CHANGE_RAD = 0.26     # ~15 degrees


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))


def _cv(xs):
    m = _mean(xs)
    return _std(xs) / m if m else 0.0


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sx, sy = _std(xs), _std(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (n - 1)
    return max(-1.0, min(1.0, cov / (sx * sy)))


def _angle_change(p0, p1, p2):
    """Angle (radians) between segments p0->p1 and p1->p2."""
    ax, ay = p1[0] - p0[0], p1[1] - p0[1]
    bx, by = p2[0] - p1[0], p2[1] - p1[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na == 0 or nb == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.acos(cosang)


def extract_features(session):
    """Compute the 12-feature vector for one session (dict or Session)."""
    events = session["events"] if isinstance(session, dict) else session.events

    # ---- parse moves and clicks ----------------------------------------
    # A move is a run of consecutive move events; a click ends it.
    # Latency is click -> the very next input event of any kind, i.e.
    # the reaction time to the next action, not the time until the
    # pointer moves again (a text burst may come in between).
    moves = []          # (duration_ms, path_len, n_dir_changes,
                        #  n_points, rms_perp_deviation)
    clicks = []         # (hover_ms, move_duration_ms)
    latencies = []      # click -> next event of any kind, ms
    cur_points = []
    cur_start = None
    last_move_t = None
    for i, ev in enumerate(events):
        typ = ev.get("type")
        if typ == "move":
            if cur_start is None:
                cur_start = ev["t"]
            cur_points.append((ev["x"], ev["y"]))
            last_move_t = ev["t"]
        elif typ == "click":
            if len(cur_points) >= 2 and cur_start is not None:
                dur = last_move_t - cur_start
                path = 0.0
                n_dir = 0
                for k in range(1, len(cur_points)):
                    path += math.hypot(cur_points[k][0] - cur_points[k - 1][0],
                                       cur_points[k][1] - cur_points[k - 1][1])
                for k in range(1, len(cur_points) - 1):
                    if _angle_change(cur_points[k - 1], cur_points[k],
                                     cur_points[k + 1]) > DIR_CHANGE_RAD:
                        n_dir += 1
                # RMS perpendicular deviation from the straight line
                # start -> end of the move
                x0, y0 = cur_points[0]
                x1, y1 = cur_points[-1]
                dx, dy = x1 - x0, y1 - y0
                seg_len = math.hypot(dx, dy)
                if seg_len > 0:
                    devs = []
                    for (px, py) in cur_points:
                        devs.append(abs((py - y0) * dx - (px - x0) * dy) / seg_len)
                    rms_dev = math.sqrt(sum(d * d for d in devs) / len(devs))
                else:
                    rms_dev = 0.0
                moves.append((max(1.0, float(dur)), path, n_dir,
                              len(cur_points), rms_dev))
                hover = ev["t"] - last_move_t
                clicks.append((max(0.0, float(hover)), max(1.0, float(dur))))
            cur_points = []
            cur_start = None
            if i + 1 < len(events) and events[i + 1].get("type") == "move":
                # reaction latency: only measurable when the pointer
                # moves next (a text burst after the click would bury
                # it under the burst's pause)
                latencies.append(events[i + 1]["t"] - ev["t"])

    # ---- text bursts ---------------------------------------------------
    bursts = []         # (pause_before_ms, n_text, n_corrections)
    text_events = [ev for ev in events if ev.get("type") == "text"]
    correct_events = [ev for ev in events if ev.get("type") == "correct"]
    cur_burst = None
    last_text_t = None
    for ev in text_events:
        if last_text_t is None or ev["t"] - last_text_t > BURST_GAP_MS:
            if cur_burst is not None:
                bursts.append(cur_burst)
            pause = (ev["t"] - last_text_t) if last_text_t is not None else 0.0
            cur_burst = {"pause": pause, "n_text": 0, "n_corr": 0,
                         "first_t": ev["t"], "last_t": ev["t"]}
        cur_burst["n_text"] += 1
        cur_burst["last_t"] = ev["t"]
        last_text_t = ev["t"]
    if cur_burst is not None:
        bursts.append(cur_burst)
    for ev in correct_events:
        # attach to the burst whose text window most recently ended
        for b in reversed(bursts):
            if ev["t"] >= b["last_t"]:
                b["n_corr"] += ev.get("n", 1)
                break

    # within-burst intervals
    within = []
    prev = None
    for ev in text_events:
        if prev is not None and ev["t"] - prev <= BURST_GAP_MS:
            within.append(ev["t"] - prev)
        prev = ev["t"]

    # ---- assemble the features -----------------------------------------
    hovers = [c[0] for c in clicks]
    move_durs = [m[0] for m in moves]
    speeds = [m[1] / m[0] for m in moves if m[0] > 0]
    dir_rates = [m[2] / max(1, m[3] - 2) for m in moves]
    rms_devs = [m[4] for m in moves]
    pauses = [b["pause"] for b in bursts[1:]]
    burst_lens = [b["n_text"] for b in bursts]

    hover_move_ratio = (_median(hovers) / _median(move_durs)) if hovers and move_durs else 0.0
    correction_rate = (len(correct_events) / float(len(text_events))) if text_events else 0.0
    pause_ratio = (_median(pauses) / _median(within)) if pauses and within else 1.0
    dir_change_rate = _mean(dir_rates) if dir_rates else 0.0
    click_move_latency_ratio = (_median(latencies) / _median(move_durs)) if latencies and move_durs else 0.0

    feats = [
        hover_move_ratio,
        correction_rate,
        pause_ratio,
        dir_change_rate,
        click_move_latency_ratio,
        _cv(speeds),
        _mean(burst_lens),
        _cv(pauses),
        _cv(hovers),
        _pearson(hovers, [c[1] for c in clicks]),
        _pearson([b["pause"] for b in bursts[1:]],
                 [b["n_text"] for b in bursts[1:]]),
        _pearson([b["pause"] for b in bursts[1:]],
                 [b["n_corr"] for b in bursts[1:]]),
        _pearson(move_durs, rms_devs),
    ]
    return feats
