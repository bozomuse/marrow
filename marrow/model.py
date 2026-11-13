"""One-class model: diagonal Mahalanobis (sum of squared z-scores).

Enrolment is in two states, fresh and tired, because evening fatigue
must not be mistaken for a stranger. A session is scored against both
state profiles and takes the better (lower) score.

THRESHOLD RULE (frozen from enrolment alone, never tuned on test data):
  1. For every enrolment session, compute its leave-one-out score:
     rebuild the profile of its own state from the OTHER enrolment
     sessions of that state, and score the session against it.
  2. threshold = mean(LOO scores) + 3 * std(LOO scores).
The threshold is a single number, computed before any test session is
looked at, and printed in the report with the LOO mean and std that
produced it.

Per-feature standard deviations get a floor so that a feature that
happens to be nearly constant in enrolment cannot produce an enormous
z-score from ordinary sampling noise. The floor reflects the sampling
noise of the statistic itself:
  * correlation features (indices 9..12) are Pearson correlations
    computed from ~15-20 points per session, whose sampling std is
    roughly 0.2 even when the true correlation is 0 -- so a floor
    of 0.15. Without it, a feature that is exactly 0.0 in most
    enrolment sessions (e.g. a stranger who never corrects) would
    get a std near 0.01 and turn one ordinary session into a
    z-score of 30.
  * all other features: std_floor = 0.05 * |mean| + 0.02.
"""

import math

from .features import FEATURE_NAMES, extract_features

# indices of the correlation (joint) features in FEATURE_NAMES
_CORR_FEATURES = frozenset((9, 10, 11, 12))
_CORR_STD_FLOOR = 0.15


class Profile(object):
    """Per-feature mean/std for one enrolment state."""

    def __init__(self, feature_rows):
        self.n = len(feature_rows)
        self.means = []
        self.stds = []
        for j in range(len(FEATURE_NAMES)):
            col = [row[j] for row in feature_rows]
            m = sum(col) / len(col)
            if len(col) > 1:
                var = sum((v - m) ** 2 for v in col) / (len(col) - 1)
                s = math.sqrt(var)
            else:
                s = 0.0
            floor = 0.05 * abs(m) + 0.02
            if j in _CORR_FEATURES:
                floor = max(floor, _CORR_STD_FLOOR)
            self.means.append(m)
            self.stds.append(max(s, floor))

    def score(self, feats):
        return sum(((f - m) / s) ** 2
                   for f, m, s in zip(feats, self.means, self.stds))


class MarrowModel(object):
    """Two-state one-class model with a frozen threshold."""

    def __init__(self):
        self.profiles = {}      # state -> Profile
        self.threshold = None
        self.loo_mean = None
        self.loo_std = None
        self.loo_max = None

    def enrol(self, sessions_by_state):
        """sessions_by_state: {"fresh": [sessions], "tired": [sessions]}.

        Builds both state profiles, then freezes the threshold from
        leave-one-out enrolment scores.
        """
        rows_by_state = {}
        for state, sessions in sessions_by_state.items():
            rows = [extract_features(s) for s in sessions]
            rows_by_state[state] = rows
            self.profiles[state] = Profile(rows)

        loo_scores = []
        for state, rows in rows_by_state.items():
            for i, row in enumerate(rows):
                others = rows[:i] + rows[i + 1:]
                if not others:
                    continue
                loo_scores.append(Profile(others).score(row))
        if loo_scores:
            self.loo_mean = sum(loo_scores) / len(loo_scores)
            var = sum((v - self.loo_mean) ** 2 for v in loo_scores) / len(loo_scores)
            self.loo_std = math.sqrt(var)
            self.loo_max = max(loo_scores)
            self.threshold = self.loo_mean + 3.0 * self.loo_std
        return self

    def score(self, session):
        """Lower is more like the enrolled user. Min over states."""
        feats = extract_features(session)
        return min(p.score(feats) for p in self.profiles.values())

    def accepts(self, session):
        return self.score(session) < self.threshold
