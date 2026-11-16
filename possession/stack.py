"""Layer 3b: stage-2 "auto-context" stacker.

The stage-1 models (models.py) classify every frame from a 1-7 s neighbourhood of tracking features.
spellF1 is decided by where possessions START and END, and that needs much longer context: a
possession is ~14 s on average and the challenge ball is often missing for 10-30 s. The stacker is
a second pair of LightGBM models whose inputs are the stage-1 log-odds summarised over centred,
past-only and future-only windows of 1, 3, 7, 15 and 30 s, plus how long the ball has been unseen.
It learns, from labelled external games, patterns such as "home log-odds positive for the past 15 s
and negative for the next 15 s = a turnover here" that a per-frame model and a constant-cost
decoder cannot express.

Training is nested so the stacker only ever sees OUT-OF-FOLD stage-1 predictions (see
run_pipeline.train). Team symmetry is kept exactly as in stage 1: the control stacker is trained on
both team orientations (home log-odds negated, label swapped) and its two predictions are averaged.
"""

import json
import os

import numpy as np

from .features import _roll
from .labels import A, D, H, X

CTX = (25, 75, 175, 375, 750)            # 1, 3, 7, 15, 30 s
AUX = ("c_obs", "c_since_obs", "c_until_obs", "c_gap", "c_obs_m175", "c_n_tracked")
STACK_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500,
    feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
    verbosity=-1, num_threads=0,
)
STACK_ROUNDS = 250
STACK_STRIDE = 4
EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def aux_from_bundle(bundle):
    """The tracking-quality columns the stacker uses, taken from a stage-1 feature bundle."""
    return np.stack([bundle.Xc[:, bundle.names_c.index(n)] for n in AUX], axis=1).astype(np.float32)


def stack_features(p_dead, p_home, period, aux, sign=1.0):
    """(T, F) float32. sign=-1 swaps the team roles (home log-odds negated)."""
    play = period > 0
    ld = np.where(play, _logit(p_dead), np.nan)
    lh = np.where(play, sign * _logit(p_home), np.nan)
    cols = [ld, lh]
    for w in CTX:
        h = w // 2
        for series in (ld, lh):
            cols.append(_roll(series, -h, h))        # centred
            cols.append(_roll(series, -w, 0))        # past only
            cols.append(_roll(series, 0, w))         # future only
    for w in (75, 375):                              # how unsettled the team signal is
        h = w // 2
        m = _roll(lh, -h, h).astype(np.float64)
        m2 = _roll(lh * lh, -h, h).astype(np.float64)
        cols.append(np.sqrt(np.maximum(m2 - m * m, 0.0)))
    X = np.column_stack([np.asarray(c, dtype=np.float32) for c in cols] + [aux.astype(np.float32)])
    return X


def feature_names():
    names = ["ld", "lh"]
    for w in CTX:
        for s in ("ld", "lh"):
            names += ["{}_c{}".format(s, w), "{}_p{}".format(s, w), "{}_f{}".format(s, w)]
    return names + ["lh_std75", "lh_std375"] + ["aux_" + n for n in AUX]


class Stacker:
    def __init__(self, dead=None, ctrl=None):
        self.dead, self.ctrl = dead, ctrl

    @classmethod
    def fit(cls, items, params=None, n_rounds=STACK_ROUNDS, stride=STACK_STRIDE):
        """items: list of dicts {p_dead, p_home, period, aux, codes} with OUT-OF-FOLD stage-1 probs."""
        import lightgbm as lgb
        params = {**STACK_PARAMS, **(params or {})}
        xd, yd, xc, yc = [], [], [], []
        for it in items:
            rows = np.arange(0, len(it["codes"]), stride)
            codes = it["codes"][rows]
            Xp = stack_features(it["p_dead"], it["p_home"], it["period"], it["aux"], 1.0)[rows]
            Xm = stack_features(it["p_dead"], it["p_home"], it["period"], it["aux"], -1.0)[rows]
            graded = codes != X
            xd.append(Xp[graded])
            yd.append((codes[graded] == D).astype(np.float32))
            live = (codes == H) | (codes == A)
            xc.append(Xp[live])
            yc.append((codes[live] == H).astype(np.float32))
            xc.append(Xm[live])
            yc.append((codes[live] == A).astype(np.float32))
        dead = lgb.train(params, lgb.Dataset(np.vstack(xd), np.concatenate(yd)), n_rounds)
        ctrl = lgb.train(params, lgb.Dataset(np.vstack(xc), np.concatenate(yc)), n_rounds)
        return cls(dead, ctrl)

    def predict(self, p_dead, p_home, period, aux):
        """-> refined (p_dead, p_home). Frames outside play keep their stage-1 values."""
        Xp = stack_features(p_dead, p_home, period, aux, 1.0)
        Xm = stack_features(p_dead, p_home, period, aux, -1.0)
        pd2 = self.dead.predict(Xp).astype(np.float64)
        ph2 = 0.5 * (self.ctrl.predict(Xp) + (1.0 - self.ctrl.predict(Xm))).astype(np.float64)
        play = period > 0
        return np.where(play, pd2, p_dead), np.where(play, ph2, p_home)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.dead.save_model(os.path.join(folder, "stack_dead.txt"))
        self.ctrl.save_model(os.path.join(folder, "stack_ctrl.txt"))
        with open(os.path.join(folder, "stack_names.json"), "w") as handle:
            json.dump(feature_names(), handle)

    @classmethod
    def load(cls, folder):
        import lightgbm as lgb
        return cls(lgb.Booster(model_file=os.path.join(folder, "stack_dead.txt")),
                   lgb.Booster(model_file=os.path.join(folder, "stack_ctrl.txt")))
