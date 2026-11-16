"""Dead-ball and control models (LightGBM), feature bundles, leave-one-game-out helpers."""

import json
import os
from dataclasses import dataclass

import numpy as np

from .ball import fill_ball
from .features import compute_common, control_features, dead_features
from .labels import A, D, H, X, build_labels
from .masking import MaskConfig, apply_masks

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
    verbosity=-1, num_threads=0,
)
N_ROUNDS = 350
SWITCH_HALO = 12          # frames (~0.5 s) either side of a direct H <-> A change count as "turnover"


def switch_targets(codes, halo=SWITCH_HALO):
    """(T,) float32: 1 within `halo` frames of a direct H<->A change (no dead ball in between)."""
    live = (codes == H) | (codes == A)
    change = np.zeros(len(codes), dtype=np.float32)
    idx = np.flatnonzero(live[1:] & live[:-1] & (codes[1:] != codes[:-1])) + 1
    change[idx] = 1.0
    return (np.convolve(change, np.ones(2 * halo + 1, dtype=np.float32), mode="same") > 0).astype(np.float32)


@dataclass
class Bundle:
    name: str
    seed: int
    rows: np.ndarray          # frame indices kept
    Xc: np.ndarray            # dead / common features
    Xo01: np.ndarray          # control features, team 0 as "own"
    Xo10: np.ndarray          # control features, team 1 as "own"
    codes: np.ndarray         # labels at `rows` (None for the challenge)
    period: np.ndarray        # period at `rows`
    names_c: list
    names_o: list
    switch: np.ndarray = None # 1 where a direct H<->A change is within SWITCH_HALO frames (training only)


def prepare(match, mask_cfg=None, seed=0, stride=1, apply_mask=True, with_labels=True, fcfg=None):
    """Mask (optionally), bridge the ball, build features. Rows are subsampled by `stride`."""
    mm = apply_masks(match, mask_cfg or MaskConfig(), seed) if apply_mask else match
    filled, observed = fill_ball(mm.ball, pitch=mm.pitch, fps=mm.fps)
    com = compute_common(mm, filled, observed, fcfg)
    Xc, nc = dead_features(com)
    Xo01, no = control_features(com, 0, 1)
    Xo10, _ = control_features(com, 1, 0)
    rows = np.arange(0, match.n_frames, stride)
    has_labels = match.events is not None or getattr(match, "labels", None) is not None
    full = build_labels(match) if (with_labels and has_labels) else None
    codes = full[rows] if full is not None else None
    switch = switch_targets(full)[rows] if full is not None else None
    return Bundle(match.name, seed, rows, Xc[rows], Xo01[rows], Xo10[rows], codes,
                  match.period[rows], nc, no, switch)


class PossessionModels:
    def __init__(self, dead=None, ctrl=None, names_c=None, names_o=None, switch=None):
        self.dead, self.ctrl, self.names_c, self.names_o = dead, ctrl, names_c, names_o
        self.switch = switch       # optional turnover detector (None for models saved before it existed)

    # -- training --------------------------------------------------------------------------
    @classmethod
    def fit(cls, bundles, n_rounds=N_ROUNDS, params=None, with_switch=True):
        import lightgbm as lgb
        params = {**LGB_PARAMS, **(params or {})}
        xd, yd, xc, yc = [], [], [], []
        for b in bundles:
            graded = b.codes != X
            xd.append(b.Xc[graded])
            yd.append((b.codes[graded] == D).astype(np.float32))
            live = (b.codes == H) | (b.codes == A)
            xc.append(np.hstack([b.Xc[live], b.Xo01[live]]))
            yc.append((b.codes[live] == H).astype(np.float32))      # team 0 in control
            xc.append(np.hstack([b.Xc[live], b.Xo10[live]]))
            yc.append((b.codes[live] == A).astype(np.float32))      # swapped roles, swapped label
        dead = lgb.train(params, lgb.Dataset(np.vstack(xd), np.concatenate(yd)), n_rounds)
        ctrl = lgb.train(params, lgb.Dataset(np.vstack(xc), np.concatenate(yc)), n_rounds)
        switch = None
        if with_switch and all(b.switch is not None for b in bundles):
            xs, ys = [], []
            for b in bundles:
                graded = b.codes != X
                for xo in (b.Xo01, b.Xo10):              # a turnover is symmetric: both orientations
                    xs.append(np.hstack([b.Xc[graded], xo[graded]]))
                    ys.append(b.switch[graded])
            switch = lgb.train(params, lgb.Dataset(np.vstack(xs), np.concatenate(ys)), n_rounds)
        return cls(dead, ctrl, bundles[0].names_c, bundles[0].names_o, switch)

    # -- inference -------------------------------------------------------------------------
    def predict(self, b):
        """-> (p_dead, p_home): P(ball dead) and P(home in control | live) per bundle row."""
        p_dead = self.dead.predict(b.Xc)
        p01 = self.ctrl.predict(np.hstack([b.Xc, b.Xo01]))
        p10 = self.ctrl.predict(np.hstack([b.Xc, b.Xo10]))
        p_home = 0.5 * (p01 + (1.0 - p10))
        return p_dead.astype(np.float64), p_home.astype(np.float64)

    def predict_switch(self, b):
        """P(direct H<->A change within ~0.5 s) per bundle row, or None if there is no detector."""
        if self.switch is None:
            return None
        a = self.switch.predict(np.hstack([b.Xc, b.Xo01]))
        c = self.switch.predict(np.hstack([b.Xc, b.Xo10]))
        return (0.5 * (a + c)).astype(np.float64)

    def importance(self, which="ctrl", top=20):
        import pandas as pd
        model = self.ctrl if which == "ctrl" else self.dead
        names = (self.names_c + self.names_o) if which == "ctrl" else self.names_c
        gain = model.feature_importance("gain")
        return pd.Series(gain, index=names).sort_values(ascending=False).head(top)

    # -- persistence -----------------------------------------------------------------------
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.dead.save_model(os.path.join(folder, "dead.txt"))
        self.ctrl.save_model(os.path.join(folder, "ctrl.txt"))
        if self.switch is not None:
            self.switch.save_model(os.path.join(folder, "switch.txt"))
        with open(os.path.join(folder, "names.json"), "w") as handle:
            json.dump({"c": self.names_c, "o": self.names_o}, handle)

    @classmethod
    def load(cls, folder):
        import lightgbm as lgb
        with open(os.path.join(folder, "names.json")) as handle:
            names = json.load(handle)
        sw_path = os.path.join(folder, "switch.txt")
        return cls(lgb.Booster(model_file=os.path.join(folder, "dead.txt")),
                   lgb.Booster(model_file=os.path.join(folder, "ctrl.txt")), names["c"], names["o"],
                   lgb.Booster(model_file=sw_path) if os.path.isfile(sw_path) else None)


def frame_metrics(b, p_dead, p_home):
    """Quick frame-level sanity: dead-ball and control log-loss / accuracy on a bundle."""
    eps = 1e-6
    graded = b.codes != X
    yd = (b.codes[graded] == D).astype(float)
    pd_ = np.clip(p_dead[graded], eps, 1 - eps)
    live = (b.codes == H) | (b.codes == A)
    yh = (b.codes[live] == H).astype(float)
    ph = np.clip(p_home[live], eps, 1 - eps)
    return {
        "dead_acc": float(np.mean((pd_ > 0.5) == (yd > 0.5))),
        "dead_logloss": float(-np.mean(yd * np.log(pd_) + (1 - yd) * np.log(1 - pd_))),
        "dead_base_rate": float(yd.mean()),
        "ctrl_acc": float(np.mean((ph > 0.5) == (yh > 0.5))),
        "ctrl_logloss": float(-np.mean(yh * np.log(ph) + (1 - yh) * np.log(1 - ph))),
    }

