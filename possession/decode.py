"""Layer 4: sticky sequence decoding, short-run clean-up, and decoder tuning against spellF1."""

import itertools
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .labels import A, D, H, X, run_lengths

_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)
import possession_metrics as pm  # noqa: E402

EPS = 1e-3


@dataclass
class DecodeConfig:
    c_ha: float = 20.0          # cost of switching H <-> A (in log-probability units)
    c_ld: float = 12.0          # cost of switching live <-> dead
    min_poss_s: float = 2.0     # live runs shorter than this are merged into a neighbour
    min_dead_s: float = 1.0     # dead runs shorter than this are merged into a neighbour
    stride: int = 1             # decode on blocks of this many frames (5 is plenty for tuning)
    sw_beta: float = 0.0        # how much the turnover detector lowers / raises c_ha (0 = constant c_ha)
    shift: int = 0              # frames: delay (+) or advance (-) every decoded boundary, tuned out-of-fold


def viterbi(logp, c_ha, c_ld):
    """Most likely H/A/D sequence. logp: (n, 3) log emission scores, columns H, A, D.
    c_ha: scalar, or (n,) cost of an H<->A switch INTO each step."""
    n = len(logp)
    L = logp.tolist()
    cha = np.broadcast_to(np.asarray(c_ha, dtype=np.float64), (n,)).tolist()
    back = [None] * n
    s0, s1, s2 = L[0]
    for t in range(1, n):
        e0, e1, e2 = L[t]
        c_ha = cha[t]
        # best predecessor of each state
        a, b, c = s0, s1 - c_ha, s2 - c_ld
        if a >= b and a >= c:
            p0, v0 = 0, a
        elif b >= c:
            p0, v0 = 1, b
        else:
            p0, v0 = 2, c
        a, b, c = s0 - c_ha, s1, s2 - c_ld
        if b >= a and b >= c:
            p1, v1 = 1, b
        elif a >= c:
            p1, v1 = 0, a
        else:
            p1, v1 = 2, c
        a, b, c = s0 - c_ld, s1 - c_ld, s2
        if c >= a and c >= b:
            p2, v2 = 2, c
        elif a >= b:
            p2, v2 = 0, a
        else:
            p2, v2 = 1, b
        s0, s1, s2 = v0 + e0, v1 + e1, v2 + e2
        back[t] = (p0, p1, p2)
    state = 0 if (s0 >= s1 and s0 >= s2) else (1 if s1 >= s2 else 2)
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        if t:
            state = back[t][state]
    return out


def merge_short_runs(s, min_live, min_dead):
    """Absorb runs shorter than the thresholds (in frames) into their longest sensible neighbour."""
    s = s.copy()
    for _ in range(100000):
        runs = run_lengths(s)
        if len(runs) <= 1:
            break
        bad = [(b - a, i) for i, (a, b, c) in enumerate(runs)
               if (b - a) < (min_dead if c == D else min_live)]
        if not bad:
            break
        _, i = min(bad)
        a, b, c = runs[i]
        cands = [runs[j] for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        if c != D:
            live = [r for r in cands if r[2] != D]
            cands = live or cands
        else:
            cands = [r for r in cands if r[2] != D] or cands
        pick = max(cands, key=lambda r: r[1] - r[0])
        s[a:b] = pick[2]
    return s


def _emissions(p_dead, p_home):
    pdead = np.clip(p_dead, EPS, 1 - EPS)
    ph = np.clip(p_home, EPS, 1 - EPS)
    live = 1.0 - pdead
    return np.log(np.stack([live * ph, live * (1.0 - ph), pdead], axis=1))


def _switch_costs(c_ha, beta, p_switch):
    """Per-frame H<->A cost: c_ha - beta * (logit p_switch - its median), floored at 0. The median is
    taken over the half being decoded, so only where a turnover is MORE likely than usual gets cheaper."""
    p = np.clip(p_switch, 1e-4, 1 - 1e-4)
    lg = np.log(p / (1 - p))
    return np.maximum(c_ha - beta * (lg - np.median(lg)), 0.0)


def decode_match(p_dead, p_home, period, cfg=None, fps=25, force_dead=(), p_switch=None):
    """(T,) codes H=0 A=1 D=2 X=3. Decodes each half separately; X outside the halves.

    force_dead: [(first, last)] inclusive frame ranges known to be dead from published facts
    (the challenge's graded pre-kick-off stretch of the 2nd half).
    """
    cfg = cfg or DecodeConfig()
    p_dead = p_dead.copy()
    for a, b in force_dead:
        p_dead[a:b + 1] = 1.0
    out = np.full(len(period), X, dtype=np.int8)
    emis = _emissions(p_dead, p_home)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        e = emis[idx[0]: idx[-1] + 1]
        n = len(e)
        if p_switch is not None and cfg.sw_beta:
            cha = _switch_costs(cfg.c_ha, cfg.sw_beta, p_switch[idx[0]: idx[-1] + 1])
        else:
            cha = np.full(n, float(cfg.c_ha))
        if cfg.stride > 1:
            starts = np.arange(0, n, cfg.stride)
            blocks = np.add.reduceat(e, starts, axis=0)
            states = np.repeat(viterbi(blocks, np.minimum.reduceat(cha, starts), cfg.c_ld), cfg.stride)[:n]
        else:
            states = viterbi(e, cha, cfg.c_ld)
        states = merge_short_runs(states, int(cfg.min_poss_s * fps), int(cfg.min_dead_s * fps))
        k = int(cfg.shift)
        if k > 0:                                   # boundaries later: repeat the first label
            states = np.concatenate([np.full(k, states[0], dtype=states.dtype), states[:-k]])
        elif k < 0:                                 # boundaries earlier: repeat the last label
            states = np.concatenate([states[-k:], np.full(-k, states[-1], dtype=states.dtype)])
        out[idx[0]: idx[-1] + 1] = states
    for a, b in force_dead:
        out[a:b + 1] = np.where(out[a:b + 1] == X, X, D)
    return out


def score(gt, pred):
    """spellF1 (mean over IoU thresholds) and macroF1 for one match, with the official code."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return {"spell_f1": spell["mean_f1"], "macro_f1": pm.macro_f1(cm3),
            "n_gt": spell["n_gt"], "n_pred": spell["n_pred"]}


def nearest_player_baseline(match, window=50, fps=25):
    """The published baseline's idea (nearest player to the ball, majority vote, D when the ball
    is missing), run on any Match, for a like-for-like yardstick on masked external games."""
    from .labels import X as XCODE
    T = match.n_frames
    d = np.linalg.norm(match.xy - match.ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.argmin(d, axis=1)
    seen = np.isfinite(d[np.arange(T), near]) & ~np.isnan(match.ball[:, 0])
    code = np.where(seen, match.team[near], -1)
    hc = np.concatenate(([0], np.cumsum(code == 0)))
    ac = np.concatenate(([0], np.cumsum(code == 1)))
    idx = np.arange(T)
    lo, hi = np.maximum(0, idx - window), np.minimum(T, idx + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(T, D, dtype=np.int8)
    sel = (h + a) > 0
    out[sel] = np.where(h[sel] >= a[sel], H, A)
    out[match.period == 0] = XCODE
    return out


def tune_decoder(oof, grid_ha=(5, 10, 20, 40, 80), grid_ld=(3, 6, 12, 25, 50),
                 grid_poss=(0.0, 1.0, 2.0, 3.0, 4.0), grid_dead=(0.0, 0.5, 1.0, 2.0),
                 tune_stride=5, top_k=3, verbose=True, grid_sw=(0.0,), grid_shift=(0,)):
    """Two-stage search maximising the mean spellF1 over out-of-fold external games.

    oof: list of dicts {name, p_dead, p_home, period, codes}. All probabilities must come from
    models that did not train on that game.
    Stage 1 tunes the switching costs with no clean-up; stage 2 tunes the clean-up on the best few.
    Returns (best DecodeConfig, DataFrame of every configuration tried).
    """
    rows = []

    def evaluate(cfg):
        scores = [score(o["codes"], decode_match(o["p_dead"], o["p_home"], o["period"], cfg,
                                                 p_switch=o.get("p_switch")))
                  for o in oof]
        return {"spell_f1": float(np.mean([s["spell_f1"] for s in scores])),
                "macro_f1": float(np.mean([s["macro_f1"] for s in scores])),
                "n_pred": int(np.mean([s["n_pred"] for s in scores])),
                "n_gt": int(np.mean([s["n_gt"] for s in scores]))}

    for c_ha, c_ld, sw in itertools.product(grid_ha, grid_ld, grid_sw):
        cfg = DecodeConfig(c_ha=c_ha, c_ld=c_ld, min_poss_s=0.0, min_dead_s=0.0, stride=tune_stride,
                           sw_beta=sw)
        rows.append({"stage": 1, "c_ha": c_ha, "c_ld": c_ld, "sw_beta": sw, "min_poss_s": 0.0,
                     "min_dead_s": 0.0, **evaluate(cfg)})
    stage1 = pd.DataFrame(rows).sort_values("spell_f1", ascending=False)
    if verbose:
        print("stage 1 best:\n", stage1.head(top_k).to_string(index=False))

    rows2 = []
    for _, r in stage1.head(top_k).iterrows():
        for mp, md in itertools.product(grid_poss, grid_dead):
            cfg = DecodeConfig(c_ha=r.c_ha, c_ld=r.c_ld, min_poss_s=mp, min_dead_s=md, stride=tune_stride,
                               sw_beta=r.sw_beta)
            rows2.append({"stage": 2, "c_ha": r.c_ha, "c_ld": r.c_ld, "sw_beta": r.sw_beta, "min_poss_s": mp,
                          "min_dead_s": md, **evaluate(cfg)})
    table = pd.concat([stage1, pd.DataFrame(rows2)], ignore_index=True)
    best = table.sort_values(["spell_f1", "macro_f1"], ascending=False).iloc[0]
    # Stage 3: shift the best decoding in time. Evaluated at full frame resolution (stride 1 decode
    # would be slow, so the stride-`tune_stride` path is used: the shift is applied after the
    # blocks are expanded back to frames, so it is still exact to the frame).
    if any(k != 0 for k in grid_shift):
        rows3 = []
        for k in grid_shift:
            cfg = DecodeConfig(c_ha=best.c_ha, c_ld=best.c_ld, min_poss_s=best.min_poss_s,
                               min_dead_s=best.min_dead_s, stride=tune_stride, sw_beta=best.sw_beta, shift=int(k))
            rows3.append({"stage": 3, "c_ha": best.c_ha, "c_ld": best.c_ld, "sw_beta": best.sw_beta,
                          "min_poss_s": best.min_poss_s, "min_dead_s": best.min_dead_s, "shift": int(k),
                          **evaluate(cfg)})
        stage3 = pd.DataFrame(rows3)
        if verbose:
            print("stage 3 (shift):\n", stage3[["shift", "spell_f1", "macro_f1"]].to_string(index=False))
        table = pd.concat([table, stage3], ignore_index=True)
        zero = stage3[stage3["shift"] == 0]
        top3 = stage3.sort_values(["spell_f1", "macro_f1"], ascending=False).iloc[0]
        # a non-zero shift must strictly beat no shift
        if len(zero) and (top3.spell_f1, top3.macro_f1) <= (zero.iloc[0].spell_f1, zero.iloc[0].macro_f1):
            top3 = zero.iloc[0]
        best = top3
    shift = int(best["shift"]) if "shift" in best.index and not pd.isna(best["shift"]) else 0
    cfg = DecodeConfig(c_ha=float(best.c_ha), c_ld=float(best.c_ld), min_poss_s=float(best.min_poss_s),
                       min_dead_s=float(best.min_dead_s), stride=1, sw_beta=float(best.sw_beta), shift=shift)
    return cfg, table

