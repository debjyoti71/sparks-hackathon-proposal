"""Rule-based hierarchical possession engine (teammate's possession_engine.py / BALL_POSSESSION_LOGIC.md),
ported to run on any `Match` so it can be validated out-of-fold on the external games, and combined
with the learned models as log-odds priors.

Rules (unchanged from possession_engine.py):
  1. Out of bounds   : ball tracked and more than `oob_margin` m outside the pitch      -> D
  2. Set-piece dead  : inside a Smart Tagging SET PIECES window and the ball is missing
                       or the nearest player is more than `sp_far` m away                 -> D
  3. Control switch  : the nearest player is within `control_r` m of the ball; an opponent
                       must hold that for `confirm` consecutive frames to take possession
  4. Otherwise       : pass / aerial / free ball / ball missing -> keep the current team

Difference from the original: the team that kicks off each half is NOT hard-coded. It is read from
the Smart Tagging `Kick Off (Start)` instance at the start of the half when it names a team, else
from the first player-ball contact in the half. Nothing here reads any ground truth.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .labels import A, D, H, X
from .smart_tagging import _parse_team


@dataclass
class RuleConfig:
    control_r: float = 2.5      # metres: max distance for physical ball control
    confirm: int = 12           # frames (~0.48 s) an opponent must control before possession switches
    oob_margin: float = 0.5     # metres outside the pitch lines before the ball counts as out
    sp_far: float = 3.0         # metres: inside a set-piece window, nearest player this far -> dead


def _nearest(match):
    """(nearest team or -1, distance or 999) per frame, using the raw (unfilled) ball."""
    ball = match.ball
    has = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.linalg.norm(match.xy - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), j]
    ok = has & np.isfinite(dist)
    return np.where(ok, match.team[j], -1).astype(np.int8), np.where(ok, dist, 999.0).astype(np.float32)


def _kickoff_team(instances, first, near_team, near_dist, last, cfg, fps):
    best, best_gap = -1, None
    for inst in instances or ():
        if (inst.get("code") or "").strip().upper() != "SET PIECES":
            continue
        tags = inst.get("tags") or {}
        if "kick off" not in str(tags.get("Type") or "").lower():
            continue
        team = _parse_team(tags.get("Team"))
        a, b = int(inst.get("start_frame", 0)), int(inst.get("end_frame", 0))
        if team in (0, 1) and a <= first + 60 * fps and b >= first - 10 * fps:
            gap = abs(a - first)
            if best_gap is None or gap < best_gap:
                best, best_gap = team, gap
    if best in (0, 1):
        return best
    touch = np.flatnonzero((near_team[first:last + 1] >= 0) & (near_dist[first:last + 1] <= cfg.control_r))
    return int(near_team[first + touch[0]]) if len(touch) else 0


def run_rules(match, instances, cfg=None):
    """-> (codes (T,) H/A/D/X, team (T,) current possessing team or -1, dead (T,) bool rule-dead)."""
    cfg = cfg or RuleConfig()
    T = match.n_frames
    W, Hh = match.pitch
    near_team, near_dist = _nearest(match)

    bx, by = match.ball[:, 0], match.ball[:, 1]
    has = ~np.isnan(bx)
    m = cfg.oob_margin
    with np.errstate(invalid="ignore"):
        oob = has & ((bx < -m) | (bx > W + m) | (by < -m) | (by > Hh + m))

    in_sp = np.zeros(T, dtype=bool)
    for inst in instances or ():
        if (inst.get("code") or "").strip().upper() == "SET PIECES":
            a = max(0, int(inst.get("start_frame", 0)))
            b = min(T - 1, int(inst.get("end_frame", 0)))
            if b >= a:
                in_sp[a:b + 1] = True
    dead = oob | (in_sp & (near_dist > cfg.sp_far))

    control = (near_team >= 0) & (near_dist <= cfg.control_r)
    codes = np.full(T, X, dtype=np.int8)
    team = np.full(T, -1, dtype=np.int8)
    for p in (1, 2):
        idx = np.flatnonzero(match.period == p)
        if not len(idx):
            continue
        first, last = int(idx[0]), int(idx[-1])
        cur = _kickoff_team(instances, first, near_team, near_dist, last, cfg, match.fps)
        cnt = 0
        dead_l, ctrl_l, nt_l = dead[first:last + 1].tolist(), control[first:last + 1].tolist(), near_team[first:last + 1].tolist()
        out_c, out_t = [], []
        for is_dead, is_ctrl, t in zip(dead_l, ctrl_l, nt_l):
            if is_dead:
                cnt = 0
                out_c.append(D)
            else:
                if is_ctrl:
                    if t == cur:
                        cnt = 0
                    else:
                        cnt += 1
                        if cnt >= cfg.confirm:
                            cur, cnt = t, 0
                else:
                    cnt = 0
                out_c.append(H if cur == 0 else A)
            out_t.append(cur)
        codes[first:last + 1] = out_c
        team[first:last + 1] = out_t
    return codes, team, dead


def apply_rule_priors(p_dead, p_home, rule_codes, rule_dead, w_team=0.0, w_dead=0.0):
    """Add the rule engine's verdict to the model probabilities in log-odds space.

    w_team: +w to logit P(home) where the rules say H, -w where they say A.
    w_dead: +w to logit P(dead) where a dead-ball rule fired.
    w_team = w_dead = 0 returns the inputs unchanged.
    """
    if not w_team and not w_dead:
        return p_dead, p_home
    eps = 1e-4
    pd_ = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    ld = np.log(pd_ / (1 - pd_)) + w_dead * rule_dead.astype(np.float64)
    sign = np.where(rule_codes == H, 1.0, np.where(rule_codes == A, -1.0, 0.0))
    lh = np.log(ph / (1 - ph)) + w_team * sign
    return 1.0 / (1.0 + np.exp(-np.clip(ld, -15, 15))), 1.0 / (1.0 + np.exp(-np.clip(lh, -15, 15)))
