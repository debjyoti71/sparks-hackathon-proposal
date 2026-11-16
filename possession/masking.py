"""Simulate the challenge's tracking failures on clean external games.

The challenge ball is missing on ~41% of graded frames, in gaps, and disproportionately where it
is fast (passes, crosses) or crowded (duels, boxes). We reproduce that so the models learn to cope
with it, and so the decoder is tuned on inputs that look like the real ones.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .ball import fill_ball


@dataclass
class MaskConfig:
    ball_missing: float = 0.41          # target fraction of in-play frames
    ball_gap_median: float = 35.0       # frames
    ball_gap_sigma: float = 0.9
    speed_weight: float = 1.5           # hazard multiplier for a fast ball
    crowd_weight: float = 1.0           # hazard multiplier for a crowded ball
    player_missing: float = 0.07        # target fraction of (frame, player) cells
    player_gap_median: float = 20.0
    player_gap_sigma: float = 0.8
    noise_sigma: float = 0.15           # metres of jitter on players
    ball_gap_max: int = 600             # frames: longest simulated ball gap
    # 'simulate': bridge the raw external ball gaps, then hide the ball independently of the label.
    # 'raw'     : keep the external tracker's own gaps (long, mostly during dead ball) and only add
    #             extra random gaps if the total is below all_missing.
    mode: str = "simulate"


def _paint(starts, lengths, T):
    """starts: (T, K) bool of gap starts, lengths aligned to np.nonzero(starts). -> (T, K) bool."""
    K = starts.shape[1]
    t_idx, k_idx = np.nonzero(starts)
    size = (T + 1) * K
    plus = np.bincount(t_idx * K + k_idx, minlength=size)
    minus = np.bincount(np.minimum(t_idx + lengths, T) * K + k_idx, minlength=size)
    diff = (plus - minus).reshape(T + 1, K)
    return np.cumsum(diff, axis=0)[:T] > 0


def _gap_lengths(rng, n, median, sigma, max_len=600):
    return np.clip(np.exp(np.log(median) + sigma * rng.standard_normal(n)), 3, max_len).astype(np.int64)


def _calibrate(u, weight, lengths_draw, play, target, eligible=None, hi=0.1, base=None):
    """Bisect the base hazard so the final missing fraction hits `target`."""
    T, K = u.shape
    lo = 1e-6
    best = None
    for _ in range(28):
        mid = np.sqrt(lo * hi)
        starts = u < np.minimum(mid * weight, 1.0)
        n = int(starts.sum())
        mask = _paint(starts, lengths_draw[:n], T)
        if eligible is None:
            frac = (mask if base is None else (mask | base))[play].mean()
        else:
            frac = (mask & eligible)[play].sum() / max(eligible[play].sum(), 1)
        best = mask
        if frac < target:
            lo = mid
        else:
            hi = mid
    return best


def _speed(xy, fps, k=2):
    out = np.full(len(xy), np.nan)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=1) * fps / (2 * k)
    return out


def apply_masks(match, cfg=None, seed=0):
    """Copy of `match` with ball gaps, player dropouts and jitter applied (ball_true is kept)."""
    cfg = cfg or MaskConfig()
    rng = np.random.default_rng(seed)
    T, N, _ = match.xy.shape
    play = match.period > 0
    xy = match.xy.astype(np.float32).copy()
    # The raw external ball is missing mostly while the ball is dead (73-88% of dead frames, 6-12%
    # of live ones), which would teach the models 'ball missing => dead'. The challenge ball goes
    # missing during live play too, so bridge the raw gaps first and then mask independently of
    # the label (hazard depends on ball speed, crowding only).
    if cfg.mode == "raw":
        ball = match.ball.astype(np.float32).copy()
        base_missing = np.isnan(ball).any(axis=1)[:, None]
        ball = np.where(np.isnan(ball), 0.0, ball)      # only for the hazard weights below
        raw_nan = base_missing[:, 0]
    else:
        ball, _ = fill_ball(match.ball.astype(np.float32), pitch=match.pitch, fps=match.fps)
        base_missing, raw_nan = None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = _speed(np.where(np.isnan(ball), 0, ball), match.fps)
        d = np.linalg.norm(xy - ball[:, None, :], axis=2)
        crowd = np.nansum(d < 6.0, axis=1)
    sp = np.nan_to_num(sp, nan=0.0)
    weight = 1.0 + cfg.speed_weight * np.clip(sp / 20.0, 0, 1.5) + cfg.crowd_weight * np.clip(crowd / 6.0, 0, 1.5)
    weight = (weight * play)[:, None]

    u = rng.random((T, 1))
    draws = _gap_lengths(rng, T, cfg.ball_gap_median, cfg.ball_gap_sigma, cfg.ball_gap_max)
    gap = _calibrate(u, weight, draws, play, cfg.ball_missing, None, hi=0.1, base=base_missing)[:, 0]
    if raw_nan is not None:
        gap = gap | raw_nan
    ball[gap] = np.nan

    base_players = np.isnan(xy[:, :, 0])
    up = rng.random((T, N))
    pdraws = _gap_lengths(rng, T * N, cfg.player_gap_median, cfg.player_gap_sigma)
    pweight = np.broadcast_to(play[:, None].astype(np.float64), (T, N))
    pmask = _calibrate(up, pweight, pdraws, play, cfg.player_missing, ~base_players, hi=0.03)
    xy[pmask] = np.nan

    if cfg.noise_sigma > 0:
        xy = xy + rng.normal(0.0, cfg.noise_sigma, xy.shape).astype(np.float32)

    return replace(match, xy=xy, ball=ball,
                   ball_true=match.ball_true if match.ball_true is not None else match.ball)


def missing_stats(match):
    play = match.period > 0
    return {
        "ball_missing_in_play_%": round(100 * float(np.isnan(match.ball[play, 0]).mean()), 1),
        "player_missing_in_play_%": round(100 * float(np.isnan(match.xy[play, :, 0]).mean()), 1),
    }


def challenge_gap_config():
    """Ball gaps shaped like the provided tactical ATD's (aggregate input statistics, no labels):
    41% of graded frames missing, gap median 10.7 s (267 frames), p90 33.8 s -> lognormal
    sigma = ln(33.8 / 10.7) / 1.2816 = 0.90, longest gap 221 s (cap 5500 frames)."""
    return MaskConfig(mode="simulate", ball_missing=0.41, ball_gap_median=267.0, ball_gap_sigma=0.9,
                      ball_gap_max=5500)
