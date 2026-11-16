"""Layer 1: bridge ball gaps using both ends of the gap (we have the whole match offline)."""

import numpy as np
import pandas as pd

BIG = 10 ** 9


def since_until(flag, clip):
    """Frames since the last / until the next True in `flag` (clipped, float32)."""
    T = len(flag)
    ar = np.arange(T)
    last = np.maximum.accumulate(np.where(flag, ar, -1))
    nxt = np.minimum.accumulate(np.where(flag, ar, BIG)[::-1])[::-1]
    since = np.where(last >= 0, ar - last, clip)
    until = np.where(nxt < BIG, nxt - ar, clip)
    return np.minimum(since, clip).astype(np.float32), np.minimum(until, clip).astype(np.float32)


def last_next_index(flag):
    T = len(flag)
    ar = np.arange(T)
    last = np.maximum.accumulate(np.where(flag, ar, -1))
    nxt = np.minimum.accumulate(np.where(flag, ar, BIG)[::-1])[::-1]
    return last, nxt


def fill_ball(ball, max_hermite_gap=100, vmax=35.0, fps=25, pitch=(105.0, 68.0)):
    """Fill NaN ball positions. Returns (filled (T,2) float32, observed (T,) bool).

    Gaps up to `max_hermite_gap` frames get a cubic Hermite curve through both end anchors, with
    end velocities taken from the observed frames next to each anchor and capped at `vmax` m/s.
    Longer gaps are linearly interpolated (the models also receive the gap length, so they know
    how little that is worth). Before the first / after the last sighting the ball is held.
    """
    T = len(ball)
    obs = ~np.isnan(ball).any(axis=1)
    out = ball.astype(np.float64).copy()
    idx = np.flatnonzero(obs)
    if len(idx) < 2:
        out[:] = (pitch[0] / 2.0, pitch[1] / 2.0)
        return out.astype(np.float32), obs

    out[:idx[0]] = out[idx[0]]
    out[idx[-1] + 1:] = out[idx[-1]]
    vcap = vmax / fps
    P = out[idx]

    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        L = t1 - t0
        if L <= 1:
            continue
        chord = (P[i + 1] - P[i]) / L
        if L > max_hermite_gap:
            s = (np.arange(t0 + 1, t1) - t0) / float(L)
            out[t0 + 1:t1] = P[i] + s[:, None] * (P[i + 1] - P[i])
            continue
        v0 = chord
        if i > 0 and t0 - idx[i - 1] <= 4:
            v0 = (P[i] - P[i - 1]) / (t0 - idx[i - 1])
        v1 = chord
        if i + 2 < len(idx) and idx[i + 2] - t1 <= 4:
            v1 = (P[i + 2] - P[i + 1]) / (idx[i + 2] - t1)
        v0 = _cap(v0, vcap)
        v1 = _cap(v1, vcap)
        s = (np.arange(t0 + 1, t1) - t0) / float(L)
        s2, s3 = s * s, s * s * s
        h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s
        h01, h11 = -2 * s3 + 3 * s2, s3 - s2
        out[t0 + 1:t1] = (h00[:, None] * P[i] + h10[:, None] * L * v0
                          + h01[:, None] * P[i + 1] + h11[:, None] * L * v1)

    out[:, 0] = np.clip(out[:, 0], -2.0, pitch[0] + 2.0)
    out[:, 1] = np.clip(out[:, 1], -2.0, pitch[1] + 2.0)
    return out.astype(np.float32), obs


def _cap(v, cap):
    n = np.linalg.norm(v)
    return v if n <= cap or n == 0 else v * (cap / n)


def gap_error_table(ball_true, filled, observed, fps=25):
    """RMSE (metres) of the filled ball on masked frames, by gap length. Layer-1 diagnostic."""
    since, until = since_until(observed, clip=10 ** 6)
    gap_len = np.where(observed, 0, since + until - 1)
    ok = ~observed & ~np.isnan(ball_true).any(axis=1)
    err = np.linalg.norm(filled - ball_true, axis=1)
    bins = [0, 12, 25, 50, 100, 10 ** 7]
    labels = ["<0.5s", "0.5-1s", "1-2s", "2-4s", ">4s"]
    rows = []
    for lo, hi, name in zip(bins[:-1], bins[1:], labels):
        sel = ok & (gap_len > lo) & (gap_len <= hi)
        rows.append({"gap": name, "frames": int(sel.sum()),
                     "rmse_m": float(np.sqrt(np.mean(err[sel] ** 2))) if sel.any() else np.nan,
                     "median_err_m": float(np.median(err[sel])) if sel.any() else np.nan})
    return pd.DataFrame(rows)

