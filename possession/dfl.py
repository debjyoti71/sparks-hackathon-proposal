"""Extra training data: the open DFL / IDSSE dataset (7 full Bundesliga / 2. Bundesliga matches).

Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data. Released under CC BY 4.0; loaded with kloppy (>= 3.19, which fetches
the files from the pysport/idsse-data mirror on Hugging Face).

Why it is useful: every tracking frame carries the provider's own ball status (alive / dead) and
ball-owning team. That is the same KIND of label as the challenge ground truth ("a frame is D where
the corrected tracking marks the ball dead, and otherwise takes the team the tracking names as
owning it"), so these labels need no reconstruction from an event log.

Labels: D where ball_state is dead; otherwise the owning team (home = first team = 0); an alive frame
with no owner keeps the last owner in that half ("possession stays with the last team that had it").

These games are used as additional TRAINING data only. Out-of-fold evaluation stays on the Metrica
games, so scores remain comparable with earlier runs.
"""

import os

import numpy as np

from .data import CACHE_DIR, Match
from .labels import A, D, H, X

DFL_MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
HALF_GAP = 250           # NaN frames (period 0) inserted between halves so windows never span the break


def _is_alive(state):
    value = getattr(state, "value", state)
    return str(value).lower() == "alive"


def load_dfl_game(match_id, cache=True):
    cache_path = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match_id))
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        return Match("DFL_" + match_id, z["xy"], z["team"], z["ball"], z["period"], 25,
                     tuple(z["pitch"].tolist()), None, z["ball"].copy(), z["labels"])

    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, coordinates="metrica", only_alive=False)
    pd_ = ds.metadata.pitch_dimensions
    pitch = (float(getattr(pd_, "pitch_length", None) or 105.0), float(getattr(pd_, "pitch_width", None) or 68.0))
    teams = ds.metadata.teams
    players = [p for t in teams for p in t.players]
    pidx = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(teams[0].players) + [1] * len(teams[1].players), dtype=np.int8)
    home_id = teams[0].team_id

    frames = ds.frames
    n_periods = len({f.period.id for f in frames})
    T = len(frames) + HALF_GAP * max(n_periods - 1, 0)
    N = len(players)
    xy = np.full((T, N, 2), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    period = np.zeros(T, dtype=np.int8)
    labels = np.full(T, X, dtype=np.int8)

    row, prev_period, owner = -1, None, -1
    for f in frames:
        pid = int(f.period.id)
        if prev_period is not None and pid != prev_period:
            row += HALF_GAP                     # leave the gap rows as NaN / period 0 / X
            owner = -1
        prev_period = pid
        row += 1
        if row >= T:
            break
        period[row] = min(pid, 2)
        bc = f.ball_coordinates
        if bc is not None and bc.x is not None and bc.y is not None:
            ball[row] = (bc.x * pitch[0], bc.y * pitch[1])
        for player, pdata in f.players_data.items():
            j = pidx.get(player.player_id)
            c = getattr(pdata, "coordinates", None)
            if j is not None and c is not None and c.x is not None and c.y is not None:
                xy[row, j] = (c.x * pitch[0], c.y * pitch[1])
        if not _is_alive(f.ball_state):
            labels[row] = D
            continue
        own = f.ball_owning_team
        if own is not None:
            owner = 0 if own.team_id == home_id else 1
        labels[row] = (H if owner == 0 else A) if owner in (0, 1) else D

    keep = period > 2            # extra periods (none expected) are dropped from play
    period[keep] = 0
    labels[keep] = X
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, xy=xy, team=team, ball=ball, period=period, labels=labels,
                 pitch=np.array(pitch))
    return Match("DFL_" + match_id, xy, team, ball, period, 25, pitch, None, ball.copy(), labels)


def smooth_labels(labels, period, min_poss_s=2.0, min_dead_s=1.0, fps=25):
    """Keep only controlled possessions: the DFL owner flips on every touch (deflections, duels), while
    the challenge definition only changes possession on CONTROL. Owner runs shorter than `min_poss_s`
    (and dead runs shorter than `min_dead_s`) are merged into their neighbours, half by half."""
    from .decode import merge_short_runs
    out = labels.copy()
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            seg = slice(idx[0], idx[-1] + 1)
            out[seg] = merge_short_runs(out[seg], int(min_poss_s * fps), int(min_dead_s * fps))
    return out


def load_dfl_games(match_ids=DFL_MATCHES, cache=True, log=print, smooth_s=0.0):
    out = []
    for mid in match_ids:
        log("  DFL {} ...".format(mid))
        g = load_dfl_game(mid, cache=cache)
        if smooth_s > 0:
            g.labels = smooth_labels(g.labels, g.period, min_poss_s=smooth_s)
        out.append(g)
    return out
