"""Event-derived H / A / D labels for the EXTERNAL games.

These are training / tuning targets for models that never see the challenge game. They are an
approximation of the challenge's definition (the challenge ground truth comes from corrected
tracking, not events), so run `label_report` and read it before trusting a score.

Rules, in order:
  owner   = the team of the latest ownership event (PASS / SHOT / SET PIECE / RECOVERY start; a
            BALL LOST hands the ball to the opponent). Challenges are ignored: an aerial duel
            won is not controlled possession.
  dead    = from a stoppage event until the next restart event.
              stoppage: BALL OUT, FAULT RECEIVED, offside / end-half BALL LOST, goals, and shots
                        that end out of play.
              restart : SET PIECE, or a goal-kick PASS.
            plus the stretch from the start of each period to its first restart (kick-off).
  X       = outside the two periods.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3

OWNER_TYPES = ("PASS", "SHOT", "SET PIECE", "RECOVERY")
MAX_DEAD_FRAMES = 25 * 150          # a stoppage is never longer than this unless the half ends


def _is_stoppage(row):
    typ, sub = row.type, row.subtype
    if typ in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if typ == "BALL LOST" and ("OFFSIDE" in sub or "END HALF" in sub):
        return True
    if typ == "SHOT" and ("GOAL" in sub and "SAVED" not in sub and "GOAL KICK" not in sub):
        return True
    if typ == "SHOT" and sub.endswith("OUT"):
        return True
    return False


def _is_restart(row):
    return row.type == "SET PIECE" or (row.type == "PASS" and "GOAL KICK" in row.subtype)


def build_labels(match):
    """(T,) int8 codes with H=0, A=1, D=2, X=3."""
    if getattr(match, "labels", None) is not None:      # provider-labelled tracking (DFL)
        return np.asarray(match.labels, dtype=np.int8).copy()
    T = match.n_frames
    ev = match.events.copy()
    ev = ev[(ev["start"] >= 0) & (ev["start"] < T)].reset_index(drop=True)
    ev = ev.sort_values("start", kind="stable").reset_index(drop=True)

    owner = np.full(T, -1, dtype=np.int8)
    changes = []
    for row in ev.itertuples():
        if row.team not in (0, 1):
            continue
        if row.type in OWNER_TYPES:
            changes.append((row.start, row.team))
        elif row.type == "BALL LOST":
            changes.append((row.start, 1 - row.team))
    mark = np.full(T, -1, dtype=np.int8)
    for frame, team in changes:
        mark[frame] = team          # later events on the same frame win
    idx = np.maximum.accumulate(np.where(mark >= 0, np.arange(T), -1))
    owner[idx >= 0] = mark[idx[idx >= 0]]

    dead = np.zeros(T, dtype=bool)
    restarts = np.array([r.start for r in ev.itertuples() if _is_restart(r)], dtype=np.int64)
    play = match.period > 0

    for p in (1, 2):
        frames = np.flatnonzero(match.period == p)
        if not len(frames):
            continue
        lo, hi = frames[0], frames[-1]
        first = restarts[(restarts >= lo) & (restarts <= hi)]
        dead[lo: (first[0] if len(first) else hi + 1)] = True        # before kick-off

    for row in ev.itertuples():
        if not _is_stoppage(row):
            continue
        frames = np.flatnonzero(match.period == row.period)
        period_end = frames[-1] + 1 if len(frames) else row.start
        nxt = restarts[restarts > row.start]
        stop = min(nxt[0] if len(nxt) else period_end, period_end, row.start + MAX_DEAD_FRAMES)
        dead[row.start:stop] = True

    codes = np.where(owner >= 0, owner, D).astype(np.int8)
    codes[dead] = D
    codes[~play] = X
    return codes


def run_lengths(codes):
    """[(start, stop, code)] constant runs of an array."""
    if not len(codes):
        return []
    edge = np.flatnonzero(np.diff(codes) != 0) + 1
    starts = np.concatenate(([0], edge))
    stops = np.concatenate((edge, [len(codes)]))
    return [(int(a), int(b), int(codes[a])) for a, b in zip(starts, stops)]


def label_report(codes, fps=25):
    """Shares, possession count and mean duration. Compare with the challenge's public numbers
    (about 17 possessions in 5 min, mean 14 s, dead about 20%) to judge how realistic the labels are."""
    graded = codes != X
    n = int(graded.sum())
    out = {"graded_frames": n}
    for name, c in (("H", H), ("A", A), ("D", D)):
        out["share_" + name] = round(float((codes == c).sum()) / max(n, 1), 3)
    spells = []
    for p_start, p_stop in _graded_spans(codes):
        for a, b, c in run_lengths(codes[p_start:p_stop]):
            if c in (H, A):
                spells.append((b - a) / float(fps))
    out["possessions"] = len(spells)
    out["mean_possession_s"] = round(float(np.mean(spells)), 1) if spells else None
    out["median_possession_s"] = round(float(np.median(spells)), 1) if spells else None
    out["possessions_per_5min"] = round(len(spells) / (n / fps / 300.0), 1) if n else None
    return out


def _graded_spans(codes):
    g = (codes != X).astype(np.int8)
    e = np.diff(np.concatenate(([0], g, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))

