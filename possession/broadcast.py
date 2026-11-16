"""Broadcast-to-Tactical tracking alignment and ball fusion.

The broadcast recording (`data/broadcast/`) has its own timeline (164,488 frames) with cuts,
replays and close-ups, and is not synchronised to the tactical video (199,319 frames). However:
  1. Both tracking files report normalised (0-1) pitch coordinates for the same two teams.
  2. Within any contiguous live camera shot (where `broadcast_homography.csv` has `valid == 1`
     and enough players are visible on the pitch), both recordings run at 25 fps, so the timeline
     mapping for that shot is a single integer frame offset:
         tactical_frame = broadcast_frame + offset
  3. We find each shot's offset by matching a low-dimensional, permutation-invariant team
     constellation signature (team centroids, spreads and visible ball coordinates) against the
     tactical timeline, and verify the match on player-to-player constellation distance and
     shared ball observations. Replays are rejected automatically because they jump backwards in
     game time relative to the surrounding monotonic live timeline or fail the constellation check.

Wherever a broadcast shot is confidently aligned and the tactical ball is NaN while the broadcast
ball is visible, we copy the broadcast ball into the tactical ball array before bridging gaps.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .data import CACHE_DIR, ROOT


def load_broadcast_raw(data_dir=None, cache=True):
    """Load broadcast ATD (xy, ball, team) and homography validity flag.

    Returns dict with keys: xy (Tb, Nb, 2), ball (Tb, 2), team (Nb,), valid (Tb,) bool.
    """
    sys.path.insert(0, ROOT)
    from games import GAME, GAMES, require, source_frames

    cache_path = os.path.join(CACHE_DIR, "challenge_broadcast.npz")
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        return {"xy": z["xy"], "ball": z["ball"], "team": z["team"], "valid": z["valid"]}

    meta_path, raw_path, homo_path = require(
        ["atd_meta", "atd_raw", "homography"], GAME, data_dir, source="broadcast"
    )
    pitch = GAMES[GAME]["pitch"]
    n_frames = source_frames("broadcast", GAME)

    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)
    n_slots = len(slots)

    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if 0 <= row < n_frames:
                vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
                xy[row] = vals.reshape(n_slots, -1)[:, :2]
                b = parts[2].split(",")
                ball[row] = (float(b[0]), float(b[1]))
    xy[:, :, 0] *= pitch[0]
    xy[:, :, 1] *= pitch[1]
    ball[:, 0] *= pitch[0]
    ball[:, 1] *= pitch[1]

    homo = pd.read_csv(homo_path, usecols=["valid"])
    valid = np.zeros(n_frames, dtype=bool)
    m = min(n_frames, len(homo))
    valid[:m] = homo["valid"].values[:m].astype(bool)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_path, xy=xy, ball=ball, team=team, valid=valid)
    return {"xy": xy, "ball": ball, "team": team, "valid": valid}


def _constellation_signature(xy, team, pitch=(105.3, 68.0)):
    """(T, 12) permutation-invariant team shape descriptor, in metres.

    For each team (0 and 1): mean x, mean y, std x, std y, min x, max x.
    Uses pure C-vectorised nan reductions so 199k frames run in ~0.2 s.
    """
    T = len(xy)
    sig = np.full((T, 12), np.nan, dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in (0, 1):
            sub = xy[:, team == k, :]  # (T, Nk, 2)
            mx = np.nanmean(sub[:, :, 0], axis=1)
            my = np.nanmean(sub[:, :, 1], axis=1)
            sx = np.nanstd(sub[:, :, 0], axis=1)
            sy = np.nanstd(sub[:, :, 1], axis=1)
            x0 = np.nanmin(sub[:, :, 0], axis=1)
            x1 = np.nanmax(sub[:, :, 0], axis=1)
            base = k * 6
            sig[:, base:base + 6] = np.stack([mx, my, sx, sy, x0, x1], axis=1)
    return sig


def _contiguous_runs(mask, min_len=50):
    """[(start, stop)] half-open runs where `mask` is True and length >= min_len."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops) if (b - a) >= min_len]


def _chamfer_distance(xy_a, team_a, xy_b, team_b):
    """Mean nearest-player distance (metres) between two frames, matched within each team."""
    dists = []
    for k in (0, 1):
        pa = xy_a[team_a == k]
        pb = xy_b[team_b == k]
        pa = pa[~np.isnan(pa[:, 0])]
        pb = pb[~np.isnan(pb[:, 0])]
        if len(pa) < 5 or len(pb) < 5:
            return np.inf
        diff = pa[:, None, :] - pb[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        dists.append(float(np.mean(np.sqrt(np.min(d2, axis=1)))))
    return 0.5 * (dists[0] + dists[1])


def align_and_fuse_broadcast(tactical_match, data_dir=None, cache=True,
                             min_shot_frames=50, stride_search=5,
                             max_sig_err=3.0, max_chamfer_m=2.5, max_ball_disagree_m=3.5):
    """Align live broadcast camera shots to the tactical timeline and fuse missing ball positions.

    Returns (fused_match, report_dict).
    """
    from dataclasses import replace

    cache_file = os.path.join(CACHE_DIR, "broadcast_fusion_v1.npz")
    if cache and os.path.isfile(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        fused_ball = z["fused_ball"]
        report = z["report"].item()
        return replace(tactical_match, ball=fused_ball), report

    try:
        bcast = load_broadcast_raw(data_dir=data_dir, cache=cache)
    except Exception as exc:
        return tactical_match, {"status": f"broadcast unavailable: {exc}", "frames_filled": 0}

    xy_t, ball_t, team_t = tactical_match.xy, tactical_match.ball.copy(), tactical_match.team
    xy_b, ball_b, team_b, valid_b = bcast["xy"], bcast["ball"], bcast["team"], bcast["valid"]

    # Live frames need valid homography and at least 12 tracked players
    n_tracked_b = (~np.isnan(xy_b[:, :, 0])).sum(axis=1)
    live_b = valid_b & (n_tracked_b >= 12)
    shots = _contiguous_runs(live_b, min_len=min_shot_frames)

    sig_t = _constellation_signature(xy_t, team_t, tactical_match.pitch)
    sig_b = _constellation_signature(xy_b, team_b, tactical_match.pitch)

    # Downsample on a coarse grid (every `stride_search` frames) for fast offset search
    T_t = len(xy_t)
    play_t = tactical_match.period > 0
    cand_t = np.flatnonzero(play_t)[::stride_search]

    matched_shots = []
    bcast_to_tac = np.full(len(xy_b), -1, dtype=np.int32)

    # Expect monotonic progress across live shots, so track rough expected tactical position
    expected_ratio = T_t / float(len(xy_b))

    for b0, b1 in shots:
        L = b1 - b0
        # Take up to 15 probe points across the shot
        probes = np.linspace(b0, b1 - 1, min(15, max(5, L // 25))).astype(int)
        sb = sig_b[probes]  # (P, 12)
        if np.isnan(sb).any():
            continue

        # Search candidate tactical starts `t0` where t0 + (probes - b0) are in play
        rel = probes - b0
        valid_cands = cand_t[(cand_t + L < T_t)]
        if not len(valid_cands):
            continue

        # Narrow search to +/- 25,000 frames (~16 minutes) around proportional timeline position
        center_t = int(b0 * expected_ratio)
        window_cands = valid_cands[np.abs(valid_cands - center_t) <= 30000]
        if not len(window_cands):
            window_cands = valid_cands

        # Vectorised signature MAE across candidate offsets
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx_matrix = window_cands[:, None] + rel[None, :]  # (C, P)
            st = sig_t[idx_matrix]  # (C, P, 12)
            mae = np.nanmean(np.abs(st - sb[None, :, :]), axis=(1, 2))
            if np.all(np.isnan(mae)):
                continue
            best_c = window_cands[int(np.nanargmin(mae))]

            # Fine single-frame refinement in [best_c - stride_search, best_c + stride_search]
            fine_cands = np.arange(max(0, best_c - stride_search),
                                   min(T_t - L, best_c + stride_search + 1))
            idx_fine = fine_cands[:, None] + rel[None, :]
            mae_fine = np.nanmean(np.abs(sig_t[idx_fine] - sb[None, :, :]), axis=(1, 2))
            best_t0 = int(fine_cands[int(np.nanargmin(mae_fine))])
            best_sig_err = float(np.nanmin(mae_fine))
        if best_sig_err > max_sig_err:
            continue

        # Verify constellation Chamfer distance on 5 check frames
        checks = np.linspace(b0, b1 - 1, 5).astype(int)
        chamfers = [_chamfer_distance(xy_b[b], team_b, xy_t[best_t0 + (b - b0)], team_t)
                    for b in checks]
        mean_chamfer = float(np.mean(chamfers))
        if mean_chamfer > max_chamfer_m:
            continue

        # If both cameras see the ball on >= 10 frames in this shot, check ball agreement
        t_slice = slice(best_t0, best_t0 + L)
        b_slice = slice(b0, b1)
        bt_sub = ball_t[t_slice]
        bb_sub = ball_b[b_slice]
        both_ball = ~np.isnan(bt_sub[:, 0]) & ~np.isnan(bb_sub[:, 0])
        if both_ball.sum() >= 10:
            ball_err = float(np.median(np.linalg.norm(bt_sub[both_ball] - bb_sub[both_ball], axis=1)))
            if ball_err > max_ball_disagree_m:
                continue
        else:
            ball_err = np.nan

        offset = best_t0 - b0
        matched_shots.append((b0, b1, offset, best_sig_err, mean_chamfer, ball_err))

    # Enforce monotonicity of tactical time to reject any replay that happened to match a static shape
    if matched_shots:
        kept = [matched_shots[0]]
        for s in matched_shots[1:]:
            prev_t1 = kept[-1][1] + kept[-1][2]
            curr_t0 = s[0] + s[2]
            if curr_t0 >= prev_t1 - 25:  # allow tiny 1s tolerance
                kept.append(s)
        matched_shots = kept

    frames_aligned = 0
    frames_filled = 0
    for b0, b1, offset, _, _, _ in matched_shots:
        for b in range(b0, b1):
            t = b + offset
            if 0 <= t < T_t and play_t[t]:
                bcast_to_tac[b] = t
                frames_aligned += 1
                if np.isnan(ball_t[t, 0]) and not np.isnan(ball_b[b, 0]):
                    ball_t[t] = ball_b[b]
                    frames_filled += 1

    play_missing_before = float(np.isnan(tactical_match.ball[play_t, 0]).mean())
    play_missing_after = float(np.isnan(ball_t[play_t, 0]).mean())
    report = {
        "total_bcast_shots": len(shots),
        "aligned_shots": len(matched_shots),
        "aligned_frames": frames_aligned,
        "ball_frames_filled": frames_filled,
        "ball_missing_before_%": round(100.0 * play_missing_before, 2),
        "ball_missing_after_%": round(100.0 * play_missing_after, 2),
    }
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache_file, fused_ball=ball_t, report=report)
    return replace(tactical_match, ball=ball_t), report
