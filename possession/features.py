"""Per-frame features. Everything is team-agnostic or expressed as own-vs-opponent, and uses
absolute pitch geometry (distance to lines, not left/right), because the external games and the
challenge attack in different directions.

Upgrades from the starter notebooks & architecture plan:
  1. Collective team velocity & coherence (`02_tracking_atd.ipynb`: "how the shape moves"):
     team centroid speed, movement coherence (||mean(v_i)|| / mean(||v_i||)), spatial spread, and
     inter-team centroid separation.
  2. Pass-bridge & deflection filtering (`CHALLENGE.md`: a simple touch or deflection without
     subsequent control does not change possession): suppresses isolated brief touches (< 0.4 s)
     sandwiched between the opposing team's touches, and explicitly flags intervals where the
     previous and next controlled touches belong to the same team (`o_pass_bridge`).

`dead_features`    : symmetric, used by the dead-ball model
`control_features` : own-vs-opponent, used by the control model; called for (0,1) and (1,0) so
                     the model is trained with team-swap augmentation and predictions are averaged.
"""

import warnings

import numpy as np

from .ball import last_next_index, since_until

CONTACT_R = 2.5          # metres: a player this close to the ball is "in contact"
CONTACT_VMAX = 20.0      # m/s: faster than this the ball is in flight, nobody controls it
CLIP_S = 20.0            # seconds, clip for since/until features
WINDOWS = (25, 75, 175)  # 1 s, 3 s, 7 s centred context


def _roll(a, lo, hi):
    """NaN-aware mean over frames [t+lo, t+hi] (inclusive)."""
    T = len(a)
    valid = ~np.isnan(a)
    x = np.where(valid, a, 0.0).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    ar = np.arange(T)
    a0 = np.clip(ar + lo, 0, T)
    a1 = np.clip(ar + hi + 1, 0, T)
    s = cs[a1] - cs[a0]
    n = cn[a1] - cn[a0]
    out = np.full(T, np.nan)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out.astype(np.float32)


def _centred(a, w):
    return _roll(a, -(w // 2), w // 2)


def _speed_series(xy, fps, k):
    """Speed in m/s from positions k frames either side. Works on (T,2) or (T,N,2)."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    out[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * fps / (2.0 * k)
    out[:k] = out[k]
    out[-k:] = out[-k - 1]
    return out


def _velocity_vec(xy, fps, k=2):
    """Velocity vector (vx, vy) in m/s from positions k frames either side. Shape matches `xy`."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2.0 * k))
    out[:k] = out[k]
    out[-k:] = out[-k - 1]
    return out


def _filter_deflections(contact, max_deflect_frames=10, max_bridge_gap_frames=100):
    """Remove isolated brief touches (<= max_deflect_frames) sandwiched between two touches by the
    other team within `max_bridge_gap_frames` (~4 s). Returns cleaned contact array."""
    out = contact.copy()
    T = len(out)
    active = out >= 0
    if not active.any():
        return out
    edges = np.flatnonzero(np.diff(out) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [T]))
    runs = [(int(a), int(b), int(out[a])) for a, b in zip(starts, stops)]

    touch_runs = [(idx, a, b, tm) for idx, (a, b, tm) in enumerate(runs) if tm in (0, 1)]
    for j in range(1, len(touch_runs) - 1):
        _, a, b, tm = touch_runs[j]
        if (b - a) <= max_deflect_frames:
            _, prev_a, prev_b, prev_tm = touch_runs[j - 1]
            _, next_a, next_b, next_tm = touch_runs[j + 1]
            if prev_tm == next_tm and prev_tm != tm and (next_a - prev_b) <= max_bridge_gap_frames:
                out[a:b] = -1  # treat as deflection / un-controlled touch
    return out


DEFAULT_FCFG = {"windows": list(WINDOWS), "clip_s": CLIP_S}
# Long-context variant: adds 15 s and 30 s windows and lets since/until features see 60 s.
LONG_FCFG = {"windows": [25, 75, 175, 375, 750], "clip_s": 60.0}
# Directional variant: adds attack-direction-aware control features (see _attack_direction).
DIR_FCFG = {"windows": [25, 75, 175], "clip_s": 20.0, "directional": True}


def _attack_direction(xy, team, period, pitch_x, k):
    """(T,) +1 if team k attacks towards x = pitch_x in that half, -1 if towards x = 0, NaN outside play.

    Inferred from the tracking alone, per half: a team's deepest player (its goalkeeper) sits near
    its own goal line, so the side where the team's minimum / maximum x is closest to a goal line
    is the side it defends. No labels, no match-specific constants."""
    out = np.full(len(xy), np.nan, dtype=np.float32)
    xk = np.where((team == k)[None, :], xy[:, :, 0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = np.nanmin(xk, axis=1)
        hi = np.nanmax(xk, axis=1)
        for p in (1, 2):
            sel = period == p
            if not sel.any():
                continue
            gap_left = np.nanmedian(lo[sel])
            gap_right = pitch_x - np.nanmedian(hi[sel])
            if np.isfinite(gap_left) and np.isfinite(gap_right):
                out[sel] = 1.0 if gap_left < gap_right else -1.0
    return out


def compute_common(match, ball, observed, fcfg=None):
    """Everything both models share. `ball` is the filled ball, `observed` the real-sighting mask."""
    fcfg = fcfg or DEFAULT_FCFG
    clip_s = float(fcfg["clip_s"])
    fps = match.fps
    xy = match.xy.astype(np.float32)
    team = match.team
    T, N, _ = xy.shape
    ar = np.arange(T)
    W, Hh = match.pitch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.linalg.norm(xy - ball[:, None, :], axis=2)
        vel = _velocity_vec(xy, fps, 2)
        sp = np.linalg.norm(vel, axis=-1)
        bvel = _velocity_vec(ball, fps, 2)
        bsp = np.linalg.norm(bvel, axis=-1)
        bsp1 = _speed_series(ball, fps, 12)
        bacc = _speed_series(bvel, fps, 2)  # ball acceleration magnitude (m/s^2)

        c = {
            "d": d, "sp": sp, "bsp": bsp, "bsp1": bsp1, "bacc": bacc,
            "obs": observed, "ball": ball, "_windows": tuple(fcfg["windows"]), "_clip_s": clip_s,
        }
        for key in ("d1", "d2", "cnt3", "cnt6", "cnt12", "spn", "tsp",
                    "centroid_sp", "coherence", "spread", "radial_v"):
            c[key] = {}

        centroids = {}
        for k in (0, 1):
            mask_k = (team[None, :] == k)
            dk = np.where(mask_k, d, np.nan)
            srt = np.sort(dk, axis=1)
            c["d1"][k] = srt[:, 0]
            c["d2"][k] = srt[:, 1] if N > 1 else srt[:, 0]
            c["cnt3"][k] = (dk < 3).sum(1).astype(np.float32)
            c["cnt6"][k] = (dk < 6).sum(1).astype(np.float32)
            c["cnt12"][k] = (dk < 12).sum(1).astype(np.float32)
            am = np.argmin(np.where(np.isnan(dk), np.inf, dk), axis=1)
            spn = sp[ar, am].copy()
            spn[np.isnan(c["d1"][k])] = np.nan
            c["spn"][k] = spn
            c["tsp"][k] = np.nanmean(np.where(mask_k, sp, np.nan), axis=1)

            # Collective team velocity, coherence, and spatial shape
            xy_k = np.where(mask_k[:, :, None], xy, np.nan)
            vel_k = np.where(mask_k[:, :, None], vel, np.nan)
            centroids[k] = np.nanmean(xy_k, axis=1)  # (T, 2)
            mean_vel_k = np.nanmean(vel_k, axis=1)   # (T, 2)
            c.setdefault("centroid", {})[k] = centroids[k].astype(np.float32)
            c.setdefault("mvel", {})[k] = mean_vel_k.astype(np.float32)
            c_sp = np.linalg.norm(mean_vel_k, axis=1).astype(np.float32)
            c["centroid_sp"][k] = c_sp
            c["coherence"][k] = (c_sp / (c["tsp"][k] + 0.15)).astype(np.float32)
            std_xy = np.nanstd(xy_k, axis=1)
            c["spread"][k] = np.hypot(std_xy[:, 0], std_xy[:, 1]).astype(np.float32)

            # Radial speed of nearest player toward the ball
            vec_to_ball = ball - xy[ar, am]
            dist_n = np.maximum(np.linalg.norm(vec_to_ball, axis=1), 0.1)
            v_nearest = vel[ar, am]
            rad = np.sum(v_nearest * (vec_to_ball / dist_n[:, None]), axis=1)
            rad[np.isnan(c["d1"][k])] = np.nan
            c["radial_v"][k] = rad.astype(np.float32)

        c["inter_centroid_d"] = np.linalg.norm(centroids[0] - centroids[1], axis=1).astype(np.float32)
        c["_W"] = float(W)
        if fcfg.get("directional"):
            c["dir"] = {k: _attack_direction(xy, team, match.period, W, k) for k in (0, 1)}

        # Who is in contact with the ball (debounced), and deflection-filtered pass bridge
        d1f = [np.nan_to_num(c["d1"][k], nan=1e3) for k in (0, 1)]
        nearest = np.where(d1f[0] <= d1f[1], 0, 1)
        ok = observed & (np.minimum(d1f[0], d1f[1]) <= CONTACT_R) & (np.nan_to_num(bsp, nan=0.0) < CONTACT_VMAX)
        raw = np.where(ok, nearest, -1)
        s0 = _roll((raw == 0).astype(np.float32), -2, 2)
        s1 = _roll((raw == 1).astype(np.float32), -2, 2)
        contact = np.where((s0 >= 0.6) & (s0 >= s1), 0, np.where(s1 >= 0.6, 1, -1))
        clean_contact = _filter_deflections(contact)

        c["contact"] = contact
        c["clean_contact"] = clean_contact
        c["since"], c["until"] = {}, {}
        for k in (0, 1):
            s, u = since_until(clean_contact == k, clip_s * fps)
            c["since"][k], c["until"][k] = s / fps, u / fps

        # Pass bridge: who had the last controlled touch, and who gets the next controlled touch?
        last_idx, next_idx = last_next_index(clean_contact >= 0)
        prev_team = np.where(last_idx >= 0, clean_contact[np.maximum(last_idx, 0)], -1)
        next_team = np.where(next_idx < T, clean_contact[np.minimum(next_idx, T - 1)], -1)
        c["prev_team"] = prev_team
        c["next_team"] = next_team

        x, y = ball[:, 0], ball[:, 1]
        bd = np.minimum.reduce([x, W - x, y, Hh - y]).astype(np.float32)
        last, nxt = last_next_index(observed)
        bd_obs = np.where(observed, bd, np.nan)
        c["bd"] = bd
        c["bd_last"] = np.where(last >= 0, bd_obs[np.maximum(last, 0)], np.nan).astype(np.float32)
        c["bd_next"] = np.where(nxt < T, bd_obs[np.minimum(nxt, T - 1)], np.nan).astype(np.float32)
        c["since_obs"], c["until_obs"] = [v / fps for v in since_until(observed, clip_s * fps)]

        c["tsp_all"] = np.nanmean(sp, axis=1)
        c["frac_slow"] = np.nanmean(np.where(np.isnan(sp), np.nan, (sp < 0.7).astype(np.float32)), axis=1)
        c["n_tracked"] = (~np.isnan(xy[:, :, 0])).sum(1).astype(np.float32)
        c["coherence_mean"] = 0.5 * (c["coherence"][0] + c["coherence"][1])
        c["centroid_sp_mean"] = 0.5 * (c["centroid_sp"][0] + c["centroid_sp"][1])
        c["spread_mean"] = 0.5 * (c["spread"][0] + c["spread"][1])
    return c


def dead_features(c):
    """Symmetric features for P(ball dead). Returns (X float32, names)."""
    ball = c["ball"]
    obs = c["obs"].astype(np.float32)
    d1min = np.fmin(c["d1"][0], c["d1"][1])
    f = {
        "c_ball_dx": np.abs(ball[:, 0] - 52.5),
        "c_ball_dy": np.abs(ball[:, 1] - 34.0),
        "c_bd": c["bd"], "c_bd_last": c["bd_last"], "c_bd_next": c["bd_next"],
        "c_bsp": c["bsp"], "c_bsp1": c["bsp1"], "c_bacc": c["bacc"],
        "c_obs": obs, "c_since_obs": c["since_obs"], "c_until_obs": c["until_obs"],
        "c_gap": np.where(c["obs"], 0.0, np.minimum(c["since_obs"] + c["until_obs"], c["_clip_s"])),
        "c_d1min": d1min, "c_tsp_all": c["tsp_all"], "c_frac_slow": c["frac_slow"],
        "c_n_tracked": c["n_tracked"],
        "c_coherence": c["coherence_mean"],
        "c_centroid_sp": c["centroid_sp_mean"],
        "c_spread": c["spread_mean"],
        "c_inter_centroid_d": c["inter_centroid_d"],
    }
    for name, series in (("bsp", c["bsp"]), ("bsp1", c["bsp1"]), ("obs", obs), ("d1min", d1min),
                         ("tsp_all", c["tsp_all"]), ("frac_slow", c["frac_slow"]), ("bd", c["bd"]),
                         ("coherence", c["coherence_mean"]), ("centroid_sp", c["centroid_sp_mean"])):
        for w in c["_windows"]:
            f["c_{}_m{}".format(name, w)] = _centred(series, w)
    f["c_bsp_past50"] = _roll(c["bsp"], -50, 0)
    f["c_bsp_fut50"] = _roll(c["bsp"], 0, 50)
    f["c_tsp_past50"] = _roll(c["tsp_all"], -50, 0)
    f["c_tsp_fut50"] = _roll(c["tsp_all"], 0, 50)
    names = list(f)
    return np.stack([np.asarray(f[n], dtype=np.float32) for n in names], axis=1), names


def control_features(c, a, b):
    """Features for P(team `a` is the one in control), with `b` the opponent."""
    dA, dB = c["d1"][a], c["d1"][b]
    gap = dB - dA
    cnt6 = c["cnt6"][a] - c["cnt6"][b]

    # Pass bridge: +1 if both previous and next controlled touch belong to `a`, -1 if both `b`
    prev_a = (c["prev_team"] == a).astype(np.float32) - (c["prev_team"] == b).astype(np.float32)
    next_a = (c["next_team"] == a).astype(np.float32) - (c["next_team"] == b).astype(np.float32)
    pass_bridge = np.where((c["prev_team"] == a) & (c["next_team"] == a), 1.0,
                           np.where((c["prev_team"] == b) & (c["next_team"] == b), -1.0, 0.0)).astype(np.float32)

    f = {
        "o_dA1": dA, "o_dB1": dB, "o_dA2": c["d2"][a], "o_dB2": c["d2"][b], "o_gap": gap,
        "o_cA3": c["cnt3"][a], "o_cB3": c["cnt3"][b], "o_cA6": c["cnt6"][a], "o_cB6": c["cnt6"][b],
        "o_cA12": c["cnt12"][a], "o_cB12": c["cnt12"][b], "o_cdiff6": cnt6,
        "o_spA": c["spn"][a], "o_spB": c["spn"][b], "o_tspA": c["tsp"][a], "o_tspB": c["tsp"][b],
        "o_cohA": c["coherence"][a], "o_cohB": c["coherence"][b],
        "o_coh_diff": c["coherence"][a] - c["coherence"][b],
        "o_spreadA": c["spread"][a], "o_spreadB": c["spread"][b],
        "o_radA": c["radial_v"][a], "o_radB": c["radial_v"][b],
        "o_since_own": c["since"][a], "o_since_opp": c["since"][b],
        "o_until_own": c["until"][a], "o_until_opp": c["until"][b],
        "o_last_diff": c["since"][b] - c["since"][a],
        "o_next_diff": c["until"][b] - c["until"][a],
        "o_contact_now": np.where(c["clean_contact"] == a, 1.0,
                                  np.where(c["clean_contact"] == b, -1.0, 0.0)),
        "o_prev_touch": prev_a,
        "o_next_touch": next_a,
        "o_pass_bridge": pass_bridge,
    }
    for name, series in (("gap", gap), ("dA1", dA), ("dB1", dB), ("cdiff6", cnt6),
                         ("last_diff", f["o_last_diff"]), ("next_diff", f["o_next_diff"]),
                         ("pass_bridge", pass_bridge), ("coh_diff", f["o_coh_diff"])):
        for w in c["_windows"]:
            f["o_{}_m{}".format(name, w)] = _centred(series, w)
    f["o_gap_past50"] = _roll(gap, -50, 0)
    f["o_gap_fut50"] = _roll(gap, 0, 50)
    if "dir" in c:
        # Everything in the attacking frame of `a` / `b`: positive = towards that team's target goal.
        half = c["_W"] / 2.0
        da, db = c["dir"][a], c["dir"][b]
        fva = c["mvel"][a][:, 0] * da                    # own block moving forward (m/s)
        fvb = c["mvel"][b][:, 0] * db                    # opponent block moving forward
        fdiff = fva - fvb
        adv_a = (c["centroid"][a][:, 0] - half) * da     # own block height (m past halfway)
        adv_b = (c["centroid"][b][:, 0] - half) * db
        ball_adv = (c["ball"][:, 0] - half) * da         # ball in own attacking half > 0
        f.update({"o_fwdvA": fva, "o_fwdvB": fvb, "o_fwdv_diff": fdiff,
                  "o_advA": adv_a, "o_advB": adv_b, "o_adv_diff": adv_a - adv_b, "o_ball_adv": ball_adv})
        for w in (75, 175, 375, 750):
            f["o_fwdv_diff_m{}".format(w)] = _centred(fdiff, w)
            f["o_adv_diff_m{}".format(w)] = _centred(adv_a - adv_b, w)
            f["o_ball_adv_m{}".format(w)] = _centred(ball_adv, w)
        for w in (50, 125):                              # turnover signature: forward motion flips
            f["o_fwdv_diff_past{}".format(w)] = _roll(fdiff, -w, 0)
            f["o_fwdv_diff_fut{}".format(w)] = _roll(fdiff, 0, w)
    names = list(f)
    return np.stack([np.asarray(f[n], dtype=np.float32) for n in names], axis=1), names
