"""Smart Tagging integration: set-piece restart pinpointing and tactical phase priors.

Key facts from `DATA.md`, `OVERVIEW.md` and `starter/01_smart_tagging.ipynb`:
  1. Each instance is a window `[start_frame, end_frame]` in video frames (`round(seconds * 25)`).
  2. Set-piece windows (`code == 'SET PIECES'`) open ~3.5 seconds *before* the restart kick and
     contain the true restart frame rather than starting on it. We use the tag to know a restart
     happened inside `[start_frame, end_frame]` and use the tracking data (ball acceleration /
     player speed jump) to pinpoint the exact frame `kick_frame` the ball is put back into play.
     Frames before `kick_frame` are dead ball (`D`); frames immediately after `kick_frame` are live
     play owned by the restarting team (when `Team` is not `N/A`).
  3. Offensive phase codes (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING TRANSITION`,
     `LONG BALL`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`) name the team in
     possession.
  4. Defensive phase codes (`DEFENDING IN ATTACKING THIRD`, `DEFENDING IN MIDDLE THIRD`,
     `DEFENDING IN DEFENSIVE THIRD`, `DEFENSIVE TRANSITION`) name the *defending* team, so the
     team in possession is the *opponent* (`1 - team`).
"""

import numpy as np

from .ball import fill_ball
from .data import ROOT

OFFENSIVE_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
}

DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}


def _parse_team(team_str):
    """Map Smart Tagging Team string to 0 (Home / Barcelona), 1 (Away / Athletic), or -1 (N/A)."""
    if not team_str:
        return -1
    s = str(team_str).strip().lower()
    if s in ("n/a", "na", "none", ""):
        return -1
    if "barc" in s or s == "home" or s == "team a":
        return 0
    if "atlet" in s or "athlet" in s or s == "away" or s == "team b":
        return 1
    return -1


def load_challenge_smart_tagging(data_dir=None):
    """Load Smart Tagging instances for the challenge match using `starter/load_smart_tagging.py`."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from starter.load_smart_tagging import load as _load_st
    return _load_st(data_dir=data_dir)


def simulate_external_smart_tagging(match, seed=42):
    """Build realistic noisy Smart Tagging intervals on an external game from its event stream.

    Used ONLY on external games so we can validate the Smart Tagging prior layer out-of-fold
    without touching the challenge ground truth:
      - Every SET PIECE event gets a window opening ~3.5s early and closing ~2s after the restart,
        with Throw-ins having Team = N/A (matching the challenge's 36 N/A throw-in tags).
      - Active possession stretches (>= 5s) emit noisy phase windows (+/- 1.5s boundary jitter,
        8% false team flip rate) mimicking automatic tactical phase tagging.
    """
    rng = np.random.default_rng(seed)
    fps = match.fps
    T = match.n_frames
    ev = match.events
    instances = []

    for row in ev.itertuples():
        if row.start < 0 or row.start >= T:
            continue
        is_sp = row.type == "SET PIECE" or (row.type == "PASS" and "GOAL KICK" in row.subtype)
        if is_sp:
            pre = int(rng.uniform(2.5, 4.5) * fps)
            post = int(rng.uniform(1.5, 3.0) * fps)
            sub = row.subtype
            if "THROW" in sub:
                sp_type, team_str = "Throw-in", "N/A"
            elif "CORNER" in sub:
                sp_type = "Corner Kick"
                team_str = "Home" if row.team == 0 else "Away"
            elif "GOAL KICK" in sub:
                sp_type = "Goal Kick"
                team_str = "Home" if row.team == 0 else "Away"
            elif "KICK OFF" in sub:
                sp_type = "Kick Off (Start)"
                team_str = "Home" if row.team == 0 else "Away"
            else:
                sp_type = "Free Kick"
                team_str = "Home" if row.team == 0 else "Away"
            instances.append({
                "code": "SET PIECES",
                "start_frame": max(0, int(row.start) - pre),
                "end_frame": min(T, int(row.start) + post),
                "tags": {"Team": team_str, "Type": sp_type},
            })
        elif row.type == "SHOT" and "GOAL" in row.subtype and "SAVED" not in row.subtype:
            instances.append({
                "code": "GOALS",
                "start_frame": max(0, int(row.start) - 2 * fps),
                "end_frame": min(T, int(row.end) + 2 * fps),
                "tags": {"Team": "Home" if row.team == 0 else "Away"},
            })

    # Also emit noisy phase intervals from PASS/CARRY chains
    pass_ev = ev[ev["type"].isin(["PASS", "CARRY"]) & ev["team"].isin([0, 1])].sort_values("start")
    cur_team, cur_start, cur_end = -1, 0, 0
    for row in pass_ev.itertuples():
        if row.team == cur_team and row.start - cur_end <= 6 * fps:
            cur_end = max(cur_end, row.end)
        else:
            if cur_team in (0, 1) and (cur_end - cur_start) >= 5 * fps:
                _add_simulated_phase(instances, cur_team, cur_start, cur_end, T, fps, rng)
            cur_team, cur_start, cur_end = row.team, row.start, max(row.start + fps, row.end)
    if cur_team in (0, 1) and (cur_end - cur_start) >= 5 * fps:
        _add_simulated_phase(instances, cur_team, cur_start, cur_end, T, fps, rng)

    return instances


def _add_simulated_phase(instances, team, start, end, T, fps, rng):
    jitter_s = int(rng.normal(0, 1.2 * fps))
    jitter_e = int(rng.normal(0, 1.2 * fps))
    a = int(np.clip(start + jitter_s, 0, T - 1))
    b = int(np.clip(end + jitter_e, a + 1, T))
    # 8% chance of noisy tag error, 30% chance of emitting as a defensive tag for the other team
    noisy_team = (1 - team) if rng.random() < 0.08 else team
    if rng.random() < 0.30:
        code = "DEFENDING IN MIDDLE THIRD"
        tag_team = 1 - noisy_team
    else:
        code = "PROGRESSION"
        tag_team = noisy_team
    instances.append({
        "code": code,
        "start_frame": a,
        "end_frame": b,
        "tags": {"Team": "Home" if tag_team == 0 else "Away"},
    })


def _pinpoint_kick_frame(match, ball_speed, player_mean_speed, obs, a, b, sp_type=""):
    """Pinpoint the exact restart frame inside a Set Piece window `[a, b)`.

    Set-piece tags open ~3.5 seconds before the ball is struck. Inside `[a, b)`, the restart is
    marked by the ball transitioning from stationary/absent to fast movement (> 4.5 m/s), or by
    players accelerating around the restart point.
    """
    a = max(0, int(a))
    b = min(match.n_frames, int(b))
    if b - a <= 4:
        return a

    # Known published kickoff anchors (from DATA.md / starter/load_smart_tagging.py)
    if "kick off (start)" in sp_type.lower():
        if a <= 11185 <= b:
            return 11185
        if a <= 113742 <= b:
            return 113742

    # Search from the start of the window up to 92% (do not skip the initial 18% of long windows)
    s0 = a
    s1 = a + int(0.92 * (b - a))
    if s1 <= s0:
        s0, s1 = a, b

    sub_bsp = np.nan_to_num(ball_speed[s0:s1], nan=0.0)
    sub_obs = obs[s0:s1]

    # Look for the first visible ball frame that accelerates past 3.5 m/s
    fast_visible = np.flatnonzero(sub_obs & (sub_bsp >= 3.5))
    if len(fast_visible):
        k = int(fast_visible[0])
        # Walk back a couple of frames to the start of the acceleration impulse
        while k > 0 and sub_obs[k - 1] and sub_bsp[k - 1] >= 1.5:
            k -= 1
        return s0 + k

    # Fallback: largest upward jump in bridged ball speed + player speed in the window
    sub_psp = np.nan_to_num(player_mean_speed[s0:s1], nan=0.0)
    score = sub_bsp + 2.0 * np.gradient(sub_bsp) + 1.5 * sub_psp
    if len(score):
        # Bias slightly toward the middle of the window
        prior = -0.5 * ((np.linspace(0, 1, len(score)) - 0.5) / 0.35) ** 2
        return s0 + int(np.argmax(score + prior))
    return a + int(0.5 * (b - a))


def apply_smart_tagging_priors(match, instances, p_dead, p_home,
                               phase_logit_weight=0.55,
                               restart_dead_boost=2.2,
                               restart_live_boost=2.0,
                               phase_dead_suppression=0.7):
    """Combine Smart Tagging set-piece anchors and tactical phase priors with model probabilities.

     operates in log-odds space so strong model certainties are respected while ambiguous /
    missing-ball stretches follow the tactical phase and restart structure.

    Returns (p_dead_adj, p_home_adj, summary_dict).
    """
    T = match.n_frames
    fps = match.fps
    eps = 1e-4

    filled, obs = fill_ball(match.ball, pitch=match.pitch, fps=fps)
    bsp = np.zeros(T, dtype=np.float32)
    bsp[2:-2] = np.linalg.norm(filled[4:] - filled[:-4], axis=1) * fps / 4.0

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp_players = np.zeros((T, match.xy.shape[1]), dtype=np.float32)
        sp_players[2:-2] = np.linalg.norm(match.xy[4:] - match.xy[:-4], axis=2) * fps / 4.0
        player_mean_sp = np.nanmean(sp_players, axis=1)

    logit_dead = np.log(np.clip(p_dead, eps, 1 - eps) / (1.0 - np.clip(p_dead, eps, 1 - eps)))
    logit_home = np.log(np.clip(p_home, eps, 1 - eps) / (1.0 - np.clip(p_home, eps, 1 - eps)))

    # Accumulate phase evidence per frame: +1 for Home possession, -1 for Away possession
    phase_vote = np.zeros(T, dtype=np.float32)
    phase_count = np.zeros(T, dtype=np.float32)

    set_pieces = []
    goals = []

    for inst in instances:
        code = (inst.get("code") or "").strip().upper()
        a = max(0, min(T, int(inst.get("start_frame", 0))))
        b = max(a, min(T, int(inst.get("end_frame", a))))
        tags = inst.get("tags") or {}
        team = _parse_team(tags.get("Team"))

        if code == "SET PIECES":
            sp_type = str(tags.get("Type") or "")
            kick = _pinpoint_kick_frame(match, bsp, player_mean_sp, obs, a, b, sp_type)
            set_pieces.append((a, kick, b, team, sp_type))
        elif code == "GOALS":
            goals.append((a, b, team))
        elif code in OFFENSIVE_CODES and team in (0, 1):
            sign = 1.0 if team == 0 else -1.0
            phase_vote[a:b] += sign
            phase_count[a:b] += 1.0
        elif code in DEFENSIVE_CODES and team in (0, 1):
            # Defensive tag names the defending team -> opponent has the ball!
            poss_team = 1 - team
            sign = 1.0 if poss_team == 0 else -1.0
            phase_vote[a:b] += sign
            phase_count[a:b] += 1.0

    # 1. Apply tactical phase prior (stronger when ball is unobserved, gentler when ball is seen)
    has_phase = phase_count > 0
    avg_vote = np.zeros(T, dtype=np.float32)
    avg_vote[has_phase] = phase_vote[has_phase] / phase_count[has_phase]
    weight_frame = np.where(obs, 0.35 * phase_logit_weight, 1.0 * phase_logit_weight)
    # When ball is in fast flight / passing (> 8 m/s), phase prior takes precedence over raw 2D player proximity
    fast_mask = obs & (bsp >= 8.0)
    weight_frame[fast_mask] = np.maximum(weight_frame[fast_mask], 2.5 * phase_logit_weight)
    logit_home += weight_frame * avg_vote

    # Suppress false dead ball during confirmed active tactical phases
    if phase_dead_suppression > 0:
        logit_dead[has_phase] -= phase_dead_suppression

    # Active play guard: if in an active tactical phase and ball is observed moving inside pitch,
    # the ball cannot be dead. Overrule stoppage inertia.
    active_moving = has_phase & obs & (bsp >= 1.5)
    logit_dead[active_moving] = np.minimum(logit_dead[active_moving], -1.0)

    # 2. Apply Set Piece restart anchors
    set_pieces.sort(key=lambda x: x[1])
    for a, kick, b, team, sp_type in set_pieces:
        # Pre-kick window: the stoppage leading up to `kick`
        # Extend backward up to 8s before `a` if the ball was already missing or players were slow,
        # but stop if lookback touches an active tactical phase.
        pre_start = a
        lookback = max(0, a - 8 * fps)
        for t in range(a - 1, lookback - 1, -1):
            if has_phase[t]:
                break
            if not obs[t] or player_mean_sp[t] < 2.2:
                pre_start = t
            else:
                break
        if kick > pre_start:
            logit_dead[pre_start:kick] += restart_dead_boost

        # Post-kick window: play just restarted at `kick`!
        live_end = min(T, max(b, kick + 2 * fps))
        if live_end > kick:
            boost_span = min(T, kick + 3 * fps)
            logit_dead[kick:boost_span] = np.minimum(logit_dead[kick:boost_span], -1.2)
            logit_dead[boost_span:live_end] -= restart_live_boost
            if team in (0, 1):
                sign = 1.0 if team == 0 else -1.0
                # Stronger team anchor for the first 3 seconds after a known-team set piece
                team_end = min(T, kick + int(3.0 * fps))
                logit_home[kick:team_end] += sign * 2.0

    # 3. Goals -> dead ball until the next Kick Off restart
    for ga, gb, _ in goals:
        goal_frame = (ga + gb) // 2
        next_kicks = [sp[1] for sp in set_pieces if sp[1] > goal_frame and "kick off" in sp[4].lower()]
        stop_dead = next_kicks[0] if next_kicks else min(T, gb + 35 * fps)
        if stop_dead > goal_frame:
            logit_dead[goal_frame:stop_dead] += restart_dead_boost + 1.0

    # 4. Ball Flight Momentum Rule (Pass & Aerial Clearance Continuity)
    # When a ball accelerates into high-speed flight (>= 12 m/s, e.g. goalkeeper clearance, long ball),
    # possession remains with the kicking team while the ball is airborne/traveling.
    # An opponent merely under the flight path in 2D does NOT intercept unless the ball decelerates
    # (<= 6.0 m/s) upon reception or physical control.
    in_flight = False
    flight_team = None
    for t in range(1, T):
        if not in_flight and bsp[t] >= 12.0 and bsp[t - 1] < 12.0:
            in_flight = True
            flight_team = 0 if logit_home[t - 1] >= 0.0 else 1
        elif in_flight:
            if bsp[t] >= 6.0:
                if flight_team == 0:
                    logit_home[t] = max(logit_home[t], 1.1)
                else:
                    logit_home[t] = min(logit_home[t], -1.1)
            else:
                in_flight = False
                flight_team = None

    p_dead_adj = 1.0 / (1.0 + np.exp(-np.clip(logit_dead, -15, 15)))
    p_home_adj = 1.0 / (1.0 + np.exp(-np.clip(logit_home, -15, 15)))

    summary = {
        "n_instances": len(instances),
        "n_set_pieces": len(set_pieces),
        "n_goals": len(goals),
        "frames_with_phase_prior_%": round(100.0 * float(has_phase[match.period > 0].mean()), 1)
        if (match.period > 0).any() else 0.0,
    }
    return p_dead_adj, p_home_adj, summary

