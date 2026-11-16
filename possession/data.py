"""Loaders. Everything ends up as a `Match`: positions in metres, one row per video frame.

Row index is the 0-based frame, so for the challenge a Match row i is submission row i.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(ROOT, "external_data", "data")
CACHE_DIR = os.path.join(ROOT, "artifacts", "cache")
EXTERNAL_PITCH = (105.0, 68.0)

PERIOD_NONE = 0


@dataclass
class Match:
    name: str
    xy: np.ndarray            # (T, N, 2) metres, NaN = not tracked
    team: np.ndarray          # (N,) int8, 0 = home / first team, 1 = away / second team
    ball: np.ndarray          # (T, 2) metres, NaN = not seen
    period: np.ndarray        # (T,) int8, 0 outside play, 1 / 2 for halves
    fps: int = 25
    pitch: tuple = EXTERNAL_PITCH
    events: pd.DataFrame = None   # external games only; see labels.py
    ball_true: np.ndarray = None  # unmasked ball, kept for diagnostics only
    labels: np.ndarray = None     # per-frame H/A/D/X codes when the source provides them (DFL data)

    @property
    def n_frames(self):
        return self.xy.shape[0]

    def summary(self):
        play = self.period > 0
        ball_nan = np.isnan(self.ball[:, 0])
        return {
            "name": self.name,
            "frames": self.n_frames,
            "players_slots": self.xy.shape[1],
            "frames_in_play": int(play.sum()),
            "ball_missing_in_play_%": round(100.0 * ball_nan[play].mean(), 1) if play.any() else None,
            "mean_tracked_in_play": round(float((~np.isnan(self.xy[play, :, 0])).sum(1).mean()), 1)
            if play.any() else None,
        }


# --------------------------------------------------------------------------- external games 1, 2

def _read_metrica_csv_tracking(path):
    df = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64, na_values=["NaN", ""])
    values = df.values
    if (values.shape[1] - 3) % 2 == 1:      # trailing empty column from the header's last comma
        values = values[:, :-1]
    return values


def _team_xy(values, pitch):
    players = values[:, 3:-2]
    n = players.shape[1] // 2
    xy = players.reshape(len(values), n, 2).copy()
    xy[:, :, 0] *= pitch[0]
    xy[:, :, 1] *= pitch[1]
    ball = values[:, -2:].copy()
    ball[:, 0] *= pitch[0]
    ball[:, 1] *= pitch[1]
    return xy, ball


def load_metrica_csv_game(n, root=EXTERNAL_DIR):
    """Sample_Game_1 / Sample_Game_2 (Metrica CSV format)."""
    folder = os.path.join(root, "Sample_Game_{}".format(n))
    stem = os.path.join(folder, "Sample_Game_{}".format(n))
    home = _read_metrica_csv_tracking(stem + "_RawTrackingData_Home_Team.csv")
    away = _read_metrica_csv_tracking(stem + "_RawTrackingData_Away_Team.csv")
    if len(home) != len(away):
        raise ValueError("home/away tracking length differs: {} vs {}".format(len(home), len(away)))

    first_frame = int(home[0, 1])
    frames = home[:, 1].astype(np.int64) - first_frame
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Sample_Game_{} tracking frames are not contiguous".format(n))

    hxy, ball = _team_xy(home, EXTERNAL_PITCH)
    axy, ball_away = _team_xy(away, EXTERNAL_PITCH)
    ball = np.where(np.isnan(ball), ball_away, ball)

    xy = np.concatenate([hxy, axy], axis=1)
    team = np.concatenate([np.zeros(hxy.shape[1], np.int8), np.ones(axy.shape[1], np.int8)])
    period = np.nan_to_num(home[:, 0], nan=0).astype(np.int8)

    ev = pd.read_csv(stem + "_RawEventsData.csv")
    events = pd.DataFrame({
        "team": ev["Team"].str.strip().str.upper().map({"HOME": 0, "AWAY": 1}),
        "type": ev["Type"].fillna("").str.strip().str.upper(),
        "subtype": ev["Subtype"].fillna("").astype(str).str.strip().str.upper(),
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int) - first_frame,
        "end": ev["End Frame"].astype(int) - first_frame,
    })
    return Match("Sample_Game_{}".format(n), xy.astype(np.float32), team, ball.astype(np.float32),
                 period, 25, EXTERNAL_PITCH, events, ball.astype(np.float32).copy())


# --------------------------------------------------------------------------------- external game 3

def _provider_params(root):
    out = {}
    for param in root.iter("ProviderParameter"):
        name = param.findtext("Name")
        if name:
            out[name] = param.findtext("Value")
    return out


def load_metrica_epts_game3(root=EXTERNAL_DIR):
    """Sample_Game_3 (FIFA / EPTS tracking + json events), parsed without kloppy."""
    folder = os.path.join(root, "Sample_Game_3")
    meta = ET.parse(os.path.join(folder, "Sample_Game_3_metadata.xml")).getroot()

    team_ids = [t.get("id") for t in meta.iter("Team")]
    player_team = {p.get("id"): p.get("teamId") for p in meta.iter("Player")}
    # 35 players in the metadata, 22 per tracking line: each DataFormatSpecification (one per
    # substitution segment) lists which players the 22 groups of its frames belong to.
    channel_index, channel_players = {}, []
    for ch in meta.iter("PlayerChannel"):
        if ch.get("channelId") == "x":
            channel_index[ch.get("id")[:-2]] = len(channel_players)
            channel_players.append(ch.get("playerId"))
    team = np.array([team_ids.index(player_team[pid]) for pid in channel_players], dtype=np.int8)

    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([channel_index[r[:-2]] for r in refs])))

    first_frame = 1
    n_total = max(s[1] for s in specs)
    xy = np.full((n_total, len(channel_players), 2), np.nan, dtype=np.float64)
    ball = np.full((n_total, 2), np.nan, dtype=np.float64)
    with open(os.path.join(folder, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) < 3:
                continue
            frame = int(parts[0])
            spec = next(s for s in specs if s[0] <= frame <= s[1])
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64).reshape(-1, 2)
            xy[frame - first_frame, spec[2]] = vals
            ball[frame - first_frame] = np.array(parts[2].split(",")[:2], dtype=np.float64)
    frame_ids = np.arange(n_total)
    xy[:, :, 0] *= EXTERNAL_PITCH[0]
    xy[:, :, 1] *= EXTERNAL_PITCH[1]
    ball[:, 0] *= EXTERNAL_PITCH[0]
    ball[:, 1] *= EXTERNAL_PITCH[1]
    params = _provider_params(meta)
    period = np.zeros(len(frame_ids), np.int8)
    for p, (a, b) in enumerate((("first_half_start", "first_half_end"),
                                ("second_half_start", "second_half_end")), start=1):
        if params.get(a) and params.get(b):
            period[int(params[a]) - first_frame: int(params[b]) - first_frame + 1] = p

    import json
    with open(os.path.join(folder, "Sample_Game_3_events.json"), encoding="utf-8") as handle:
        raw = json.load(handle)["data"]

    def _subtype(value):
        if not value:
            return ""
        if isinstance(value, dict):
            return str(value.get("name", "")).strip().upper()
        return "-".join(str(v.get("name", "")).strip().upper() for v in value)

    events = pd.DataFrame({
        "team": [team_ids.index(e["team"]["id"]) if e.get("team") else -1 for e in raw],
        "type": [e["type"]["name"].strip().upper() for e in raw],
        "subtype": [_subtype(e.get("subtypes")) for e in raw],
        "period": [int(e["period"]) for e in raw],
        "start": [int(e["start"]["frame"]) - first_frame for e in raw],
        "end": [int(e["end"]["frame"]) - first_frame for e in raw],
    })
    return Match("Sample_Game_3", xy.astype(np.float32), team, ball.astype(np.float32), period, 25,
                 EXTERNAL_PITCH, events, ball.astype(np.float32).copy())


def load_external_games(root=EXTERNAL_DIR, games=(1, 2, 3), cache=True):
    """Load the Metrica open sample games, cached as npz because the text parse is slow."""
    out = []
    for n in games:
        cache_path = os.path.join(CACHE_DIR, "external_game_{}.npz".format(n))
        ev_path = os.path.join(CACHE_DIR, "external_game_{}_events.pkl".format(n))
        if cache and os.path.isfile(cache_path) and os.path.isfile(ev_path):
            z = np.load(cache_path)
            out.append(Match("Sample_Game_{}".format(n), z["xy"], z["team"], z["ball"], z["period"],
                             25, EXTERNAL_PITCH, pd.read_pickle(ev_path), z["ball"].copy()))
            continue
        match = load_metrica_epts_game3(root) if n == 3 else load_metrica_csv_game(n, root)
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache_path, xy=match.xy, team=match.team, ball=match.ball, period=match.period)
            match.events.to_pickle(ev_path)
        out.append(match)
    return out


# ------------------------------------------------------------------------------ challenge (tactical)

def load_challenge(data_dir=None, cache=True):
    """The provided tactical ATD as a Match, row i = submission row i = video frame i.

    Positions only (speed is recomputed from positions so external and challenge features are
    computed identically). The half boundaries are the published graded blocks.
    """
    sys.path.insert(0, ROOT)
    from games import GAME, GAMES, require, half_video_blocks, source_frames

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    pitch = GAMES[GAME]["pitch"]
    n_frames = source_frames("tactical", GAME)

    cache_path = os.path.join(CACHE_DIR, "challenge_tactical.npz")
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        xy, ball, team = z["xy"], z["ball"], z["team"]
    else:
        root = ET.parse(meta_path).getroot()
        team_ids = [t.get("id") for t in root.iter("Team")]
        if len(team_ids) != 2:
            raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
        slots = [p.get("teamId") for p in root.iter("Player")]
        team = np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)  # first = home
        n_slots = len(slots)

        xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
        ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
        with open(raw_path) as handle:
            for line in handle:
                parts = line.split(":")
                if len(parts) != 3:
                    continue
                row = int(parts[0]) - 1                     # ATD frame_id counts from 1
                values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
                xy[row] = values.reshape(n_slots, -1)[:, :2]
                b = parts[2].split(",")
                ball[row] = (float(b[0]), float(b[1]))
        xy[:, :, 0] *= pitch[0]
        xy[:, :, 1] *= pitch[1]
        ball[:, 0] *= pitch[0]
        ball[:, 1] *= pitch[1]
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache_path, xy=xy, ball=ball, team=team)

    period = np.zeros(n_frames, np.int8)
    for half, first, last in half_video_blocks(GAME):
        period[first:last + 1] = half
    return Match("BAR-ATH", xy, team, ball, period, 25, tuple(pitch), None, None)
