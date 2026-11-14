"""Hierarchical Possession Detection Engine based on BALL_POSSESSION_LOGIC.md.

Implements:
1. Match Boundary Handling (pre-kickoff, halftime, post-match -> X)
2. Dead Ball Detection (Out-of-bounds, Smart Tagging set pieces -> D)
3. Ball Trajectory & Situation Classification (PASS_IN_PROGRESS, CONTROLLED_BALL)
4. Possession State Machine with Temporal Confirmation / Hysteresis
5. Full Match 199,319 frames CSV generation
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

import games
from starter.load_smart_tagging import load as load_smart_tags

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm

# Pitch dimensions in meters
PITCH_X = 105.3
PITCH_Y = 68.0

# Match Periods (video frame indices)
# Halves from games.py: H1: 11185 to 90651, H2: 110808 to 190197
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197

# Engineering parameters from BALL_POSSESSION_LOGIC.md
CONTROL_DISTANCE_THRESHOLD = 2.5   # meters: max distance for physical ball control
CONFIRMATION_FRAMES = 12           # ~0.48 sec: temporal confirmation required to switch possession
MAX_OUT_OF_BOUNDS_MARGIN = 0.5     # meters: boundary tolerance


def load_team_slots(meta_path):
    """Map each track slot index (0..45) to 0 (Home) or 1 (Away)."""
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    home_id = teams[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if s == home_id else 1 for s in slots], dtype=np.int8)


def get_set_piece_intervals():
    """Extract set piece frame intervals from Smart Tagging to assist Dead Ball classification."""
    try:
        events = load_smart_tags()
        sp_intervals = []
        for e in events:
            if e.get("code") == "SET PIECES":
                s = max(0, e["start_frame"])
                e_f = min(games.source_frames("tactical"), e["end_frame"])
                sp_intervals.append((s, e_f))
        return sp_intervals
    except Exception as err:
        print(f"Warning: could not load smart tagging: {err}")
        return []


def run_possession_pipeline():
    print("=" * 70)
    print("RUNNING HIERARCHICAL FOOTBALL POSSESSION ENGINE (BALL_POSSESSION_LOGIC.md)")
    print("=" * 70)

    paths = games.asset_paths()
    meta_path = paths["atd_meta"]
    raw_path = paths["atd_raw"]
    n_frames = games.source_frames("tactical")

    print(f"Loading track metadata: {os.path.basename(meta_path)}")
    team_of_slot = load_team_slots(meta_path)

    print("Extracting Smart Tagging set-piece intervals...")
    set_pieces = get_set_piece_intervals()
    print(f"  Found {len(set_pieces)} set piece intervals.")

    # Arrays for full match (199,319 frames)
    print(f"Processing {n_frames} frames from tracking...")
    closest_team = np.full(n_frames, -1, dtype=np.int8)
    closest_dist = np.full(n_frames, 999.0, dtype=np.float32)
    ball_pos = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as f:
        for fnum, line in enumerate(f):
            if fnum >= n_frames:
                break
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue

            ball_str = parts[2]
            if ball_str[0] != "N":  # Ball is tracked
                b_tokens = ball_str.split(",")
                bx = float(b_tokens[0]) * PITCH_X
                by = float(b_tokens[1]) * PITCH_Y
                ball_pos[fnum] = [bx, by]

                # Find closest player
                best_d = 999.0
                best_t = -1
                for s_idx, p_str in enumerate(parts[1].split(";")):
                    if p_str[0] == "N":
                        continue
                    p_tokens = p_str.split(",")
                    px = float(p_tokens[0]) * PITCH_X
                    py = float(p_tokens[1]) * PITCH_Y
                    d = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
                    if d < best_d:
                        best_d = d
                        best_t = team_of_slot[s_idx]

                closest_team[fnum] = best_t
                closest_dist[fnum] = best_d

    print("Tracking data parsed. Now running Hierarchical State Machine...")

    # Step 5: Possession State Machine with Temporal Confirmation
    output_labels = np.full(n_frames, "X", dtype="<U1")

    # Process Half 1 and Half 2
    for half_idx, (start_f, end_f, kickoff_team) in enumerate([
        (H1_START, H1_END, 1),   # H1: Athletic Club kicked off (team 1)
        (H2_START, H2_END, 0),   # H2: Barcelona kicked off (team 0)
    ], start=1):
        print(f"  Processing Half {half_idx} (frames {start_f} to {end_f})...")

        current_team_possession = kickoff_team
        confirmation_counter = 0

        for f in range(start_f, end_f + 1):
            bx, by = ball_pos[f]

            # 1. Dead Ball condition: Out of Bounds
            if not np.isnan(bx):
                if (bx < -MAX_OUT_OF_BOUNDS_MARGIN or bx > PITCH_X + MAX_OUT_OF_BOUNDS_MARGIN or
                    by < -MAX_OUT_OF_BOUNDS_MARGIN or by > PITCH_Y + MAX_OUT_OF_BOUNDS_MARGIN):
                    output_labels[f] = "D"
                    confirmation_counter = 0
                    continue

            # 2. Dead Ball condition: Check if inside a set-piece dead ball stoppage
            # (only if ball is missing or players are far/preparing)
            in_set_piece = any(s <= f <= e for s, e in set_pieces)
            if in_set_piece and (np.isnan(bx) or closest_dist[f] > 3.0):
                output_labels[f] = "D"
                confirmation_counter = 0
                continue

            # 3. Active Play: Evaluate Control & Turnover Hysteresis
            t = closest_team[f]
            d = closest_dist[f]

            # If opponent is in physical control distance
            if t != -1 and d <= CONTROL_DISTANCE_THRESHOLD:
                if t == current_team_possession:
                    # Current controlling team maintains ball
                    confirmation_counter = 0
                else:
                    # Opponent challenging or taking control
                    confirmation_counter += 1
                    # Temporal confirmation: require sustained control to switch teams
                    if confirmation_counter >= CONFIRMATION_FRAMES:
                        current_team_possession = t
                        confirmation_counter = 0
            else:
                # PASS_IN_PROGRESS / AERIAL / FREE BALL:
                # Ball is travelling or in flight -> preserve current team possession
                confirmation_counter = 0

            output_labels[f] = "H" if current_team_possession == 0 else "A"

    # Step 6: Save Submission CSV
    out_dir = os.path.join(ROOT, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "hierarchical_possession_v1.csv")

    print(f"Writing full submission file to: {out_csv}")
    with open(out_csv, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(output_labels))

    print("Submission generated successfully.")

    # Step 7: Score against public ground truth
    print("\nScoring against public ground truth...")
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_df = pm.read_label_csv(gt_path)
    gt_codes = pm.label_codes(gt_df)
    pred_df = pm.read_label_csv(out_csv)
    pred_codes = pm.label_codes(pred_df)

    res = pm.evaluate(gt_codes, pred_codes)
    runs = pm.valid_runs(gt_codes)
    pred_spells = pm.possession_spells(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)

    print("-" * 50)
    print("RESULTS:")
    print(f"  spellF1 (Rank Metric) : {res['spell_f1_mean']:.4f}")
    print(f"  macroF1               : {res['macro_f1']:.4f}")
    print(f"  Possessions Detected  : {len(pred_spells)} (GT has {len(gt_spells)})")
    print("-" * 50)


if __name__ == "__main__":
    run_possession_pipeline()

