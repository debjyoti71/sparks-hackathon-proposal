"""Generate a 1-minute visual clip comparing:
1. Ground Truth
2. Shivain's 5-GRU Pipeline (debjyoti_team.csv)
3. Your Hierarchical Rules Engine (hierarchical_possession_v1.csv)
4. Original Nearest-Player Baseline (baseline_smooth50.csv)
"""

import os
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd

from starter.load_homography import load as load_homography

# Configuration
START_FRAME = 11500  # right after kickoff during active gameplay
N_FRAMES = 1500      # 1500 frames = 60 seconds at 25 fps
OUTPUT_VIDEO = "tracking_overlay_1min.mp4"
VIDEO_PATH = "data/tactical/BAR-ATH_tactical-002.mp4"
RAW_ATD_PATH = "data/tactical/BAR-ATH_tactical_atd.txt"
META_ATD_PATH = "data/tactical/BAR-ATH_tactical_atd.xml"

# CSV predictions
GT_PATH = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
SHIVAIN_PRED_PATH = "evaluation/results/debjyoti_team.csv"
RULES_PRED_PATH = "evaluation/results/hierarchical_possession_v1.csv"
BASELINE_PATH = "evaluation/results/baseline_smooth50.csv"

# Colors (BGR)
COLOR_BARCA = (220, 50, 50)       # Reddish/Burgundy in BGR
COLOR_ATLETIC = (50, 200, 50)     # Green for Athletic
COLOR_BALL = (0, 255, 255)        # Bright Yellow
COLOR_BG = (20, 20, 20)           # Dark gray badge background

def get_team_mapping(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    home_id = teams[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return [0 if s == home_id else 1 for s in slots]

def project_point(H, norm_x, norm_y):
    """Project normalized pitch coordinate (0..1) to pixel coordinate (u, v)."""
    p = np.array([norm_x, norm_y, 1.0], dtype=np.float64)
    res = H @ p
    if abs(res[2]) < 1e-6:
        return None
    u = res[0] / res[2]
    v = res[1] / res[2]
    return int(round(u)), int(round(v))

def main():
    print(f"Loading homographies for frames {START_FRAME} to {START_FRAME + N_FRAMES}...")
    H_all, valid_h, _ = load_homography()

    team_map = get_team_mapping(META_ATD_PATH)
    gt_df = pd.read_csv(GT_PATH)
    shivain_df = pd.read_csv(SHIVAIN_PRED_PATH)
    rules_df = pd.read_csv(RULES_PRED_PATH)
    base_df = pd.read_csv(BASELINE_PATH)

    # Read tracking records for the window
    print("Reading ATD tracking lines...")
    tracking_data = {}
    with open(RAW_ATD_PATH) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1  # 0-indexed frame
            if START_FRAME <= fid < START_FRAME + N_FRAMES:
                # Players
                players = []
                for s_idx, p_str in enumerate(parts[1].split(";")):
                    if p_str[0] == "N":  # NaN
                        continue
                    tokens = p_str.split(",")
                    players.append((team_map[s_idx], float(tokens[0]), float(tokens[1])))

                # Ball
                ball = None
                if parts[2][0] != "N":
                    tokens = parts[2].split(",")
                    ball = (float(tokens[0]), float(tokens[1]))

                tracking_data[fid] = (players, ball)
            elif fid >= START_FRAME + N_FRAMES:
                break

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Rendering {N_FRAMES} frames (~60 seconds) to {OUTPUT_VIDEO}...")
    for i in range(N_FRAMES):
        current_frame = START_FRAME + i
        ret, frame = cap.read()
        if not ret:
            print("Video ended prematurely.")
            break

        H = H_all[current_frame]
        players, ball = tracking_data.get(current_frame, ([], None))

        # 1. Draw Players
        for team, px, py in players:
            pt = project_point(H, px, py)
            if pt and 0 <= pt[0] < width and 0 <= pt[1] < height:
                color = COLOR_BARCA if team == 0 else COLOR_ATLETIC
                cv2.circle(frame, pt, 8, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                lbl = "H" if team == 0 else "A"
                cv2.putText(frame, lbl, (pt[0]-4, pt[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Ball
        if ball is not None:
            bpt = project_point(H, ball[0], ball[1])
            if bpt and 0 <= bpt[0] < width and 0 <= bpt[1] < height:
                cv2.circle(frame, bpt, 9, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, bpt, 7, COLOR_BALL, -1, cv2.LINE_AA)
                cv2.circle(frame, bpt, 3, (0, 0, 255), -1, cv2.LINE_AA)

        # 3. Possession Info Overlay (Top-Right HUD)
        gt_label = gt_df.iloc[current_frame]["label"]
        shivain_label = shivain_df.iloc[current_frame]["label"]
        rules_label = rules_df.iloc[current_frame]["label"]
        base_label = base_df.iloc[current_frame]["label"]

        # HUD Box
        hud_w, hud_h = 450, 195
        hud_x = width - hud_w - 30
        hud_y = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 2)

        def label_desc(lbl):
            if lbl == "H": return "BARCELONA (H)", (100, 100, 255)
            if lbl == "A": return "ATHLETIC (A)", (100, 255, 100)
            if lbl == "D": return "DEAD BALL (D)", (160, 160, 160)
            return "UNKNOWN (X)", (200, 200, 200)

        gt_text, gt_col = label_desc(gt_label)
        shivain_text, shivain_col = label_desc(shivain_label)
        rules_text, rules_col = label_desc(rules_label)
        base_text, _ = label_desc(base_label)

        match_time_s = (current_frame - 11185) / 25.0
        mm = int(max(0, match_time_s) // 60)
        ss = int(max(0, match_time_s) % 60)

        cv2.putText(frame, f"TIME: {mm:02d}:{ss:02d}  (Frame {current_frame})", (hud_x + 15, hud_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"GROUND TRUTH : {gt_text}", (hud_x + 15, hud_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gt_col, 2, cv2.LINE_AA)

        cv2.putText(frame, f"5-GRU PIPELINE: {shivain_text}", (hud_x + 15, hud_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, shivain_col, 2, cv2.LINE_AA)

        cv2.putText(frame, f"YOUR RULES    : {rules_text}", (hud_x + 15, hud_y + 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, rules_col, 2, cv2.LINE_AA)

        cv2.putText(frame, f"OLD BASELINE  : {base_text}", (hud_x + 15, hud_y + 154),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)

        # Ball visibility status
        ball_status = "TRACKED" if ball is not None else "LOST (NaN)"
        ball_col = (0, 255, 0) if ball is not None else (0, 0, 255)
        cv2.putText(frame, f"BALL: {ball_status}", (hud_x + 15, hud_y + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_col, 1, cv2.LINE_AA)

        out.write(frame)
        if (i + 1) % 250 == 0:
            print(f"Rendered {i + 1}/{N_FRAMES} frames ({int((i+1)/N_FRAMES*100)}%)...")

    cap.release()
    out.release()
    print(f"DONE! Saved 1-minute clip to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
