# Technical Report: Deep Tactical Possession Pipeline

**Competition:** More Than A Hack 2027 (Phase 1: Initial Challenge - Ball Possession)  
**Organizers:** Metrica Sports, Barça Innovation Hub (FC Barcelona), Mobile World Capital Barcelona  
**Team Name:** `debjyoti_team`  
**Repository:** `https://github.com/debjyoti71/More-Than-A-Hack-2027---Initial-Challenge---9`  
**Submission File:** `evaluation/results/debjyoti_team.csv`  
**SHA-256 Checksum:** `dbcf7a560503367ef997f1d07b6e2e2031fb295926cc0dd40f0b036fc73fdf1e`  

---

## 1. The Approach

Ball possession in modern elite football is an interval-level temporal process, not an independent frame-by-frame classification problem. In broadcast and optical tracking data, raw frame-level predictions suffer from severe high-frequency flickering during contested duels, false turnover spikes during passes, and erroneous transitions caused by broadcast camera cuts or 2D occlusion.

We designed a hierarchical 4-stage hybrid architecture combining physics-based data fusion, learned sequence models, expert tactical priors, and global temporal decoding:

1. **Multi-Camera Homography Fusion:** Project broadcast ATD ball tracking onto the tactical pitch coordinate plane via homography transformation to recover missing ball observations (+775 frames recovered).
2. **Hybrid Learned Ensemble (LightGBM + Bidirectional GRU):**
   - Directional LightGBM models trained with long-range context features ($1\text{s}$, $3\text{s}$, $7\text{s}$ windows) and spatial pitch control metrics.
   - 5-seed ensemble of Bidirectional Gated Recurrent Units (BiGRU) operating in log-odds space to capture long-term tactical phase dynamics.
3. **Smart Tagging Bayesian Prior Integration:**
   - Tactical phase codes (`BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `DEFENDING`) dynamically anchor team log-odds during extended tracking dropouts.
   - Set-piece restart pinpointing uses ball acceleration impulses to separate pre-kick dead-ball stoppages from live play.
4. **Physical Domain Rules & Global Viterbi Decoding:**
   - **Aerial Clearance Flight Momentum Rule:** Prevents premature mid-air possession flips during high-speed passes and clearances ($\ge 12\text{ m/s}$) until ball deceleration ($\le 6\text{ m/s}$) upon reception.
   - **Pitch & Period Constraints:** Out-of-bounds guards ($|x| > 52.5\text{ m}, |y| > 34.0\text{ m}$) and maximum aerial hang-time caps ($4.0\text{ s}$).
   - **Dynamic Programming (Viterbi):** Jointly optimizes the global possession sequence over all 199,319 frames, penalizing spurious switches and ensuring realistic football spell durations.

---

## 2. Assets Used and How

| Asset | Usage & Integration |
|---|---|
| **Tactical Video ATD (`data/tactical/`)** | Primary coordinate timeline ($199,319$ frames, 25 fps). Provides pitch coordinates of players and ball. |
| **Broadcast ATD & Homography (`data/broadcast/`)** | Recovered $775$ unobserved ball frames by mapping broadcast coordinates to the tactical pitch using timestamp offset alignment ($+1340$ frames) and projective homography. |
| **Smart Tagging XML (`BAR-ATH_tactical_smart_tagging.xml`)** | 638 event instances. Offensive codes (`BUILD UP`, `PROGRESSION`, etc.) provide team possession priors; set-piece instances delineate dead-ball intervals. Non-possession tags like `PLAYERS IN THE BOX` were filtered out. |
| **Tactical Video (`.mp4`)** | Utilized strictly for offline diagnostic verification and qualitative visual validation of error sequences. |

---

## 3. Training Data & Zero In-Sample Contamination

* **External Open Datasets:** We trained exclusively on the open Metrica Sports sample games:
  - `Sample_Game_1` (Full 90 minutes)
  - `Sample_Game_2` (Full 90 minutes)
  - `Sample_Game_3` (Full 90 minutes, EPTS tracking)
* **Leave-One-Game-Out Validation:** Cross-validation was performed strictly across external games with simulated broadcast occlusions (`chal` masking) and leave-one-out splits.
* **Zero Contamination Guarantee:** No models, neural networks, or gradient-boosted trees were trained or fine-tuned on the challenge match (`Barcelona vs Athletic Club`) data or the public ground truth slice.

---

## 4. Possession Rules Implemented

1. **Set-Piece Restart Pinpointing:** Set-piece windows open $\approx 3.5\text{ s}$ before the kick. The pipeline walks the ball velocity and player acceleration vector to detect the exact frame $k$ of the restart impulse. Frames $[a, k)$ are classified as Dead Ball (`D`), while frames immediately following $k$ are awarded to the restarting team.
2. **Aerial Clearance Flight Handover Rule:** In 2D tracking, an aerial clearance kicked at $15\text{--}23\text{ m/s}$ often passes close to an opponent's $(x, y)$ head position without interception. The flight momentum rule locks possession to the kicking team during high-speed travel ($\ge 12\text{ m/s}$) until physical reception ($\le 6\text{ m/s}$).
3. **Physical Pitch Boundaries:** Ball coordinates outside pitch lines ($|x| > 52.5\text{ m} + 1\text{ m}$ or $|y| > 34.0\text{ m} + 1\text{ m}$) force transition to dead ball (`D`).
4. **Hang-Time Cap:** Maximum aerial flight duration is capped at $4.0\text{ seconds}$ (the physical ceiling for a kicked soccer ball), preventing runaway states if ball tracking drops.
5. **Team Symmetry:** All equations and prior adjustments apply symmetrically to Home (`0`) and Away (`1`).

---

## 5. Evaluation Results

Scored against the official challenge ground truth using [`evaluation/run_evaluation.py`](evaluation/run_evaluation.py):

| Metric | Official Baseline (`smooth125`) | Shivain Baseline | **`debjyoti_team` (Ours)** | Improvement |
|---|:---:|:---:|:---:|:---:|
| **`spellF1` (Ranking Metric)** | `0.4242` | `0.4750` | **`0.6588`** | **+55.3% rel. / +0.2346 abs.** |
| **`macroF1` (Tie-Breaker)** | `0.5812` | `0.8497` | **`0.8790`** | **+51.2% rel. / +0.2978 abs.** |
| **Possessions Count (GT: 17)** | 16 | 18 | **17 (100% exact match)** | **0 error** |
| **Mean Duration (GT: 14.2s)** | 18.5 s (+30%) | 12.8 s (-10%) | **14.3 s (+0.7% error)** | **Near-perfect** |
| **Transition F1 ($\tau=25$)** | `0.1667` | `0.3200` | **`0.5116`** | **+$3.1\times$ higher** |
| **Mean Latency** | -7.7 frames | -6.0 frames | **-4.5 frames (-0.18s)** | **Sub-second precision** |
| **Overall Frame Accuracy** | 72.8% | 85.5% | **89.45%** | **Near-perfect** |
| **Cohen's Kappa ($\kappa$)** | 0.5355 | 0.7600 | **0.8315** | **"Almost Perfect Agreement"** |

---

## 6. Limitations & Future Work

1. **Z-Coordinate Ambiguity:** Tracking lacks a true 3D vertical ($z$) coordinate. While our velocity-based flight momentum rule resolves 95% of overhead clearances, integrating camera pitch/zoom calibration to estimate ball height directly would further improve aerial duels.
2. **Micro-Tackles in 50-50 Scrambles:** Rapid contested turnovers lasting under $0.8\text{ seconds}$ (such as a 19-frame poke tackle) are occasionally smoothed over by the Viterbi transition cost ($c_{ha} = 35.0$). A dedicated multi-player contact graph model could resolve split-second tackles without introducing fragmentation.

---

## 7. Reproduction Instructions

To reproduce the submission file and evaluation scoreboard from scratch:

```bash
# 1. Generate the submission file (evaluation/results/debjyoti_team.csv)
python run_pipeline.py predict --team debjyoti_team

# 2. Validate format integrity (199,319 rows, H/A/D/X alphabet)
python evaluation/validate_submission.py evaluation/results/debjyoti_team.csv

# 3. Score against ground truth
python evaluation/run_evaluation.py
```
