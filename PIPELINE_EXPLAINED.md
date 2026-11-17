# How our possession pipeline works, what we tried, and where we are

This document explains the code on branch `combined-pipeline`: what each part does, how to run it,
every experiment we ran, and what each one achieved. It is internal, for the team, and is not the
technical document for submission (that one still needs writing).

---

## 1. The task in one paragraph

For each of the 199,319 frames of the tactical video of Barcelona vs Athletic Club, we output `H`
(Barcelona has the ball), `A` (Athletic has it), `D` (ball dead) or `X` (outside the graded halves).
The ranking metric is **spellF1**. It treats every maximal run of `H` or `A` as one possession and
matches our possessions to the true ones by time overlap (IoU 0.3 to 0.7). That means **flicker and
late or early switches are expensive**, while long correct stretches are cheap. The baseline to beat
is 0.2433 on the private full match.

## 2. The rules we work under

Short version (details in `RULES.md` and the organisers' clarification email):

- **No training, tuning or choosing between versions using the provided match.** That includes the
  public 5-minute ground truth. We use it only once, at the end, to check that the file is aligned.
- **All training and tuning happens on external games.** Every number in this document is measured
  on games the model did not train on.
- **No hand labels, no manual fixes, no checking our outputs against the match video.**
- The repo must regenerate the exact CSV from `data/`. It must not contain `data/`,
  `external_data/` or `artifacts/cache/`.

## 3. How we measure progress without touching the match

We have three external games with labels (Metrica Sports open sample data). For every experiment:

1. **Leave one game out:** train on two games, predict the third, and repeat for all three. This
   gives **out-of-fold (OOF)** scores.
2. **Break each held-out game like the challenge tracking is broken.** We hide the ball in three
   different ways ("masks"), because the challenge ball is missing on 41% of frames:
   - `sim`: short gaps, mostly while the ball is fast or crowded;
   - `raw`: the external tracker's own long gaps, mostly during dead ball;
   - `chal`: gaps shaped like the challenge's (median 10.7 s, p90 34 s). This is the most realistic
     of the three.
3. **The headline number** is the mean OOF spellF1 over 3 games × 3 masks, called **row 7** in the
   training output. We decided in advance that a change is kept only if it raises row 7.

The scorer documentation says spellF1 differences under ~0.04 are noise *on the full match*. Our OOF
numbers average 9 runs, so they are steadier, but small gains (~0.01) are still only moderately
certain.

## 4. The pipeline, step by step

The order is the same for the external games (validation) and the challenge match (prediction).

| # | Step | File | What it does |
|---|---|---|---|
| 0 | Load | `possession/data.py` | Reads the Metrica games and the challenge tactical ATD into one `Match` structure (positions in metres, one row per frame). |
| 0b | Labels | `possession/labels.py` | Builds H/A/D training labels for the external games from their event logs (passes, recoveries, ball out, fouls...). Never used on the challenge. |
| 0c | Masking | `possession/masking.py` | Hides the ball and some players on the external games (`sim`, `raw`, `chal`) so the models practise on challenge-like gaps. |
| 1 | Broadcast fusion | `possession/broadcast.py` | Challenge only. Aligns each live broadcast camera shot to the tactical timeline by matching team shapes, and fills missing tactical ball positions with the broadcast ball. Replays are rejected because they go backwards in time. |
| 2 | Ball bridging | `possession/ball.py` | Fills remaining ball gaps: a curve through both ends for gaps up to 4 s, a straight line for longer ones. The models are told how long each gap is. |
| 3 | Features | `possession/features.py` | About 130 per-frame features: distances from each team to the ball, who touched it last and who touches it next, pass bridges, team speed, shape and coherence, plus the **attack-direction features** (section 6). |
| 4 | Models | `possession/models.py` | Three LightGBM models: **P(dead)**, **P(home has it, given alive)**, and the **turnover detector**, P(the ball passes directly to the other team within ±0.5 s). Team symmetry is enforced by training on both team orientations. |
| 4b | GRU ensemble | `possession/gru.py` | An **ensemble of 5 bidirectional GRUs** (recurrent neural networks, different random seeds, averaged) reads the same features as a sequence over the whole half and predicts the same three probabilities. They are blended with the tree models' (weight 0.8 on the GRUs, in log-odds), so every later step sees the blend. |
| 5 | Smart Tagging priors | `possession/smart_tagging.py` | Challenge tags nudge the probabilities. Set-piece windows mark dead ball until the restart kick, which is found from ball acceleration. Attacking-phase tags favour the tagged team; defending-phase tags favour the other team. On the external games we simulate tags from their events. |
| 6 | Decoder | `possession/decode.py` | Viterbi (most-likely-sequence) decoding over H/A/D with a cost per switch, then very short runs are merged. **The H↔A switch cost is cheap where the turnover detector fires and expensive elsewhere.** All decoder settings are tuned on OOF predictions. |
| 7 | Output | `possession/pipeline.py`, `run_pipeline.py` | Forces the published pre-kick-off part of the 2nd half to `D`, writes the CSV, runs the official validator and prints the file's SHA-256. |

Parts that are in the code but **switched off**, because they lost out of fold (section 7):
`possession/stack.py` (stage-2 stacker), `possession/rules.py` (your rule engine as priors) and
`possession/dfl.py` (extra Bundesliga training data).

### Current final configuration (`artifacts/models/`)

- Training variant `dir_chal`: 6 masked copies of each training game (`sim`×2, `raw`×2, `chal`×2),
  base features plus attack-direction features, 350 boosting rounds.
- GRU ensemble: **5 networks** (seeds 0–4), 128 units per direction, 1 layer, 15 epochs, trained
  on GPU. Blended in with weight **0.8** (`pipeline_cfg.json` → `gru_w`).
- Decoder (`decode_cfg.json`): `c_ha = 40`, `c_ld = 50`, `sw_beta = 6.0` (turnover-detector
  weight), `min_poss_s = 0.0`, `min_dead_s = 0.0`.
- Stacker off, rule priors 0, no DFL data.
- Reference SHA-256 of the submission CSV this produces (two `predict` runs gave the same hash):
  `5bbb327135ae54ec2557a012b047cc176d4b0dc16b4f69c38eaa8475c961f840`.

### The models themselves (what is in `artifacts/models/`)

There are three LightGBM gradient-boosted decision-tree models (350 trees each, 31 leaves, learning
rate 0.05, saved as plain-text files) and an ensemble of five small neural networks (the GRUs). Each
answers yes/no questions for **every frame** with a probability. None of them decides the final
label alone; the blend and the decoder (step 6) combine them.

| File | Question it answers | Trained on which frames | Input features |
|---|---|---|---|
| `dead.txt` | "Is the ball dead in this frame?" → **P(dead)** | every frame in play | 51 team-neutral features: ball speed and acceleration, how long the ball has been unseen, ball distance to the lines, how slow and spread out the players are, all over 1/3/7 s windows |
| `ctrl.txt` | "Given the ball is alive, does the home team have it?" → **P(home)** | only frames where the ball is alive | the same 51 plus 82 **own-vs-opponent** features (59 base + 23 attack-direction): each team's distance to the ball, last and next touch, pass bridges, team shape and speed, forward motion and block height |
| `switch.txt` | "Does the ball pass directly to the other team within ±0.5 s of this frame?" → **P(turnover)** | every frame in play; positive within 0.5 s of a direct H↔A change | same inputs as `ctrl.txt` |
| `gru.pt`, `gru_1.pt` … `gru_4.pt` (+ `gru_cfg.json`, `gru_norm.npz`) | all three questions at once: P(dead), P(own team has it), P(turnover) | every 4th frame of each half, as one sequence | the same 133 features, standardised (`gru_norm.npz` holds the means and spreads) |

**The GRUs in more detail.** Each network is a linear layer (133 → 128, ReLU, dropout 0.1) feeding
a 1-layer bidirectional GRU with 128 units per direction, then a linear layer that outputs 3
log-odds per step. Because it is bidirectional and reads the whole half, each frame's prediction
uses what happened before *and* after it. That is the context a possession boundary needs, and the
trees can't see it. Each is trained on 41-second chunks (256 steps), batch 32, AdamW (learning rate
1e-3), 15 epochs. Each sequence is fed twice, with home and with away as "own" team, and the two
predictions are averaged. Five networks are trained identically except for the random seed (0–4)
and their probabilities averaged. A single small network is noisy; averaging five removes much of
that noise. Predictions are made every 4th frame and interpolated in between.

How they are trained:

- **Data:** only the 3 Metrica open games, each copied 6 times with different simulated tracking
  failures (masks `sim`, `raw` and `chal`, two random seeds each), every 4th frame. Labels come from
  the games' event logs (`possession/labels.py`).
- **Team symmetry:** `ctrl.txt` and `switch.txt` are trained twice over on each frame, once with
  home as "own" and once with away as "own" (label swapped). At prediction time both views are
  averaged, so the model cannot learn a bias for one team.
- **No match data:** the challenge match is never used for training or tuning.

How they are combined, frame by frame:

1. Each tree probability is blended with the GRU ensemble's: log-odds = 0.2·tree + 0.8·GRU. This
   applies to P(dead), P(home) and P(turnover).
   The blended P(dead) and P(home) give three scores per frame: P(H) = (1−P(dead))·P(home),
   P(A) = (1−P(dead))·(1−P(home)), P(D) = P(dead).
2. Smart Tagging nudges those probabilities (set pieces → dead until the restart; phase tags →
   favour the tagged team).
3. The Viterbi decoder picks the single most likely H/A/D sequence for each half. Switching costs
   are `c_ld` between live and dead, and `c_ha` between H and A. **`switch.txt` makes `c_ha`
   frame-dependent:** cheaper where P(turnover) is above its usual level, dearer where it is below.
   `sw_beta = 6` sets how strongly.
4. Runs shorter than `min_poss_s` / `min_dead_s` are merged into a neighbour.

The other files in the folder are not models: `names.json` (feature order each model expects),
`feature_cfg.json` (which features to compute), `decode_cfg.json` (decoder settings),
`rules_cfg.json` (rule-engine weights, both 0) and `pipeline_cfg.json` (which options are on).

## 5. How to run it

From the repo root (Windows PowerShell; on Linux or macOS use `venv/bin/python`):

```powershell
# get the code (everything lives on this branch, not on main)
git fetch
git switch combined-pipeline

# one-time setup (PyTorch first, from its own index; the CPU build is enough to reproduce the CSV)
python -m venv venv
venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python -m pip install -r requirements-pipeline.txt

# data (neither folder is in git)
#   data\                 the competition bundle, laid out as in DATA.md
#   external_data\        only needed to retrain:
#                         git clone https://github.com/metrica-sports/sample-data external_data

# regenerate the submission from the committed models (a few minutes)
venv\Scripts\python run_pipeline.py predict --team <team_name> --no-cache

# retrain everything from the external games (~1 h with a GPU, many hours on CPU)
venv\Scripts\python run_pipeline.py train --gru-device cuda
```

**Check you reproduced it:** the last line of `predict` prints the CSV's SHA-256. It should be
`5bbb327135ae54ec2557a012b047cc176d4b0dc16b4f69c38eaa8475c961f840`. If it differs, first compare
`pip freeze` against `requirements-pipeline.txt` (numpy, pandas, lightgbm, torch) and the Python
version (3.13.6).

**Reproducibility:** `predict` from the committed models should give the same CSV on any machine
(it runs on CPU). Retraining reproduces the **tree models exactly**, but the **GRU weights will differ
slightly** on another machine or GPU, so a retrained pipeline gives a very similar CSV, not a
byte-identical one. The committed models are the reference.

`predict` writes `evaluation/results/<team_name>.csv`, validates it, and prints its SHA-256 so two
runs can be compared byte for byte. `--compare` also writes a rules-only version and a
without-rule-priors version, to look at side by side.

Useful `train` flags, all optional. The defaults are the final configuration.

| flag | meaning |
|---|---|
| `--models DIR --reports DIR` | write to a separate folder (use for experiments, so `artifacts/models` stays intact) |
| `--variant base / chal / chal_long / dir / dir_chal` | which training masks and features (section 7) |
| `--stack auto / on / off` | stage-2 stacker; `auto` keeps it only if it wins OOF |
| `--rule-search` | also search weights for your rule engine as priors |
| `--dfl [--dfl-smooth 2]` | add the 7 open DFL matches as training data (needs `kloppy==3.19.1`, downloads data) |
| `--rounds N` | LightGBM boosting rounds |
| `--no-gru` | skip the GRU (trees only); by default it is trained and kept only if it wins OOF |
| `--gru-device cpu / cuda` | where the GRU trains (prediction always runs on CPU) |
| `--gru-seeds N` | GRU ensemble size (default 5) |
| `--gru-hidden N`, `--gru-layers N` | GRU size (default 128 units, 1 layer) |
| `--gru-epochs N`, `--torch-threads N` | GRU training length (default 15); CPU threads (fixed, for determinism) |

### Reading the training output

The last table, "mean over all held-out games and masks", has one row per method:

| row | what it is |
|---|---|
| 1 | the published baseline idea (nearest player + smoothing), on the same masked games |
| 2 | your rule engine on its own |
| 3 | our models with no decoder (argmax per frame): good macroF1, terrible spellF1 because of flicker |
| 4 | models + untuned decoder |
| 5 | models + Smart Tagging + tuned decoder (with the GRU blend when it is selected) |
| 5c | the same with trees only (GRU weight 0): the comparison point for the GRU |
| 6 | the same with the stacker |
| 6b | the selected pipeline with the turnover detector forced off |
| **7** | **the selected pipeline: the number we compare** |
| 8 | oracle: our team decisions, but a perfect dead/live split |
| 9 | oracle: our dead/live split, but perfect team decisions |

Rows 8 and 9 are not achievable scores. They swap in the true labels for half of the problem to
show where the remaining errors are. In the final run, row 9 (0.89) against row 7 (0.69) says
**most of the remaining loss is still in H vs A decisions (turnovers)**. Row 8 (0.73) says perfect
dead/live boundaries would add only about +0.04.

## 6. The three changes that worked, explained

**Attack-direction features (+0.026).** The original features were deliberately direction-free.
But the team with the ball pushes toward the opponent's goal while the other team retreats. We now
infer each team's attacking direction per half from the tracking alone (a team's goalkeeper is its
deepest player). From it we compute, per team: forward speed of the block, how high the block sits
past halfway, whether the ball is in its attacking half, and whether forward motion flips sign
(a typical turnover signature). These became some of the model's most important features.

**Turnover detector plus turnover-aware decoder (+0.011).** Before, the decoder charged the same
penalty for an H↔A switch at every frame. Now a third model predicts "a direct turnover happens
within ±0.5 s", and the switch penalty is lowered where it fires and raised where it doesn't. The
detector's weight (`sw_beta`) is tuned OOF, and 0 (detector ignored) is always one of the options.
Row 6b vs row 7 isolates the gain.

**Bidirectional GRU blended with the trees (+0.069), the biggest single gain.** The tree models
classify each frame from a few seconds around it. The GRU reads the whole half as a sequence in
both directions, so it can use "home were clearly in control for the last 20 s and away for the
next 20 s" to place a turnover. We expected little from it, given only 3 training games. It won in
**all 9** held-out runs (+0.04 to +0.09 each, including every challenge-like `chal` run), and the
blend-weight curve is smooth (weight 0 / 0.33 / 0.67 / 1 → 0.604 / 0.646 / 0.676 / 0.672). Both
point to a real effect, not luck. In every fold it trains only on the two other games, including
its feature normalisation. The earlier stacker, which only saw the tree outputs, lost; the GRU sees
the raw features over time.

**Then: 5 GRUs instead of 1, and bigger ones (+0.017 more).** Averaging five networks trained with
different seeds took it to 0.6876. Making each one bigger (64 → 128 units) and training longer
(10 → 15 epochs) took it to **0.6926**. A finer blend-weight search also mattered: both runs chose a
weight (0.9 and 0.8) between the old grid points 0.67 and 1.0. Against the single GRU, the final
ensemble is better in 7 of 9 held-out runs, including all three challenge-like `chal` runs (+0.009,
+0.023, +0.023). One run ties, and one is slightly worse (−0.003).

## 7. Everything we tried

All numbers are mean OOF spellF1 over 3 Metrica games × 3 masks (sim, raw, chal), unless noted.

| # | Experiment | Result | Kept? | Why |
|---|---|---|---|---|
| 0 | Published baseline idea (nearest player, w=50) on the same masked games | 0.258 | — | reference |
| 1 | **Original pipeline** (models + Smart Tagging + tuned decoder) | **0.5707** (chal 0.522 · raw 0.636 · sim 0.555) | starting point | |
| 2 | Your rule engine (`possession_engine.py`) on its own | 0.225 | no | almost never says `D`, and creates about twice the true number of possessions |
| 3 | Your rules added as priors to our models (weights 0–2) | every non-zero weight was worse (down to 0.516); best weight = 0 | no | adds no information the models lack |
| 4 | Training with challenge-shaped ball gaps (`chal`) | 0.5667 | no | +0.004 on chal masks, −0.004 overall: noise |
| 5 | `chal` + long-context features (15 s / 30 s windows) | 0.5691 | no | same |
| 6 | Stage-2 "auto-context" stacker | 0.5582 (and lost in every later variant too) | no | overfits with only 3 games |
| 7 | **Attack-direction features** (`dir`) | 0.5925 | superseded by 8 | |
| 8 | **Attack direction + challenge-shaped gaps** (`dir_chal`) | **0.5962** (chal 0.555) | yes | +0.026 overall, +0.033 on chal |
| 9 | Wider decoder search (`c_ha` down to 2) | 0.5962, identical | no change | optimum was already inside the grid |
| 10 | 700 boosting rounds instead of 350 | 0.5962, macroF1 slightly lower | no | not under-trained |
| 11 | **Turnover detector + turnover-aware decoder** | **0.6074** (chal 0.570 · raw 0.665 · sim 0.587) | yes | +0.011 overall, +0.016 on chal |
| 12 | Wider detector-weight search (up to 8) | 0.6074, identical (weight 3 still best) | no change | |
| 13 | + 7 open DFL Bundesliga matches as extra training data | 0.5655 | no | DFL labels switch owner on every touch; the challenge only changes possession on control |
| 14 | DFL with labels smoothed (owner runs < 2 s merged) | 0.5632 | no | still a definition mismatch |
| 15 | **Bidirectional GRU blended with the trees** (1 network, 64 units, weight 0.67, trained on GPU) | **0.6759** (chal 0.629 · raw 0.745 · sim 0.653) | superseded by 17 | +0.069 overall, +0.059 on chal; wins in all 9 held-out runs |
| 16 | GRU ensemble: 5 networks (64 units, 10 epochs), finer blend-weight grid | 0.6876 (chal 0.636 · raw 0.758 · sim 0.669) | superseded by 17 | +0.012 over one GRU |
| 17 | **GRU ensemble: 5 networks, 128 units, 15 epochs** | **0.6926** (chal 0.648 · raw 0.752 · sim 0.678) | **yes: current final** | +0.017 over one GRU; best on the challenge-like masks and on macroF1 |

**Net achievement: 0.5707 → 0.6926 OOF (+0.122), and 0.522 → 0.648 on the challenge-like masks
(+0.126).** For comparison, the nearest-player baseline idea scores 0.258 on the same masked games.
Reports for every run are in `reports/<run name>/`, including `oof_layers_summary.csv` per mask,
`decoder_tuning_*.csv` and, for the GRU runs, `gru_weight_search.csv`. The final run is
`reports/gru5_big/`. The single-GRU run is `reports/gru/` and the 64-unit ensemble `reports/gru5/`.

### Final run in full (`reports/gru5_big/`)

Mean over all 3 held-out games × 3 masks:

| Row | Method | spellF1 | macroF1 |
|---|---|---|---|
| 1 | nearest-player baseline (w=50) | 0.2584 | 0.5433 |
| 2 | rule engine alone (teammate) | 0.2247 | 0.4872 |
| 3 | models, argmax (no decoder) | 0.5450 | 0.8596 |
| 4 | models + default decoder | 0.6288 | 0.8587 |
| 5 | models + Smart Tagging + tuned decoder | 0.6926 | 0.8706 |
| 5c | trees only (GRU weight 0) | 0.6074 | 0.8490 |
| 6 | models + stacker + Smart Tagging + tuned decoder | 0.5622 | 0.8505 |
| 6b | selected, but turnover detector off | 0.6586 | 0.8678 |
| **7** | **selected pipeline** | **0.6926** | **0.8706** |
| 8 | oracle: selected + perfect dead/live | 0.7313 | 0.9310 |
| 9 | oracle: selected + perfect team | 0.8914 | 0.9043 |

GRU blend-weight searches (`gru_weight_search.csv`; the decoder is re-tuned for each weight):

| GRU weight | 0 (trees only) | 0.33 | 0.5 | 0.67 | 0.8 | 0.9 | 1.0 (GRU only) |
|---|---|---|---|---|---|---|---|
| 1 GRU, 64 units (`reports/gru`) | 0.6040 | 0.6455 | — | **0.6755** | — | — | 0.6716 |
| 5 GRUs, 64 units (`reports/gru5`) | 0.6040 | — | 0.6679 | 0.6798 | 0.6842 | **0.6859** | 0.6829 |
| **5 GRUs, 128 units, 15 ep** (`reports/gru5_big`) | 0.6040 | — | 0.6837 | 0.6927 | **0.6968** | 0.6966 | 0.6955 |

Every held-out run, spellF1: trees only → one GRU → final ensemble:

| Game | chal | raw | sim |
|---|---|---|---|
| Sample_Game_1 | 0.592 → 0.640 → **0.649** | 0.688 → 0.772 → **0.779** | 0.604 → 0.662 → **0.711** |
| Sample_Game_2 | 0.574 → 0.614 → **0.637** | 0.676 → 0.743 → **0.760** | 0.585 → 0.647 → **0.672** |
| Sample_Game_3 | 0.546 → 0.634 → **0.657** | 0.631 → 0.721 → **0.718** | 0.573 → 0.651 → **0.651** |

Things worth noticing. Row 3 jumped from about 0.19 (trees only) to 0.55: the GRUs' per-frame output
is already much less flickery than the trees'. And the turnover detector still adds +0.034 on top
of the GRUs (row 6b vs row 7).

Other things established along the way:

- **The tree models retrain deterministically.** The same command gave byte-identical tree models
  twice, and the turnover run gave the identical table twice. The GRU is reproducible from its
  committed weights. Retrained on another machine or GPU, its weights will differ slightly.
- **Prediction is deterministic:** `predict` from the committed models runs on CPU, and two runs
  printed the same SHA-256, `5bbb327135ae54ec2557a012b047cc176d4b0dc16b4f69c38eaa8475c961f840`.
- **The broadcast video was used** (layer 1) to fill missing tactical ball positions.
- **The raw ATD has no hidden possession or alive channel.** The ball field is just `x,y,speed`.
- **Public 5-minute sanity check:** the *original* pipeline scored spellF1 0.4387 there, and the
  single-GRU pipeline 0.5935 (14 possessions vs 17 true, transitions about 12 frames early on
  average). That is an alignment check only. It is never used to pick a version, and one possession
  moves it by about 6 points.

## 8. What to expect, and limitations

- The private score will most likely be **below** our OOF number of 0.693. The challenge tracking
  is noisier than our simulated masks, and Smart Tagging on the external games is simulated by our
  own code. Earlier, the original pipeline scored about 0.57 OOF but 0.44 on the public 5 minutes.
  If a similar drop applies, expect something like 0.5–0.6 on the full match, against the 0.2433
  baseline. That is an estimate, not a measurement. **0.7 on the private match is possible but
  unlikely.**
- **Turnovers are still the bottleneck** (oracle row 9: 0.89 vs 0.69). Fixing dead/live boundaries
  perfectly would add only about +0.04.
- **Only 3 labelled games.** That limits every learned part. The stacker overfit; the GRU didn't,
  but a sequence model can also learn quirks of the Metrica games that don't carry over.
- **GRU retraining is not bit-exact across machines.** The submission is reproducible from the
  committed weights, not from a retrain.
- **Label definition gap.** Our training labels come from event logs; the challenge GT comes from
  corrected tracking. They agree in spirit but not frame for frame.

Ideas not tried yet, roughly by expected value:

1. **Push the GRU a little further:** 10 networks instead of 5, longer training chunks (512 steps,
   ~82 s), or 2 layers. These are single flags or small changes. Expected gains are now small
   (+0.005 to +0.01), so returns are diminishing. Each is judged by row 7, as always.
2. **Our own ball detector on the tactical video** (allowed by the rules) to cut the 41% missing
   ball. Highest potential, but a big job.
3. More turnover-specific features: pass-target inference, a pressing measure on the ball carrier.
4. Self-training on the challenge match with algorithmic pseudo-labels. Allowed, but it must be
   validated on the external games first.

## 9. Repo map (what is ours)

```
run_pipeline.py              CLI: train / predict / all
possession/                  the pipeline package (table in section 4)
  gru.py                     the bidirectional GRU (training, prediction, blending; CPU or GPU training)
  rules.py                   your rule engine, ported to run on any match (kick-off team from tags, not hard-coded)
  stack.py                   stage-2 stacker (off)
  dfl.py                     open DFL data loader + label smoothing (off)
artifacts/models/            final models + configs (committed; trained only on external data)
reports/                     CV tables for every experiment (one folder per run)
notebooks/possession_pipeline.ipynb   interactive version of the ORIGINAL pipeline (no detector,
                                      no direction features; the CLI is the source of truth)
requirements-pipeline.txt    pinned versions (CPython 3.13.6); install torch first (see section 5)
possession_engine.py, BALL_POSSESSION_LOGIC.md   your original engine and design doc
create_visual_clip.py        see open items
```

## 10. Open items


- **Team name** (the CSV and the document must be named after it), and which repo goes public.
- **The technical document** for the submission still needs writing.
- If DFL data is ever mentioned in the submission document: cite Bassek, Rein, Weber & Memmert
  (2025), *Scientific Data*, CC BY 4.0. The license comes from kloppy's docs and is not yet
  confirmed on the dataset page.
