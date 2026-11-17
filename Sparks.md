# Sparks: Technical Document

**Challenge:** More Than A Hack 2027, Phase 1: Ball Possession
**Team:** Sparks
**Repository:** https://github.com/debjyoti71/sparks-hackathon-proposal (branch `main`)
**Submission:** `Sparks.csv`, SHA-256 `5bbb327135ae54ec2557a012b047cc176d4b0dc16b4f69c38eaa8475c961f840`

---

## 1. Approach

spellF1 scores whole possessions, so flicker and badly placed switches cost far more than a few
wrong frames. We therefore split the problem in two:

- **per-frame probabilities**: P(dead), P(home | alive) and P(turnover);
- **one global decoding** of the whole half into H / A / D runs.

The pipeline, in order:

1. **Broadcast fusion.** Each live broadcast camera shot is aligned to the tactical timeline by
   matching team shapes, which gives a frame offset per shot. Replays are rejected because they run
   backwards in game time. Where the tactical ball is missing, the aligned broadcast ball fills it.
2. **Ball bridging.** Remaining gaps are filled with a curve (up to 4 s) or a straight line. The
   models are told how long each gap is.
3. **Features.** About 130 per frame: each team's distance to the ball, last and next touch, pass
   bridges, team speed, shape and coherence over 1, 3 and 7 s windows. **Attack-direction
   features** are also included. Each team's attacking direction is inferred per half from
   goalkeeper depth. From it we compute forward speed of the block, block height, ball in the
   attacking half, and flips of forward motion (a turnover signature).
4. **Tree models (LightGBM).** Three models: P(dead), P(home | alive), and a **turnover detector**,
   P(direct H↔A change within ±0.5 s). Team symmetry is enforced: every frame is trained and
   predicted in both team orientations, and the two predictions are averaged.
5. **Sequence models.** An ensemble of **5 bidirectional GRUs** reads the same features over the
   whole half in both directions, so each frame sees what happened before *and* after it. It
   predicts the same three probabilities. The GRUs are blended with the trees in log-odds
   (0.2 · trees + 0.8 · GRU).
6. **Smart Tagging priors.** Set-piece tags mark dead ball up to the restart kick. The kick frame is
   found from ball acceleration and player speed inside the tag window. After the kick, play is
   live and favours the restarting team. Attacking-phase tags favour the tagged team, and
   defending-phase tags favour the opponent.
7. **Viterbi decoding.** The decoder finds the most likely H / A / D sequence per half. It charges a
   cost for each live↔dead switch and each H↔A switch. The **H↔A cost varies by frame**: it is
   lower where the turnover detector fires and higher elsewhere.
8. **Output.** The published pre-kick-off stretch of the 2nd half (110808–113741) is forced to `D`.
   The file is checked with the official validator, and its SHA-256 is printed.

**Why this design.** We have only three labelled games. Trees are robust on that much data, and the
GRU adds the long context that possession boundaries need. A global decoder turns noisy per-frame
probabilities into stable spells. Every choice was settled out of fold (section 3).

## 2. Assets used

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (`data/tactical/`) | yes | primary timeline: players and ball, 25 fps |
| Broadcast ATD + homography (`data/broadcast/`) | yes | fills missing tactical ball positions in aligned live shots (step 1) |
| Smart Tagging XML | yes | set-piece and tactical-phase priors (step 6) |
| Tactical / broadcast video (`.mp4`) | **no** | not read by the pipeline, and outputs were not checked against it |
| Public 5-minute ground truth | alignment check only | never used to train, tune or choose between versions |

**Tried and dropped:**
- a hand-written rule engine, alone and as priors;
- a stage-2 stacker;
- extra DFL / Bundesliga training data;
- the other items listed as dropped in section 4.

## 3. Training data and validation

**Labelled data: the 3 Metrica Sports open sample games**
(`github.com/metrica-sports/sample-data`):
- about 270 minutes of tracking in total, with event data annotated by Metrica Sports;
- H / A / D frame labels are derived from those event logs by our code (`possession/labels.py`);
- nothing comes from the provided Barcelona vs Athletic game.

**Simulated challenge conditions.** The challenge ball is missing on 41% of frames. Each training
game is therefore copied 6 times with simulated tracking failures:
- `sim`: short gaps;
- `raw`: the external tracker's own gaps;
- `chal`: gaps shaped like the challenge's (median 10.7 s, p90 34 s).

Smart Tagging is also simulated on the external games from their events.

**Validation: leave one game out.** We train on two games and predict the third, for every game and
every mask, giving 3 × 3 = 9 out-of-fold (OOF) runs. All hyperparameters are chosen on these
predictions: decoder costs, the turnover-detector weight and the GRU blend weight. We decided in
advance that **a change is kept only if it raises mean OOF spellF1**.

## 4. Iterations

Mean OOF spellF1 over the 9 held-out runs:

| # | Change | OOF spellF1 | Kept? |
|---|---|---|---|
| 0 | Published baseline idea (nearest player + smoothing) | 0.258 | reference |
| 1 | LightGBM + Smart Tagging + tuned Viterbi decoder | 0.571 | start |
| 2 | Rule engine alone / as priors to the models | 0.225 / best weight 0 | dropped: adds nothing the models lack |
| 3 | Challenge-shaped gaps alone; 15 / 30 s context windows | 0.567 / 0.569 | dropped: noise |
| 4 | Stage-2 stacker on model outputs | 0.558 | dropped: overfits on 3 games |
| 5 | **Attack-direction features + challenge-shaped gaps** | **0.596** | **kept** (+0.026) |
| 6 | Wider decoder grid; 700 boosting rounds | no change | dropped |
| 7 | **Turnover detector + turnover-aware decoder** | **0.607** | **kept** (+0.011) |
| 8 | + 7 DFL Bundesliga matches (raw, then smoothed labels) | 0.566 / 0.563 | dropped: DFL changes owner on every touch, not on control |
| 9 | **Bidirectional GRU blended with the trees** | **0.676** | **kept** (+0.069; better in all 9 runs) |
| 10 | **5-GRU ensemble, 128 units, 15 epochs, finer blend grid** | **0.693** | **kept: final** |
| 11 | Boundary time-shift search; 10 GRUs with 82 s training chunks | marginal | dropped |

**Final: 0.571 → 0.693 OOF (+0.122), and 0.522 → 0.648 on the challenge-like `chal` masks.**

Oracle checks on the final model:
- perfect team decisions would give 0.89;
- perfect dead / live boundaries would give only 0.73.

So the remaining error is mostly in H vs A decisions (turnovers).

## 5. Possession rules

These definitions are used for training labels and enforced by the decoder:

- **Possession = control.** The owner is the team of the latest controlling action: pass, shot,
  set piece or recovery. A *ball lost* event hands the ball to the opponent.
- **Challenges.** Duels, including aerial duels won, are *not* possession. A deflection does not
  change the owner.
- **Loose balls.** The last team in control keeps possession until the opponent gains control.
- **Dead ball.** Starts at a stoppage: ball out, foul, offside, end of half, a goal, or a shot that
  goes out. It lasts until the restart kick, and kick-offs are dead until the ball is played.
- **Restarts.** Dead up to the kick frame, found from ball acceleration inside the Smart Tagging
  set-piece window. From the kick, play is live and favours the restarting team.
- **Spell stability.** The decoder enforces it through switch costs, which are lower where a
  turnover is likely. No minimum spell length is imposed: OOF tuning chose 0.
- **Symmetry.** Home and away are treated identically everywhere.

The decoder settings are `c_ha = 40`, `c_ld = 50` and turnover weight 6.

## 6. Limitations

- **Turnovers are the main error source.** Perfect team calls would lift OOF spellF1 from 0.69
  to 0.89. Quick contested turnovers are often placed late or merged away.
- **The private score will likely be below 0.69 OOF.** The challenge tracking is noisier than our
  simulated masks, and Smart Tagging on the training games is simulated by our own code.
- **Only 3 labelled games.** This limits every learned component.
- **Label definition gap.** Our labels come from event logs; the challenge ground truth comes from
  corrected tracking.
- **With more time:**
  - our own ball detector on the tactical video, to cut the 41% missing ball (allowed by the
    rules);
  - turnover features such as pass-target inference and pressure on the ball carrier;
  - more labelled external games that use the same possession definition.

## 7. Reproduction

Python 3.13.6. All pinned versions are in `requirements-pipeline.txt`.

```bash
git clone https://github.com/debjyoti71/More-Than-A-Hack-2027---Initial-Challenge---9
cd More-Than-A-Hack-2027---Initial-Challenge---9 && git switch combined-pipeline
python -m venv venv
venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r requirements-pipeline.txt
# place the competition bundle in data/ as described in DATA.md

# regenerate the submission from the committed models (CPU, a few minutes)
venv/bin/python run_pipeline.py predict --team Sparks --no-cache
```

`predict` writes `evaluation/results/Sparks.csv`, validates it and prints its SHA-256. The hash
should be `5bbb327135ae54ec2557a012b047cc176d4b0dc16b4f69c38eaa8475c961f840`. Prediction is
deterministic: two runs produced the same hash.

To retrain from the external games:

```bash
git clone https://github.com/metrica-sports/sample-data external_data
venv/bin/python run_pipeline.py train --gru-device cuda    # ~1 h on a GPU
```

The tree models retrain exactly. GRU weights differ slightly across machines and GPUs. The committed
models in `artifacts/models/` are therefore the reference for the submitted file.
