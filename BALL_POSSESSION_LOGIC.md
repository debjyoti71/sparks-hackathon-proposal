# Football Ball Possession Detection Logic

## 1. Project Overview

This document defines the hierarchical logic for detecting and calculating ball possession in football (soccer) using a 25 FPS match recording.

The system uses:
- Ball detection and tracking.
- Player detection and tracking.
- Team identification.
- Player-ball spatial association.
- Ball movement and trajectory analysis.
- Possession state classification.
- Temporal confirmation and smoothing.

**Primary objective:** Determine which team has possession of the ball at any given time, while correctly handling passes, interceptions, tackles, aerial balls, loose balls, contested situations, and stoppages.

The system must not assign possession based solely on the nearest player to the ball.

---

## 2. System Processing Hierarchy

The possession detection pipeline follows this order:

1. Ball Tracking and Validation
2. Player Tracking and Team Identification
3. Player-Ball Association
4. Ball Situation Classification
5. Possession State Evaluation
6. Temporal Confirmation
7. Possession Output Generation
8. Possession Percentage Calculation

### Processing Flow

```text
Input: 25 FPS Football Video
           |
           v
Ball Detection & Tracking
           |
           v
Player Detection & Tracking
           |
           v
Team Identification
           |
           v
Player-Ball Association
           |
           v
Ball Situation Classification
           |
           v
Possession State Machine
           |
           v
Temporal Confirmation
           |
           v
Possession Output
           |
           v
Team Possession Statistics
```

---

## 3. Step 1: Ball Tracking and Validation

For every frame, maintain the following ball information:

- Bounding box: `(x1, y1, x2, y2)`
- Ball centre: `(cx, cy)`
- Detection confidence.
- Tracking ID.
- Previous ball positions.
- Ball velocity.
- Ball trajectory.
- Visibility status.

### Ball Velocity Calculation

At 25 FPS:

```text
Frame interval = 1 / 25 = 0.04 seconds
```

Calculate ball velocity using consecutive positions:

\[
V_t = \frac{P_t-P_{t-1}}{0.04}
\]

### Ball Visibility Classification

Classify the ball into the following conditions:

| Condition | Description |
|---|---|
| VISIBLE | Ball detected reliably |
| OCCLUDED | Ball temporarily hidden |
| REAPPEARED | Ball detected after occlusion |
| MISSING | Ball not detected for an extended period |
| OUT_OF_PLAY | Ball confirmed outside the playing area |

**Important:** Do not assign a new possession state when ball detection is unreliable. Maintain the previous state temporarily where justified, while reducing confidence.

---

## 4. Step 2: Player Tracking and Team Identification

For every tracked player, maintain:

- Player tracking ID.
- Bounding box.
- Player centre.
- Bottom-centre ground-contact estimate.
- Team ID.
- Player movement history.
- Player velocity.
- Detection confidence.

### Player Ground Position

Use the bottom-centre of the bounding box as an initial ground-contact estimate:

\[
P_i = \left(\frac{x_1+x_2}{2},y_2\right)
\]

Team identification should be consistent across frames.

**Important:** Player tracking IDs and team IDs must remain stable during temporary occlusions and player movement.

---

## 5. Step 3: Player-Ball Association

Determine which player is most likely interacting with the ball.

### Distance Calculation

Ball position:

\[
B=(c_x,c_y)
\]

Player ground position:

\[
P_i=(P_{ix},P_{iy})
\]

Euclidean distance:

\[
d_i=\sqrt{(P_{ix}-B_x)^2+(P_{iy}-B_y)^2}
\]

### Association Logic

For every frame:

1. Calculate the distance between the ball and all visible players.
2. Identify the closest player.
3. Identify the second-closest player.
4. Retrieve their team IDs.
5. Calculate ball velocity.
6. Compare ball movement with player movement.
7. Evaluate whether the ball is approaching or moving away from a player.
8. Evaluate possible physical ball control.

### Important Considerations

- Nearest player does not automatically mean possession.
- A player may be close to a loose ball without controlling it.
- Multiple players may be competing for the same ball.
- An airborne ball should not be associated with the player directly underneath it.
- Pixel distance is affected by camera perspective.

### Recommended Improvement: Pitch Calibration

Use homography/pitch calibration to transform image coordinates into pitch coordinates.

This improves spatial distance calculations because players at different camera depths appear at different scales in the image.

For airborne balls, use a separate ground-projection estimate rather than treating the ball centre as its ground position.

---

## 6. Step 4: Ball Situation Classification

Before assigning possession, classify the current ball situation.

### Supported Situations

| State | Description |
|---|---|
| CONTROLLED_BALL | A player has established control |
| PASS_IN_PROGRESS | Ball has been deliberately played toward another player |
| INTERCEPTION | Opponent cuts out a pass and establishes control |
| CONTESTED | Opposing players are competing for the ball |
| FREE_BALL | Ball is moving without established player control |
| AERIAL_BALL | Ball is travelling through the air |
| CLEARANCE | Player kicks the ball away from danger |
| DEAD_BALL | Play has stopped or ball is out of play |
| UNKNOWN | Insufficient evidence for classification |

### 6.1 Controlled Ball

Conditions:

- Ball is close to a player's feet.
- Player and ball movement show a consistent relationship.
- Ball remains within a reasonable control radius.
- Repeated touches or controlled movement provide supporting evidence.

Result:

- Identify the controlling player.
- Assign their team as the possession team.
- Update the possession state.

### 6.2 Pass in Progress

Conditions:

- Ball accelerates or changes direction following a likely kick.
- Ball moves away from the previous controlling player.
- A teammate may be positioned along the ball trajectory.

Result:

- Physical controlling player: None during flight.
- Possession team: Previous team, provisionally.
- Ball situation: PASS_IN_PROGRESS.

Possible outcomes:

1. Teammate receives the ball.
2. Opponent intercepts the ball.
3. Ball becomes contested.
4. Ball becomes free.

**Important:** A pass does not automatically mean the team has lost possession.

### 6.3 Interception

Conditions:

- Ball is travelling toward a player from the opposing team.
- The opponent becomes a plausible receiver.
- The opponent establishes control.
- The previous team was attempting a pass or retaining possession.

Result:

- Change possession to the opponent's team after confirming control.

### 6.4 Contested Ball

Conditions:

- Multiple players are close to the ball.
- Players belong to opposing teams.
- Neither player demonstrates established control.

Result:

- Set state to CONTESTED.
- Do not immediately assign possession to either team.
- Continue evaluating subsequent frames.

### 6.5 Free Ball

Conditions:

- Ball moves independently.
- No player demonstrates control.
- No confirmed reception or interception exists.

Result:

- Set state to FREE_BALL.
- Keep the previous possession information as historical context only.
- Do not claim current physical control.

### 6.6 Aerial Ball

Conditions:

- Ball is airborne.
- Ball is travelling above the playing surface.
- No player has established contact or control.

Result:

- Set state to AERIAL_BALL.
- Do not assign possession based solely on the nearest player.

### 6.7 Clearance

Conditions:

- A player kicks the ball away from an immediate dangerous area.
- Ball trajectory indicates an attempt to remove danger rather than retain control.

Result:

- Identify the clearance event.
- Evaluate the next ball interaction.
- Do not automatically award possession to the opposing team.

### 6.8 Dead Ball

Conditions:

- Ball crosses the boundary line.
- Referee stops play.
- A foul or other stoppage occurs.
- Another confirmed stoppage event is identified.

Result:

- Set state to DEAD_BALL.
- Pause active possession calculation until play restarts.

---

## 7. Step 5: Possession State Machine

The possession state machine is responsible for maintaining continuity between frames.

### Team Possession States

| State | Meaning |
|---|---|
| TEAM_A_CONTROL | Team A controls the ball |
| TEAM_A_PASS | Team A has played a pass |
| TEAM_B_CONTROL | Team B controls the ball |
| TEAM_B_PASS | Team B has played a pass |
| CONTESTED | Both teams are competing |
| FREE_BALL | No established control |
| AERIAL_BALL | Ball is airborne |
| UNKNOWN | Insufficient evidence |
| DEAD_BALL | Play is stopped |

### State Transition Logic

```text
TEAM_A_CONTROL
      |
      | Pass detected
      v
TEAM_A_PASS
      |
      +---------------------+
      |                     |
      v                     v
Teammate receives     Opponent intercepts
      |                     |
      v                     v
TEAM_A_CONTROL        TEAM_B_CONTROL


TEAM_A_CONTROL
      |
      | Opponent challenge
      v
CONTESTED
      |
      +---------------------+
      |                     |
      v                     v
Team A wins            Team B wins
      |                     |
      v                     v
TEAM_A_CONTROL        TEAM_B_CONTROL


Any Active State
      |
      +--> FREE_BALL
      |
      +--> AERIAL_BALL
      |
      +--> UNKNOWN
      |
      +--> DEAD_BALL
```

Apply equivalent transitions for Team B.

### State Transition Rules

1. Do not switch teams because of a single-frame distance change.
2. A possession switch requires evidence of actual control.
3. A pass should preserve provisional team possession.
4. An interception should be confirmed before changing possession.
5. A contested ball should not automatically belong to the nearest player.
6. Temporary ball occlusion should not immediately reset possession.
7. Dead-ball situations must be separated from active play.

---

## 8. Step 6: Temporal Confirmation and Smoothing

A possession system operating at 25 FPS must evaluate multiple consecutive frames.

| Frames | Duration | Suggested Purpose |
|---:|---:|---|
| 1 | 0.04 sec | Individual observation |
| 3 | 0.12 sec | Initial candidate detection |
| 5 | 0.20 sec | Short-term evidence |
| 8 | 0.32 sec | Potential control confirmation |
| 12 | 0.48 sec | Stronger possession confirmation |
| 25 | 1 sec | Longer-term stability analysis |

These are initial engineering parameters, not fixed football rules. They must be tuned using labelled footage.

### Recommended Logic

1. Identify a candidate controlling player.
2. Evaluate distance and movement evidence.
3. Accumulate evidence over consecutive frames.
4. Confirm possession when evidence is sufficiently consistent.
5. Detect possible possession changes.
6. Require new control evidence before switching teams.
7. Preserve state briefly during temporary occlusion.
8. Transition to FREE_BALL, CONTESTED or UNKNOWN when appropriate.

### Confidence Calculation

Use a weighted confidence model:

\[
C=w_1D+w_2V+w_3T+w_4K
\]

Where:

- D = Distance-based confidence.
- V = Movement correlation.
- T = Temporal consistency.
- K = Ball-control/contact evidence.
- w = Corresponding weights.

The weights must be calibrated using manually annotated match footage.

Do not treat this formula as a trained probability unless it has been calibrated.

---

## 9. Step 7: Possession Output

Maintain three independent variables:

1. Physical ball control.
2. Team possession.
3. Ball situation.

### Recommended Output Structure

```json
{
  "timestamp": 125.40,
  "frame_id": 3135,
  "controlling_player": null,
  "possession_team": "TEAM_A",
  "ball_situation": "PASS_IN_PROGRESS",
  "possession_confidence": 0.91,
  "is_possession_confirmed": false
}
```

### Output Explanation

- `timestamp`: Video timestamp.
- `frame_id`: Current frame number.
- `controlling_player`: Player currently controlling the ball, if known.
- `possession_team`: Team currently credited with possession.
- `ball_situation`: Current ball situation.
- `possession_confidence`: Confidence in the possession decision.
- `is_possession_confirmed`: Whether the decision has passed temporal confirmation.

During a pass, physical control can be null while the previous team retains provisional possession.

---

## 10. Step 8: Possession Percentage Calculation

Calculate possession using confirmed team possession time.

### Example Match Segment

| Situation | Duration |
|---|---:|
| Team A control | 30 sec |
| Team A pass | 15 sec |
| Team B control | 35 sec |
| Team B pass | 10 sec |
| Contested / Free ball | 5 sec |
| Dead ball | 5 sec |
| Total | 100 sec |

For active-play possession:

\[
T_{active}=30+15+35+10=90
\]

Team A:

\[
P_A=\frac{30+15}{90}\times100=50\%
\]

Team B:

\[
P_B=\frac{35+10}{90}\times100=50\%
\]

### Calculation Rules

- Count controlled possession time.
- Include provisional team possession during passes when appropriate.
- Exclude dead-ball time.
- Define a consistent policy for contested and free-ball periods.
- Avoid counting the same frame for both teams.
- Use timestamps rather than assuming every frame is processed successfully.

The exact treatment of loose-ball and contested periods should be documented because possession-statistics providers may use different conventions.

---

## 11. Recommended Processing Algorithm

```python
for frame in video:

    # 1. Get detections and tracking information
    ball = get_ball_detection(frame)
    players = get_tracked_players(frame)

    # 2. Validate ball
    if not ball.is_reliable:
        update_missing_ball_state()
        continue

    # 3. Player-ball association
    candidates = calculate_player_ball_distances(
        ball, players
    )

    # 4. Analyze movement
    ball_velocity = calculate_ball_velocity()
    player_velocities = calculate_player_velocities(
        players
    )

    # 5. Classify ball situation
    situation = classify_situation(
        ball,
        candidates,
        ball_velocity,
        player_velocities
    )

    # 6. Evaluate possession transition
    new_state = possession_state_machine(
        current_state,
        situation,
        candidates
    )

    # 7. Temporal confirmation
    confirmed_state = temporal_confirmation(
        new_state,
        previous_states
    )

    # 8. Update possession
    update_possession(
        confirmed_state,
        frame_timestamp
    )
```

This is architectural pseudocode. Each function requires its own implementation and validation.

---

## 12. Implementation Priority

| Priority | Component | Importance |
|---|---|---|
| 1 | Reliable ball tracking | Critical |
| 2 | Player tracking | Critical |
| 3 | Team identification | Critical |
| 4 | Player-ball association | Critical |
| 5 | Pass and interception detection | Critical |
| 6 | Possession state machine | Critical |
| 7 | Temporal confirmation | Critical |
| 8 | Aerial and contested-ball classification | High |
| 9 | Pitch calibration | High |
| 10 | Possession percentage calculation | High |

---

## 13. Validation and Evaluation

Before considering the possession system complete, validate it against manually annotated match footage.

### Important Evaluation Metrics

- Team possession classification accuracy.
- Possession transition detection accuracy.
- False possession switch rate.
- Pass completion and interception classification accuracy.
- Possession duration error.
- Unknown/uncertain classification frequency.
- Possession percentage error against reference annotations.

### Important Test Scenarios

1. Normal dribbling.
2. Short and long passes.
3. Successful and unsuccessful passes.
4. Opponent interceptions.
5. Tackles and physical challenges.
6. Contested balls.
7. Aerial balls and headers.
8. Goalkeeper possession.
9. Clearances.
10. Temporary ball occlusion.
11. Ball going out of play.
12. Rapid counter-attacks.

---

## 14. Final Architecture Recommendation

Implement a rule-based hierarchical possession system first.

Do not immediately introduce another ML model for the entire possession pipeline.

Start with:

- Existing ball detector and tracker.
- Existing player tracker.
- Team classification.
- Spatial association.
- Ball movement analysis.
- Rule-based state machine.
- Temporal confirmation.

After evaluating the rule-based implementation, identify the scenarios where it consistently fails.

Introduce ML-based classification only where necessary, such as ambiguous ball control, pass detection, aerial-ball classification or contested situations.

**Core architectural principle:**

Ball detection identifies where the ball is.

Player tracking identifies who is nearby.

Ball situation classification identifies what is happening.

The possession state machine determines which team has possession.

Temporal confirmation ensures the decision remains reliable across consecutive frames.

The final objective is not simply to identify the nearest player, but to reconstruct the sequence of ball-control events throughout the match.