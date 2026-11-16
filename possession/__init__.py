"""Possession pipeline: broadcast fusion -> ball-gap bridging -> dead-ball / control models ->
Smart Tagging priors -> sticky sequence decoder.

Training and tuning use ONLY external games (Metrica open sample data, ./external_data). Nothing
here reads the challenge ground truth or fits anything on the challenge match.

Modules
-------
data           load external games and the challenge ATD into one `Match` structure
labels         event-derived H / A / D frame labels for the external games (training targets only)
masking        simulate the challenge's missing ball and dropped players on the external games
ball           bidirectional ball-gap bridging, gap diagnostics
broadcast      align live broadcast tracking shots to tactical frames and fuse missing ball coords
smart_tagging  set-piece restart pinpointing and tactical phase priors (offensive vs defensive)
features       symmetric (team-agnostic) per-frame features with collective motion & pass bridging
models         LightGBM dead-ball and control models, leave-one-game-out helpers
decode         Viterbi decoder, short-run clean-up, decoder tuning against spellF1
pipeline       challenge inference -> submission CSV
"""
