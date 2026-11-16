"""Challenge inference: Match -> Broadcast fusion -> features -> models -> Smart Tagging priors
-> rule-engine priors -> decoder -> CSV.

Nothing here reads the ground truth. The only match-specific constants are the published facts in
DATA.md / FAQ.md: the graded half boundaries and the 2nd-half kick-off frame.
"""

import os

import numpy as np

from .broadcast import align_and_fuse_broadcast
from .decode import DecodeConfig, decode_match
from .labels import A, D, H, X
from .models import PossessionModels, prepare
from .rules import RuleConfig, apply_rule_priors, run_rules
from .smart_tagging import apply_smart_tagging_priors, load_challenge_smart_tagging
from .stack import aux_from_bundle

LETTERS = np.array(["H", "A", "D", "X"])

# FAQ.md: the 2nd half is graded from 110808 but kicks off at 113742; those frames are all D.
SECOND_HALF_DEAD = [(110808, 113741)]


def predict_challenge(models, cfg, match, force_dead=SECOND_HALF_DEAD,
                      use_broadcast=True, use_smart_tagging=True, data_dir=None, cache=True,
                      rule_weights=None, rule_cfg=None, feature_cfg=None, stacker=None,
                      gru=None, gru_w=0.0):
    """Run full pipeline on the challenge match.

    rule_weights: {"w_team": float, "w_dead": float} for the rule-engine priors (None / zeros = off).
    The rule engine runs on the original tactical ball (as in possession_engine.py), not the fused one.
    cache=False recomputes the broadcast alignment instead of reading artifacts/cache/.

    Returns (codes, p_dead, p_home, diagnostics_dict, variants) where variants holds the decoded
    outputs of each stage for side-by-side comparison: "rules_only", "models_st" (no rule priors)
    and "combined" (== codes).
    """
    diagnostics, variants = {}, {}
    if use_broadcast:
        match_used, bcast_report = align_and_fuse_broadcast(match, data_dir=data_dir, cache=cache)
        diagnostics["broadcast"] = bcast_report
    else:
        match_used = match

    bundle = prepare(match_used, apply_mask=False, stride=1, with_labels=False, fcfg=feature_cfg)
    p_dead, p_home = models.predict(bundle)
    p_switch = models.predict_switch(bundle)     # None for models without a turnover detector
    if gru is not None and gru_w:                # experimental GRU, blended in log-odds space
        from .gru import blend
        g_dead, g_home, g_switch = gru.predict(bundle)
        p_dead, p_home = blend(p_dead, g_dead, gru_w), blend(p_home, g_home, gru_w)
        p_switch = blend(p_switch, g_switch, gru_w)
    diagnostics["gru_w"] = float(gru_w) if gru is not None else 0.0
    if stacker is not None:                      # stage-2 long-context refinement
        p_dead, p_home = stacker.predict(p_dead, p_home, match_used.period, aux_from_bundle(bundle))
    diagnostics["stacker"] = stacker is not None

    instances = []
    if use_smart_tagging:
        try:
            instances = load_challenge_smart_tagging(data_dir=data_dir)
            p_dead, p_home, st_report = apply_smart_tagging_priors(
                match_used, instances, p_dead, p_home
            )
            diagnostics["smart_tagging"] = st_report
        except Exception as exc:
            diagnostics["smart_tagging"] = {"status": f"skipped: {exc}"}

    rule_codes, _, rule_dead = run_rules(match, instances, rule_cfg)
    variants["rules_only"] = rule_codes
    variants["models_st"] = decode_match(p_dead, p_home, match_used.period, cfg, match_used.fps, force_dead,
                                         p_switch=p_switch)

    w = rule_weights or {}
    p_dead, p_home = apply_rule_priors(p_dead, p_home, rule_codes, rule_dead,
                                       w.get("w_team", 0.0), w.get("w_dead", 0.0))
    diagnostics["rule_priors"] = {"w_team": w.get("w_team", 0.0), "w_dead": w.get("w_dead", 0.0),
                                  "rule_dead_frames": int(rule_dead[match.period > 0].sum())}

    codes = decode_match(p_dead, p_home, match_used.period, cfg, match_used.fps, force_dead,
                         p_switch=p_switch)
    variants["combined"] = codes
    return codes, p_dead, p_home, diagnostics, variants


def write_submission(codes, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    letters = LETTERS[codes]
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    return path
