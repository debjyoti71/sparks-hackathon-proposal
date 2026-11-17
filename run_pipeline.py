"""Command-line entry point: reproduce the possession submission without Jupyter.

    python run_pipeline.py predict --team <team_name>            # saved models -> evaluation/results/<team_name>.csv
    python run_pipeline.py predict --team <team_name> --compare  # also write the rules-only and models-only variants
    python run_pipeline.py train                                 # retrain + retune on external games only
    python run_pipeline.py all --team <team_name>                # train, then predict

`train` reads ONLY the Metrica open sample games in external_data/: leave-one-game-out CV, a nested
stage-2 stacker, decoder tuning on out-of-fold predictions, selection of the better pipeline by
out-of-fold spellF1, final fit on all three games. It never opens data/ or any ground truth.

`predict` reads the provided match from data/ (tactical ATD, broadcast ATD + homography validity,
Smart Tagging), applies the saved models (+ stacker if selected), Smart Tagging priors and decoder,
writes the CSV, runs the official validator and prints the file's SHA-256.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_MODELS = os.path.join("artifacts", "models")
DEFAULT_REPORTS = "reports"

# ---- training configuration ---------------------------------------------------------------------
TRAIN_RUNS = [("sim", 1), ("sim", 2), ("raw", 3), ("raw", 4)]   # (mask mode, seed) per training game
TEST_SEED = 100                 # mask seed for out-of-fold evaluation
STRIDE_TRAIN = 4                # keep every 4th frame for training (frames are highly correlated)
N_ROUNDS = 350
SMART_TAG_SEED = 42             # seed for simulated Smart Tagging on the external games
GRID_LD = (6, 12, 25, 50, 100)  # live<->dead switching-cost grid
GRID_HA = (2, 3, 5, 10, 20, 40)  # H<->A switching-cost grid
DFL_STRIDE = 8                  # extra DFL games: keep every 8th frame (7 games, limits memory)
GRID_GRU_W = (0.0, 0.5, 0.67, 0.8, 0.9, 1.0)  # GRU blend weight (0 = trees only, 1 = GRU only), chosen out-of-fold
GRID_SW = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)  # turnover-detector weight in the decoder; 0 = detector ignored
# Rule-engine prior weights (teammate's engine). Rejected out-of-fold; searched only with --rule-search.
GRID_W_TEAM = (0.0, 0.5, 1.0, 2.0)
GRID_W_DEAD = (0.0, 1.0, 2.0)
# Training variants. All are scored on the same three OOF test masks (sim, raw, chal).
#   base      : the original training masks and features
#   chal      : + 2 training copies per game with challenge-shaped ball gaps (median 10.7 s)
#   chal_long : chal + long-context features (15 s / 30 s windows, 60 s since/until clip)
VARIANTS = {
    "base": (TRAIN_RUNS, None),
    "chal": (TRAIN_RUNS + [("chal", 5), ("chal", 6)], None),
    "chal_long": (TRAIN_RUNS + [("chal", 5), ("chal", 6)], "long"),
    # dir      : + attack-direction-aware control features (direction inferred per half from tracking)
    "dir": (TRAIN_RUNS, "dir"),
    "dir_chal": (TRAIN_RUNS + [("chal", 5), ("chal", 6)], "dir"),
}


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _oracles(gt, pred, p_home):
    """Error attribution: replace one half of the problem with the truth and re-score.
    'perfect dead/live' keeps our team labels; 'perfect team' keeps our dead/live split."""
    from possession.labels import A, D, H, X
    live = (gt == H) | (gt == A)
    ours_live = (pred == H) | (pred == A)
    perfect_dead = np.where(gt == D, D, np.where(live & ours_live, pred, X)).astype(np.int8)
    perfect_team = np.where(pred == D, D, np.where(live, gt, pred)).astype(np.int8)
    return perfect_dead, perfect_team


# --------------------------------------------------------------------------------------- train

def train(models_dir, reports_dir, threads=0, variant="dir_chal", search_rules=False, use_stack="off",
          n_rounds=N_ROUNDS, dfl=False, dfl_smooth=0.0, gru=True, gru_epochs=15, torch_threads=4,
          gru_device="cpu", gru_seeds=5, gru_hidden=128, gru_layers=1):
    from possession.data import load_external_games
    from possession.decode import DecodeConfig, decode_match, nearest_player_baseline, score, tune_decoder
    from possession.features import DEFAULT_FCFG, DIR_FCFG, LONG_FCFG
    from possession.labels import A, D, H, X
    from possession.masking import MaskConfig, apply_masks, challenge_gap_config
    from possession.models import PossessionModels, frame_metrics, prepare
    from possession.rules import apply_rule_priors, run_rules
    from possession.smart_tagging import apply_smart_tagging_priors, simulate_external_smart_tagging
    from possession.stack import Stacker, aux_from_bundle

    params = {"num_threads": threads}
    mask_cfgs = {"sim": MaskConfig(mode="simulate"), "raw": MaskConfig(mode="raw"),
                 "chal": challenge_gap_config()}
    train_runs, fname = VARIANTS[variant]
    fcfg = {"long": LONG_FCFG, "dir": DIR_FCFG}.get(fname, DEFAULT_FCFG)
    log("variant {}: train masks {}, features {}, {} boosting rounds".format(variant, train_runs, fcfg, n_rounds))
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    log("loading external games (Metrica Sample_Game_1..3)")
    games = load_external_games()

    log("building features")
    train_bundles = {g.name: [prepare(g, mask_cfgs[m], seed=s, stride=STRIDE_TRAIN, fcfg=fcfg)
                              for m, s in train_runs] for g in games}
    test_bundles = {m: {g.name: prepare(g, mask_cfgs[m], seed=TEST_SEED, stride=1, fcfg=fcfg) for g in games}
                    for m in mask_cfgs}

    extra = []
    if dfl:
        from possession.dfl import DFL_MATCHES, load_dfl_games
        from possession.labels import label_report
        log("loading the open DFL games (extra TRAINING data only; first run downloads them)")
        for g in load_dfl_games(DFL_MATCHES, log=log, smooth_s=dfl_smooth):
            log("  {}: {}".format(g.name, label_report(g.labels)))
            extra += [prepare(g, mask_cfgs[m], seed=s, stride=DFL_STRIDE, fcfg=fcfg) for m, s in train_runs]
            del g
        log("  {} extra DFL training bundles".format(len(extra)))

    single = {}

    def single_game_model(name):
        if name not in single:
            single[name] = PossessionModels.fit(train_bundles[name], n_rounds=n_rounds, params=params)
        return single[name]

    log("leave-one-game-out training (stage 1) with a nested stage-2 stacker")
    oof, frame_rows = [], []
    for g in games:
        tr = [bd for h in games if h.name != g.name for bd in train_bundles[h.name]] + extra
        m = PossessionModels.fit(tr, n_rounds=n_rounds, params=params)
        g_model = None
        if gru:
            from possession.gru import GRUConfig, GRUModel
            log("  training GRU for held-out {}".format(g.name))
            g_model = GRUModel.fit(tr, GRUConfig(epochs=gru_epochs, threads=torch_threads, device=gru_device, seeds=gru_seeds,
                          hidden=gru_hidden, layers=gru_layers), log=log)

        # Stacker for this fold: trained on stage-1 predictions for the two training games, each
        # produced by a model fitted on the OTHER training game only -- the held-out game is never seen.
        inner = [h for h in games if h.name != g.name] if use_stack != "off" else []
        items = []
        for h in inner:
            other = next(x for x in inner if x.name != h.name)
            m1 = single_game_model(other.name)
            for mode in mask_cfgs:
                tbh = test_bundles[mode][h.name]
                a, b = m1.predict(tbh)
                items.append(dict(p_dead=a, p_home=b, period=tbh.period, aux=aux_from_bundle(tbh), codes=tbh.codes))
        st = Stacker.fit(items, params=params) if items else None

        tags = simulate_external_smart_tagging(g, seed=SMART_TAG_SEED)
        for mode in mask_cfgs:
            tb = test_bundles[mode][g.name]
            aux = aux_from_bundle(tb)
            p_dead, p_home = m.predict(tb)
            p_dead2, p_home2 = st.predict(p_dead, p_home, tb.period, aux) if st else (p_dead, p_home)
            p_switch = m.predict_switch(tb)
            g_dead, g_home, g_switch = g_model.predict(tb) if g_model else (None, None, None)
            mm = apply_masks(g, mask_cfgs[mode], TEST_SEED)
            p_dead_st, p_home_st, _ = apply_smart_tagging_priors(mm, tags, p_dead, p_home)
            p_dead2_st, p_home2_st, _ = apply_smart_tagging_priors(mm, tags, p_dead2, p_home2)
            rule_codes, _, rule_dead = run_rules(mm, tags)
            fm1 = frame_metrics(tb, p_dead, p_home)
            fm2 = frame_metrics(tb, p_dead2, p_home2)
            frame_rows.append({"game": g.name, "mode": mode, "stage": 1, **fm1})
            frame_rows.append({"game": g.name, "mode": mode, "stage": 2, **fm2})
            log("  held out {} [{}]: stage1 dead_acc {:.3f} ctrl_acc {:.3f} | stage2 dead_acc {:.3f} ctrl_acc {:.3f}".format(
                g.name, mode, fm1["dead_acc"], fm1["ctrl_acc"], fm2["dead_acc"], fm2["ctrl_acc"]))
            oof.append(dict(name=g.name, mode=mode, match=mm, aux=aux, p_dead=p_dead, p_home=p_home,
                            p_switch=p_switch, g_dead=g_dead, g_home=g_home, g_switch=g_switch, tags=tags,
                            p_dead_st=p_dead_st, p_home_st=p_home_st,
                            p_dead2_st=p_dead2_st, p_home2_st=p_home2_st,
                            rule_codes=rule_codes, rule_dead=rule_dead, period=tb.period, codes=tb.codes))
    pd.DataFrame(frame_rows).round(4).to_csv(os.path.join(reports_dir, "oof_frame_metrics.csv"), index=False)

    def tune(key_dead, key_home):
        items = [dict(name=o["name"], mode=o["mode"], p_dead=o[key_dead], p_home=o[key_home],
                      p_switch=o["p_switch"], period=o["period"], codes=o["codes"]) for o in oof]
        cfg, table = tune_decoder(items, grid_ha=GRID_HA, grid_ld=GRID_LD, grid_sw=GRID_SW, verbose=False)
        table.round(4).to_csv(os.path.join(reports_dir, "decoder_tuning_{}.csv".format(key_dead)), index=False)
        top = table.sort_values(["spell_f1", "macro_f1"], ascending=False).iloc[0]
        return cfg, (float(top.spell_f1), float(top.macro_f1))

    gru_w, cfg_g0 = 0.0, None
    if gru:
        from possession.gru import blend
        log("searching the GRU blend weight (0 = trees only) on out-of-fold predictions")
        for o in oof:
            o["tree"] = (o["p_dead"], o["p_home"], o["p_switch"], o["p_dead_st"], o["p_home_st"])

        def set_blend(w):
            for o in oof:
                pdb, phb = blend(o["tree"][0], o["g_dead"], w), blend(o["tree"][1], o["g_home"], w)
                o["p_dead"], o["p_home"] = pdb, phb
                o["p_switch"] = blend(o["tree"][2], o["g_switch"], w)
                o["p_dead_st"], o["p_home_st"], _ = apply_smart_tagging_priors(o["match"], o["tags"], pdb, phb)

        rows_g, best_g = [], None
        for w in GRID_GRU_W:
            set_blend(w)
            cfg_w, s_w = tune("p_dead_st", "p_home_st")
            if w == 0.0:
                cfg_g0 = cfg_w
            rows_g.append({"gru_w": w, "spell_f1": s_w[0], "macro_f1": s_w[1], "decoder": str(cfg_w)})
            log("  GRU weight {}: spellF1 {:.4f} macroF1 {:.4f}".format(w, *s_w))
            if best_g is None or s_w > best_g[0]:      # strict improvement keeps the lower weight on ties
                best_g = (s_w, w)
        pd.DataFrame(rows_g).round(4).to_csv(os.path.join(reports_dir, "gru_weight_search.csv"), index=False)
        gru_w = best_g[1]
        log("  selected GRU weight {}".format(gru_w))
        set_blend(gru_w)                               # everything below uses the selected blend

    log("tuning decoder: models + Smart Tagging")
    cfg_base, s_base = tune("p_dead_st", "p_home_st")
    log("  {} -> spellF1 {:.4f} macroF1 {:.4f}".format(cfg_base, *s_base))
    log("tuning decoder: models + stacker + Smart Tagging")
    cfg_stack, s_stack = tune("p_dead2_st", "p_home2_st") if use_stack != "off" else (cfg_base, (-1.0, -1.0))
    log("  {} -> spellF1 {:.4f} macroF1 {:.4f}".format(cfg_stack, *s_stack))

    if use_stack == "auto":
        stack_on = s_stack > s_base          # strict improvement on (spellF1, macroF1) required
    else:
        stack_on = use_stack == "on"
    log("stacker {} ({})".format("ON" if stack_on else "OFF", "selected out-of-fold" if use_stack == "auto" else "forced"))
    kd, kh, best_cfg = ("p_dead2_st", "p_home2_st", cfg_stack) if stack_on else ("p_dead_st", "p_home_st", cfg_base)

    rule_w = {"w_team": 0.0, "w_dead": 0.0}
    if search_rules:
        log("searching rule-prior weights on top of the selected pipeline")
        rows, best = [], None
        for w_team, w_dead in itertools.product(GRID_W_TEAM, GRID_W_DEAD):
            items = []
            for o in oof:
                a, b = apply_rule_priors(o[kd], o[kh], o["rule_codes"], o["rule_dead"], w_team, w_dead)
                items.append(dict(name=o["name"], mode=o["mode"], p_dead=a, p_home=b, p_switch=o["p_switch"],
                                  period=o["period"], codes=o["codes"]))
            cfg, table = tune_decoder(items, grid_ha=GRID_HA, grid_ld=GRID_LD, grid_sw=GRID_SW, verbose=False)
            top = table.sort_values(["spell_f1", "macro_f1"], ascending=False).iloc[0]
            rows.append({"w_team": w_team, "w_dead": w_dead, "spell_f1": float(top.spell_f1), "macro_f1": float(top.macro_f1)})
            if best is None or (rows[-1]["spell_f1"], rows[-1]["macro_f1"]) > best[0]:
                best = ((rows[-1]["spell_f1"], rows[-1]["macro_f1"]), {"w_team": w_team, "w_dead": w_dead}, cfg)
        pd.DataFrame(rows).round(4).to_csv(os.path.join(reports_dir, "rule_weight_search.csv"), index=False)
        _, rule_w, best_cfg = best
        log("  selected rule weights {}".format(rule_w))

    log("layer ablation on out-of-fold predictions")

    def argmax_codes(o):
        out = np.where(o["p_dead"] > 0.5, D, np.where(o["p_home"] > 0.5, H, A)).astype(np.int8)
        out[o["period"] == 0] = X
        return out

    rows = []
    for o in oof:
        final_pd, final_ph = apply_rule_priors(o[kd], o[kh], o["rule_codes"], o["rule_dead"], **rule_w)
        selected = decode_match(final_pd, final_ph, o["period"], best_cfg, p_switch=o["p_switch"])
        no_detector = DecodeConfig(**{**best_cfg.__dict__, "sw_beta": 0.0})
        oracle_dead, oracle_team = _oracles(o["codes"], selected, final_ph)
        for name, pred in (
            ("1. nearest-player baseline (w=50)", nearest_player_baseline(o["match"], 50)),
            ("2. rule engine alone (teammate)", o["rule_codes"]),
            ("3. models, argmax", argmax_codes(o)),
            ("4. models + default decoder", decode_match(o["p_dead"], o["p_home"], o["period"], DecodeConfig(stride=5))),
            ("5. models + Smart Tagging + tuned decoder", decode_match(o["p_dead_st"], o["p_home_st"], o["period"], cfg_base, p_switch=o["p_switch"])),
            ("6. models + STACKER + Smart Tagging + tuned decoder", decode_match(o["p_dead2_st"], o["p_home2_st"], o["period"], cfg_stack, p_switch=o["p_switch"])),
            ("6b. SELECTED but turnover detector off (sw_beta=0)", decode_match(final_pd, final_ph, o["period"], no_detector)),
            ("7. SELECTED pipeline", selected),
            ("8. oracle: selected + perfect dead/live", oracle_dead),
            ("9. oracle: selected + perfect team", oracle_team),
        ):
            rows.append({"game": o["name"], "mode": o["mode"], "method": name, **score(o["codes"], pred)})
        if gru:
            trees_only = decode_match(o["tree"][3], o["tree"][4], o["period"], cfg_g0, p_switch=o["tree"][2])
            rows.append({"game": o["name"], "mode": o["mode"], "method": "5c. trees only (GRU weight 0)",
                         **score(o["codes"], trees_only)})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(reports_dir, "oof_layers.csv"), index=False)
    summary = res.groupby(["mode", "method"])[["spell_f1", "macro_f1", "n_pred", "n_gt"]].mean().round(4)
    summary.to_csv(os.path.join(reports_dir, "oof_layers_summary.csv"))
    print(summary.to_string(), flush=True)
    overall = res.groupby("method")[["spell_f1", "macro_f1"]].mean().round(4)
    overall.to_csv(os.path.join(reports_dir, "oof_layers_overall.csv"))
    print("\nmean over all held-out games and masks:\n" + overall.to_string(), flush=True)

    log("fitting final stage-1 models on all three external games")
    final = PossessionModels.fit([bd for g in games for bd in train_bundles[g.name]] + extra,
                                 n_rounds=n_rounds, params=params)
    final.save(models_dir)
    if gru_w > 0:
        from possession.gru import GRUConfig, GRUModel
        log("fitting final GRU on all three external games")
        GRUModel.fit([bd for g in games for bd in train_bundles[g.name]] + extra,
                     GRUConfig(epochs=gru_epochs, threads=torch_threads, device=gru_device, seeds=gru_seeds,
                          hidden=gru_hidden, layers=gru_layers), log=log).save(models_dir)
    if stack_on:
        log("fitting final stacker on all out-of-fold stage-1 predictions")
        Stacker.fit([dict(p_dead=o["p_dead"], p_home=o["p_home"], period=o["period"], aux=o["aux"],
                          codes=o["codes"]) for o in oof], params=params).save(models_dir)
    with open(os.path.join(models_dir, "decode_cfg.json"), "w") as handle:
        json.dump(best_cfg.__dict__, handle, indent=2)
    with open(os.path.join(models_dir, "rules_cfg.json"), "w") as handle:
        json.dump(rule_w, handle, indent=2)
    with open(os.path.join(models_dir, "feature_cfg.json"), "w") as handle:
        json.dump(fcfg, handle, indent=2)
    with open(os.path.join(models_dir, "pipeline_cfg.json"), "w") as handle:
        json.dump({"use_stack": bool(stack_on), "variant": variant, "n_rounds": n_rounds, "gru_w": gru_w, "dfl": bool(dfl), "dfl_smooth_s": dfl_smooth,
                   "oof_spell_f1_models_st": s_base[0], "oof_spell_f1_stacker_st": s_stack[0]}, handle, indent=2)
    final.importance("ctrl", 25).round(1).to_frame("gain").to_csv(os.path.join(reports_dir, "importance_ctrl.csv"))
    final.importance("dead", 25).round(1).to_frame("gain").to_csv(os.path.join(reports_dir, "importance_dead.csv"))
    log("saved models + configs to {}; reports to {}".format(models_dir, reports_dir))


# ------------------------------------------------------------------------------------- predict

def _describe(codes, period, fps=25):
    from possession.labels import A, D, H, run_lengths
    g = codes[period > 0]
    spells = [(b - a) / fps for p in (1, 2) for a, b, c in run_lengths(codes[period == p]) if c in (H, A)]
    return {"H_%": round(100 * float((g == H).mean()), 1), "A_%": round(100 * float((g == A).mean()), 1),
            "D_%": round(100 * float((g == D).mean()), 1), "possessions": len(spells),
            "mean_poss_s": round(float(np.mean(spells)), 1) if spells else 0.0}


def _validate(path):
    check = subprocess.run([sys.executable, os.path.join("evaluation", "validate_submission.py"), path],
                           capture_output=True, text=True)
    print(check.stdout.strip() or check.stderr.strip(), flush=True)
    return check.returncode == 0


def _load_json(path, default):
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    return default


def predict(team, models_dir, data_dir=None, use_cache=True, compare=False,
            out_dir=os.path.join("evaluation", "results")):
    from possession.data import load_challenge
    from possession.decode import DecodeConfig
    from possession.models import PossessionModels
    from possession.pipeline import predict_challenge, write_submission
    from possession.stack import Stacker

    log("loading models from {}".format(models_dir))
    models = PossessionModels.load(models_dir)
    cfg = DecodeConfig(**_load_json(os.path.join(models_dir, "decode_cfg.json"), {}))
    rule_w = _load_json(os.path.join(models_dir, "rules_cfg.json"), {"w_team": 0.0, "w_dead": 0.0})
    fcfg = _load_json(os.path.join(models_dir, "feature_cfg.json"), None)
    pcfg = _load_json(os.path.join(models_dir, "pipeline_cfg.json"), {"use_stack": False})
    stacker = Stacker.load(models_dir) if pcfg.get("use_stack") else None
    gru_w = float(pcfg.get("gru_w", 0.0))
    gru_model = None
    if gru_w > 0:
        from possession.gru import GRUModel
        gru_model = GRUModel.load(models_dir)
    log("  decoder: {}  rule priors: {}  features: {}  stacker: {}".format(
        cfg, rule_w, fcfg or "default", "on" if stacker else "off"))

    log("loading provided match (cache={})".format(use_cache))
    match = load_challenge(data_dir=data_dir, cache=use_cache)

    log("running pipeline: broadcast fusion -> models -> stacker -> Smart Tagging -> decoder")
    codes, _, _, diag, variants = predict_challenge(models, cfg, match, data_dir=data_dir, cache=use_cache,
                                                    rule_weights=rule_w, feature_cfg=fcfg, stacker=stacker,
                                                    gru=gru_model, gru_w=gru_w)
    print(json.dumps(diag, indent=2), flush=True)

    path = write_submission(codes, os.path.join(out_dir, team + ".csv"))
    ok = _validate(path)
    log("sha256 {}  {}".format(sha256(path), path))

    if compare:
        rows = []
        for name, file_name in (("rules_only", team + "_rules_only.csv"),
                                ("models_st", team + "_models_only.csv"),
                                ("combined", team + ".csv")):
            p = path if name == "combined" else write_submission(variants[name], os.path.join(out_dir, file_name))
            rows.append({"variant": name, "file": os.path.basename(p), **_describe(variants[name], match.period)})
        print("\nside-by-side (prediction statistics only, no ground truth):")
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if not ok:
        sys.exit("validation failed")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "predict", "all"])
    ap.add_argument("--team", default="myteam", help="team name; output is evaluation/results/<team>.csv")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="model folder (default: %(default)s)")
    ap.add_argument("--reports", default=DEFAULT_REPORTS, help="where train writes its CV reports")
    ap.add_argument("--data-dir", default=None, help="match data folder (default: ./data)")
    ap.add_argument("--no-cache", action="store_true",
                    help="predict: re-parse the match files instead of reading artifacts/cache/")
    ap.add_argument("--compare", action="store_true",
                    help="predict: also write <team>_rules_only.csv and <team>_models_only.csv")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="dir_chal", help="train: training variant")
    ap.add_argument("--stack", choices=["auto", "on", "off"], default="off",
                    help="train: stage-2 stacker; auto = keep it only if it wins out-of-fold")
    ap.add_argument("--rule-search", action="store_true", help="train: also search teammate rule-prior weights")
    ap.add_argument("--dfl", action="store_true",
                    help="train: add the 7 open DFL matches as extra training data (needs kloppy>=3.19)")
    ap.add_argument("--dfl-smooth", type=float, default=0.0,
                    help="train: merge DFL owner runs shorter than this many seconds (0 = raw labels)")
    ap.add_argument("--no-gru", dest="gru", action="store_false",
                    help="train: skip the bidirectional GRU (by default it is trained and blended in if it wins out-of-fold)")
    ap.add_argument("--gru-device", default="cpu", choices=["cpu", "cuda"],
                    help="train: device for GRU training (prediction always runs on CPU)")
    ap.add_argument("--gru-seeds", type=int, default=5, help="train: GRU ensemble size (networks averaged)")
    ap.add_argument("--gru-hidden", type=int, default=128, help="train: GRU units per direction")
    ap.add_argument("--gru-layers", type=int, default=1, help="train: stacked GRU layers")
    ap.add_argument("--gru-epochs", type=int, default=15, help="train: GRU epochs")
    ap.add_argument("--torch-threads", type=int, default=4, help="train: CPU threads for the GRU (fixed for determinism; saved with the model)")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS, help="train: LightGBM boosting rounds per model")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM threads for train (0 = all cores)")
    args = ap.parse_args()

    t0 = time.time()
    if args.command in ("train", "all"):
        train(args.models, args.reports, threads=args.threads, variant=args.variant,
              search_rules=args.rule_search, use_stack=args.stack, n_rounds=args.rounds, dfl=args.dfl, dfl_smooth=args.dfl_smooth,
              gru=args.gru, gru_epochs=args.gru_epochs, torch_threads=args.torch_threads,
              gru_device=args.gru_device, gru_seeds=args.gru_seeds, gru_hidden=args.gru_hidden,
              gru_layers=args.gru_layers)
    if args.command in ("predict", "all"):
        predict(args.team, args.models, data_dir=args.data_dir, use_cache=not args.no_cache,
                compare=args.compare)
    log("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
