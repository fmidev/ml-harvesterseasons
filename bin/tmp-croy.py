#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for SWE (snow water equivalent) regression with XGBoost.

Trains three models from a single training table, selected via a domain / tag column:
  1. scandi : rows where tag == "scandi"
  2. alps   : rows where tag == "alps"
  3. both   : scandi + alps pooled

Splitting (SPLIT_MODE):
  * "time"  (default) : split by SNOW YEAR (Sep 1 boundary), whole seasons kept
                        together. Chronological by default: oldest -> train,
                        most recent -> test. Mirrors operational forecasting.
  * "group"           : split by location (leakage-safe for new-terrain tests).

For each model:
  * Optuna study (resumable via SQLite) optimising back-transformed RMSE on val
  * early stopping (+ optional Optuna pruning) driven by the true, back-transformed metric
  * final retrain with best params, full test-set metrics
  * SHAP (TreeExplainer): beeswarm, mean|SHAP| bar, top-N dependence, importance CSV

SNOWCAP conventions honoured:
  * domain tag in every output filename (critical separation rule)
  * run-tag namespacing of outputs
  * resumable studies as defensive checkpointing

Usage:
  python optuna_swe.py <run_tag>
  python optuna_swe.py exp01 --domains scandi alps both --n-trials 80
  python optuna_swe.py gpu_run --device cuda --n-trials 120
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import shap
    _HAVE_SHAP = True
except Exception as exc:  # noqa: BLE001
    warnings.warn(f"shap not importable ({exc}); SHAP step will be skipped.")
    _HAVE_SHAP = False


# ======================================================================
# CONFIG  (adjust column names / paths to match your training table)
# ======================================================================

# --- data ---
INPUT_PATH = "training_swe_merged.parquet"   # .parquet or .csv
TARGET_COL = "swe"                            # SWE target column
TAG_COL    = "tag"                            # domain / tag column

# Which raw tag values map to each domain. Edit if your tags differ
# (e.g. "scandinavia" / "alps", or numeric codes).
TAG_VALUES = {
    "scandi": ["scandi", "scandinavia"],
    "alps":   ["alps"],
    "both":   ["scandi", "scandinavia", "alps"],
}

# Columns that are NOT model features (ids, bookkeeping, extra coords).
# TARGET_COL, TAG_COL and TIME_COL are dropped automatically. lat/lon are
# KEPT as features by default (valid SWE predictors); add them here to drop.
DROP_COLS = []

# ----------------------------------------------------------------------
# SPLIT SETTINGS
# ----------------------------------------------------------------------
SPLIT_MODE = "time"          # "time" (by snow year) or "group" (by location)

# --- time split ---
TIME_COL = "date"            # a date column: datetime, YYYYMMDD, or a year int
SNOW_YEAR_START_MONTH = 9    # snow season starts Sep 1; Sep 2016 -> snow year 2016

# Option A: name the snow years explicitly (start year of each season).
# Leave all three empty to use Option B (chronological fractions).
TRAIN_YEARS = []             # e.g. [2016, 2017, 2018, 2019, 2020]
VAL_YEARS   = []             # e.g. [2021]
TEST_YEARS  = []             # e.g. [2022, 2023]

# Option B: chronological fractions of the available snow years.
# Most recent -> test, the block before it -> val, the rest (oldest) -> train.
TEST_FRAC = 0.15
VAL_FRAC  = 0.15

# --- group (location) split, only used when SPLIT_MODE == "group" ---
GROUP_COL = None
LATLON_COLS = ("lat", "lon")
GROUP_LATLON_ROUND = 2       # decimals for deriving location groups

# --- objective / transform ---
# "reg:squarederror" (default, paired with log1p) or "reg:tweedie".
OBJECTIVE = "reg:squarederror"
# "log1p" (default; SHAP values then live in log space) or "none".
TARGET_TRANSFORM = "log1p"

# Add a numeric is_alps feature to the pooled "both" model.
ADD_DOMAIN_DUMMY_FOR_BOTH = False

# --- xgboost / search ---
NUM_BOOST_ROUND_MAX = 5000     # ceiling; early stopping picks the real count
EARLY_STOPPING_ROUNDS = 100
MAX_BIN = 256                  # fixed so the QuantileDMatrix cache stays valid
DEFAULT_N_TRIALS = 60
DEFAULT_DEVICE = "cpu"         # or "cuda"
DEFAULT_NTHREAD = 0            # 0 = all cores
SEED = 42

# --- optuna pruning ---
USE_PRUNING = True

# --- shap ---
SHAP_SAMPLE_N = 5000
SHAP_MAX_DISPLAY = 20
SHAP_TOP_DEPENDENCE = 6

# --- output ---
OUTPUT_ROOT = "optuna_swe"


# ======================================================================
# transforms
# ======================================================================

def fwd_transform(y):
    if TARGET_TRANSFORM == "log1p":
        return np.log1p(np.clip(y, 0, None))
    return np.asarray(y, dtype=float)


def inv_transform(y):
    if TARGET_TRANSFORM == "log1p":
        return np.clip(np.expm1(y), 0, None)
    return np.clip(np.asarray(y, dtype=float), 0, None)  # SWE is non-negative


# ======================================================================
# data loading + domain / feature selection
# ======================================================================

def load_table(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p.resolve()}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def select_domain(df, domain):
    values = TAG_VALUES[domain]
    sub = df[df[TAG_COL].isin(values)].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for domain '{domain}' with {TAG_COL} in {values}. "
            f"Present tag values: {sorted(df[TAG_COL].unique())[:20]}"
        )
    return sub


def feature_columns(df, domain):
    non_features = {TARGET_COL, TAG_COL, TIME_COL, *DROP_COLS}
    if GROUP_COL is not None:
        non_features.add(GROUP_COL)
    cols = [c for c in df.columns
            if c not in non_features and pd.api.types.is_numeric_dtype(df[c])]
    if ADD_DOMAIN_DUMMY_FOR_BOTH and domain == "both":
        df["is_alps"] = df[TAG_COL].isin(TAG_VALUES["alps"]).astype(np.int8)
        cols.append("is_alps")
    return cols


# ======================================================================
# snow-year derivation + time split
# ======================================================================

def derive_snow_year(df):
    """Map each row to a snow-year start year (Sep boundary)."""
    if TIME_COL not in df.columns:
        raise KeyError(
            f"SPLIT_MODE='time' needs TIME_COL '{TIME_COL}' in the table.")
    col = df[TIME_COL]

    # plain year integers (e.g. 2016..2024)
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        vals = col.astype("Int64")
        if vals.dropna().between(1900, 2100).all():
            return vals.astype(float).astype(int).values, "year column"
        dt = pd.to_datetime(vals.astype(str), format="%Y%m%d", errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(col):
        dt = col
    else:  # strings: try YYYYMMDD, then general parsing
        dt = pd.to_datetime(col, format="%Y%m%d", errors="coerce")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(col, errors="coerce")

    sy = dt.dt.year.where(dt.dt.month >= SNOW_YEAR_START_MONTH, dt.dt.year - 1)
    n_bad = int(sy.isna().sum())
    if n_bad:
        warnings.warn(f"{n_bad:,} rows have an unparseable {TIME_COL}; dropped.")
    return sy.values, f"snow-year from {TIME_COL} (Sep 1 boundary)"


def time_train_val_test(df, feats):
    X = df[feats]
    y = df[TARGET_COL].values.astype(float)
    sy, how = derive_snow_year(df)
    sy = pd.Series(sy, index=df.index)
    print(f"  time split via {how}")

    if TRAIN_YEARS or VAL_YEARS or TEST_YEARS:
        train_years, val_years, test_years = TRAIN_YEARS, VAL_YEARS, TEST_YEARS
    else:
        years = np.sort(pd.unique(sy.dropna())).astype(int)
        n = len(years)
        if n < 3:
            raise ValueError(f"Need >= 3 snow years for a time split, found {n}.")
        n_test = max(1, int(round(n * TEST_FRAC)))
        n_val = max(1, int(round(n * VAL_FRAC)))
        if n_test + n_val >= n:
            raise ValueError("TEST_FRAC + VAL_FRAC leave no training years.")
        test_years = list(years[-n_test:])
        val_years = list(years[-(n_test + n_val):-n_test])
        train_years = list(years[:-(n_test + n_val)])

    print(f"  train years {train_years}")
    print(f"  val   years {val_years}")
    print(f"  test  years {test_years}")

    tr = sy.isin(train_years).values
    va = sy.isin(val_years).values
    te = sy.isin(test_years).values
    for name, mask in (("train", tr), ("val", va), ("test", te)):
        if not mask.any():
            raise ValueError(f"Time split produced an empty {name} set.")

    return (X[tr].reset_index(drop=True), y[tr],
            X[va].reset_index(drop=True), y[va],
            X[te].reset_index(drop=True), y[te])


# ======================================================================
# group (location) split
# ======================================================================

def make_group_ids(df):
    if GROUP_COL is not None and GROUP_COL in df.columns:
        return df[GROUP_COL].astype(str).values, f"column '{GROUP_COL}'"
    la, lo = LATLON_COLS
    if la in df.columns and lo in df.columns:
        g = (df[la].round(GROUP_LATLON_ROUND).astype(str) + "_"
             + df[lo].round(GROUP_LATLON_ROUND).astype(str))
        return g.values, f"rounded {la}/{lo} @ {GROUP_LATLON_ROUND} dp"
    return None, None


def group_train_val_test(df, feats, seed):
    X = df[feats]
    y = df[TARGET_COL].values.astype(float)
    groups, how = make_group_ids(df)
    if groups is None:
        raise ValueError(
            "SPLIT_MODE='group' but no GROUP_COL and no lat/lon columns found.")
    print(f"  group split by {how} ({pd.Series(groups).nunique():,} groups)")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    rest_i, test_i = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=seed)
    rel_tr, rel_val = next(gss2.split(X.iloc[rest_i], y[rest_i], groups[rest_i]))
    tr_i = rest_i[rel_tr]
    val_i = rest_i[rel_val]
    return (X.iloc[tr_i].reset_index(drop=True), y[tr_i],
            X.iloc[val_i].reset_index(drop=True), y[val_i],
            X.iloc[test_i].reset_index(drop=True), y[test_i])


def make_splits(df, feats, seed):
    if SPLIT_MODE == "time":
        return time_train_val_test(df, feats)
    if SPLIT_MODE == "group":
        return group_train_val_test(df, feats, seed)
    raise ValueError(f"Unknown SPLIT_MODE '{SPLIT_MODE}'.")


# ======================================================================
# metric + pruning callback (both on the back-transformed scale)
# ======================================================================

def rmse_orig_metric(y_pred, dmat):
    """Custom eval metric: RMSE on the original (back-transformed) SWE scale."""
    y_true = inv_transform(dmat.get_label())
    y_hat = inv_transform(y_pred)
    return "rmse_orig", float(np.sqrt(mean_squared_error(y_true, y_hat)))


class OptunaPruningCallback(xgb.callback.TrainingCallback):
    """Report val rmse_orig to Optuna each round and prune unpromising trials."""

    def __init__(self, trial, eval_name="val", metric="rmse_orig"):
        self.trial = trial
        self.key = (eval_name, metric)

    def after_iteration(self, model, epoch, evals_log):
        name, metric = self.key
        try:
            value = evals_log[name][metric][-1]
        except (KeyError, IndexError):
            return False
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return False


# ======================================================================
# optuna objective (data built once per domain, reused across trials)
# ======================================================================

def build_params(trial, device, nthread):
    params = {
        "objective": OBJECTIVE,
        "tree_method": "hist",
        "device": device,
        "nthread": nthread,
        "max_bin": MAX_BIN,
        "disable_default_eval_metric": 1,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 256.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "seed": SEED,
    }
    if OBJECTIVE == "reg:tweedie":
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.05, 1.95)
    return params


def make_objective(dtrain, dval, device, nthread):
    def objective(trial):
        params = build_params(trial, device, nthread)
        callbacks = []
        if USE_PRUNING:
            callbacks.append(OptunaPruningCallback(trial))
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND_MAX,
            evals=[(dtrain, "train"), (dval, "val")],
            custom_metric=rmse_orig_metric,
            maximize=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            callbacks=callbacks,
            verbose_eval=False,
        )
        trial.set_user_attr("best_iteration", int(bst.best_iteration))
        return float(bst.best_score)
    return objective


# ======================================================================
# final fit + evaluation
# ======================================================================

def fit_final(best_params, dtrain, dval, device, nthread):
    params = {
        "objective": OBJECTIVE,
        "tree_method": "hist",
        "device": device,
        "nthread": nthread,
        "max_bin": MAX_BIN,
        "disable_default_eval_metric": 1,
        "seed": SEED,
        **best_params,
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND_MAX,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=rmse_orig_metric,
        maximize=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    return bst


def evaluate(bst, dtest, y_test):
    y_hat = inv_transform(
        bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1)))
    y_true = np.clip(y_test, 0, None)
    snow = y_true > 10.0  # operationally meaningful snow-covered subset (mm)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
        "mae": float(mean_absolute_error(y_true, y_hat)),
        "r2": float(r2_score(y_true, y_hat)),
        "bias_mean_pred_minus_true": float(np.mean(y_hat) - np.mean(y_true)),
        "n_test": int(len(y_true)),
        "rmse_snow_gt_10mm": (
            float(np.sqrt(mean_squared_error(y_true[snow], y_hat[snow])))
            if snow.any() else None),
        "n_snow_gt_10mm": int(snow.sum()),
    }
    return metrics


# ======================================================================
# shap
# ======================================================================

def run_shap(bst, X_test, outdir, tag, seed):
    if not _HAVE_SHAP:
        return
    n = min(SHAP_SAMPLE_N, len(X_test))
    Xs = X_test.sample(n=n, random_state=seed).reset_index(drop=True)
    explainer = shap.TreeExplainer(bst)
    sv = explainer(Xs)

    mean_abs = np.abs(sv.values).mean(axis=0)
    imp = (pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp.to_csv(outdir / f"shap_importance_{tag}.csv", index=False)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP beeswarm ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_beeswarm_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(sv, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP mean|value| ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_bar_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feat in imp["feature"].head(SHAP_TOP_DEPENDENCE):
        plt.figure()
        shap.plots.scatter(sv[:, feat], show=False)
        plt.title(f"SHAP dependence: {feat} ({tag})")
        plt.tight_layout()
        plt.savefig(outdir / f"shap_dep_{tag}_{feat}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    note = ("SHAP values are on the LOG1P scale (target was log1p-transformed); "
            "interpret magnitudes accordingly."
            if TARGET_TRANSFORM == "log1p"
            else "SHAP values are on the native SWE scale.")
    (outdir / f"shap_note_{tag}.txt").write_text(note + "\n")


# ======================================================================
# per-domain driver
# ======================================================================

def run_domain(domain, df_all, args, storage):
    tag = domain  # scandi / alps / both -> used in every filename
    print(f"\n{'=' * 60}\nDOMAIN: {domain}\n{'=' * 60}")

    df = select_domain(df_all, domain)
    feats = feature_columns(df, domain)
    print(f"  rows: {len(df):,}   features: {len(feats)}")

    Xtr, ytr, Xval, yval, Xte, yte = make_splits(df, feats, args.seed)
    print(f"  split rows  train={len(Xtr):,}  val={len(Xval):,}  test={len(Xte):,}")

    # DMatrices built ONCE and reused across all trials
    dtrain = xgb.QuantileDMatrix(
        Xtr, label=fwd_transform(ytr), max_bin=MAX_BIN, feature_names=list(feats))
    dval = xgb.QuantileDMatrix(
        Xval, label=fwd_transform(yval), ref=dtrain, feature_names=list(feats))
    dtest = xgb.DMatrix(Xte, feature_names=list(feats))

    outdir = Path(OUTPUT_ROOT) / args.run_tag / domain
    outdir.mkdir(parents=True, exist_ok=True)

    pruner = (optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=50)
              if USE_PRUNING else optuna.pruners.NopPruner())
    study = optuna.create_study(
        direction="minimize",
        study_name=f"swe_{domain}_{args.run_tag}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
    )
    study.optimize(
        make_objective(dtrain, dval, args.device, args.n_jobs),
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print(f"  best val rmse_orig: {study.best_value:.4f}")
    study.trials_dataframe().to_csv(outdir / f"optuna_trials_{tag}.csv", index=False)
    (outdir / f"best_params_{tag}.json").write_text(
        json.dumps(study.best_params, indent=2))

    bst = fit_final(study.best_params, dtrain, dval, args.device, args.n_jobs)
    bst.save_model(str(outdir / f"model_swe_{tag}.json"))

    metrics = evaluate(bst, dtest, yte)
    metrics.update({
        "domain": domain,
        "run_tag": args.run_tag,
        "split_mode": SPLIT_MODE,
        "objective": OBJECTIVE,
        "target_transform": TARGET_TRANSFORM,
        "n_features": len(feats),
        "best_val_rmse_orig": float(study.best_value),
        "best_iteration": int(bst.best_iteration),
    })
    (outdir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(f"  TEST rmse={metrics['rmse']:.3f}  mae={metrics['mae']:.3f}  "
          f"r2={metrics['r2']:.4f}  bias={metrics['bias_mean_pred_minus_true']:.3f}")

    run_shap(bst, Xte, outdir, tag, args.seed)
    print(f"  outputs -> {outdir}")
    return metrics


# ======================================================================
# main
# ======================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Optuna XGBoost SWE tuning (3 domains).")
    ap.add_argument("run_tag", help="experiment tag; namespaces all outputs")
    ap.add_argument("--input", default=INPUT_PATH)
    ap.add_argument("--domains", nargs="+", default=["scandi", "alps", "both"],
                    choices=["scandi", "alps", "both"])
    ap.add_argument("--split-mode", default=None, choices=["time", "group"],
                    help="override SPLIT_MODE")
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    ap.add_argument("--n-jobs", type=int, default=DEFAULT_NTHREAD,
                    help="xgboost nthread (0 = all cores)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target", default=None, help="override TARGET_COL")
    ap.add_argument("--tag-col", default=None, help="override TAG_COL")
    ap.add_argument("--time-col", default=None, help="override TIME_COL")
    ap.add_argument("--group-col", default=None, help="override GROUP_COL")
    return ap.parse_args()


def main():
    args = parse_args()

    global TARGET_COL, TAG_COL, TIME_COL, GROUP_COL, SPLIT_MODE
    if args.target:
        TARGET_COL = args.target
    if args.tag_col:
        TAG_COL = args.tag_col
    if args.time_col:
        TIME_COL = args.time_col
    if args.group_col:
        GROUP_COL = args.group_col
    if args.split_mode:
        SPLIT_MODE = args.split_mode

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"loading {args.input} ...")
    df_all = load_table(args.input)
    print(f"  {len(df_all):,} rows, {df_all.shape[1]} columns  (split_mode={SPLIT_MODE})")
    if TARGET_COL not in df_all.columns:
        raise KeyError(f"TARGET_COL '{TARGET_COL}' not in table.")
    if TAG_COL not in df_all.columns:
        raise KeyError(f"TAG_COL '{TAG_COL}' not in table.")

    root = Path(OUTPUT_ROOT) / args.run_tag
    root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(root / f'optuna_swe_{args.run_tag}.db').as_posix()}"

    summary = {}
    for domain in args.domains:
        summary[domain] = run_domain(domain, df_all, args, storage)

    (root / f"summary_{args.run_tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll domains done. Summary -> {root / f'summary_{args.run_tag}.json'}")


if __name__ == "__main__":
    main()