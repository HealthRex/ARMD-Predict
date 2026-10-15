"""Step 3: train one pooled four-site XGBoost model per cell (full and short form).

Pipeline per cell, in this exact order (it determines the fitted trees):
  1. load the cell's rows from each site's model table, sites in SITES order;
  2. keep the feature columns every contributing site has (no site_id);
  3. split PATIENTS at random 60/20/20 (seed 20260806); thresholds per partition;
  4. fit the encoder (numeric vs categorical, one-hot vocabulary) on train only;
  5. fill NaN with 99999 days on the monotone-increasing days-since columns;
  6. fit XGBoost with monotone constraints, early stopping on the validation part;
  7. report test AUROC.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from .config import DEFAULT_CONFIG_DIR, FEATURE_SET, SITES, cell_key, load_cells
from .leakage import PROTECTED_METADATA_COLUMNS, assert_no_protected_metadata

log = logging.getLogger(__name__)

SPLIT_SEED = 20260806
SPLIT_FRACTIONS = (0.60, 0.20)          # train, validation; test is the rest
XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": -1, "eval_metric": "auc", "early_stopping_rounds": 50,
    # Unweighted: re-weighting the positive class distorts the displayed probability.
    "scale_pos_weight": 1.0,
}
META_COLS = ({"site_id", "antibiotic", "culture_type", "ordering_mode", "y"}
             | set(PROTECTED_METADATA_COLUMNS))
ANTIBIOTIC_ALIASES = {
    "trimeth_sulfamethoxazole": "trimethoprim_sulfamethoxazole",
    "trimeth_sulf": "trimethoprim_sulfamethoxazole",
    "amoxicillin_clavulanate": "amoxicillin_clavulanic_acid",
    "amox_clavulanate": "amoxicillin_clavulanic_acid",
    "augmentin": "amoxicillin_clavulanic_acid",
    "pip_tazobactam": "piperacillin_tazobactam",
    "amp_sulbactam": "ampicillin_sulbactam",
}

# ── Monotone constraints ─────────────────────────────────────────────────────
# Without them the trees satisfied the loss through ~100 correlated history
# columns and left the clinically meaningful one pointing the wrong way (a prior
# resistant culture LOWERED predicted resistance); retraining alone reproduced
# the inversion. y = 1 is SUSCEPTIBLE, so "more prior resistance / exposure" is
# -1 and "more days since that event" is +1. Getting the sign backwards hardens
# the inversion. Companion columns (has_prior_event, event_count) are pinned
# too: pinning a family but leaving a companion free just moves the leak.
# These are the lists the published models were trained with.
MONOTONE_DECREASING_PREFIXES = (
    "prior_resistance_any_within", "prior_resistance_same_abx_within",
    "prior_resistance_event_count", "microbial_resistance_resistant_count",
    "antibiotic_class_exposure_any_prior", "antibiotic_subtype_exposure_any_prior",
    "antibiotic_exposure_any_prior", "prior_infect_org_any_within",
    "prior_med_any_within", "prior_med_any_prior", "prior_med_count_prior_events",
    "prior_resistance_has_prior_event", "prior_infect_org_has_prior_event",
    "prior_infect_org_event_count",
)
MONOTONE_INCREASING_PREFIXES = (
    "prior_resistance_min_days", "microbial_resistance_resistant_time_to_culture",
    "antibiotic_class_exposure_min_days_prior", "antibiotic_subtype_exposure_min_days_prior",
    "prior_infect_org_min_days", "prior_med_min_days_prior",
)
# Monotone constraints bind numeric moves only; XGBoost sends NaN down a
# learned default branch that they do not cover, so "never" -> "15 days ago"
# could still move the wrong way. "Never" (NaN) on the constrained days-since
# columns is therefore encoded as 99999 days, at the susceptible end of the
# axis, in training and at every prediction.
NEVER_SENTINEL_DAYS = 99999.0

# ── Encoder settings ─────────────────────────────────────────────────────────
MISSING_TOKENS = {"null", "none", "na", "n/a", "nan", "<na>", "nat", ""}
MAX_CATEGORIES = 50
MIN_CATEGORY_FREQ = 30
OTHER_CATEGORY = "__other__"
CATEGORY_FLOAT_DECIMALS = 3


def monotone_constraints(feature_names: List[str]) -> Tuple[int, ...]:
    out = []
    for name in map(str, feature_names):
        if name.startswith(MONOTONE_INCREASING_PREFIXES):
            out.append(1)
        elif name.startswith(MONOTONE_DECREASING_PREFIXES):
            out.append(-1)
        else:
            out.append(0)
    return tuple(out)


def canonical_antibiotic(name: str) -> str:
    return ANTIBIOTIC_ALIASES.get(name, name)


# ── Data ─────────────────────────────────────────────────────────────────────
def load_site_cell(tables_dir: Path | str, site: str, culture: str, setting: str,
                   antibiotic: str) -> Optional[pd.DataFrame]:
    path = Path(tables_dir) / site / culture / setting / FEATURE_SET / "model_table.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["antibiotic"] = df["antibiotic"].map(canonical_antibiotic)
    df = df[df["antibiotic"] == antibiotic].copy()
    return df if len(df) else None


def common_feature_columns(site_frames: Dict[str, pd.DataFrame]) -> List[str]:
    """Columns every contributing site carries, minus metadata and site_id (sorted)."""
    common = set.intersection(*(set(d.columns) for d in site_frames.values()))
    exclude = META_COLS | {c for c in common if c.startswith("site_id")}
    return sorted(common - exclude)


def patient_groups(df: pd.DataFrame) -> np.ndarray:
    """Site-qualified patient ids (two sites may reuse the same local id)."""
    if "patient_id" not in df.columns or df["patient_id"].isna().any():
        raise ValueError("every row needs a patient_id for the patient-grouped split")
    return (df["site_id"].astype(str) + "::" + df["patient_id"].astype(str)).values


def split_patients(groups: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """Random patient-grouped 60/20/20 assignment ('train' / 'val' / 'test') per row."""
    shuffled = np.random.default_rng(seed).permutation(np.unique(groups))
    n_train = int(round(len(shuffled) * SPLIT_FRACTIONS[0]))
    n_val = int(round(len(shuffled) * SPLIT_FRACTIONS[1]))
    bucket = {p: "train" for p in shuffled[:n_train]}
    bucket.update({p: "val" for p in shuffled[n_train:n_train + n_val]})
    bucket.update({p: "test" for p in shuffled[n_train + n_val:]})
    assign = np.array([bucket[p] for p in groups])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(groups[assign == a]) & set(groups[assign == b]):
            raise RuntimeError(f"a patient appears in both {a} and {b}")
    return assign


def counts(y: np.ndarray) -> Dict[str, int]:
    return {"n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


# ── Encoder (fitted on the training partition only) ──────────────────────────
def _blank_missing(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        txt = s.astype(str).str.strip().str.lower()
        return s.where(~txt.isin(MISSING_TOKENS) & s.notna())
    return s


def _as_category_text(s: pd.Series) -> pd.Series:
    # Round floats first so 1.1 / 1.10 / "1.1" are one level, not three.
    if pd.api.types.is_float_dtype(s):
        s = s.round(CATEGORY_FLOAT_DECIMALS)
    out = s.astype(object)
    return out.where(s.notna(), "Null").astype(str)


def fit_encoder(train: pd.DataFrame, feature_cols: List[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """A column is numeric only if ALL its non-missing values parse as numbers.

    The old ">50% parse" rule coerced MGB's "Female"/"Male" to NaN (every MGB
    patient became one sex). A column mixing numbers and text is an error.
    """
    col_types: Dict[str, str] = {}
    mixed = []
    for col in feature_cols:
        s = train[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            col_types[col] = "numeric"
            continue
        cleaned = _blank_missing(s)
        n_nonnull = int(cleaned.notna().sum())
        n_parsed = int(pd.to_numeric(cleaned, errors="coerce").notna().sum())
        if n_nonnull == 0 or n_parsed == n_nonnull:
            col_types[col] = "numeric"
        elif n_parsed == 0:
            col_types[col] = "categorical"
        else:
            mixed.append(f"{col} ({n_parsed:,}/{n_nonnull:,} values numeric)")
            col_types[col] = "categorical"
    if mixed:
        raise ValueError("columns mix numbers and text (harmonise them first): " + "; ".join(mixed))
    vocab: Dict[str, List[str]] = {}
    for col, kind in col_types.items():
        if kind != "categorical":
            continue
        vc = _as_category_text(train[col]).value_counts()
        frequent = vc[vc >= MIN_CATEGORY_FREQ].index.tolist()
        if len(frequent) > MAX_CATEGORIES:
            # Without the explicit bucket a rare level encodes exactly like "missing".
            frequent = frequent[:MAX_CATEGORIES] + [OTHER_CATEGORY]
        vocab[col] = sorted(frequent)
    return col_types, vocab


def encode(df: pd.DataFrame, feature_cols: List[str], col_types: Dict[str, str],
           vocab: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    """float32 matrix: numeric columns (NaN kept) then one-hot columns."""
    sub = df[feature_cols].copy()
    numeric, categorical = [], []
    for col in feature_cols:
        if col_types.get(col, "numeric") == "numeric":
            sub[col] = pd.to_numeric(_blank_missing(sub[col]), errors="coerce")
            numeric.append(col)
        else:
            categorical.append(col)
    X_num = (sub[numeric].to_numpy(dtype=np.float32) if numeric
             else np.empty((len(sub), 0), dtype=np.float32))
    names, arrays = list(numeric), []
    for col in categorical:
        cats = vocab.get(col, [])
        if not cats:
            continue
        txt = _as_category_text(sub[col])
        in_vocab = txt.isin([c for c in cats if c != OTHER_CATEGORY])
        for cat in cats:
            arr = (~in_vocab) if cat == OTHER_CATEGORY else (txt == cat)
            arrays.append(arr.to_numpy(dtype=np.float32))
            names.append(f"{col}__{cat}")
    # NaN is NOT zero-filled: XGBoost learns a default branch for "not measured".
    X = np.hstack([X_num, np.column_stack(arrays)]) if arrays else X_num
    return X, names


def apply_never_sentinel(X: np.ndarray, names: List[str], columns: List[str]) -> np.ndarray:
    for c in columns:
        if c in names:
            col = X[:, names.index(c)]
            col[np.isnan(col)] = NEVER_SENTINEL_DAYS
    return X


# ── Training ─────────────────────────────────────────────────────────────────
def load_cell_pool(tables_dir: Path | str, cell: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    frames = {}
    for site in SITES:
        d = load_site_cell(tables_dir, site, cell["culture"], cell["setting"], cell["antibiotic"])
        if d is not None:
            frames[site] = d
    return frames


def train_cell(cell: Dict[str, str], variant: str, tables_dir: Path | str, out_dir: Path | str,
               config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Optional[Dict[str, Any]]:
    """Train and save one model; returns a summary dict, or None if the cell is excluded."""
    spec = load_cells(config_dir)
    min_n, min_pos, min_neg = spec["min_n"], spec["min_susceptible"], spec["min_resistant"]
    key = cell_key(cell)
    frames = load_cell_pool(tables_dir, cell)
    if not frames:
        log.warning("%s: no rows", key)
        return None
    feature_cols = common_feature_columns(frames)
    if variant == "lite":
        prefixes = tuple(spec["lite_block_prefixes"])
        feature_cols = [c for c in feature_cols if c.startswith(prefixes)]
    assert_no_protected_metadata(feature_cols, key)
    pool = pd.concat(list(frames.values()), ignore_index=True)

    y_all = pool["y"].values
    cohort = counts(y_all)
    if cohort["n"] < min_n or cohort["n_pos"] < min_pos or cohort["n_neg"] < min_neg:
        log.warning("%s/%s excluded: %s", key, variant, cohort)
        return None
    groups = patient_groups(pool)
    assign = split_patients(groups)
    parts = {name: pool.iloc[np.flatnonzero(assign == name)].copy()
             for name in ("train", "val", "test")}
    per_split = {name: counts(parts[name]["y"].values) for name in parts}
    for name, c in per_split.items():
        if c["n"] < min_n or c["n_pos"] < min_pos or c["n_neg"] < min_neg:
            log.warning("%s/%s excluded: %s partition %s", key, variant, name, c)
            return None

    col_types, vocab = fit_encoder(parts["train"], feature_cols)
    X, y = {}, {}
    for name, part in parts.items():
        X[name], names = encode(part, feature_cols, col_types, vocab)
        y[name] = part["y"].values
    sentinel_cols = [c for c in names if c.startswith(MONOTONE_INCREASING_PREFIXES)]
    for name in X:
        apply_never_sentinel(X[name], names, sentinel_cols)

    params = dict(XGB_PARAMS)
    mono = monotone_constraints(names)
    if any(mono):
        params["monotone_constraints"] = mono
    model = xgb.XGBClassifier(**params)
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    p_test = model.predict_proba(X["test"])[:, 1]
    auroc = float(roc_auc_score(y["test"], p_test))
    log.info("%s/%s: test AUROC %.6f (n=%d), best iteration %s", key, variant, auroc,
             len(p_test), model.best_iteration)

    out = Path(out_dir) / variant / key
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    meta = {
        "cell": key, **cell, "variant": variant,
        "features_used": feature_cols,
        "column_types": col_types,
        "category_vocabulary": vocab,
        "encoded_feature_names": names,
        "never_sentinel": {"value": NEVER_SENTINEL_DAYS, "columns": sentinel_cols},
        "monotone_constraints": {
            "decreasing": [n for n, m in zip(names, mono) if m == -1],
            "increasing": [n for n, m in zip(names, mono) if m == 1]},
        "split": {"type": "random_patient_grouped", "seed": SPLIT_SEED,
                  "fractions": [0.6, 0.2, 0.2], "counts": per_split},
        "cohort": cohort,
        # pi_train for local-prevalence re-anchoring (P(resistant) in the pooled cell).
        "resistance_rate": float((y_all == 0).mean()),
        "train_susceptible_rate": float(y["train"].mean()),
        "auroc_test": auroc,
        "n_test": int(len(p_test)),
        "best_iteration": int(model.best_iteration),
        "xgboost_params": {k: v for k, v in params.items() if k != "monotone_constraints"},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    return {"cell": key, "variant": variant, "auroc_test": auroc, "n_test": len(p_test),
            "n_features": len(names), "best_iteration": int(model.best_iteration)}


# ── Replay (shared by calibrate, evaluate and baselines) ─────────────────────
def load_model(model_dir: Path | str) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    model_dir = Path(model_dir)
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "model.json")
    return model, json.loads((model_dir / "metadata.json").read_text())


def prepare_features(df: pd.DataFrame, meta: Dict[str, Any]) -> np.ndarray:
    """Model-table rows -> the exact matrix the model was trained on."""
    missing = [c for c in meta["features_used"] if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} model features are missing from the table "
                         f"(e.g. {missing[:3]}); these are not the tables the model was trained on")
    X, names = encode(df, meta["features_used"], meta["column_types"], meta["category_vocabulary"])
    if names != meta["encoded_feature_names"]:
        raise ValueError("encoded feature names differ from the training run")
    return apply_never_sentinel(X, names, meta["never_sentinel"]["columns"])


def replay(model_dir: Path | str, tables_dir: Path | str) -> Dict[str, Any]:
    """Reload a model's cell, re-deal its recorded split and check the counts."""
    model, meta = load_model(model_dir)
    frames = load_cell_pool(tables_dir, meta)
    pool = pd.concat(list(frames.values()), ignore_index=True)
    groups = patient_groups(pool)
    assign = split_patients(groups, meta["split"]["seed"])
    y = pool["y"].values.astype(int)
    for name, want in meta["split"]["counts"].items():
        if counts(y[assign == name]) != want:
            raise RuntimeError(f"{meta['cell']}: replayed {name} partition does not match the "
                               f"recorded counts; refusing to score possibly-trained-on rows")
    return {"model": model, "meta": meta, "pool": pool, "groups": groups,
            "assign": assign, "y": y, "X": prepare_features(pool, meta)}
