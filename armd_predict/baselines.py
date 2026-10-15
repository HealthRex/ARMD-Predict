"""Simple-rule comparators on each model's own test orders.

Antibiogram rule: every order gets its cell's susceptibility rate in the
TRAINING partition (a constant, AUROC 0.5).
Last-result rule: the patient's most recent earlier result for the same
antibiotic, from any culture in the CDM recorded on an EARLIER calendar day
(same-day results excluded, as in the feature day-zero rule; resistant wins
within a day; conflicting labels dropped). Prior S -> P(S) = 1, prior R/I ->
P(S) = 0, no prior result -> the antibiogram rate. At the rule's alert rate
(share of orders with a prior R/I result) the model flags the same number of
orders, ranked by predicted resistance (ties broken at random, fixed seed).

Aggregates only: counts of 1-19 are written "<20" and proportions that would
reveal such a count "masked(<20)".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import BOOTSTRAP_SEED, N_BOOT, _Ranks, bootstrap_weights, percentile_ci
from .train import canonical_antibiotic, replay

MIN_CELL = 20


def mask_count(n: int):
    n = int(n)
    return n if n == 0 or n >= MIN_CELL else "<20"


def masked_rate(num: int, den: int):
    num, den = int(num), int(den)
    if den == 0:
        return float("nan")
    if any(0 < v < MIN_CELL for v in (num, den - num, den)):
        return "masked(<20)"
    return num / den


def load_history(cdm_dir: Path | str, site: str, antibiotics: Iterable[str]) -> pd.DataFrame:
    """Day-level same-drug results for one site: pid, abx, date, prior_y (resistant wins)."""
    c = pd.read_parquet(Path(cdm_dir) / site / "cohort_long.parquet",
                        columns=["patient_id", "culture_time", "antibiotic", "y", "had_conflict"])
    c["abx"] = c["antibiotic"].astype(str).map(canonical_antibiotic)
    c = c[c["abx"].isin(set(antibiotics)) & c["y"].notna() & ~(c["had_conflict"] == True)]  # noqa: E712
    c = c.assign(pid=c["patient_id"].astype(str),
                 date=pd.to_datetime(c["culture_time"]).dt.normalize())
    return c.groupby(["pid", "abx", "date"], sort=False).agg(prior_y=("y", "min")).reset_index()


def _last_result(test: pd.DataFrame, antibiotic: str, history: Dict[str, pd.DataFrame]) -> np.ndarray:
    test = test.assign(_r=np.arange(len(test)), pid=test["patient_id"].astype(str),
                       idx_date=pd.to_datetime(test["culture_time"]).dt.normalize())
    prior = np.full(len(test), np.nan)
    for site, g in test.groupby("site_id", sort=False):
        h = history[site]
        h = h[h["abx"] == antibiotic][["pid", "date", "prior_y"]].rename(columns={"date": "prior_date"})
        m = pd.merge_asof(g[["_r", "pid", "idx_date"]].sort_values("idx_date"),
                          h.sort_values("prior_date"), left_on="idx_date", right_on="prior_date",
                          by="pid", allow_exact_matches=False, direction="backward")
        prior[m["_r"].to_numpy()] = m["prior_y"].to_numpy(dtype=float)
    return prior


def compare_cell(model_dir: Path | str, tables_dir: Path | str,
                 history: Dict[str, pd.DataFrame], n_boot: int = N_BOOT) -> Dict[str, Any]:
    r = replay(model_dir, tables_dir)
    meta, test = r["meta"], r["assign"] == "test"
    y = r["y"][test]
    p = r["model"].predict_proba(r["X"][test])[:, 1].astype(float)
    rate_s = float(r["y"][r["assign"] == "train"].mean())
    prior = _last_result(r["pool"][test], meta["antibiotic"], history)
    has, pr_r, pr_s = np.isfinite(prior), prior == 0, prior == 1
    rule = np.where(pr_s, 1.0, np.where(pr_r, 0.0, rate_s))
    n, n_s = len(y), int(y.sum())
    n_r = n - n_s
    out: Dict[str, Any] = {
        "cell": meta["cell"], "n_test": mask_count(n), "n_resistant": mask_count(n_r),
        "antibiogram_train_susceptible_rate": rate_s, "antibiogram_auroc": 0.5,
        "share_with_prior_same_drug": masked_rate(has.sum(), n),
        "share_prior_resistant": masked_rate(pr_r.sum(), n),
        "auroc_model": float(roc_auc_score(y, p)), "auroc_last_result": float(roc_auc_score(y, rule)),
    }
    ys = y[has]
    sub_ok = ys.sum() >= MIN_CELL and len(ys) - ys.sum() >= MIN_CELL
    if sub_ok:
        out["auroc_model_with_prior"] = float(roc_auc_score(ys, p[has]))
        out["auroc_last_result_with_prior"] = float(roc_auc_score(ys, prior[has]))

    # Paired patient-clustered bootstrap of the AUROC difference (model - rule).
    rm, rr = _Ranks(p, y), _Ranks(rule, y)
    rms, rrs = (_Ranks(p[has], ys), _Ranks(prior[has], ys)) if sub_ok else (None, None)
    d_all, d_sub, r_all = [], [], []
    for w in bootstrap_weights(r["groups"][test], n_boot, BOOTSTRAP_SEED):
        a_r = rr.auroc(w)
        r_all.append(a_r)
        d_all.append(rm.auroc(w) - a_r)
        if sub_ok:
            d_sub.append(rms.auroc(w[has]) - rrs.auroc(w[has]))
    out["auroc_last_result_ci_lo"], out["auroc_last_result_ci_hi"] = percentile_ci(r_all)
    out["delta_auroc"] = out["auroc_model"] - out["auroc_last_result"]
    out["delta_auroc_ci_lo"], out["delta_auroc_ci_hi"] = percentile_ci(d_all)
    if sub_ok:
        out["delta_auroc_with_prior"] = out["auroc_model_with_prior"] - out["auroc_last_result_with_prior"]
        out["delta_auroc_with_prior_ci_lo"], out["delta_auroc_with_prior_ci_hi"] = percentile_ci(d_sub)

    # Matched alert rate: the model flags as many orders as the rule does.
    k = int(pr_r.sum())
    out["alert_rate"] = masked_rate(k, n)
    if k > 0:
        tie = np.random.default_rng(BOOTSTRAP_SEED).random(n)
        flag_m = np.zeros(n, bool)
        flag_m[np.lexsort((tie, p))[:k]] = True     # lowest P(S) = highest P(R)
        for name, flags in (("model", flag_m), ("rule", pr_r)):
            out[f"{name}_sensitivity"] = masked_rate((flags & (y == 0)).sum(), n_r)
            out[f"{name}_ppv"] = masked_rate((flags & (y == 0)).sum(), k)
            out[f"{name}_susceptible_flagged"] = masked_rate((flags & (y == 1)).sum(), n_s)
    return out
