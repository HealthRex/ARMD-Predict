"""Step 4b: held-out evaluation of each model on its recorded test partition.

Susceptible (y = 1) is the positive class throughout. Reported per cell:
AUROC and AUPRC with patient-clustered bootstrap 95% CIs (1,000 replicates,
all of a patient's test orders drawn together), specificity at sensitivity
0.90 / 0.80, Brier score, ECE (10 equal-width bins), calibration intercept and
slope, for the raw and the served (calibrated if accepted) probability, and the
two exploratory operating thresholds. Thresholds and their yields are derived
on the same test rows, so they need validation in independent data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

BOOTSTRAP_SEED = 20260806
N_BOOT = 1000
PPV_TARGET = 0.95            # narrowing threshold: PPV for "susceptible" >= 95%
MIN_PREDICTED_SUSCEPTIBLE = 20
ESCALATION_BUDGET = 0.20     # at most 20% of susceptible orders flagged


# ── Point metrics ────────────────────────────────────────────────────────────
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins [0, 0.1], (0.1, 0.2], ..."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray, eps: float = 1e-7,
                                iters: int = 100) -> Tuple[float, float]:
    """Unpenalised logistic recalibration y ~ a + b * logit(p), ideal (0, 1).

    Newton-Raphson to convergence; a default-tolerance lbfgs fit stops about
    2e-4 short of the maximum-likelihood values reported in the paper.
    """
    q = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    x, y = np.log(q / (1 - q)), np.asarray(y, dtype=float)
    a, b = 0.0, 1.0
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(a + b * x)))
        r, v = y - mu, mu * (1 - mu)
        step = np.linalg.solve(np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]]),
                               np.array([r.sum(), (r * x).sum()]))
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def specificity_at_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    for i, sens in enumerate(tpr):
        if sens >= target:
            return float(1 - fpr[i])
    return float(1 - fpr[-1])


# ── Patient-clustered bootstrap ──────────────────────────────────────────────
class _Ranks:
    """Weighted AUROC / average precision on fixed scores (integer weights =
    exactly the metric on the resampled data, without materialising it)."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        self.y = np.asarray(y, dtype=np.float64)
        _, self.gid = np.unique(np.asarray(score, dtype=np.float64), return_inverse=True)
        self.ng = int(self.gid.max()) + 1

    def auroc(self, w: np.ndarray) -> float:
        wp = np.bincount(self.gid, weights=w * self.y, minlength=self.ng)
        wn = np.bincount(self.gid, weights=w * (1.0 - self.y), minlength=self.ng)
        P, N = wp.sum(), wn.sum()
        if P <= 0 or N <= 0:
            return float("nan")
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (P * N))

    def ap(self, w: np.ndarray) -> float:
        tp = np.bincount(self.gid, weights=w * self.y, minlength=self.ng)[::-1]
        fp = np.bincount(self.gid, weights=w * (1.0 - self.y), minlength=self.ng)[::-1]
        P = tp.sum()
        if P <= 0:
            return float("nan")
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        denom = ctp + cfp
        prec = np.divide(ctp, denom, out=np.zeros_like(ctp), where=denom > 0)
        return float((tp / P * prec).sum())


def bootstrap_weights(groups: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED):
    """Yield per-row weights; each replicate resamples PATIENTS with replacement."""
    _, code = np.unique(np.asarray(groups), return_inverse=True)
    n_cl = int(code.max()) + 1
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl).astype(np.float64)
        yield c[code]


def percentile_ci(values) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def clustered_ci(y, p, groups, n_boot: int = N_BOOT) -> Dict[str, float]:
    r = _Ranks(p, y)
    au, ap = [], []
    for w in bootstrap_weights(groups, n_boot):
        au.append(r.auroc(w))
        ap.append(r.ap(w))
    (a_lo, a_hi), (p_lo, p_hi) = percentile_ci(au), percentile_ci(ap)
    return {"auroc_ci_lo": a_lo, "auroc_ci_hi": a_hi, "auprc_ci_lo": p_lo, "auprc_ci_hi": p_hi}


# ── Operating thresholds ─────────────────────────────────────────────────────
def operating_thresholds(y: np.ndarray, p_susc: np.ndarray) -> Optional[Dict[str, Any]]:
    """Narrowing: lowest T with PPV >= 95% among P(S) >= T (>= 20 predicted S).
    Escalation: the T, among those flagging <= 20% of susceptible orders
    (P(S) < T), that flags the most resistant orders."""
    y, p = np.asarray(y), np.asarray(p_susc)
    if len(y) < 50 or y.sum() in (0, len(y)):
        return None
    n_s, n_r = int(y.sum()), int(len(y) - y.sum())
    # Candidate cut-offs, highest first: a fixed grid plus every observed score.
    ts = np.sort(np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), p])))[::-1]
    ps = np.sort(p[y == 1].astype(np.float64))
    pr = np.sort(p[y == 0].astype(np.float64))
    tp = len(ps) - np.searchsorted(ps, ts, side="left")        # susceptible with P(S) >= t
    fp = len(pr) - np.searchsorted(pr, ts, side="left")        # resistant with P(S) >= t
    caught = np.searchsorted(pr, ts, side="left") / max(n_r, 1)  # resistant with P(S) < t
    sens = tp / max(n_s, 1)
    ppv = np.divide(tp, tp + fp, out=np.zeros(len(ts)), where=(tp + fp) > 0)
    ok_high = ((tp + fp) >= MIN_PREDICTED_SUSCEPTIBLE) & (ppv >= PPV_TARGET)
    ok_low = (1 - sens) <= ESCALATION_BUDGET
    best_high = best_low = None
    if ok_high.any():
        i = int(np.flatnonzero(ok_high)[-1])                   # lowest qualifying cut-off
        best_high = (float(ts[i]), float(sens[i]), float(ppv[i]))
    if ok_low.any():
        i = int(np.argmax(np.where(ok_low, caught, -np.inf)))   # most resistant flagged; ties -> highest t
        best_low = (float(ts[i]), float(1 - sens[i]), float(caught[i]))
    out = {"narrowing_attained": best_high is not None,
           "T_narrow": np.nan, "narrowed_pct_of_susceptible": np.nan, "narrowing_ppv": np.nan,
           "T_escalate": np.nan, "flagged_pct_of_resistant": np.nan,
           "susceptible_flagged_pct": np.nan}
    if best_high is not None:
        out.update(T_narrow=best_high[0], narrowed_pct_of_susceptible=100 * best_high[1],
                   narrowing_ppv=best_high[2])
    if best_low is not None:
        out.update(T_escalate=best_low[0], flagged_pct_of_resistant=100 * best_low[2],
                   susceptible_flagged_pct=100 * best_low[1])
    return out


# ── One model ────────────────────────────────────────────────────────────────
def evaluate_model(model_dir: Path | str, tables_dir: Path | str, n_boot: int = N_BOOT) -> Dict[str, Any]:
    from .calibrate import Calibrator
    from .train import replay

    r = replay(model_dir, tables_dir)
    test = r["assign"] == "test"
    y = r["y"][test]
    p_raw = r["model"].predict_proba(r["X"][test])[:, 1]
    cal = Calibrator.load(Path(model_dir) / "calibrator.json")
    p_srv = cal.predict(p_raw) if cal is not None else p_raw
    meta = r["meta"]
    row: Dict[str, Any] = {
        "cell": meta["cell"], "variant": meta["variant"], "antibiotic": meta["antibiotic"],
        "culture": meta["culture"], "setting": meta["setting"],
        "n_test": int(len(y)), "n_susceptible": int(y.sum()), "n_resistant": int(len(y) - y.sum()),
        "n_patients_test": int(len(np.unique(r["groups"][test]))),
        "auroc": float(roc_auc_score(y, p_raw)),
        "auprc": float(average_precision_score(y, p_raw)),
    }
    row.update(clustered_ci(y, p_raw, r["groups"][test], n_boot))
    row.update({
        "spec_at_sens_0.90": specificity_at_sensitivity(y, p_raw, 0.90),
        "spec_at_sens_0.80": specificity_at_sensitivity(y, p_raw, 0.80),
        "brier_raw": brier(y, p_raw), "ece_raw": ece(y, p_raw),
    })
    row["cal_intercept_raw"], row["cal_slope_raw"] = calibration_intercept_slope(y, p_raw)
    row["calibration"] = cal.method if cal is not None else "raw"
    row.update({"brier_served": brier(y, p_srv), "ece_served": ece(y, p_srv)})
    row["cal_intercept_served"], row["cal_slope_served"] = calibration_intercept_slope(y, p_srv)
    row.update(operating_thresholds(y, p_srv) or {})
    return row
