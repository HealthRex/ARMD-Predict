"""Step 4a: recalibrate each model on its validation partition.

Two candidate maps of P(susceptible) are fitted on the VALIDATION partition
(never on training rows, where the model is over-confident): isotonic
regression and 3-parameter beta calibration (Kull et al., 2017). As in the
paper, the candidate with the lower TEST-partition ECE is kept (beta wins
ties), and only if it is no worse than the raw output; otherwise the raw
probability is served. Because the selection reuses the test partition, the
reported calibrated ECE is optimistic (stated in the paper's Methods).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .evaluate import brier, ece
from .train import replay

_EPS = 1e-6


class Calibrator:
    """JSON-serialisable isotonic or beta map, P(susceptible) -> P(susceptible)."""

    def __init__(self, method: str, params: Dict[str, Any]):
        self.method, self.params = method, params

    @classmethod
    def fit_isotonic(cls, p, y) -> "Calibrator":
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return cls("isotonic", {"x_min": float(iso.X_min_), "x_max": float(iso.X_max_),
                                "x": iso.X_thresholds_.astype(float).tolist(),
                                "y": iso.y_thresholds_.astype(float).tolist()})

    @classmethod
    def fit_beta(cls, p, y) -> "Calibrator":
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(np.column_stack([np.log(p), -np.log(1.0 - p)]), np.asarray(y).astype(int))
        a, b = map(float, lr.coef_[0])
        return cls("beta", {"a": a, "b": b, "c": float(lr.intercept_[0])})

    @property
    def strictly_monotone(self) -> bool:
        return self.method != "beta" or (self.params["a"] >= 0 and self.params["b"] >= 0)

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            x, yv = np.asarray(self.params["x"]), np.asarray(self.params["y"])
            t = np.clip(p, self.params["x_min"], self.params["x_max"])
            return yv.repeat(t.shape) if len(yv) == 1 else interp1d(x, yv, kind="linear")(t)
        q = np.clip(p, _EPS, 1.0 - _EPS)
        return expit(self.params["a"] * np.log(q) - self.params["b"] * np.log(1.0 - q)
                     + self.params["c"])

    def to_json(self) -> str:
        return json.dumps({"method": self.method, "params": self.params})

    @classmethod
    def load(cls, path: Path | str) -> Optional["Calibrator"]:
        path = Path(path)
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(d["method"], d["params"])


def calibrate_model(model_dir: Path | str, tables_dir: Path | str) -> Dict[str, Any]:
    """Fit, select and write calibrator.json beside the model (or remove a stale one)."""
    model_dir = Path(model_dir)
    r = replay(model_dir, tables_dir)
    i_val, i_test = r["assign"] == "val", r["assign"] == "test"
    p_val = r["model"].predict_proba(r["X"][i_val])[:, 1]
    p_test = r["model"].predict_proba(r["X"][i_test])[:, 1]
    y_val, y_test = r["y"][i_val], r["y"][i_test]

    candidates = [("isotonic", Calibrator.fit_isotonic(p_val, y_val))]
    beta = Calibrator.fit_beta(p_val, y_val)
    if beta.strictly_monotone:
        candidates.append(("beta", beta))
    raw_ece = ece(y_test, p_test)
    scored = sorted(((ece(y_test, c.predict(p_test)), 0 if name == "beta" else 1, name, c)
                     for name, c in candidates), key=lambda t: (t[0], t[1]))
    cal_ece, _, method, best = scored[0]
    accepted = cal_ece <= raw_ece
    out = model_dir / "calibrator.json"
    if accepted:
        out.write_text(best.to_json())
    elif out.exists():
        out.unlink()
    return {"cell": r["meta"]["cell"], "variant": r["meta"]["variant"],
            "status": "accepted" if accepted else "rejected", "method": method if accepted else "",
            "n_val": int(i_val.sum()), "n_test": int(i_test.sum()),
            "raw_ece": raw_ece, "candidate_ece": cal_ece,
            "raw_brier": brier(y_test, p_test), "candidate_brier": brier(y_test, best.predict(p_test))}
