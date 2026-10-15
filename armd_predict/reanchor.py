"""Local-prevalence re-anchoring (prior-probability shift; Saerens et al., 2002).

    logit(p_local) = logit(p_model) - logit(pi_train) + logit(pi_local)

p_model is the model's calibrated P(resistant) = 1 - P(susceptible); pi_train
is the pooled training resistance rate of the cell (metadata.json
"resistance_rate"); pi_local is the user's local resistance rate. The map is
monotone in p_model, so ranking (AUROC) is unchanged, and it is the identity
when pi_local == pi_train. It assumes only the marginal prevalence differs
between sites (P(x | R) and P(x | S) are stable).

Dependency-free on purpose: standard library only.
"""
from __future__ import annotations

import math
from typing import Iterable, List

# Priors of exactly 0 or 1 would assert that resistance is impossible or
# certain and erase the patient-specific signal; both priors are bounded.
PRIOR_FLOOR = 0.005
_EPS = 1e-6


def _logit(p: float, eps: float = _EPS) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def reanchor(p_model: float, pi_train: float, pi_local: float) -> float:
    """Re-anchor one calibrated P(resistant) from the training to the local prevalence."""
    pi_train = min(max(pi_train, PRIOR_FLOOR), 1.0 - PRIOR_FLOOR)
    pi_local = min(max(pi_local, PRIOR_FLOOR), 1.0 - PRIOR_FLOOR)
    z = _logit(p_model) - _logit(pi_train) + _logit(pi_local)
    return 1.0 / (1.0 + math.exp(-z))


def reanchor_many(p_model: Iterable[float], pi_train: float, pi_local: float) -> List[float]:
    return [reanchor(float(p), pi_train, pi_local) for p in p_model]
