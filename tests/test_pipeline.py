"""Unit checks plus an end-to-end smoke test on synthetic data (about a minute).

    pytest -q tests/
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from armd_predict import features, tables, train  # noqa: E402
from armd_predict.__main__ import main  # noqa: E402
from armd_predict.reanchor import reanchor  # noqa: E402
import synthetic  # noqa: E402

CELLS = ["ceftriaxone_urine_inpatient", "ciprofloxacin_urine_outpatient"]


# ── Unit checks ──────────────────────────────────────────────────────────────
def test_reanchor_identity_monotone_formula():
    assert abs(reanchor(0.2, 0.1, 0.1) - 0.2) < 1e-12
    assert reanchor(0.2, 0.1, 0.3) > 0.2 > reanchor(0.2, 0.1, 0.05)
    lo = lambda p: math.log(p / (1 - p))  # noqa: E731
    z = lo(0.2) - lo(0.1) + lo(0.3)
    assert abs(reanchor(0.2, 0.1, 0.3) - 1 / (1 + math.exp(-z))) < 1e-12


def test_resistant_wins_collapse():
    df = pd.DataFrame({"site_id": "s", "order_id": [1, 1, 2], "antibiotic": "ceftriaxone",
                       "organism": ["E COLI", "K PNEUMONIAE", "E COLI"], "y": [1, 0, 1],
                       "patient_id": ["a", "a", "b"], "culture_time": pd.Timestamp("2020-01-01")})
    out = tables.collapse_to_order_level(df, ["site_id", "order_id", "antibiotic"])
    assert out.set_index("order_id")["y"].to_dict() == {1: 0, 2: 1}


def test_history_day_zero_and_never():
    block = pd.DataFrame({"order_id": [1, 1, 2], "days": [0, 40, 0], "drug": ["a", "b", "c"]})
    out, _ = features._history(block, ["order_id"], "days")
    out = out.set_index("order_id")
    assert out.loc[1, "count_prior_events"] == 1 and out.loc[1, "min_days_prior"] == 40
    assert out.loc[1, "drug"] == "b"                       # most recent qualifying row
    assert out.loc[2, "count_prior_events"] == 0           # only a same-day row: no event
    assert np.isnan(out.loc[2, "min_days_prior"])          # never -> NaN, not 0


def test_patient_grouped_split():
    groups = np.repeat([f"s::{i}" for i in range(1000)], 3)
    a, b = train.split_patients(groups), train.split_patients(groups)
    assert (a == b).all()
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(groups[a == x]) & set(groups[a == y])
    assert abs((a == "train").mean() - 0.6) < 0.01


# ── End to end ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("armd")
    raw = synthetic.make_all(base / "raw", n_patients=500)
    cfg = base / "configs"
    shutil.copytree(ROOT / "configs" / "sites", cfg / "sites")
    shutil.copy(ROOT / "configs" / "leakage_policy.yaml", cfg)
    spec = yaml.safe_load((ROOT / "configs" / "cells.yaml").read_text())
    spec["cells"] = [c for c in spec["cells"] if train.cell_key(c) in CELLS]
    spec.update(min_n=100, min_susceptible=10, min_resistant=10)   # small synthetic cohort
    (cfg / "cells.yaml").write_text(yaml.safe_dump(spec))
    c = ["--config-dir", str(cfg)]
    main(c + ["prepare-mgb", "--raw-dir", str(raw)])
    main(c + ["harmonize", "--raw-dir", str(raw), "--out", str(base / "cdm")])
    main(c + ["tables", "--raw-dir", str(raw), "--cdm-dir", str(base / "cdm"),
              "--out", str(base / "tables"), "--stratum", "urine/inpatient", "urine/outpatient"])
    main(c + ["train", "--tables", str(base / "tables"), "--out", str(base / "models")])
    main(c + ["calibrate", "--tables", str(base / "tables"), "--models", str(base / "models")])
    main(c + ["evaluate", "--tables", str(base / "tables"), "--models", str(base / "models"),
              "--out", str(base / "eval"), "--n-boot", "50"])
    main(c + ["baselines", "--tables", str(base / "tables"), "--models", str(base / "models"),
              "--cdm-dir", str(base / "cdm"), "--out", str(base / "eval"), "--n-boot", "50"])
    return base


def test_cdm(run):
    for site in synthetic.SITES:
        cdm = pd.read_parquet(run / "cdm" / site / "cohort_long.parquet")
        assert len(cdm) and set(cdm["y"].unique()) <= {0, 1}
        assert cdm["culture_time"].notna().all()
        assert set(cdm["ordering_mode"]) <= {"Inpatient", "Outpatient", "Unknown"}
        assert (cdm["is_implied"] == 1).any()


def test_tables_have_no_leaks(run):
    for site in synthetic.SITES:
        df = pd.read_parquet(run / "tables" / site / "urine" / "inpatient" / "3of4" / "model_table.parquet")
        assert {"patient_id", "culture_time", "y"} <= set(df.columns)
        bad = {"organism", "order_id", "encounter_id", "had_conflict", "susceptibility_raw"}
        assert not bad & set(df.columns)
        assert not [c for c in df.columns if "_shifted" in c or "jittered" in c]
        assert not [c for c in df.columns if "Congestive heart failure" in c or "Resistance to" in c]
        assert "prior_resistance_min_days" in df.columns
        assert (df["prior_resistance_min_days"].dropna() > 0).all()       # day-0 excluded
        assert df.groupby(["antibiotic", "culture_time", "patient_id"]).size().max() >= 1


def test_models(run):
    summary = pd.read_csv(run / "models" / "training_summary.csv")
    assert len(summary) == 4                                          # 2 cells x full/lite
    for _, r in summary.iterrows():
        meta = json.loads((run / "models" / r["variant"] / r["cell"] / "metadata.json").read_text())
        assert not {"patient_id", "culture_time", "site_id"} & set(meta["features_used"])
        assert meta["never_sentinel"]["columns"] and meta["monotone_constraints"]["decreasing"]
        assert r["auroc_test"] > 0.55                                 # synthetic signal is learnt
        rp = train.replay(run / "models" / r["variant"] / r["cell"], run / "tables")
        p = rp["model"].predict_proba(rp["X"][rp["assign"] == "test"])[:, 1]
        from sklearn.metrics import roc_auc_score
        assert abs(roc_auc_score(rp["y"][rp["assign"] == "test"], p) - meta["auroc_test"]) < 1e-12


def test_outputs(run):
    ev = pd.read_csv(run / "eval" / "evaluation.csv")
    assert len(ev) == 4 and (ev["auroc_ci_lo"] <= ev["auroc"]).all() and (ev["auroc"] <= ev["auroc_ci_hi"]).all()
    assert ev["ece_raw"].between(0, 1).all()
    bl = pd.read_csv(run / "eval" / "baselines.csv")
    assert len(bl) == 2 and (bl["antibiogram_auroc"] == 0.5).all()
    assert (pd.read_csv(run / "models" / "calibration.csv")["n_val"] > 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
