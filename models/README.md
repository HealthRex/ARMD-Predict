# Trained models

No trained models are distributed in this repository. Whether and how the
published models are released is decided separately by the authors.

## Layout

`python -m armd_predict train` writes, and every later step reads:

```text
models/
  full/<cell>/            e.g. full/ceftriaxone_urine_inpatient/
    model.json            XGBoost model (native JSON; includes best_iteration)
    metadata.json         feature list, encoder state, never-sentinel columns,
                          monotone constraints, split seed and partition counts,
                          pooled resistance rate (pi_train), test AUROC
    calibrator.json       only if a calibration map was accepted (see calibrate)
  lite/<cell>/            same files for the short-form variant
  training_summary.csv
  calibration.csv
```

`<cell>` is `<antibiotic>_<culture>_<setting>`, as listed in `configs/cells.yaml`.
`metadata.json` contains no patient-level values: only column names, category
vocabularies, partition counts and cohort-level rates.

## Loading a model and scoring rows

Rows must have the model-table columns written by `python -m armd_predict tables`.

```python
from armd_predict.train import load_model, prepare_features
from armd_predict.calibrate import Calibrator
from armd_predict.reanchor import reanchor

model, meta = load_model("models/full/ceftriaxone_urine_inpatient")
X = prepare_features(rows, meta)            # same encoding + 99999 never-sentinel as training
p_susceptible = model.predict_proba(X)[:, 1]
cal = Calibrator.load("models/full/ceftriaxone_urine_inpatient/calibrator.json")
if cal is not None:
    p_susceptible = cal.predict(p_susceptible)
p_resistant = 1 - p_susceptible

# Optional: re-anchor to a local resistance rate (e.g. 0.25 from a local antibiogram).
p_local = [reanchor(p, meta["resistance_rate"], 0.25) for p in p_resistant]
```

`y = 1` means susceptible: the model outputs P(susceptible). A missing value in
a days-since column means "never happened"; `prepare_features` turns it into
99999 days on the constrained columns listed in `metadata.json`, exactly as in
training. Do not substitute 0, which means "happened today".

## Regenerating the published models

Training is deterministic. Running `train` on the paper's model tables with the
pinned package versions regenerates the published models: in our check the
retrained ceftriaxone/urine/inpatient and meropenem/blood/inpatient models gave
test predictions identical to the published ones (maximum absolute difference
0), with 1, 4 or 16 threads. A different XGBoost version may change the trees
at floating-point level.
