# ARMD-Predict: usage

This repository rebuilds the empiric-susceptibility models of the ARMD-Predict
paper (Table 1: 23 antibiotic x specimen x care-setting cells, each with a
full-feature and a short-form variant) from the four ARMD datasets, and
evaluates them.

```text
raw site files --harmonize--> CDM --tables--> model tables --train--> models
                                                     |                  |
                                                     +--calibrate / evaluate / baselines
```

Conventions used throughout:

- **`y = 1` means susceptible**; `y = 0` means resistant or intermediate. Every
  model outputs P(susceptible). P(resistant) = 1 - P(susceptible).
- **A missing days-since value means the event never happened.** `0` means it
  happened on the day of the culture. The two are never substituted for each
  other. Same-day (day-0) history records are excluded.
- **`patient_id` and `culture_time` are split keys, never features.** They stay
  in the model tables only so that the split can be grouped by patient.
- **Split:** random, patient-grouped, 60/20/20 (train / validation / test),
  seed 20260806. Not chronological: no temporal validation is implied.

## 1. Install

Python 3.11. The pinned versions are the ones used for the paper; XGBoost
reproduces the published models exactly only with the same version.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e ".[test]"
pytest -q tests/                         # synthetic end-to-end check (minutes)
```

## 2. Data

The models were trained on four sites. Obtain each dataset from its
repository under its own data use agreement:

| Site | Dataset | DOI |
| --- | --- | --- |
| Stanford Health Care | ARMD (Stanford) | [10.5061/dryad.jq2bvq8kp](https://doi.org/10.5061/dryad.jq2bvq8kp) |
| East Carolina University Health | ARMD-ECUH | [10.5061/dryad.7sqv9s55x](https://doi.org/10.5061/dryad.7sqv9s55x) |
| UT Southwestern | ARMD-UTSW | [10.5061/dryad.0rxwdbsd5](https://doi.org/10.5061/dryad.0rxwdbsd5) |
| Mass General Brigham | ARMD-MGB | [10.13026/2r5k-b955](https://doi.org/10.13026/2r5k-b955) |

Differences between the public releases and the files used for the paper:

- **MGB.** The paper used a 2026 MGB extraction with its own identifiers that
  follows the same `microbiology_cultures_*` layout as the other sites (plus
  laboratory and vital-sign files). `configs/sites/mgb.yaml` describes that
  layout. The public PhysioNet release uses a different layout (`*_deid_tj.csv`)
  and needs its own site configuration.
- **Stanford.** The prior-medication and comorbidity files were later re-pulls
  of the same tables; the prior-medication re-pull contains same-day rows,
  which the day-zero rule removes. The prior-infecting-organism file is the
  April-2025 version (`microbiology_culture_prior_infecting_organism.csv`).
- **ECUH and UTSW.** The implied-susceptibility files are optional; when
  present they may be truncated at 1,048,575 rows.

### Expected layout

One directory per site under a common `RAW` directory. File names are those in
`configs/sites/<site>.yaml` (`file_mappings`); edit that file if yours differ.

```text
RAW/
  stanford/  microbiology_cultures_cohort.csv
             microbiology_cultures_{demographics,ward_info,labs,vitals,comorbidity,prior_med,
                                    priorprocedures,nursing_home_visits,microbial_resistance,
                                    antibiotic_class_exposure,antibiotic_subtype_exposure,
                                    adi_scores,implied_susceptibility}.csv
             microbiology_culture_prior_infecting_organism.csv
  ecuh/      00_microbiology_cultures_cohort.csv, 01_..._adi_scores.csv, 02_..._antibiotic_class_exposure.csv,
             03_..._antibiotic_subtype_exposure.csv, 04_..._comorbidity.csv, 05_..._demographics.csv,
             06_..._labs.csv, 07_..._microbial_resistance.csv, 08_..._nursing_home_visits.csv,
             09_..._prior_infecting_organism.csv, 010_..._prior_med.csv, 011_..._prior_procedures.csv,
             012_..._ward_info.csv, 013_..._vitals.csv, [014_..._implied_susceptibility.csv]
  utsw/      microbiology_cultures_{cohort,demographics,ward_info,labs,comorbidity,prior_med,
                                    prior_procedures,nursing_home_visits,prior_infecting_organism,
                                    microbial_resistance,antibiotic_class_exposure,
                                    antibiotic_subtype_exposure,adi_scores}.csv
             [microbiology_implied_susceptibility.csv]          (no vitals file)
  mgb/       microbiology_cultures_{cohort,demographics,ward_info,labs,vitals,priorprocedures,
                                    adi_scores,comorbidity}.csv
             microbiology_cultures_{prior_med,nursing_home_visits,prior_infecting_organism,
                                    microbial_resistance,antibiotic_class_exposure,
                                    antibiotic_subtype_exposure}_augmented.csv
             microbiology_cultures_cohort_implied_susceptibility.csv
```

Each cohort file has one row per culture order x organism x antibiotic with the
identifiers `anon_id`, `pat_enc_csn_id_coded`, `order_proc_id_coded`, the order
time, `ordering_mode`, `culture_description`, `was_positive`, `organism`,
`antibiotic` and `susceptibility`. History files carry a whole-day offset to the
culture; the offset column per file is named in the site configuration.

The intermediate outputs (CDM, model tables) are row-level patient data. Keep
them inside the environment your data use agreement allows. The evaluation and
baseline outputs are aggregate tables.

## 3. Steps

All commands accept `--config-dir` (default: the repository's `configs/`; pass
it explicitly after a non-editable install) and `-v` for progress logging.
`WORK` is any output directory.

### 3.1 MGB comorbidity (once)

MGB's comorbidity file carries the AHRQ-CCSR strings in `category` and no
per-diagnosis timing. This writes `microbiology_cultures_comorbidity_harmonized.csv`
next to it with the other sites' column name:

```bash
python -m armd_predict prepare-mgb --raw-dir RAW
```

### 3.2 Harmonise into the common data model

```bash
python -m armd_predict harmonize --raw-dir RAW --out WORK/cdm            # all four sites
python -m armd_predict harmonize --raw-dir RAW --out WORK/cdm --site ecuh
```

Output: `WORK/cdm/<site>/cohort_long.parquet`, one row per culture order x
organism x antibiotic. Antibiotic and organism names are normalised (resistance
qualifiers such as ESBL or MRSA move to `organism_phenotype`); only positive
cultures with a named antibiotic are kept; site ordering modes are mapped to
Inpatient / Outpatient (missing markers to Unknown, which joins neither cohort);
**only the first culture order of each inpatient stay is kept** (by time);
S -> 1, R/I -> 0; implied-susceptibility rows are added for drugs not tested on
that order; duplicates are resolved with measured results preferred over implied
ones and, when measured results disagree, the resistant one kept (the row is
flagged `had_conflict`).

### 3.3 Build model tables

```bash
python -m armd_predict tables --raw-dir RAW --cdm-dir WORK/cdm --out WORK/tables
python -m armd_predict tables --raw-dir RAW --cdm-dir WORK/cdm --out WORK/tables \
       --site ecuh --stratum urine/inpatient
```

Default strata: urine/inpatient, urine/outpatient, blood/inpatient,
resp/inpatient, resp/outpatient. Output:
`WORK/tables/<site>/<culture>/<mode>/3of4/model_table.parquet` (one row per
culture order x antibiotic) and an aggregate `build_manifest.json`.

What happens, in order:

1. rows with conflicting labels are dropped;
2. **resistant wins:** organisms of a polymicrobial order are collapsed to one
   row per order x antibiotic, resistant if any organism is resistant or
   intermediate;
3. the feature blocks provided by at least three sites (`has_<block>` in the
   site configurations) are joined. Demographics, ward, ADI, laboratory values
   (14-day window) and vital signs are one row per order. Comorbidities become
   multi-hot indicators plus a count of distinct components. Nursing-home
   history becomes a presence flag. Every other history block (prior
   antibiotics, class and subtype exposure, procedures, prior organisms, prior
   resistance) becomes a count, days since the most recent event, an any-prior
   flag and any-within-30/90/365-day flags, **after removing same-day (day-0)
   records**, with descriptive values taken from the most recent qualifying
   event. `prior_resistance_same_abx_within_365d` flags a prior resistant
   result for the same antibiotic;
4. site values are harmonised (gender coding, age bins, literal "Null");
5. the leakage guard drops identifiers, culture results, label provenance,
   outcome-derived diagnoses (ICD-10 Z16, Z22) and absolute or shifted
   timestamps, and stops if a label-like column survives.

Antibiotic inclusion thresholds and the cross-site column intersection are
applied at training time.

### 3.4 Train

```bash
python -m armd_predict train --tables WORK/tables --out WORK/models                     # 23 cells x full/lite
python -m armd_predict train --tables WORK/tables --out WORK/models \
       --cell ceftriaxone_urine_inpatient --variant full
```

Per cell (`configs/cells.yaml`): the four sites' rows are pooled in the order
Stanford, ECUH, UTSW, MGB; the features are the columns every contributing
site has (no site indicator); the short form keeps the demographics, ward,
prior-resistance, prior-infection and prior-medication blocks. A cell needs at
least 500 orders with 25 susceptible and 25 resistant, overall and in each
partition. The encoder (numeric vs categorical, one-hot vocabulary with at most
50 levels seen at least 30 times) is fitted on the training partition only. On
the constrained days-since columns, "never" is encoded as 99999 days so that
the monotone constraints cover it. XGBoost: 500 trees, depth 6, learning rate
0.05, row and column subsampling 0.8, min_child_weight 1, L1 0.1, L2 1.0, seed
42, no class weighting, early stopping after 50 rounds on validation AUROC,
monotone constraints on the prior-resistance, prior-infection and aggregate
prior-antibiotic columns (decreasing for counts and flags, increasing for days
since).

Output: `WORK/models/<variant>/<cell>/{model.json, metadata.json}` and
`WORK/models/training_summary.csv` (test AUROC per model). See
`models/README.md` for loading a model.

### 3.5 Calibrate

```bash
python -m armd_predict calibrate --tables WORK/tables --models WORK/models
```

Isotonic and beta calibration are fitted on the validation partition. As in
the paper, the map with the lower test ECE is kept only if it does not worsen
the test ECE of the raw output; otherwise the raw probability is used.
Because this selection uses the test partition, the calibrated ECE it reports
is optimistic. Output: `calibrator.json` beside each accepted model and
`WORK/models/calibration.csv`.

### 3.6 Evaluate

```bash
python -m armd_predict evaluate --tables WORK/tables --models WORK/models --out WORK/eval
```

The recorded split is replayed and checked against the counts saved at
training time before anything is scored. Output `WORK/eval/evaluation.csv`, one
row per model, on the test partition: AUROC and AUPRC (susceptible = positive
class) with 95% CIs from 1,000 bootstrap replicates that resample patients
(seed 20260806); specificity at sensitivity 0.90 and 0.80; Brier score, ECE (10
equal-width bins) and calibration intercept and slope for the raw and the served
probability; and two exploratory operating thresholds on the served
probability: narrowing (lowest cut-off with PPV >= 95% for "susceptible" and at
least 20 predicted-susceptible orders; yield = share of susceptible orders
above it) and escalation (the cut-off flagging the most resistant orders while
flagging at most 20% of susceptible orders). The thresholds and their yields
come from the same test rows and need independent validation.

### 3.7 Simple-rule baselines

```bash
python -m armd_predict baselines --tables WORK/tables --models WORK/models \
       --cdm-dir WORK/cdm --out WORK/eval
```

For each full-feature model, on its test orders: the antibiogram rule (the
cell's training-partition susceptibility rate) and the last-result rule (the
patient's most recent same-antibiotic result from an earlier calendar day,
falling back to the antibiogram rate), AUROC differences with paired
patient-level bootstrap CIs, and a comparison at the rule's alert rate.
Output `WORK/eval/baselines.csv`; counts of 1-19 are written `<20`.

### 3.8 Local-prevalence re-anchoring

```bash
python -m armd_predict reanchor --p-resistant 0.12 --train-prevalence 0.18 --local-prevalence 0.30
```

`armd_predict/reanchor.py` is standard-library only:
`logit(p_local) = logit(p_model) - logit(pi_train) + logit(pi_local)`, with
p_model the calibrated P(resistant) and pi_train the cell's pooled resistance
rate (`metadata.json` `resistance_rate`). It assumes that only the marginal
resistance rate differs between hospitals; it does not change the ranking.

## 4. Resources and reproducibility

- On a shared 16-core machine the ECUH CDM took about 8 minutes and the ECUH
  urine/inpatient model table about 13 minutes, using at least 7 GB of memory.
  Stanford takes longer and needs more memory (its comorbidity file is about
  20 GB). `tables` re-reads every block file for each stratum.
- Training one cell takes one to a few minutes, most of it reading the four
  model tables.
- The pipeline is deterministic. From the raw ECUH files it rebuilt the paper's
  ECUH CDM and urine/inpatient model table exactly, and retraining from the
  paper's model tables reproduced the published models exactly (identical test
  predictions; e.g. ceftriaxone/urine/inpatient AUROC 0.772936, n = 27,960).
  The result did not depend on the number of XGBoost threads (1, 4 or 16); it
  may change with another XGBoost version.

## 5. Not included

The web application, manuscript and figure code, SHAP analyses, the
Stanford-only and leave-one-site-out models, GroupDRO and other exploratory
analyses, and the organism-identification and resistance-phenotype models
(available from the authors on request).
