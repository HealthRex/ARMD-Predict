<p align="center">
  <img src="assets/armd-predict-logo.svg" alt="ARMD-Predict logo: a teal culture plate with three antibiotic discs, circled by a purple orbit" width="200">
</p>

<h1 align="center">ARMD-Predict: Multi-Site AI to Predict Antibiotic Resistance before Culture Results</h1>

<p align="center">
  <a href="https://healthrex.github.io/ARMD-Predict/"><img src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=googlechrome&logoColor=white&color=082A5E" alt="Website"></a>
  <a href="https://armd-predict.stanford.edu"><img src="https://img.shields.io/badge/Web%20app-000000?style=for-the-badge&logo=googlechrome&logoColor=white&color=0E7F81" alt="Web app"></a>
  <a href="#data"><img src="https://img.shields.io/badge/Data-000000?style=for-the-badge&logo=openaccess&logoColor=white&color=7457E8" alt="Data"></a>
  <!-- TODO: confirm the license before release -->
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-000000?style=for-the-badge&color=4B5563" alt="License: Apache-2.0"></a>
</p>

---

## Overview

Clinicians choose empiric antibiotics before culture results are available, guided by hospital antibiograms that give every patient the same estimate. ARMD-Predict estimates, at the time a culture is ordered and from information recorded up to the order, the probability that all tested organisms recovered from that culture will be susceptible to each candidate antibiotic: a patient-specific antibiogram. The models were trained on 7.4 million culture–antibiotic results from 415,952 culture orders in 241,227 patients at four U.S. health systems: Stanford Health Care, East Carolina University Health (ECUH), UT Southwestern Medical Center (UTSW) and Mass General Brigham (MGB). Monotonic constraints prevent prior resistance from lowering predicted resistance, and a prior-probability correction re-anchors predictions to a hospital's own antibiogram without retraining. Separately trained and evaluated models predict the likely organism and organism-specific resistance phenotypes, including proxies for categories in the 2024 IDSA guidance.

<p align="center">
  <a href="assets/framework.png"><img src="assets/framework.png" alt="ARMD-Predict framework diagram. Inputs: patient features recorded up to the culture order (demographics, prior resistance and infection, prior antibiotic exposure, laboratory values, Area Deprivation Index and care setting) from four sites: Stanford, ECUH, UTSW and MGB. Two parallel pipelines. Part 1, empiric susceptibility per antibiotic: a pooled four-site XGBoost with no site indicator, with urine, blood and respiratory models, giving a calibrated probability of susceptibility per antibiotic. Part 2, organism and resistance phenotype, IDSA-aligned: stage 1 Gram stain, stage 2 organism identification (top 20 plus Other), stage 3 thirty IDSA-aligned phenotype models (ESBL, AmpC-risk, CRE, P. aeruginosa, MRSA, VRE); each stage was trained and evaluated separately, and phenotype models were evaluated given the laboratory-reported organism. Unified output: research outputs at culture ordering, the probability of susceptibility to each candidate antibiotic with exploratory high and low thresholds, and ranked organism labels with organism-specific phenotype probabilities." width="760"></a>
</p>

*The ARMD-Predict framework. A shared set of features recorded up to the culture order feeds two research pipelines: empiric susceptibility per antibiotic (Part 1) and organism and resistance-phenotype prediction (Part 2). Each stage was trained and evaluated separately.*

## Main Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/main_results_dark.png">
    <img src="assets/main_results.png" alt="Held-out AUROC with 95% confidence intervals for the 23 pooled empiric-susceptibility models, in four panels. Urine inpatient (median 0.773): meropenem 0.831, vancomycin 0.825, cefepime 0.807, ceftriaxone 0.773, piperacillin–tazobactam 0.699, gentamicin 0.679, ampicillin 0.629. Urine outpatient (median 0.708): levofloxacin 0.747, ciprofloxacin 0.743, cefpodoxime 0.733, cefazolin 0.708, TMP-SMX 0.673, nitrofurantoin 0.662, amoxicillin–clavulanate 0.638. Blood inpatient (median 0.703): meropenem 0.835, vancomycin 0.825, cefepime 0.704, ceftriaxone 0.703, piperacillin–tazobactam 0.658, gentamicin 0.656, ampicillin 0.637. Respiratory: ceftriaxone inpatient 0.825, levofloxacin outpatient 0.822." width="880">
  </picture>
</p>

*Held-out AUROC of the 23 pooled, full-feature empiric-susceptibility models (paper Table 1), on a random patient-grouped test split. Lines show 95% CIs from 1,000 bootstrap replicates that resampled patients. \* Blood meropenem: most resistant labels came from orders with Gram-positive organisms, mainly staphylococci whose meropenem result was derived from oxacillin testing; a model retrained on Gram-negative orders only had AUROC 0.757 (95% CI, 0.691–0.824). † Cefpodoxime labels were inferred from laboratory implied-susceptibility rules at all four sites.*

- **Discrimination.** Median held-out AUROC was 0.77 for inpatient urine cultures, 0.70 for inpatient blood cultures and 0.71 for outpatient urine cultures; each of the two respiratory models reached 0.82.
- **Against a simple rule.** The models ranked orders better than repeating the patient's last result for the same antibiotic (median AUROC, 0.71 versus 0.60). That rule had no earlier result for 64% of orders. At the rule's own alert rate, the two flagged similar proportions of resistant orders (medians, 23.5% and 23.0%).
- **Calibration and local re-anchoring.** For the ceftriaxone inpatient urine model, raw output was close to calibrated overall (expected calibration error, 0.004) and at each site (0.004–0.013). In a controlled prevalence sweep, re-anchoring to a 40% local resistance rate lowered median expected calibration error from 0.20 to 0.06 without changing discrimination.
- **Subgroups.** The median within-model AUROC range was 0.02 by sex, 0.07 by age band and 0.05 by Area Deprivation Index quintile.

See the [project website](https://healthrex.github.io/ARMD-Predict/) for the calibration, operating-point and simple-rule results.

## Data

Each health system has released a de-identified dataset, the Antibiotic Resistance Microbiology Dataset (ARMD), available under the access terms of its repository.

| Health system | Dataset | Repository | DOI |
|---|---|---|---|
| Stanford Health Care | ARMD | Dryad | [10.5061/dryad.jq2bvq8kp](https://doi.org/10.5061/dryad.jq2bvq8kp) |
| East Carolina University Health | ARMD-ECUH | Dryad | [10.5061/dryad.7sqv9s55x](https://doi.org/10.5061/dryad.7sqv9s55x) |
| UT Southwestern Medical Center | ARMD-UTSW | Dryad | [10.5061/dryad.0rxwdbsd5](https://doi.org/10.5061/dryad.0rxwdbsd5) |
| Mass General Brigham | ARMD-MGB | PhysioNet (credentialed access) | [10.13026/2r5k-b955](https://doi.org/10.13026/2r5k-b955) |

The paper's Stanford, ECUH and UTSW analyses used these public releases, with later versions of some files that are available from the corresponding author on request. The paper's MGB analysis used a 2026 extraction from the same health system that is **not** part of the public ARMD-MGB release; access to that extraction requires approval from Mass General Brigham under a data-use agreement.

## Getting started

Installation, the command-line interface and examples are documented in [USAGE.md](USAGE.md).

```bash
git clone https://github.com/HealthRex/ARMD-Predict.git
cd ARMD-Predict
pip install -e .
python -m armd_predict --help
```

The ARMD datasets are obtained separately, from the repositories listed under [Data](#data), under each repository's access terms.

## Intended use and limitations

ARMD-Predict is a **research prototype**. It is intended for research and for independent technical evaluation, and it is **not for clinical decision-making**.

- The models were developed and evaluated retrospectively. They have not been prospectively validated, integrated into an electronic health record or shown to change prescribing or patient outcomes.
- The pooled models used a random patient-grouped split from the same period, so the results do not estimate performance in future years. In an earlier analysis with a simpler feature set, chronological splitting reduced AUROC by approximately 0.04.
- All four health systems are U.S. academic medical centers. Performance in community hospitals and in other countries is unknown.
- Reported calibration across the pooled models is optimistic: the choice between a calibration transform and the raw output was made on test-set calibration error, which was then reported. The exploratory high and low operating points were derived and evaluated on the same test orders.
- The organism-specific phenotype models were evaluated given the laboratory-reported organism, not as an end-to-end pipeline from predicted organism to phenotype.
- Intermediate results were grouped with resistant. Race and ethnicity were unavailable, and sex labels were inferred at three sites.
- The local-prevalence adjustment assumes that only the overall resistance rate differs between hospitals; its retrospective evaluation used each held-out site's observed prevalence.

## Citation

If you use ARMD-Predict, please cite:

```bibtex
@unpublished{nateghiharedasht2026armdpredict,
  title  = {{ARMD-Predict}: Multi-Site AI to Predict Antibiotic Resistance before Culture Results},
  author = {Nateghi Haredasht, Fateme and Marshall, Nicholas P. and Chen, Wenyuan and Cooper, Lauren N. and Wei, Ziming and Konsker, Harrison and Chen, Jacob S. and Klompas, Michael and McKenna, Cara and Medford, Richard J. and Kanjilal, Sanjat and Schoeffler, Austin J. and Ma, Stephen P. and Banaei, Niaz and Asch, Steven M. and Chang, Amy and Deresinski, Stanley C. and Goldstein, Mary K. and Tibshirani, Robert and Chen, Jonathan H.},
  year   = {2026},
  note   = {Manuscript submitted for publication},
  url    = {https://healthrex.github.io/ARMD-Predict/}
}
```

Fateme Nateghi Haredasht and Nicholas P. Marshall contributed equally. A machine-readable citation is in [CITATION.cff](CITATION.cff).

## Contact

Questions about the paper or the code: Fateme Nateghi Haredasht ([fnateghi@stanford.edu](mailto:fnateghi@stanford.edu)), or open an [issue](https://github.com/HealthRex/ARMD-Predict/issues).

## Acknowledgments

We thank the data-engineering and informatics teams at Stanford Health Care, ECUH, UTSW and MGB for assembling and harmonizing the de-identified EHR extracts, and the antimicrobial stewardship and infectious diseases teams at each site for guidance on prescribing logic and resistance-phenotype definitions.

<!-- TODO: add funding sources and grant numbers -->
*Funding: to be added.*

## License

See [LICENSE](LICENSE).
