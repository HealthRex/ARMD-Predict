"""Fully synthetic ARMD-like raw files for the four sites (no real patient data).

Writes <out>/<site>/<files> using each site's file names and column spellings
from configs/sites/<site>.yaml, including the site quirks the pipeline handles:
literal "Null" tokens (ECUH/UTSW), "18-24 years" vs "18-24" age bins,
"Female"/"Male" gender at MGB, extra MGB ordering modes, a 30-day lab window
next to the 14-day one, same-day (day-0) history rows, polymicrobial cultures,
implied-susceptibility rows and MGB's raw comorbidity layout.

Resistance depends on a latent patient propensity that also drives prior
resistance and antibiotic exposure, so the history features carry signal.

    python tests/synthetic.py --out /tmp/armd_synth --patients 600
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SITES = ["stanford", "ecuh", "utsw", "mgb"]
ABX = {  # raw spelling -> base resistance probability
    "Ceftriaxone": 0.12, "Ciprofloxacin": 0.18, "Nitrofurantoin": 0.10,
    "Trimethoprim/Sulfamethoxazole": 0.20, "Cefazolin": 0.15, "Meropenem": 0.03,
    "Pip/Tazobactam": 0.08, "Vancomycin": 0.10, "Gentamicin": 0.07, "Ampicillin": 0.45,
}
ORGANISMS = ["ESCHERICHIA COLI", "KLEBSIELLA PNEUMONIAE", "ESBL ESCHERICHIA COLI",
             "PROTEUS MIRABILIS", "STAPH AUREUS {MRSA}", "ENTEROCOCCUS FAECALIS"]
CULTURES = ["URINE CULTURE", "BLOOD CULTURE", "RESPIRATORY CULTURE"]
COMORBIDITIES = ["Diabetes", "Chronic kidney disease", "Heart failure", "Hypertension",
                 "Congestive heart failure", "Resistance to antimicrobial drugs"]
DRUGS = ["Cefazolin", "Ceftriaxone", "Vancomycin", "Ciprofloxacin", "Nitrofurantoin"]
AGE_BINS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85-89", "above 90"]


def _cfg(site: str) -> Dict:
    return yaml.safe_load((CONFIG_DIR / "sites" / f"{site}.yaml").read_text())


def _col(cfg: Dict, block: str, key: str, default: str) -> str:
    return cfg["column_mappings"].get(block, {}).get(key, default)


def make_site(site: str, out_dir: Path, n_patients: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cfg = _cfg(site)
    files, cm = cfg["file_mappings"], cfg["column_mappings"]
    d = out_dir / site
    d.mkdir(parents=True, exist_ok=True)
    null = "Null" if site in ("ecuh", "utsw") else np.nan
    ts_col = cm["cohort"]["timestamp"]

    orders: List[Dict] = []
    order_no = 0
    for i in range(n_patients):
        pid = int(5_000_000 + i) if site == "mgb" else f"SYN{site[:2].upper()}{i:06d}"
        risk = 3.0 if rng.random() < 0.25 else 0.3       # latent resistance propensity
        age = AGE_BINS[min(int(rng.integers(0, 9) + risk), 8)]
        sex = int(rng.random() < 0.55)
        for e in range(int(rng.integers(1, 4))):
            enc = f"E{site[:2]}{i:06d}{e}"
            start = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
            mode = rng.choice(["Inpatient", "Outpatient"], p=[0.6, 0.4])
            if site == "mgb" and rng.random() < 0.05:
                mode = "Emergency"
            if site != "mgb" and rng.random() < 0.02:
                mode = "Null"
            for k in range(int(rng.integers(1, 3))):
                order_no += 1
                orders.append({
                    "pid": pid, "enc": enc, "order": f"O{site[:2]}{order_no:08d}",
                    "time": start + pd.Timedelta(hours=int(rng.integers(0, 72)) + 24 * k),
                    "mode": mode, "culture": rng.choice(CULTURES, p=[0.7, 0.2, 0.1]),
                    "positive": int(rng.random() < 0.85), "risk": risk, "age": age, "sex": sex,
                })
    O = pd.DataFrame(orders)
    keys = {"anon_id": O["pid"], "pat_enc_csn_id_coded": O["enc"], "order_proc_id_coded": O["order"]}
    stamp = O["time"].dt.strftime("%Y-%m-%d %H:%M:%S") + ("+00:00" if "utc" in ts_col else "")

    # ── cohort: order x organism x antibiotic ──
    rows, implied = [], []
    for j, o in O.iterrows():
        base = {"anon_id": o["pid"], "pat_enc_csn_id_coded": o["enc"], "order_proc_id_coded": o["order"],
                ts_col: stamp[j], "ordering_mode": o["mode"], "culture_description": o["culture"],
                "was_positive": o["positive"]}
        if not o["positive"]:
            rows.append({**base, "organism": null, "antibiotic": null, "susceptibility": null})
            continue
        for org in rng.choice(ORGANISMS, size=int(rng.integers(1, 3)), replace=False):
            for abx, p in ABX.items():
                if rng.random() < 0.15:
                    continue
                pr = min(0.9, p * o["risk"])
                s = rng.choice(["Resistant", "Intermediate"], p=[0.8, 0.2]) if rng.random() < pr else "Susceptible"
                rows.append({**base, "organism": org, "antibiotic": abx, "susceptibility": s})
            if rng.random() < 0.5:
                implied.append({"anon_id": o["pid"], "pat_enc_csn_id_coded": o["enc"],
                                "order_proc_id_coded": o["order"], "organism": org,
                                "antibiotic": "Amoxicillin/Clavulanate", "susceptibility": null,
                                "implied_susceptibility": rng.choice(["Susceptible", "Resistant"], p=[0.8, 0.2])})
    pd.DataFrame(rows).to_csv(d / files["cohort"], index=False)
    if implied:
        pd.DataFrame(implied).to_csv(d / files["implied_susceptibility"], index=False)

    K = pd.DataFrame(keys)

    def history(block: str, n_mean: float, cols: Dict[str, object], day0_share=0.1) -> None:
        reps = rng.poisson(n_mean * O["risk"].to_numpy())
        idx = np.repeat(np.arange(len(O)), reps)
        h = K.iloc[idx].reset_index(drop=True)
        days = rng.integers(1, 800, len(idx)).astype(float)
        days[rng.random(len(idx)) < day0_share] = 0
        for name, gen in cols.items():
            h[name] = gen(len(idx)) if callable(gen) else gen
        h[_col(cfg, block, "time_to_culture", f"{block}_days_to_culture")] = days
        h.to_csv(d / files[block], index=False)

    pick = lambda vals: (lambda n: rng.choice(vals, n))  # noqa: E731
    history("prior_med", 1.5, {_col(cfg, "prior_med", "medication_name", "medication_name"): pick(DRUGS),
                               _col(cfg, "prior_med", "medication_category", "medication_category"):
                                   pick(["CEP1", "CEP3", "VAN", "FQ", "NIT"])})
    history("priorprocedures", 0.5, {_col(cfg, "priorprocedures", "procedure_name",
                                          _col(cfg, "priorprocedures", "procedure_description",
                                               "procedure_description")): pick(["foley", "cvc", "picc"])})
    history("prior_infecting_organism", 0.8, {"prior_organism": pick(["Escherichia", "Klebsiella", "Proteus"])},
            day0_share=0.0)
    history("microbial_resistance", 1.0, {"organism": pick(["ESCHERICHIA COLI", "KLEBSIELLA PNEUMONIAE"]),
                                          "antibiotic": pick(["Ceftriaxone", "Ciprofloxacin", "Cefazolin"])})
    history("antibiotic_class_exposure", 1.2, {"antibiotic_class": pick(["Cephalosporin", "Fluoroquinolone"]),
                                               "medication_name": pick(DRUGS),
                                               "medication_category": pick(["CEP3", "FQ"])})
    history("antibiotic_subtype_exposure", 1.2, {
        "antibiotic_subtype": pick(["Cephalosporin 3rd Gen", "Fluoroquinolone"]),
        "antibiotic_subtype_category": pick(["CEP3", "FQ"])})

    nh = K.copy()
    nh["nursing_home_visit_culture"] = np.where(rng.random(len(O)) < 0.1, rng.integers(0, 30, len(O)), -1)
    nh["nursing_home_visit_culture"] = nh["nursing_home_visit_culture"].astype(object).where(
        nh["nursing_home_visit_culture"] >= 0, null)
    nh.to_csv(d / files["nursing_home_visits"], index=False)

    demo = K.copy()
    demo["age"] = [a if site in ("ecuh", "utsw") or a == "above 90" else f"{a} years" for a in O["age"]]
    demo["gender"] = [("Male" if s else "Female") for s in O["sex"]] if site == "mgb" else O["sex"]
    demo.to_csv(d / files["demographics"], index=False)

    ward = K.copy()
    ward[_col(cfg, "ward_info", "timestamp", "order_time")] = stamp
    ip = (O["mode"] == "Inpatient").astype(int)
    ward["hosp_ward_IP"], ward["hosp_ward_OP"] = ip, 1 - ip
    ward["hosp_ward_ER"] = (rng.random(len(O)) < 0.2).astype(int)
    ward["hosp_ward_ICU"] = (ip & (rng.random(len(O)) < 0.2)).astype(int)
    ward.to_csv(d / files["ward_info"], index=False)

    adi = K.copy()
    adi[_col(cfg, "adi_scores", "timestamp", "order_time")] = stamp
    adi["adi_score"] = rng.integers(1, 101, len(O)).astype(object)
    adi["adi_state_rank"] = rng.integers(1, 11, len(O)).astype(object)
    adi.loc[rng.random(len(O)) < 0.1, ["adi_score", "adi_state_rank"]] = null
    adi.to_csv(d / files["adi_scores"], index=False)

    windows = [14, 30] if site in ("ecuh", "mgb") else [14]
    labs = pd.concat([K.assign(Period_Day=w) for w in windows], ignore_index=True)
    for lab in ("wbc", "cr"):
        v = np.round(rng.normal(9 if lab == "wbc" else 1.1, 2 if lab == "wbc" else 0.3, len(labs)), 2)
        for stat in ("median", "Q25", "Q75", "first", "last"):
            labs[f"{stat}_{lab}"] = v
    labs.to_csv(d / files["labs"], index=False)

    if files.get("vitals"):
        vit = K.copy()
        vit["median_heartrate"] = np.round(rng.normal(85, 12, len(O)), 1)
        vit["median_temp"] = np.round(rng.normal(37.1, 0.6, len(O)), 1)
        vit.to_csv(d / files["vitals"], index=False)

    reps = rng.poisson(2.0 + O["risk"].to_numpy())
    idx = np.repeat(np.arange(len(O)), reps)
    com = K.iloc[idx].reset_index(drop=True)
    comp = rng.choice(COMORBIDITIES, len(idx))
    if site == "mgb":   # raw MGB layout; `prepare-mgb` projects category -> comorbidity_component
        com["ICD10"] = "E11.9"
        com["category"] = comp
        com["order_time_jittered_utc_shifted"] = stamp.iloc[idx].values
        com.to_csv(d / "microbiology_cultures_comorbidity.csv", index=False)
    else:
        com["comorbidity_component"] = comp
        com["comorbidity_component_start_days_culture"] = rng.integers(0, 2000, len(idx))
        com["comorbidity_component_end_days_culture"] = rng.integers(-100, 500, len(idx))
        com.to_csv(d / files["comorbidity"], index=False)


def make_all(out_dir: Path | str, n_patients: int = 600, seed: int = 7) -> Path:
    out_dir = Path(out_dir)
    for k, site in enumerate(SITES):
        make_site(site, out_dir, n_patients, seed + k)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=600)
    a = ap.parse_args()
    print(make_all(a.out, a.patients))
