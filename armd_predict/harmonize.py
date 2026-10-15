"""Step 1: harmonise one site's raw cohort file into the common data model (CDM).

Output: <cdm-dir>/<site>/cohort_long.parquet, one row per
culture order x organism x antibiotic, positive cultures only, with
y = 1 (susceptible) or 0 (resistant or intermediate).

Steps: normalise antibiotic and organism names; keep positive cultures with a
named antibiotic; resolve the culture timestamp; map the site's ordering modes
to Inpatient / Outpatient / Unknown; keep the first culture per inpatient stay;
binarise the label; add implied-susceptibility rows; de-duplicate, preferring
measured over implied results and resistant over susceptible when measured
results disagree.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SiteConfig

log = logging.getLogger(__name__)

# ── Antibiotic names ─────────────────────────────────────────────────────────
ANTIBIOTIC_SYNONYMS = {
    "trimeth_sulfamethoxazole": "trimethoprim_sulfamethoxazole",
    "trimeth_sulf": "trimethoprim_sulfamethoxazole",
    "tmp_smx": "trimethoprim_sulfamethoxazole",
    "trimethoprim_sulfa": "trimethoprim_sulfamethoxazole",
    "amoxicillin_clavulanate": "amoxicillin_clavulanic_acid",
    "amox_clavulanate": "amoxicillin_clavulanic_acid",
    "augmentin": "amoxicillin_clavulanic_acid",
    "amoxicillin_k_clavulanate_aug": "amoxicillin_clavulanic_acid",
    "pip_tazobactam": "piperacillin_tazobactam",
    "pip_tazo": "piperacillin_tazobactam",
    "piperacillin_tazo": "piperacillin_tazobactam",
    "amp_sulbactam": "ampicillin_sulbactam",
    "ticarcil_clavulanate": "ticarcillin_clavulanate",
    "ticarcillin_clavulanic_acid": "ticarcillin_clavulanate",
    "ticacarcillin_clavulanic_acid": "ticarcillin_clavulanate",
    "ticarillin_clavulanic_acid": "ticarcillin_clavulanate",
    "synercid": "quinupristin_dalfopristin",
    "imipenem_ebactam": "imipenem_relebactam",
    "polymixin_e": "colistin",
    "fluconizole": "fluconazole",
    "bedaquilline": "bedaquiline",
    "caspofungin_acetate": "caspofungin",
    "5_fluorocytosine": "flucytosine",
    "gent_screen": "gentamicin_screen",
}
# Sites write the literal "Null" as well as true NaN; a row with no drug has no label.
ANTIBIOTIC_NULL_TOKENS = {"", "null", "none", "nan", "na", "n_a", "unknown",
                          "not_tested", "not_done", "no_growth"}


def normalize_antibiotic(x) -> str:
    """'Pip/Tazobactam' -> 'piperacillin_tazobactam'; 'Null' -> '' (rejected later)."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return ""
    x = re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower())
    x = re.sub(r"_+", "_", x).strip("_")
    x = re.sub(r"_rm$", "", x)  # UTSW "_RM" (repeat method): same drug
    if x in ANTIBIOTIC_NULL_TOKENS:
        return ""
    return ANTIBIOTIC_SYNONYMS.get(x, x)


# ── Organism names ───────────────────────────────────────────────────────────
# Resistance qualifiers baked into organism strings ("ESBL ESCHERICHIA COLI",
# "STAPH AUREUS {MRSA}") are moved to a separate phenotype column.
ORGANISM_PHENOTYPE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bMETHICILLIN[\s\-]*RESISTANT\b", "MRSA"),
    (r"\bMRSA\b", "MRSA"),
    (r"\bMETHICILLIN[\s\-]*(?:SENSITIVE|SUSCEPTIBLE)\b", "MSSA"),
    (r"\bMSSA\b", "MSSA"),
    (r"\bESBL(?:\s*PRODUC(?:ER|ING))?\b", "ESBL"),
    (r"\bCARBAPENEMASE[\s\-]*PRODUC(?:ER|ING)\b", "CARBAPENEM_RESISTANT"),
    (r"\bCARBAPENEM[\s\-]*RESISTANT\b", "CARBAPENEM_RESISTANT"),
    (r"\bVANCO(?:MYCIN)?[\s\-]*RESISTANT\b", "VANCOMYCIN_RESISTANT"),
    (r"\bVRE\b", "VANCOMYCIN_RESISTANT"),
]
ORGANISM_GROWTH_DESCRIPTORS = [
    r"\bCOLONY\s+VARIANT[^)]*", r"\bSMALL\s+COLONY\s+VARIANT\b",
    r"\bNON[\s\-]*MUCOID(?:\s+CF)?\b", r"\bNONMUCOID\b", r"\bMUCOID\b",
    r"\b(?:RARE|FEW|LIGHT|MODERATE|HEAVY|SCANT)\s+GROWTH\b", r"\bBY\s+DNA\s+PROBE\b",
    r"\bISOLATED\b", r"\bPROBABLE\b", r"\bPRESUMPTIVE\b",
]
ORGANISM_SYNONYMS = {
    "ENTEROBACTER AEROGENES": "KLEBSIELLA AEROGENES",
    "STAPH AUREUS": "STAPHYLOCOCCUS AUREUS",
    "STAPH EPIDERMIDIS": "STAPHYLOCOCCUS EPIDERMIDIS",
    "COAG NEGATIVE STAPHYLOCOCCUS": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "COAG NEG STAPHYLOCOCCUS": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "STAPHYLOCOCCUS COAGULASE NEGATIVE": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "STAPHYLOCOCCUS SPECIES COAGULASE NEGATIVE": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "COAGULASE NEGATIVE STAPHYLOCOCCUS SPECIES": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "COAGULASE NEGATIVE STAPHYLOCOCCUS OTHER": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
    "ACINETOBACTER BAUMANNI": "ACINETOBACTER BAUMANNII",
    "STAPHYLOCOCCUS LUGDENENSIS": "STAPHYLOCOCCUS LUGDUNENSIS",
}
ORGANISM_NULL_TOKENS = {"", "NULL", "NAN", "NONE", "NA", "UNKNOWN", "NO GROWTH"}


def normalize_organism(x) -> Tuple[str, str]:
    """Return (organism, phenotype), e.g. 'STAPH AUREUS {MRSA}' -> ('STAPHYLOCOCCUS AUREUS', 'MRSA')."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return "", ""
    s = str(x).strip().upper()
    if s in ORGANISM_NULL_TOKENS:
        return "", ""
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = re.sub(r"^Z{2,}\s*", "", s)  # Stanford LIS sort prefix
    phenotypes: List[str] = []
    for pattern, token in ORGANISM_PHENOTYPE_PATTERNS:
        if re.search(pattern, s):
            if token not in phenotypes:
                phenotypes.append(token)
            s = re.sub(pattern, " ", s)
    for pattern in ORGANISM_GROWTH_DESCRIPTORS:
        s = re.sub(pattern, " ", s)
    s = re.sub(r"\bCOMPLEX\b", " ", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ORGANISM_NULL_TOKENS:
        s = ""
    return ORGANISM_SYNONYMS.get(s, s), "|".join(phenotypes)


# ── Ordering mode ────────────────────────────────────────────────────────────
# A site's own ordering modes are mapped to Inpatient / Outpatient through
# capabilities.ordering_mode_map. Missing markers go to "Unknown" (excluded from
# both cohorts); any other unlisted value falls back to the substring rule
# ("inpatient" -> Inpatient, else Outpatient).
INPATIENT, OUTPATIENT, UNKNOWN = "Inpatient", "Outpatient", "Unknown"
DEFAULT_ORDERING_MODE_MAP = {
    "inpatient": INPATIENT, "psych inpatient": INPATIENT, "surgery admit": INPATIENT,
    "outpatient": OUTPATIENT, "emergency": OUTPATIENT, "ed observation": OUTPATIENT,
    "observation": OUTPATIENT, "urgent care": OUTPATIENT, "day surgery": OUTPATIENT,
}
ORDERING_MODE_MISSING_TOKENS = {"", "null", "nan", "none", "na", "n/a", "unknown"}

CDM_COLUMNS_BASE = [
    "site_id", "patient_id", "encounter_id", "order_id", "culture_time",
    "ordering_mode", "ordering_mode_raw", "culture_type", "organism", "organism_raw",
    "organism_phenotype", "antibiotic", "susceptibility_raw", "y", "label_source",
    "is_implied",
]
CDM_COLUMNS = CDM_COLUMNS_BASE + [
    "n_sources", "labels_set", "had_conflict", "label_source_resolved",
    "label_sources", "n_measured_sources", "n_implied_sources",
]


def encode_susceptibility(value) -> Optional[int]:
    """S -> 1; R or I -> 0; anything else -> None (row dropped)."""
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in ["s", "susceptible", "sensitive", "1"]:
        return 1
    if v in ["r", "resistant", "i", "intermediate", "0"]:
        return 0
    return None


def _map_ordering_mode(raw_values: pd.Series, caps: Dict) -> Tuple[pd.Series, pd.Series]:
    raw = raw_values.fillna("").astype(str).str.strip()
    custom = caps.get("ordering_mode_map")
    mode_map = ({str(k).strip().lower(): str(v) for k, v in custom.items()}
                if custom else dict(DEFAULT_ORDERING_MODE_MAP))
    key = raw.str.lower()
    mapped = key.map(mode_map)
    missing = mapped.isna() & key.isin(ORDERING_MODE_MISSING_TOKENS)
    if missing.any():
        mapped = mapped.mask(missing, UNKNOWN)
    unmapped = mapped.isna()
    if unmapped.any():
        log.warning("ordering mode: %d rows (%d distinct values) not in the site map; "
                    "substring rule applied", int(unmapped.sum()),
                    raw[unmapped].nunique())
        fallback = pd.Series(np.where(key.str.contains("inpatient", na=False),
                                      INPATIENT, OUTPATIENT), index=raw.index)
        mapped = mapped.where(~unmapped, fallback)
    return mapped.astype(str), raw


def _keep_first_culture_per_stay(df: pd.DataFrame, cm: Dict) -> pd.DataFrame:
    """First culture order (by time, then order id) per inpatient stay; outpatient rows kept."""
    ip = df["_ordering_mode"] == INPATIENT
    df_ip, df_op = df[ip].copy(), df[~ip].copy()
    if len(df_ip):
        p = cm.get("patient_id", "anon_id")
        e = cm.get("encounter_id", "pat_enc_csn_id_coded")
        o = cm.get("order_id", "order_proc_id_coded")
        # Sorted by time, not file order: the extracts are not in time order.
        first = (df_ip[[p, e, o, "_culture_time"]]
                 .sort_values([p, e, "_culture_time", o], na_position="last", kind="mergesort")
                 .groupby([p, e], as_index=False).head(1)
                 .rename(columns={o: "_first_order"}))
        df_ip = df_ip.merge(first[[p, e, "_first_order"]], on=[p, e], how="left")
        df_ip = df_ip[df_ip[o] == df_ip["_first_order"]].copy().drop(columns=["_first_order"])
    return pd.concat([df_ip, df_op], ignore_index=True)


def _add_implied(df: pd.DataFrame, cfg: SiteConfig) -> Tuple[pd.DataFrame, int]:
    """Add implied-susceptibility rows for antibiotics not tested on that order."""
    path = cfg.file("implied_susceptibility")
    im = cfg.column_mappings.get("implied_susceptibility", {})
    imp = pd.read_csv(path)
    p_col, abx_col = im.get("patient_id", "anon_id"), im.get("antibiotic", "antibiotic")
    e_col, o_col = im.get("encounter_id", "pat_enc_csn_id_coded"), im.get("order_id", "order_proc_id_coded")
    org_col = im.get("organism", "organism")
    val_col = im.get("implied_susceptibility", "implied_susceptibility")
    if abx_col in imp.columns:
        imp[abx_col] = imp[abx_col].apply(normalize_antibiotic)
        if int((imp[abx_col] == "").sum()):
            imp = imp[imp[abx_col] != ""].copy()
    # Same organism normaliser as the cohort, or the organism join below misses.
    if org_col in imp.columns:
        norm = imp[org_col].map(normalize_organism)
        imp["_organism_raw"] = imp[org_col]
        imp[org_col] = [n for n, _ in norm]
        imp["_organism_phenotype"] = [p for _, p in norm]
    else:
        imp["_organism_raw"], imp["_organism_phenotype"] = "", ""

    keys = ["patient_id", "encounter_id", "order_id"]
    cdm_orders = df[keys + ["organism", "culture_time", "ordering_mode",
                            "ordering_mode_raw", "culture_type"]].drop_duplicates()
    order_attrs = (df[keys + ["culture_time", "ordering_mode", "ordering_mode_raw", "culture_type"]]
                   .sort_values("culture_time").drop_duplicates(subset=keys, keep="first"))
    tested = df[keys + ["antibiotic"]].drop_duplicates()
    tested["_tested"] = 1

    imp = imp.rename(columns={p_col: "patient_id", e_col: "encounter_id", o_col: "order_id",
                              org_col: "organism", abx_col: "antibiotic",
                              val_col: "implied_susceptibility"})
    needed = keys + ["organism", "antibiotic", "implied_susceptibility",
                     "_organism_raw", "_organism_phenotype"]
    imp = imp[[c for c in needed if c in imp.columns]]
    imp = imp.merge(cdm_orders[keys].drop_duplicates(), on=keys, how="inner")
    imp = imp.merge(tested, on=keys + ["antibiotic"], how="left")
    imp = imp[imp["_tested"].isna()].drop(columns=["_tested"])
    rows = imp.merge(cdm_orders, on=keys + ["organism"], how="left")
    # Implied rows whose organism matched no cohort row take the order's attributes.
    rows = rows.merge(order_attrs.rename(columns={
        "culture_time": "_ct", "ordering_mode": "_om", "ordering_mode_raw": "_omr",
        "culture_type": "_cty"}), on=keys, how="left")
    for target, fb in [("culture_time", "_ct"), ("ordering_mode", "_om"),
                       ("ordering_mode_raw", "_omr"), ("culture_type", "_cty")]:
        rows[target] = rows[target].fillna(rows[fb])
    rows = rows.drop(columns=["_ct", "_om", "_omr", "_cty"])
    rows["site_id"] = cfg.site_id
    rows["organism_raw"] = rows.get("_organism_raw", rows["organism"])
    rows["organism_phenotype"] = rows.get("_organism_phenotype", "")
    rows["susceptibility_raw"] = rows["implied_susceptibility"]
    rows["y"] = rows["implied_susceptibility"].apply(encode_susceptibility)
    rows["label_source"] = "implied"
    rows["is_implied"] = 1
    rows = rows[rows["y"].notna()].copy()
    rows["y"] = rows["y"].astype(int)
    rows = rows[CDM_COLUMNS_BASE].copy()
    return pd.concat([df, rows], ignore_index=True), len(rows)


def _conservative_label(group: pd.DataFrame) -> str:
    labels = group["susceptibility_raw"].dropna().unique()
    for label in labels:
        if str(label).lower() in ["r", "resistant"]:
            return str(label)
    for label in labels:
        if str(label).lower() in ["i", "intermediate"]:
            return str(label)
    return str(labels[0]) if len(labels) > 0 else "Unknown"


def _resolve_group(group: pd.DataFrame) -> Dict:
    """Measured beats implied; disagreeing measured results resolve to resistant."""
    base = group.iloc[0].to_dict()
    y_values = group["y"].unique().tolist()
    implied_values = group["is_implied"].unique().tolist()
    has_observed, has_implied = 0 in implied_values, 1 in implied_values
    if has_observed and has_implied:
        observed = group[group["is_implied"] == 0]
        observed_y = observed["y"].unique().tolist()
        if len(observed_y) == 1:
            y, susc, source, conflict = (observed_y[0], observed.iloc[0]["susceptibility_raw"],
                                         "mixed_prefer_observed", False)
        else:
            y, susc, source, conflict = (0 if 0 in observed_y else 1, _conservative_label(observed),
                                         "conflict_conservative", True)
    elif len(y_values) == 1:
        y, susc, conflict = y_values[0], group.iloc[0]["susceptibility_raw"], False
        source = group.iloc[0]["label_source"] if has_observed else "implied"
    else:
        y, susc, source, conflict = (0 if 0 in y_values else 1, _conservative_label(group),
                                     "conflict_conservative", True)
    base.update({
        "n_sources": len(group),
        "labels_set": ",".join(sorted(str(v) for v in group["susceptibility_raw"].dropna().unique())),
        "had_conflict": conflict,
        "label_source_resolved": source,
        "y": y,
        "susceptibility_raw": susc,
        "is_implied": 0 if has_observed else 1,
        "label_sources": ",".join(sorted(str(s) for s in group["label_source"].dropna().unique())),
        "n_measured_sources": int((group["is_implied"] == 0).sum()),
        "n_implied_sources": int((group["is_implied"] == 1).sum()),
    })
    return base


def _deduplicate(df: pd.DataFrame, key: List[str]) -> Tuple[pd.DataFrame, int]:
    dup = df.duplicated(subset=key, keep=False)

    def defaults(frame: pd.DataFrame) -> pd.DataFrame:
        frame["n_sources"] = 1
        frame["labels_set"] = frame["susceptibility_raw"].astype(str)
        frame["had_conflict"] = False
        frame["label_source_resolved"] = frame["label_source"]
        frame["label_sources"] = frame["label_source"].astype(str)
        frame["n_measured_sources"] = (frame["is_implied"] == 0).astype(int)
        frame["n_implied_sources"] = (frame["is_implied"] == 1).astype(int)
        return frame

    if not dup.any():
        return defaults(df)[CDM_COLUMNS], 0
    unique = defaults(df[~dup].copy())
    resolved = pd.DataFrame([_resolve_group(g) for _, g in df[dup].groupby(key)])
    n_conflict = int((resolved["had_conflict"] == True).sum())  # noqa: E712
    out = pd.concat([unique, resolved], ignore_index=True)[CDM_COLUMNS]
    return out, n_conflict


def build_cdm(cfg: SiteConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build the organism-grain CDM for one site from its raw cohort file."""
    cm, caps, stats = cfg.cohort, cfg.capabilities, {}
    path = cfg.file("cohort")
    if path is None or not path.exists():
        raise FileNotFoundError(f"[{cfg.site_id}] cohort file not found: {path}")
    df = pd.read_csv(path)
    stats["raw_rows"] = len(df)

    abx_col, org_col = cm.get("antibiotic", "antibiotic"), cm.get("organism", "organism")
    for col in (abx_col, org_col):
        if col not in df.columns:
            raise ValueError(f"[{cfg.site_id}] cohort file has no '{col}' column")
    df[abx_col] = df[abx_col].apply(normalize_antibiotic)
    df["_organism_raw"] = df[org_col]
    norm = df[org_col].map(normalize_organism)
    df[org_col] = [n for n, _ in norm]
    df["_organism_phenotype"] = [p for _, p in norm]

    pos_col = cm.get("was_positive")
    if not pos_col or pos_col not in df.columns:
        raise ValueError(f"[{cfg.site_id}] cohort file has no culture-positivity column")
    df = df[df[pos_col] == 1].copy()
    stats["positive_rows"] = len(df)

    blank = df[abx_col].fillna("").astype(str).str.strip() == ""
    if blank.any():
        df = df[~blank].copy()
    stats["rows_with_antibiotic"] = len(df)

    ts_col = cm.get("timestamp", "order_time_jittered_utc")
    if ts_col not in df.columns:
        for alt in ("order_time_jittered_utc", "order_time_jittered_utc_shifted", "order_time_jittered"):
            if alt in df.columns:
                ts_col = alt
                break
    if ts_col not in df.columns:
        raise ValueError(f"[{cfg.site_id}] no culture timestamp column")
    df["_culture_time"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_localize(None)

    mode_col = cm.get("ordering_mode")
    if not mode_col or mode_col not in df.columns:
        raise ValueError(f"[{cfg.site_id}] cohort file has no ordering-mode column")
    df["_ordering_mode"], df["_ordering_mode_raw"] = _map_ordering_mode(df[mode_col], caps)

    df = _keep_first_culture_per_stay(df, cm)
    stats["after_first_culture_per_stay"] = len(df)

    if cfg.label_strategy != "single_column":
        raise ValueError(f"[{cfg.site_id}] unsupported label strategy {cfg.label_strategy!r}")
    susc_col = cm.get("susceptibility", "susceptibility")
    df["_susceptibility_raw"] = df[susc_col]
    df["_y"] = df["_susceptibility_raw"].apply(encode_susceptibility)
    df = df[df["_y"].notna()].copy()
    df["_y"] = df["_y"].astype(int)

    cdm = pd.DataFrame({
        "site_id": cfg.site_id,
        "patient_id": df[cm.get("patient_id", "anon_id")].values,
        "encounter_id": df[cm.get("encounter_id", "pat_enc_csn_id_coded")].values,
        "order_id": df[cm.get("order_id", "order_proc_id_coded")].values,
        "culture_time": df["_culture_time"].values,
        "ordering_mode": df["_ordering_mode"].values,
        "ordering_mode_raw": df["_ordering_mode_raw"].values,
        "culture_type": df[cm.get("culture_type", "culture_description")].values,
        "organism": df[org_col].values,
        "organism_raw": df["_organism_raw"].values,
        "organism_phenotype": df["_organism_phenotype"].values,
        "antibiotic": df[abx_col].values,
        "susceptibility_raw": df["_susceptibility_raw"].values,
        "y": df["_y"].values,
        "label_source": "susceptibility",
        "is_implied": 0,
    })
    stats["measured_rows"] = len(cdm)

    stats["implied_rows_added"] = 0
    if caps.get("has_implied_susceptibility", False) and cfg.has_file("implied_susceptibility"):
        cdm, stats["implied_rows_added"] = _add_implied(cdm, cfg)

    cdm, stats["groups_with_conflict"] = _deduplicate(cdm, cfg.dedup_key)
    n_null_time = int(cdm["culture_time"].isna().sum())
    if n_null_time:
        raise AssertionError(f"[{cfg.site_id}] {n_null_time:,} CDM rows have no culture_time")
    stats["final_rows"] = len(cdm)
    stats["patients"] = int(cdm["patient_id"].nunique())
    stats["orders"] = int(cdm["order_id"].nunique())
    return cdm, stats


def save_cdm(cdm: pd.DataFrame, cdm_dir: Path | str, site_id: str) -> Path:
    out = Path(cdm_dir) / site_id / "cohort_long.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cdm.to_parquet(out, index=False)
    return out


def prepare_mgb_comorbidity(mgb_dir: Path | str, chunk_size: int = 1_000_000) -> Path:
    """Write MGB's comorbidity file in the other sites' schema (category -> comorbidity_component).

    MGB's raw comorbidity file carries the same AHRQ-CCSR strings in `category`
    but no per-diagnosis timing; row order is preserved.
    """
    mgb_dir = Path(mgb_dir)
    src = mgb_dir / "microbiology_cultures_comorbidity.csv"
    out = mgb_dir / "microbiology_cultures_comorbidity_harmonized.csv"
    cols = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "category"]
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        for i, chunk in enumerate(pd.read_csv(src, usecols=cols, chunksize=chunk_size, low_memory=False)):
            chunk[cols].rename(columns={"category": "comorbidity_component"}).to_csv(
                fh, header=(i == 0), index=False)
    tmp.replace(out)
    return out
