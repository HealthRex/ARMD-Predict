"""Feature blocks: load each site file, reduce it to one row per culture order, join.

Missing-value convention for history blocks (keep it at every boundary):
  <block>_count_prior_events   number of qualifying prior events
  <block>_min_days_prior       days since the most recent qualifying event;
                               NaN when there was none, NEVER 0
  <block>_any_prior            1 / 0
  <block>_any_within_{30,90,365}d
For an order with a record in the block file the count and flags are real
numbers (0 when nothing qualifies); for an order absent from the file they are
NaN. "Never happened" (NaN) and "happened today" (0) are different patients.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SiteConfig

log = logging.getLogger(__name__)

# History files record whole-day offsets, so 0 means "same calendar day", which
# may be after the order (the antibiotic started because of this culture, this
# culture's own AST). Day-0 records are dropped from every history block.
EXCLUDE_DAY_ZERO = True
# Nursing-home offsets are mostly stamped with the culture day; the block keeps
# only "any contact recorded" instead of a timing we cannot trust.
PRESENCE_ONLY_BLOCKS = {"nursing_home_visits"}
HISTORY_WINDOW_DAYS = (30, 90, 365)
# ECUH and MGB ship 14/30/180-day lab summaries, Stanford and UTSW only 14.
LABS_LOOKBACK_DAYS = 14
# Blocks expanded to multi-hot indicators over ALL rows of the order.
MULTIHOT_CATEGORY_COL = {"comorbidity": "comorbidity_component"}
# The upstream Elixhauser crosswalk attaches "Congestive heart failure" to
# >1,400 unrelated ICD-10 codes (it is true for ~80% of patients); the separate,
# correct CCSR "Heart failure" category is kept.
CONTAMINATED_CATEGORIES = {"comorbidity_component": frozenset({"Congestive heart failure"})}
# "Prior resistance to THIS antibiotic within 365 days" needs the set of drugs
# over all of the order's rows, not one surviving row.
HISTORY_SET_COLS = {"microbial_resistance": {"same_abx_lookup_365d": ("antibiotic", 365)}}

_DAY_OFFSET_KEYS = ("time_to_culture", "start_days")
_DAY_OFFSET_FALLBACK = re.compile(
    r"(?i)(days_to_culture|days_to_culutre|time_to_culture|time_to_culturetime"
    r"|days_culture|visit_culture)$")
# The comorbidity END offset is signed (negative = ends after the culture) and
# is turned into a "still open" flag instead.
_END_DAYS = re.compile(r"(?i)_end_days_culture$")
_KEEP_TIME_PATTERNS = ["days_to_culture", "days_to_culutre", "time_to_culture",
                       "time_to_culturetime"]


def _day_offset_column(block: pd.DataFrame, mapping: Dict[str, Any]) -> Optional[str]:
    for key in _DAY_OFFSET_KEYS:
        col = mapping.get(key)
        if col and col in block.columns:
            return col
    for col in block.columns:
        if not _END_DAYS.search(col) and _DAY_OFFSET_FALLBACK.search(col):
            return col
    return None


def _days_since(s: pd.Series) -> pd.Series:
    # All sites store positive days before the culture; abs() is a guard.
    return pd.to_numeric(s, errors="coerce").abs()


def _multihot(block: pd.DataFrame, keys: List[str], cat_col: str,
              max_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """One 0/1 column per category present in ANY row of the order, plus unique_count."""
    max_cols = [c for c in (max_cols or []) if c in block.columns]
    other = [c for c in block.columns if c not in keys and c != cat_col and c not in max_cols]
    if other:
        out = block.drop_duplicates(subset=keys, keep="first")[keys + other]
    else:
        out = block[keys].drop_duplicates(subset=keys, keep="first")
    if max_cols:
        out = out.merge(block.groupby(keys, as_index=False)[max_cols].max(), on=keys, how="left")
        other = other + max_cols
    present = block.loc[block[cat_col].notna(), keys + [cat_col]]
    bad = CONTAMINATED_CATEGORIES.get(cat_col, frozenset())
    if bad:
        present = present.loc[~present[cat_col].isin(bad)]
    present = present.drop_duplicates(subset=keys + [cat_col])
    if len(present):
        dummies = pd.get_dummies(present[cat_col].astype(str), prefix=cat_col,
                                 prefix_sep="__").astype("int8")
        mh = pd.concat([present[keys].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        mh = mh.groupby(keys, as_index=False).max()
        indicators = [c for c in mh.columns if c not in keys]
        mh["unique_count"] = mh[indicators].sum(axis=1).astype("int64")
    else:
        mh = block[keys].drop_duplicates(subset=keys, keep="first").copy()
        indicators = []
        mh["unique_count"] = 0
    out = out.merge(mh, on=keys, how="left")
    # Presence flags and their count: an order with no rows has none (0 is true).
    for c in indicators:
        out[c] = out[c].fillna(0).astype("int8")
    out["unique_count"] = out["unique_count"].fillna(0).astype("int64")
    return out, other + indicators + ["unique_count"]


def _history(block: pd.DataFrame, keys: List[str], offset_col: str,
             set_cols: Optional[Dict[str, Tuple[str, int]]] = None
             ) -> Tuple[pd.DataFrame, List[str]]:
    """Count / most-recent / any-within aggregates over all qualifying rows.

    The descriptive columns (drug name, organism, the offset itself) come from
    the MOST RECENT qualifying row. The extracts list same-day rows first, so
    first-listed selection returned the row closest to (or after) the order.
    """
    work = block.copy()
    work["_days_since"] = _days_since(work[offset_col])
    # Built before the day-0 exclusion: "record exists, nothing qualifies" stays
    # distinguishable from "no record at all".
    skeleton = work[keys].drop_duplicates()
    if EXCLUDE_DAY_ZERO:
        work = work[work["_days_since"] != 0]
    qual = work[work["_days_since"].notna()]
    if len(qual):
        agg = (qual.groupby(keys, dropna=False)["_days_since"].agg(["size", "min"]).reset_index()
               .rename(columns={"size": "count_prior_events", "min": "min_days_prior"}))
    else:
        agg = skeleton.copy()
        agg["count_prior_events"] = 0
        agg["min_days_prior"] = np.nan
    out = skeleton.merge(agg, on=keys, how="left")
    out["count_prior_events"] = out["count_prior_events"].fillna(0).astype("int64")
    out["any_prior"] = (out["count_prior_events"] > 0).astype("int8")
    agg_cols = ["count_prior_events", "min_days_prior", "any_prior"]
    for w in HISTORY_WINDOW_DAYS:
        out[f"any_within_{w}d"] = (out["min_days_prior"] <= w).astype("int8")  # NaN -> 0
        agg_cols.append(f"any_within_{w}d")
    other = [c for c in block.columns if c not in keys]
    if len(qual):
        rep = qual.sort_values("_days_since", ascending=True, kind="mergesort")
        rep = rep.drop_duplicates(subset=keys, keep="first")
        out = out.merge(rep[keys + other], on=keys, how="left")
    else:
        for c in other:
            out[c] = np.nan
    for name, (src, window) in (set_cols or {}).items():
        if src not in qual.columns:
            continue
        sub = qual.loc[qual["_days_since"] <= window, keys + [src]].copy()
        sub[src] = sub[src].astype(str).str.strip().str.lower()
        sub = sub[~sub[src].isin(["", "nan", "none", "null"])]
        if len(sub):
            sub = sub.drop_duplicates(subset=keys + [src]).sort_values(keys + [src], kind="mergesort")
            joined = sub.groupby(keys, sort=False)[src].agg("|".join).reset_index(name=name)
            out = out.merge(joined, on=keys, how="left")
            out[name] = out[name].fillna("")
        else:
            out[name] = ""
        agg_cols.append(name)
    return out, other + agg_cols


def _presence_only(block: pd.DataFrame, keys: List[str],
                   offset_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    skeleton = block[keys].drop_duplicates()
    if offset_col is not None and offset_col in block.columns:
        present = pd.to_numeric(block[offset_col], errors="coerce").notna()
    else:
        present = pd.Series(True, index=block.index)
    flag = block.loc[present, keys].drop_duplicates()
    flag["any_contact"] = np.int8(1)
    out = skeleton.merge(flag, on=keys, how="left")
    out["any_contact"] = out["any_contact"].fillna(0).astype("int8")
    return out, ["any_contact"]


def _apply_value_maps(df: pd.DataFrame, cfg: SiteConfig) -> pd.DataFrame:
    """Rewrite site values onto the shared vocabulary (gender coding, age bins, "Null")."""
    vmaps, tokens = cfg.value_maps or {}, {str(t).strip() for t in cfg.missing_value_tokens or []}
    if not vmaps and not tokens:
        return df
    for col in df.columns:
        if col not in vmaps:
            continue
        rules = vmaps.get(col)
        s = df[col]
        key = s.map(lambda v: str(v).strip() if pd.notna(v) else v)
        if tokens:
            s = s.mask(key.isin(tokens))
            key = key.mask(key.isin(tokens))
        if rules:
            lookup = {str(k).strip(): v for k, v in rules.items()}
            unmapped = set(key.dropna().unique()) - set(lookup) - {None}
            if unmapped:
                # Matters only for recoding maps (gender, age); a "Null"-only map
                # passes every real value through by design.
                level = logging.WARNING if any(v is not None for v in rules.values()) else logging.DEBUG
                log.log(level, "[%s] %s: %d distinct value(s) without a harmonisation rule "
                        "passed through", cfg.site_id, col, len(unmapped))
            s = key.map(lambda v: lookup.get(v, v) if pd.notna(v) else v)
        if not s.equals(df[col]):
            df = df.assign(**{col: s})
    return df


def load_block(block_name: str, cohort: pd.DataFrame, cfg: SiteConfig) -> Tuple[pd.DataFrame, Dict]:
    """Reduce one block to one row per order and left-join it onto the cohort."""
    stats = {"block": block_name, "rows_loaded": 0, "orders_matched": 0, "n_columns": 0}
    path = cfg.file(block_name)
    if not cfg.capabilities.get(f"has_{block_name}", False) or path is None or not path.exists():
        log.info("[%s] block %s not available", cfg.site_id, block_name)
        return cohort, stats
    block = pd.read_csv(path, low_memory=False)
    stats["rows_loaded"] = len(block)
    mapping = cfg.column_mappings.get(block_name, {})

    if block_name == "labs":
        period_col = mapping.get("period_day") or "Period_Day"
        if period_col in block.columns:
            period = pd.to_numeric(block[period_col], errors="coerce")
            if (period == LABS_LOOKBACK_DAYS).any():
                block = block[period == LABS_LOOKBACK_DAYS].copy()
            else:
                log.warning("[%s] labs: no %d-day rows; window filter skipped",
                            cfg.site_id, LABS_LOOKBACK_DAYS)
            block = block.drop(columns=[period_col])

    open_flag = None
    if block_name == "comorbidity":
        end_col = mapping.get("end_days")
        if not end_col or end_col not in block.columns:
            found = [c for c in block.columns if _END_DAYS.search(c)]
            end_col = found[0] if found else None
        if end_col is not None:
            end = pd.to_numeric(block[end_col], errors="coerce")
            flag = (end <= 0).astype("float32")  # 0 counts as open (day-zero rule)
            flag[end.isna()] = np.nan
            block = block.drop(columns=[end_col])
            block["still_open_at_culture"] = flag
            open_flag = "still_open_at_culture"

    pairs = [(k, mapping[k]) for k in ("patient_id", "encounter_id", "order_id")
             if cfg.cohort.get(k) and mapping.get(k) and mapping[k] in block.columns]
    if not pairs:
        log.warning("[%s] block %s has no join keys", cfg.site_id, block_name)
        return cohort, stats
    keys = [k for k, _ in pairs]
    join_cols = [c for _, c in pairs]
    # Identifier-like columns and absolute times are never features; relative
    # day offsets to the culture are kept.
    feature_cols = [c for c in block.columns
                    if c not in join_cols
                    and not any(x in c.lower() for x in ["_id", "_coded", "csn"])
                    and ("time" not in c.lower()
                         or any(p in c.lower() for p in _KEEP_TIME_PATTERNS))]
    if not feature_cols:
        return cohort, stats
    block = block.rename(columns={c: k for k, c in pairs})
    block = block[[c for c in keys + feature_cols if c in block.columns]]

    offset_col = _day_offset_column(block, mapping)
    cat_col = MULTIHOT_CATEGORY_COL.get(block_name)
    if block_name in PRESENCE_ONLY_BLOCKS:
        block, feature_cols = _presence_only(block, keys, offset_col)
    elif cat_col and cat_col in block.columns:
        if offset_col is not None and EXCLUDE_DAY_ZERO:
            block = block[_days_since(block[offset_col]) != 0]
        block, feature_cols = _multihot(block, keys, cat_col,
                                        max_cols=[open_flag] if open_flag else None)
    elif offset_col is not None:
        block, feature_cols = _history(block, keys, offset_col, HISTORY_SET_COLS.get(block_name))
    else:  # one row per order already (demographics, ward, labs, vitals, ADI)
        block = block.drop_duplicates(subset=keys, keep="first")

    prefix = f"{block_name}_"
    block = block.rename(columns={c: prefix + c for c in feature_cols if c in block.columns})
    block = _apply_value_maps(block, cfg)
    merged = cohort.merge(block, on=keys, how="left")
    new_cols = [c for c in merged.columns if c.startswith(prefix)]
    if new_cols:
        stats["orders_matched"] = int(merged[new_cols].notna().any(axis=1).sum())
    stats["n_columns"] = len(new_cols)
    log.info("[%s] block %s: %d columns, %d of %d rows matched", cfg.site_id, block_name,
             len(new_cols), stats["orders_matched"], len(merged))
    return merged, stats


# ── Prior-history summaries ──────────────────────────────────────────────────
# Public names for the loader's per-order aggregates.
_SUMMARY_NAMES = {
    "prior_infecting_organism": ("prior_infect_org", {
        "event_count": "count_prior_events", "min_days": "min_days_prior",
        "has_prior_event": "any_prior", "any_within_30d": "any_within_30d",
        "any_within_90d": "any_within_90d", "any_within_365d": "any_within_365d"}),
    "microbial_resistance": ("prior_resistance", {
        "event_count": "count_prior_events", "min_days": "min_days_prior",
        "has_prior_event": "any_prior", "any_within_30d": "any_within_30d",
        "any_within_90d": "any_within_90d", "any_within_365d": "any_within_365d"}),
}
_TIME_CANDIDATES = {
    "prior_infecting_organism": (
        ["prior_infecting_organism_prior_infecting_organism_days_to_culture",
         "prior_infecting_organism_prior_infecting_organism_days_to_culutre",
         "prior_infecting_organism_prior_org_days_to_culture",
         "prior_infecting_organism_days_to_culture",
         "prior_infecting_organism_days_to_culutre", "prior_org_days_to_culture"],
        r"(?i)prior.*(days_to_culture|time_to_culture|days_to_culutre)"),
    "microbial_resistance": (
        ["microbial_resistance_resistant_time_to_culturetime",
         "microbial_resistance_prior_AST_time_to_culture",
         "resistant_time_to_culturetime", "prior_AST_time_to_culture"],
        r"(?i)(resistant|prior).*(time_to_culture|time_to_culturetime)"),
}
SAME_ABX_LOOKUP = "microbial_resistance_same_abx_lookup_365d"


def _resolve_time_column(df: pd.DataFrame, block: str) -> Optional[str]:
    candidates, pattern = _TIME_CANDIDATES[block]
    for c in candidates:
        if c in df.columns:
            return c
    rx = re.compile(pattern)
    return next((c for c in df.columns if rx.search(c)), None)


def add_history_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Copy the loader aggregates onto prior_infect_org_* / prior_resistance_* and
    derive prior_resistance_same_abx_within_365d; drop raw descriptive columns."""
    for block in ("prior_infecting_organism", "microbial_resistance"):
        time_col = _resolve_time_column(df, block)
        if time_col is None:
            log.warning("no %s time column; summaries skipped", block)
            continue
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        public, mapping = _SUMMARY_NAMES[block]
        created: List[str] = []
        if f"{block}_count_prior_events" in df.columns:
            for name, suffix in mapping.items():
                if f"{block}_{suffix}" in df.columns:
                    df[f"{public}_{name}"] = df[f"{block}_{suffix}"]
                    created.append(f"{public}_{name}")
        if not created:
            log.warning("%s: per-order aggregates missing; summaries not created", block)
        used_loader = bool(created)
        if block == "prior_infecting_organism":
            raw = [c for c in df.columns if c.startswith(f"{block}_") and c not in created
                   and "organism" in c.lower() and c != time_col]
            df = df.drop(columns=raw, errors="ignore")
            continue
        if SAME_ABX_LOOKUP in df.columns and "antibiotic" in df.columns:
            # KNOWN LIMITATION, kept because the published models were trained
            # with it: the history spelling is only lower-cased, the target is
            # the normalised CDM name, so multi-word drugs written with "/" or
            # "-" in the history file ("Piperacillin/Tazobactam") never match
            # and the flag is 0 for them at Stanford, ECUH and UTSW.
            lookup = "|" + df[SAME_ABX_LOOKUP].fillna("").astype(str) + "|"
            target = df["antibiotic"].astype(str).str.strip().str.lower()
            same = pd.Series(0, index=df.index, dtype="int8")
            for abx in target.dropna().unique():
                if not abx or abx in {"nan", "none", ""}:
                    continue
                m = target == abx
                same.loc[m] = lookup.loc[m].str.contains(f"|{abx}|", regex=False).astype("int8")
            df["prior_resistance_same_abx_within_365d"] = same
            created.append("prior_resistance_same_abx_within_365d")
        if SAME_ABX_LOOKUP in df.columns:
            df = df.drop(columns=[SAME_ABX_LOOKUP])
        loader_suffixes = tuple(mapping.values())
        drop = [c for c in df.columns
                if c.startswith(f"{block}_") and c not in created and c != time_col
                and (any(x in c.lower() for x in ["organism", "antibiotic", "drug_code"])
                     or (used_loader and c[len(block) + 1:] in loader_suffixes))]
        df = df.drop(columns=drop, errors="ignore")
    return df
