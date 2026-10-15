"""Leakage guard: remove identifiers, post-result columns and absolute timestamps.

patient_id and culture_time are PROTECTED METADATA: kept in the saved table so
the patient-grouped split is possible, and never allowed into a feature matrix.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import yaml

from .config import DEFAULT_CONFIG_DIR

log = logging.getLogger(__name__)

PROTECTED_METADATA_COLUMNS = frozenset({"patient_id", "culture_time"})


class LeakageError(Exception):
    """A column matching a label pattern survived sanitisation."""


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns or []]


@dataclass
class LeakagePolicy:
    forbidden_exact: Set[str]
    forbidden_patterns: List[re.Pattern]
    forbidden_timestamp_patterns: List[re.Pattern]
    safe_time_patterns: List[re.Pattern]
    microbial_resistance_drop: Set[str]
    microbial_resistance_keep: Set[str]
    microbial_resistance_safe_patterns: List[re.Pattern]
    metadata_columns: Set[str]
    protected_metadata: Set[str] = field(default_factory=set)
    outcome_derived_patterns: List[re.Pattern] = field(default_factory=list)
    drop_outcome_derived: bool = True

    @classmethod
    def load(cls, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> "LeakagePolicy":
        raw = yaml.safe_load((Path(config_dir) / "leakage_policy.yaml").read_text())
        return cls(
            forbidden_exact=set(raw.get("forbidden_exact", [])),
            forbidden_patterns=_compile(raw.get("forbidden_patterns")),
            forbidden_timestamp_patterns=_compile(raw.get("forbidden_timestamp_patterns")),
            safe_time_patterns=_compile(raw.get("safe_time_patterns")),
            microbial_resistance_drop=set(raw.get("microbial_resistance_drop", [])),
            microbial_resistance_keep=set(raw.get("microbial_resistance_keep", [])),
            microbial_resistance_safe_patterns=_compile(raw.get("microbial_resistance_safe_patterns")),
            metadata_columns=set(raw.get("metadata_columns", [])),
            protected_metadata=set(raw.get("protected_metadata", [])),
            outcome_derived_patterns=_compile(raw.get("outcome_derived_diagnosis_patterns")),
            drop_outcome_derived=bool(raw.get("drop_outcome_derived_diagnoses", True)),
        )


def sanitize(df: pd.DataFrame, policy: LeakagePolicy, strict: bool = True
             ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Drop leaky columns in a fixed order, then raise on any surviving label pattern."""
    protected = {p.lower() for p in policy.protected_metadata}
    meta = {m.lower() for m in policy.metadata_columns}

    def is_protected(c: str) -> bool:
        return c.lower() in protected

    def is_safe_time(c: str) -> bool:
        return any(p.search(c) for p in policy.safe_time_patterns)

    report: Dict[str, List[str]] = {}

    forbidden = {f.lower() for f in policy.forbidden_exact}
    exact = [c for c in df.columns if not is_protected(c) and c.lower() in forbidden]
    df = df.drop(columns=exact, errors="ignore")
    report["forbidden_exact"] = exact

    # Prior-resistance block: keep list, then safe patterns, then drop list / label patterns.
    mr_drop, mr_keep = [], []
    for c in df.columns:
        if not (c.startswith("microbial_resistance_") or c.startswith("prior_resistance_")):
            continue
        if c in policy.microbial_resistance_keep or any(
                p.search(c) for p in policy.microbial_resistance_safe_patterns):
            mr_keep.append(c)
        elif c in policy.microbial_resistance_drop or any(
                p.search(c) for p in policy.forbidden_patterns):
            mr_drop.append(c)
    df = df.drop(columns=mr_drop, errors="ignore")
    report["microbial_resistance"] = mr_drop

    outcome = []
    if policy.drop_outcome_derived:
        outcome = [c for c in df.columns if not is_protected(c)
                   and any(p.search(c) for p in policy.outcome_derived_patterns)]
    df = df.drop(columns=outcome, errors="ignore")
    report["outcome_derived"] = outcome

    def screened(c: str) -> bool:
        return not (c.lower() in meta or is_protected(c) or c in mr_keep or is_safe_time(c))

    stamps = [c for c in df.columns if screened(c)
              and any(p.search(c) for p in policy.forbidden_timestamp_patterns)]
    df = df.drop(columns=stamps, errors="ignore")
    report["timestamps"] = stamps

    violations = [c for c in df.columns if screened(c)
                  and any(p.search(c) for p in policy.forbidden_patterns)]
    if violations:
        msg = f"label-like columns survived sanitisation: {violations}"
        if strict:
            raise LeakageError(msg)
        log.warning(msg)
    for name, cols in report.items():
        if cols:
            log.info("leakage guard dropped %d %s column(s)", len(cols), name)
    return df, report


def assert_no_protected_metadata(columns, context: str = "") -> None:
    """patient_id / culture_time in a feature list is leakage: refuse."""
    leaked = sorted({c for c in columns if str(c).lower() in PROTECTED_METADATA_COLUMNS})
    if leaked:
        raise RuntimeError(f"[{context}] protected metadata in the feature list: {leaked}")
