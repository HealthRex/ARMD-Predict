"""Step 2: build an order-level model table for one site x specimen x care setting.

<tables>/<site>/<culture>/<mode>/3of4/model_table.parquet has one row per
culture order x antibiotic: metadata (site_id, antibiotic, y, culture_time,
ordering_mode, culture_type, patient_id) plus the feature columns of every
block available at the site. Antibiotic inclusion thresholds and the
cross-site column intersection are applied at training time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG_DIR, FEATURE_SET, SiteConfig, feature_blocks
from .features import add_history_summaries, load_block
from .leakage import LeakagePolicy, sanitize

log = logging.getLogger(__name__)

CULTURE_PATTERNS = {
    "urine": ["urine"],
    "blood": ["blood"],
    "resp": ["respiratory", "sputum", "bronchial", "bal", "tracheal"],
}
# organism is loaded only to count organisms per order, then removed: it is a
# culture result. had_conflict is used to drop conflicting labels.
CDM_COLUMNS = ["site_id", "patient_id", "culture_time", "encounter_id", "order_id",
               "antibiotic", "culture_type", "ordering_mode", "y", "had_conflict", "organism"]


def collapse_to_order_level(df: pd.DataFrame, key: List[str]) -> pd.DataFrame:
    """One row per order x antibiotic, "resistant wins": y = 0 if ANY organism is R/I.

    The empiric question is asked of the culture order, not of one isolate: a
    drug covers a polymicrobial culture only if it covers every organism grown.
    """
    work = df.copy()
    order_key = [c for c in key if c != "antibiotic"]
    n_org = (work.groupby(order_key, dropna=False)["organism"].nunique()
             .rename("n_organisms").reset_index())
    grouped = work.groupby(key, dropna=False)
    agg = grouped.agg(y=("y", "min"), _y_max=("y", "max"), _n_rows=("y", "size")).reset_index()
    agg = agg.merge(n_org, on=order_key, how="left")
    agg["n_organisms"] = agg["n_organisms"].fillna(1).astype(int)
    agg["had_multiorg_conflict"] = (agg["n_organisms"] > 1) & (agg["y"] != agg["_y_max"])
    passthrough = [c for c in ["culture_time", "ordering_mode", "ordering_mode_raw",
                               "culture_type", "patient_id", "encounter_id"]
                   if c in work.columns and c not in agg.columns]
    if passthrough:
        agg = agg.merge(grouped[passthrough].first().reset_index(), on=key, how="left")
    return agg.drop(columns=["_y_max", "_n_rows"])


def build_model_table(cfg: SiteConfig, culture: str, mode: str, cdm_dir: Path | str,
                      config_dir: Path | str = DEFAULT_CONFIG_DIR,
                      blocks: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    stats: Dict = {"site": cfg.site_id, "culture": culture, "mode": mode}
    cdm = pd.read_parquet(Path(cdm_dir) / cfg.site_id / "cohort_long.parquet")
    df = cdm[[c for c in CDM_COLUMNS if c in cdm.columns]].copy()
    del cdm
    stats["cdm_rows"] = len(df)

    df = df[df["ordering_mode"].str.lower().str.contains(mode.lower(), na=False)].copy()
    pattern = "|".join(CULTURE_PATTERNS[culture])
    df = df[df["culture_type"].str.lower().str.contains(pattern, case=False, na=False)].copy()
    stats["after_mode_and_culture"] = len(df)
    df = df[~(df["had_conflict"] == True)].copy()  # noqa: E712  conflicting labels
    df = df[~df["y"].isna()].copy()
    stats["after_conflict_drop"] = len(df)

    df = collapse_to_order_level(df, cfg.order_level_key)
    stats["polymicrobial_rows"] = int((df["n_organisms"] > 1).sum())
    stats["label_changed_by_collapse"] = int(df["had_multiorg_conflict"].sum())
    # n_organisms is not knowable at order time: recorded above, never a feature.
    df = df.drop(columns=[c for c in ("n_organisms", "had_multiorg_conflict", "organism")
                          if c in df.columns])
    stats["order_level_rows"] = len(df)

    block_stats = []
    for block in (blocks or feature_blocks(config_dir)):
        df, bs = load_block(block, df, cfg)
        block_stats.append(bs)
    stats["blocks"] = block_stats
    df = add_history_summaries(df)
    df, dropped = sanitize(df, LeakagePolicy.load(config_dir))
    stats["leakage_dropped"] = {k: len(v) for k, v in dropped.items()}
    stats["final_rows"], stats["n_columns"] = len(df), int(df.shape[1])
    return df, stats


def save_model_table(df: pd.DataFrame, stats: Dict, tables_dir: Path | str) -> Path:
    out = (Path(tables_dir) / stats["site"] / stats["culture"] / stats["mode"]
           / FEATURE_SET / "model_table.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.parent / "build_manifest.json").write_text(json.dumps(stats, indent=2, default=int))
    return out
