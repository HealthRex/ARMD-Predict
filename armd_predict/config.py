"""Site configuration, cell list and shared constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Pooling order of the sites. The pooled training frame is concatenated in this
# order, and XGBoost's row subsampling depends on row order, so changing it
# changes the fitted trees.
SITES = ["stanford", "ecuh", "utsw", "mgb"]

# The repository's configs/ (checkout or editable install), else ./configs.
_REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_DIR = _REPO_CONFIGS if _REPO_CONFIGS.exists() else Path.cwd() / "configs"

# Model tables are written to <tables>/<site>/<culture>/<mode>/3of4/. "3of4"
# names the block policy: a feature block is built when at least three sites
# provide it. The pooled models then keep only the columns every contributing
# site shares (see train.common_feature_columns).
FEATURE_SET = "3of4"
MIN_BLOCK_SITES = 3

# The strata the 23 cells are drawn from.
STRATA = [("urine", "inpatient"), ("urine", "outpatient"), ("blood", "inpatient"),
          ("resp", "inpatient"), ("resp", "outpatient")]


@dataclass
class SiteConfig:
    site_id: str
    file_mappings: Dict[str, Optional[str]]
    column_mappings: Dict[str, Dict[str, Any]]
    label_strategy: str
    capabilities: Dict[str, Any]
    value_maps: Dict[str, Dict[str, Any]]
    missing_value_tokens: List[str]
    data_dir: Path

    def file(self, key: str) -> Optional[Path]:
        name = self.file_mappings.get(key)
        return None if name is None else self.data_dir / name

    def has_file(self, key: str) -> bool:
        path = self.file(key)
        return path is not None and path.exists()

    @property
    def cohort(self) -> Dict[str, Any]:
        return self.column_mappings.get("cohort", {})

    @property
    def dedup_key(self) -> List[str]:
        """Organism-grain uniqueness key of the CDM."""
        return list(self.capabilities.get("dedup_key")
                    or ["site_id", "order_id", "organism", "antibiotic"])

    @property
    def order_level_key(self) -> List[str]:
        """Order-grain key: the dedup key without organism (MGB keeps encounter_id)."""
        key = [c for c in self.dedup_key if c != "organism"]
        if "antibiotic" not in key:
            key.append("antibiotic")
        return key


def load_site_config(site_id: str, raw_dir: Path | str,
                     config_dir: Path | str = DEFAULT_CONFIG_DIR) -> SiteConfig:
    path = Path(config_dir) / "sites" / f"{site_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no site configuration at {path}")
    raw = yaml.safe_load(path.read_text())
    caps = raw.get("capabilities", {}) or {}
    return SiteConfig(
        site_id=raw.get("site_id", site_id),
        file_mappings=raw.get("file_mappings", {}) or {},
        column_mappings=raw.get("column_mappings", {}) or {},
        label_strategy=(raw.get("label_policy") or {}).get("strategy", "single_column"),
        capabilities=caps,
        value_maps=raw.get("value_maps") or caps.get("value_maps") or {},
        missing_value_tokens=(raw.get("missing_value_tokens")
                              or caps.get("missing_value_tokens") or []),
        data_dir=Path(raw_dir) / site_id,
    )


def feature_blocks(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> List[str]:
    """Blocks declared (has_<block>: true) by at least MIN_BLOCK_SITES sites, sorted."""
    counts: Dict[str, int] = {}
    for path in sorted((Path(config_dir) / "sites").glob("*.yaml")):
        caps = (yaml.safe_load(path.read_text()) or {}).get("capabilities", {}) or {}
        for key, value in caps.items():
            # The implied-susceptibility file is a label source, not a feature block.
            if key.startswith("has_") and value is True and key != "has_implied_susceptibility":
                counts[key[4:]] = counts.get(key[4:], 0) + 1
    return sorted(b for b, n in counts.items() if n >= MIN_BLOCK_SITES)


def load_cells(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Dict[str, Any]:
    """The cell list, short-form block prefixes and inclusion thresholds."""
    return yaml.safe_load((Path(config_dir) / "cells.yaml").read_text())


def cell_key(cell: Dict[str, str]) -> str:
    return f"{cell['antibiotic']}_{cell['culture']}_{cell['setting']}"
