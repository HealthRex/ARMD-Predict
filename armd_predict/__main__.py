"""Command line: python -m armd_predict <command> [options]   (see USAGE.md)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import (DEFAULT_CONFIG_DIR, SITES, STRATA, cell_key, load_cells,
                     load_site_config)


def _cells(args):
    cells = load_cells(args.config_dir)["cells"]
    if args.cell:
        cells = [c for c in cells if cell_key(c) in set(args.cell)]
        if not cells:
            sys.exit(f"no cell matches {args.cell}")
    return cells


def _model_dirs(args):
    for variant in args.variant:
        for cell in _cells(args):
            d = Path(args.models) / variant / cell_key(cell)
            if (d / "model.json").exists():
                yield d


def cmd_prepare_mgb(args):
    from .harmonize import prepare_mgb_comorbidity
    out = prepare_mgb_comorbidity(Path(args.raw_dir) / "mgb")
    print(f"wrote {out.name}")


def cmd_harmonize(args):
    from .harmonize import build_cdm, save_cdm
    for site in args.site:
        cdm, stats = build_cdm(load_site_config(site, args.raw_dir, args.config_dir))
        save_cdm(cdm, args.out, site)
        print(site, json.dumps(stats))


def cmd_tables(args):
    from .tables import build_model_table, save_model_table
    strata = [tuple(s.split("/")) for s in args.stratum] if args.stratum else STRATA
    for site in args.site:
        cfg = load_site_config(site, args.raw_dir, args.config_dir)
        for culture, mode in strata:
            df, stats = build_model_table(cfg, culture, mode, args.cdm_dir, args.config_dir)
            save_model_table(df, stats, args.out)
            print(f"{site}/{culture}/{mode}: {stats['final_rows']:,} rows, {stats['n_columns']} columns")
            del df


def cmd_train(args):
    from .train import train_cell
    rows = []
    for variant in args.variant:
        for cell in _cells(args):
            r = train_cell(cell, variant, args.tables, args.out, args.config_dir)
            if r:
                rows.append(r)
                print(f"{variant}/{r['cell']}: test AUROC {r['auroc_test']:.6f} (n={r['n_test']:,})")
    if rows:
        _write(pd.DataFrame(rows), Path(args.out) / "training_summary.csv")


def cmd_calibrate(args):
    from .calibrate import calibrate_model
    rows = [calibrate_model(d, args.tables) for d in _model_dirs(args)]
    for r in rows:
        print(f"{r['variant']}/{r['cell']}: {r['status']} {r['method']}  "
              f"ECE raw {r['raw_ece']:.4f} -> candidate {r['candidate_ece']:.4f}")
    if rows:
        _write(pd.DataFrame(rows), Path(args.models) / "calibration.csv")


def cmd_evaluate(args):
    from .evaluate import evaluate_model
    rows = []
    for d in _model_dirs(args):
        r = evaluate_model(d, args.tables, args.n_boot)
        rows.append(r)
        print(f"{r['variant']}/{r['cell']}: AUROC {r['auroc']:.3f} "
              f"({r['auroc_ci_lo']:.3f}-{r['auroc_ci_hi']:.3f}), ECE {r['ece_served']:.4f}")
    if rows:
        _write(pd.DataFrame(rows), Path(args.out) / "evaluation.csv")


def cmd_baselines(args):
    from .baselines import compare_cell, load_history
    args.variant = ["full"]
    dirs = list(_model_dirs(args))
    drugs = {json.loads((d / "metadata.json").read_text())["antibiotic"] for d in dirs}
    history = {s: load_history(args.cdm_dir, s, drugs) for s in SITES
               if (Path(args.cdm_dir) / s / "cohort_long.parquet").exists()}
    rows = [compare_cell(d, args.tables, history, args.n_boot) for d in dirs]
    for r in rows:
        print(f"{r['cell']}: model {r['auroc_model']:.3f} vs last result {r['auroc_last_result']:.3f}")
    if rows:
        _write(pd.DataFrame(rows), Path(args.out) / "baselines.csv")


def cmd_reanchor(args):
    from .reanchor import reanchor
    print(f"{reanchor(args.p_resistant, args.train_prevalence, args.local_prevalence):.6f}")


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m armd_predict", description=__doc__)
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    def p(name, func, help_):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(func=func)
        return s

    s = p("prepare-mgb", cmd_prepare_mgb, "write MGB's harmonised comorbidity file")
    s.add_argument("--raw-dir", required=True)

    s = p("harmonize", cmd_harmonize, "raw site files -> CDM (cohort_long.parquet)")
    s.add_argument("--raw-dir", required=True)
    s.add_argument("--out", required=True, help="CDM directory")
    s.add_argument("--site", nargs="+", default=SITES, choices=SITES)

    s = p("tables", cmd_tables, "CDM -> order-level model tables")
    s.add_argument("--raw-dir", required=True)
    s.add_argument("--cdm-dir", required=True)
    s.add_argument("--out", required=True, help="model-table directory")
    s.add_argument("--site", nargs="+", default=SITES, choices=SITES)
    s.add_argument("--stratum", nargs="*", help="e.g. urine/inpatient (default: the five used)")

    for name, func, help_ in (("train", cmd_train, "train the cells"),
                              ("calibrate", cmd_calibrate, "fit and select calibrators"),
                              ("evaluate", cmd_evaluate, "held-out metrics"),
                              ("baselines", cmd_baselines, "antibiogram and last-result rules")):
        s = p(name, func, help_)
        s.add_argument("--tables", required=True, help="model-table directory")
        s.add_argument("--cell", nargs="*", help="cell keys, e.g. ceftriaxone_urine_inpatient")
        s.add_argument("--variant", nargs="+", default=["full", "lite"], choices=["full", "lite"])
        if name == "train":
            s.add_argument("--out", required=True, help="model directory")
        else:
            s.add_argument("--models", required=True, help="model directory")
        if name in ("evaluate", "baselines"):
            s.add_argument("--out", required=True, help="output directory (aggregate CSV)")
            s.add_argument("--n-boot", type=int, default=1000)
        if name == "baselines":
            s.add_argument("--cdm-dir", required=True)

    s = p("reanchor", cmd_reanchor, "re-anchor one P(resistant) to a local prevalence")
    s.add_argument("--p-resistant", type=float, required=True)
    s.add_argument("--train-prevalence", type=float, required=True)
    s.add_argument("--local-prevalence", type=float, required=True)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
