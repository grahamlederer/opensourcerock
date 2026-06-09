"""Top-level CLI entry point for the CIPW normative mineralogy pipeline.

Usage
-----
    # Default Fe partition (fixed_ratio)
    python run_norm.py --data dataverse_files/ --output data/processed/

    # Specify Fe partition method
    python run_norm.py --data dataverse_files/ --fe-method kress_carmichael

    # Run tests
    pytest tests/
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

from src.cipw import NORM_COLS, compute_cipw
from src.fe_partition import partition_fe
from src.ingest import clean_and_renorm, load_georoc
from src.utils import mg_number

# Final column order as specified in the plan
_OUTPUT_COLS = [
    "sample_id", "rock_type", "source_file",
    "SiO2", "TiO2", "Al2O3", "Fe2O3_calc", "FeO_calc",
    "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "CO2",
    "oxide_total_raw", "fe2_fetotal", "mg_number",
    "Q", "Or", "Ab", "An", "Ne", "Lc", "Kp",
    "Di", "Hy", "Ol", "ol_Fo",
    "Mt", "Il", "Hm", "Tn", "Pf", "Ru", "Ap", "Cc",
]


def run_pipeline(
    data_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    fe_method: str = "fixed_ratio",
) -> pathlib.Path:
    """Execute the full ingest → Fe partition → CIPW pipeline.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing raw GEOROC CSV files.
    output_dir : str or Path
        Directory where ``cipw_norm_output.csv`` will be written.
    fe_method : str
        Iron partition method passed to :func:`src.fe_partition.partition_fe`.

    Returns
    -------
    pathlib.Path
        Path to the written output CSV.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading GEOROC files from {data_dir} …")
    t0 = time.time()
    raw = load_georoc(data_dir)
    print(f"      {len(raw):,} rows loaded in {time.time()-t0:.1f}s")

    print("[2/5] Cleaning and renormalising …")
    clean = clean_and_renorm(raw)
    print(f"      {len(clean):,} rows after dropping zero/missing SiO2")

    print(f"[3/5] Fe partition ({fe_method}) …")
    fe_df = partition_fe(clean, method=fe_method)

    print("[4/5] CIPW norm …")
    t0 = time.time()
    norm = compute_cipw(fe_df)
    print(f"      Done in {time.time()-t0:.1f}s")

    print("[5/5] Computing Mg# and writing output …")
    norm["mg_number"] = mg_number(norm).round(4)

    # Round fe2_fetotal for output
    norm["fe2_fetotal"] = norm["fe2_fetotal"].round(4)
    norm["oxide_total_raw"] = norm["oxide_total_raw"].round(4)

    # Select and order output columns (skip any missing)
    out_cols = [c for c in _OUTPUT_COLS if c in norm.columns]
    out = norm[out_cols]

    out_path = output_dir / "cipw_norm_output.csv"
    out.to_csv(out_path, index=False)
    print(f"\nOutput written to {out_path}  ({len(out):,} samples)")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute CIPW normative mineralogy from GEOROC data."
    )
    p.add_argument(
        "--data",
        default="dataverse_files/",
        metavar="DIR",
        help="Directory containing GEOROC CSV files (default: dataverse_files/)",
    )
    p.add_argument(
        "--output",
        default="data/processed/",
        metavar="DIR",
        help="Output directory (default: data/processed/)",
    )
    p.add_argument(
        "--fe-method",
        default="fixed_ratio",
        choices=("fixed_ratio", "middlemost", "kress_carmichael"),
        help="Iron partition method (default: fixed_ratio)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        run_pipeline(args.data, args.output, fe_method=args.fe_method)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
