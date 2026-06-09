"""Load and standardise GEOROC CSV files into a single analysis-ready DataFrame."""

from __future__ import annotations

import hashlib
import pathlib
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

# GEOROC stores every oxide as  OXIDE(WT%)  in ALL CAPS.
# Strategy: strip the "(WT%)" suffix from every column header, then apply
# this map from GEOROC uppercase stub → canonical mixed-case name.
# This avoids enumerating every "SIO2(WT%)" form explicitly.
_STUB_TO_CANONICAL: dict[str, str] = {
    "SIO2": "SiO2",
    "TIO2": "TiO2",
    "AL2O3": "Al2O3",
    "FE2O3": "Fe2O3",
    "FEO": "FeO",
    "FEOT": "FeOT",
    "MNO": "MnO",
    "MGO": "MgO",
    "CAO": "CaO",
    "NA2O": "Na2O",
    "K2O": "K2O",
    "P2O5": "P2O5",
    "CO2": "CO2",
}

# Non-oxide metadata columns kept verbatim (GEOROC name → canonical name)
_META_COLS: dict[str, str] = {
    "SAMPLE NAME": "sample_name",
    "ROCK NAME": "rock_name",
    "ROCK TYPE": "georoc_rock_type",
}

# All oxides used downstream in CIPW / Fe partition (filled with 0 when absent)
OXIDE_COLS: list[str] = [
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "FeOT",
    "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "CO2",
]

# Oxides included in the anhydrous renormalisation (excludes CO2 and FeOT
# which are volatile / redundant)
_RENORM_OXIDES: list[str] = [
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO",
    "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rock_type_from_filename(path: pathlib.Path) -> str:
    """Extract rock type label from a GEOROC filename.

    Parameters
    ----------
    path : pathlib.Path
        E.g. ``2025-12-2JETOA_BASALT_part1.csv``

    Returns
    -------
    str
        E.g. ``BASALT``
    """
    stem = path.stem  # drop .csv
    # Remove leading date-and-prefix token (everything up to first underscore)
    parts = stem.split("_", 1)
    label = parts[1] if len(parts) > 1 else stem
    # Strip trailing _partN
    label = re.sub(r"_part\d+$", "", label, flags=re.IGNORECASE)
    return label.upper()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename GEOROC column headers to canonical oxide and metadata names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as read from a GEOROC CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns; unrecognised columns are dropped.
    """
    rename_map: dict[str, str] = {}

    for col in df.columns:
        # Strip (WT%) suffix (case-insensitive) to get the oxide stub
        stub = re.sub(r"\(WT%\)$", "", col, flags=re.IGNORECASE).strip()
        if stub in _STUB_TO_CANONICAL:
            rename_map[col] = _STUB_TO_CANONICAL[stub]
        elif col in _META_COLS:
            rename_map[col] = _META_COLS[col]

    keep = list(rename_map.keys())
    return df[keep].rename(columns=rename_map)


def _make_sample_id(source_file: str, sample_name: str, row_index: int) -> str:
    """Generate a stable, unique sample ID.

    Parameters
    ----------
    source_file : str
        Basename of the source CSV.
    sample_name : str
        Value of the SAMPLE NAME column (may be empty).
    row_index : int
        Zero-based row position within the source file.

    Returns
    -------
    str
        Eight-character hex hash prefixed with ``ORS_``.
    """
    key = f"{source_file}|{sample_name}|{row_index}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"ORS_{h}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_georoc(data_dir: str | pathlib.Path) -> pd.DataFrame:
    """Glob all CSVs in *data_dir*, standardise columns, and concatenate.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing raw GEOROC CSV files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with canonical column names, ``rock_type``,
        ``source_file``, and ``sample_id`` columns added.

    Notes
    -----
    GEOROC files use latin-1 encoding (not UTF-8); this function reads
    them with ``encoding='latin-1'``.
    """
    data_dir = pathlib.Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        raw = pd.read_csv(
            path,
            encoding="latin-1",
            low_memory=False,
        )
        df = _standardize_columns(raw)

        rock_type = _rock_type_from_filename(path)
        df["rock_type"] = rock_type
        df["source_file"] = path.name

        # Ensure metadata columns exist even if absent in this file
        for meta in ("sample_name", "rock_name", "georoc_rock_type"):
            if meta not in df.columns:
                df[meta] = pd.NA

        # Assign sample IDs using original row positions within this file
        df["sample_id"] = [
            _make_sample_id(path.name, str(row.get("sample_name", "")), i)
            for i, row in enumerate(raw.to_dict("records"))
        ]

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean_and_renorm(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce oxides to numeric, fill missing with 0, renormalise to 100 wt%.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_georoc`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:

        - All oxide columns coerced to ``float64``; missing → 0.
        - ``oxide_total_raw`` : sum of renorm oxides before normalisation.
        - Oxide columns renormalised anhydrous to 100 wt%.
        - Rows where SiO2 is missing or zero are dropped.

    Notes
    -----
    ``FeOT`` is excluded from the renormalisation sum because it is
    redundant with ``FeO`` + ``Fe2O3`` and would double-count iron.
    ``CO2`` is also excluded (volatile).  The renormalisation set matches
    ``_RENORM_OXIDES``.
    """
    df = df.copy()

    # Ensure all oxide columns exist before coercing
    for col in OXIDE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce to numeric; non-numeric strings → NaN
    for col in OXIDE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where SiO2 is missing or zero
    df = df[df["SiO2"].notna() & (df["SiO2"] > 0)].copy()

    # Record raw total before filling zeros
    df["oxide_total_raw"] = df[_RENORM_OXIDES].sum(axis=1, min_count=1)

    # Fill remaining NaNs with 0 (missing oxide → assume absent)
    for col in OXIDE_COLS:
        df[col] = df[col].fillna(0.0)

    # Anhydrous renormalisation
    totals = df[_RENORM_OXIDES].sum(axis=1)
    valid_total = totals.replace(0, np.nan)
    for col in _RENORM_OXIDES:
        df[col] = df[col] / valid_total * 100.0

    return df.reset_index(drop=True)
