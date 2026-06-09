"""Split FeOT into FeO + Fe2O3 and compute Fe²⁺/Fe_total.

Three partition methods are provided:

``fixed_ratio``
    Fe2O3 / FeOT = 0.15 by weight (conservative default).
``middlemost``
    Fe2O3 / FeO = 0.15 after Middlemost (1989).
``kress_carmichael``
    Oxidation state approximated from SiO2 content (simplified
    from Kress & Carmichael 1991).

Rows that already report both ``Fe2O3`` and ``FeO`` are used directly;
the partition step runs only on rows where only ``FeOT`` is populated.
"""

import numpy as np
import pandas as pd

# Molecular weight ratio used to convert between Fe2O3 and FeO equivalents.
# FeOT = FeO + Fe2O3 * (2 * MW_FeO / MW_Fe2O3)
_MW_FEO = 71.844
_MW_FE2O3 = 159.688
_FE2O3_TO_FEO = 2 * _MW_FEO / _MW_FE2O3  # ≈ 0.8998

_METHODS = ("fixed_ratio", "middlemost", "kress_carmichael")


# ---------------------------------------------------------------------------
# Partition helpers (operate on 1-D arrays of FeOT values)
# ---------------------------------------------------------------------------

def _split_fixed_ratio(feot: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Fe2O3 / FeOT = 0.15 by weight.

    Parameters
    ----------
    feot : np.ndarray
        Total iron as FeO (wt%).

    Returns
    -------
    fe2o3, feo : tuple of np.ndarray
        Partitioned Fe2O3 and FeO in wt%.
    """
    fe2o3 = 0.15 * feot
    feo = feot - fe2o3 * _FE2O3_TO_FEO
    return fe2o3, feo


def _split_middlemost(feot: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
    """Fe2O3 / FeO = 0.15 after Middlemost (1989).

    Parameters
    ----------
    feot : np.ndarray
        Total iron as FeO (wt%).

    Returns
    -------
    fe2o3, feo : tuple of np.ndarray
        Partitioned Fe2O3 and FeO in wt%.
    """
    # FeOT = FeO + Fe2O3 * k  and  Fe2O3 = 0.15 * FeO
    # FeOT = FeO * (1 + 0.15 * k)  →  FeO = FeOT / (1 + 0.15 * k)
    feo = feot / (1.0 + 0.15 * _FE2O3_TO_FEO)
    fe2o3 = 0.15 * feo
    return fe2o3, feo


def _split_kress_carmichael(
    feot: np.ndarray, sio2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SiO2-based Fe³⁺/Fetotal approximation (simplified Kress & Carmichael 1991).

    A linear ramp from Fe³⁺/Fetotal = 0.10 at SiO2 = 45 wt% to 0.25 at
    SiO2 = 70 wt% is used as a first-order proxy for the compositional
    control on oxidation state. The full Kress–Carmichael model requires
    temperature and fO₂, which are not available in bulk-rock databases.

    Parameters
    ----------
    feot : np.ndarray
        Total iron as FeO (wt%).
    sio2 : np.ndarray
        SiO2 in wt% (renormalised).

    Returns
    -------
    fe2o3, feo : tuple of np.ndarray
        Partitioned Fe2O3 and FeO in wt%.
    """
    # Fe3_frac = molar Fe³⁺ / Fetotal; clamp to [0.05, 0.40]
    fe3_frac = np.clip(0.10 + (sio2 - 45.0) * (0.15 / 25.0), 0.05, 0.40)

    # Convert molar fraction to wt% split
    # Fetotal_mol (per 100 g) = FeOT / MW_FeO
    # Fe3_mol = fe3_frac * Fetotal_mol  →  Fe2O3 = Fe3_mol / 2 * MW_Fe2O3
    # Fe2_mol = (1-fe3_frac) * Fetotal_mol  →  FeO = Fe2_mol * MW_FeO
    fetotal_mol = feot / _MW_FEO
    fe3_mol = fe3_frac * fetotal_mol
    fe2_mol = (1.0 - fe3_frac) * fetotal_mol
    fe2o3 = (fe3_mol / 2.0) * _MW_FE2O3
    feo = fe2_mol * _MW_FEO
    return fe2o3, feo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def partition_fe(df: pd.DataFrame, method: str = "fixed_ratio") -> pd.DataFrame:
    """Partition iron and compute Fe²⁺/Fetotal.

    Rows that already have **both** ``Fe2O3`` > 0 and ``FeO`` > 0 are used
    directly (no partitioning applied).  Rows where only ``FeOT`` is
    populated are split using *method*.

    Original values are preserved as ``Fe2O3_original`` and ``FeO_original``.
    Working columns written to the DataFrame are ``Fe2O3_calc`` and
    ``FeO_calc``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`src.ingest.clean_and_renorm`.
    method : str
        One of ``"fixed_ratio"`` (default), ``"middlemost"``, or
        ``"kress_carmichael"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns:

        - ``Fe2O3_original``, ``FeO_original`` — values from ingest
        - ``Fe2O3_calc``, ``FeO_calc``  — values used in CIPW
        - ``fe2_fetotal``              — molar Fe²⁺ / (Fe²⁺ + Fe³⁺)
        - ``fe_partition_method``     — which method was applied per row

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}")

    df = df.copy()

    # Preserve originals
    df["Fe2O3_original"] = df["Fe2O3"].copy()
    df["FeO_original"] = df["FeO"].copy()

    fe2o3 = df["Fe2O3"].values.copy()
    feo = df["FeO"].values.copy()
    feot = df["FeOT"].values.copy()
    sio2 = df["SiO2"].values.copy()

    # Classify rows
    has_split = (fe2o3 > 0) & (feo > 0)           # both explicit → use directly
    needs_partition = (~has_split) & (feot > 0)    # only FeOT → partition
    # Rows with nothing are left as zero

    fe2o3_calc = fe2o3.copy()
    feo_calc = feo.copy()
    partition_method = np.where(has_split, "direct", np.where(needs_partition, method, "none"))

    if needs_partition.any():
        idx = np.where(needs_partition)[0]
        feot_sub = feot[idx]
        sio2_sub = sio2[idx]

        if method == "fixed_ratio":
            fe2o3_part, feo_part = _split_fixed_ratio(feot_sub)
        elif method == "middlemost":
            fe2o3_part, feo_part = _split_middlemost(feot_sub)
        else:  # kress_carmichael
            fe2o3_part, feo_part = _split_kress_carmichael(feot_sub, sio2_sub)

        fe2o3_calc[idx] = fe2o3_part
        feo_calc[idx] = feo_part

    df["Fe2O3_calc"] = fe2o3_calc
    df["FeO_calc"] = feo_calc
    df["fe_partition_method"] = partition_method

    # Molar Fe²⁺/Fetotal
    fe2_mol = feo_calc / _MW_FEO
    fe3_mol = (fe2o3_calc / _MW_FE2O3) * 2.0
    fetotal_mol = fe2_mol + fe3_mol
    with np.errstate(invalid="ignore", divide="ignore"):
        fe2_fetotal = np.where(fetotal_mol > 0, fe2_mol / fetotal_mol, np.nan)

    df["fe2_fetotal"] = fe2_fetotal

    return df
