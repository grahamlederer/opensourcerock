"""Utility functions: Mg number, renormalisation, and miscellaneous helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MW_FEO = 71.844
_MW_MGO = 40.304


def mg_number(df: pd.DataFrame) -> pd.Series:
    """Compute the magnesium number on a molar basis.

    .. math::

        Mg\\# = \\frac{Mg_{mol}}{Mg_{mol} + Fe^{2+}_{mol}}

    where :math:`Fe^{2+}` comes from the partitioned ``FeO_calc`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``MgO`` and ``FeO_calc`` columns (wt%).

    Returns
    -------
    pd.Series
        Mg# in [0, 1]; ``NaN`` where iron or magnesium sum to zero.
    """
    mg_mol = df["MgO"] / _MW_MGO
    fe2_mol = df["FeO_calc"] / _MW_FEO
    denom = mg_mol + fe2_mol
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0, mg_mol / denom, np.nan)
    return pd.Series(result, index=df.index, name="mg_number")


def renorm(df: pd.DataFrame, columns: list[str], target: float = 100.0) -> pd.DataFrame:
    """Renormalise *columns* in *df* so their row-sum equals *target*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to renormalise.
    target : float
        Target sum (default 100.0).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *columns* rescaled.
    """
    df = df.copy()
    row_sum = df[columns].sum(axis=1).replace(0, np.nan)
    for col in columns:
        df[col] = df[col] / row_sum * target
    return df
