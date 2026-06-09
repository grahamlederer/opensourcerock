"""CIPW normative mineralogy calculation.

Implements the standard CIPW norm algorithm following Le Bas & Streckeisen
(1991) and Verma et al. (2002).  All intermediate values are kept as exact
floats; rounding is applied only to the final output.

Mineral column order in output:
    Q  Or  Ab  An  Ne  Lc  Kp  Di  Hy  Ol  ol_Fo  Mt  Il  Hm  Tn  Pf  Ru  Ap  Cc
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Molecular weights (IUPAC 2021)
# ---------------------------------------------------------------------------

_MW = {
    "SiO2": 60.084,
    "TiO2": 79.866,
    "Al2O3": 101.961,
    "Fe2O3": 159.688,
    "FeO": 71.844,
    "MnO": 70.937,
    "MgO": 40.304,
    "CaO": 56.077,
    "Na2O": 61.979,
    "K2O": 94.196,
    "P2O5": 141.944,
    "CO2": 44.010,
}

# Mineral end-member molecular weights (g/mol per CIPW formula unit)
# Each "unit" is defined by how many moles of each oxide are consumed.
# See module docstring for derivations.
_MWM = {
    # Tectosilicates
    "Or": _MW["K2O"] + _MW["Al2O3"] + 6 * _MW["SiO2"],   # K2O·Al2O3·6SiO2
    "Ab": _MW["Na2O"] + _MW["Al2O3"] + 6 * _MW["SiO2"],  # Na2O·Al2O3·6SiO2
    "An": _MW["CaO"] + _MW["Al2O3"] + 2 * _MW["SiO2"],   # CaO·Al2O3·2SiO2
    "Ne": _MW["Na2O"] + _MW["Al2O3"] + 2 * _MW["SiO2"],  # Na2O·Al2O3·2SiO2
    "Lc": _MW["K2O"] + _MW["Al2O3"] + 4 * _MW["SiO2"],   # K2O·Al2O3·4SiO2
    "Kp": _MW["K2O"] + _MW["Al2O3"] + 2 * _MW["SiO2"],   # K2O·Al2O3·2SiO2
    "Q": _MW["SiO2"],
    # Pyroxenes — Mg and Fe end-members tracked separately, summed for output
    "Di_Mg": _MW["CaO"] + _MW["MgO"] + 2 * _MW["SiO2"],  # CaMgSi2O6
    "Di_Fe": _MW["CaO"] + _MW["FeO"] + 2 * _MW["SiO2"],  # CaFeSi2O6
    "Hy_Mg": _MW["MgO"] + _MW["SiO2"],                    # MgSiO3 (enstatite)
    "Hy_Fe": _MW["FeO"] + _MW["SiO2"],                    # FeSiO3 (ferrosilite)
    # Olivines — true moles of end-member formula unit
    "Fo": 2 * _MW["MgO"] + _MW["SiO2"],                   # Mg2SiO4
    "Fa": 2 * _MW["FeO"] + _MW["SiO2"],                   # Fe2SiO4
    # Oxides / accessories
    "Mt": _MW["Fe2O3"] + _MW["FeO"],                       # FeO·Fe2O3
    "Il": _MW["TiO2"] + _MW["FeO"],                        # FeO·TiO2
    "Hm": _MW["Fe2O3"],
    "Tn": _MW["CaO"] + _MW["TiO2"] + _MW["SiO2"],         # CaTiSiO5
    "Pf": _MW["CaO"] + _MW["TiO2"],                        # CaTiO3
    "Ru": _MW["TiO2"],
    # Ap: CIPW unit = 1 mol P2O5 consuming 10/3 mol CaO
    "Ap": _MW["P2O5"] + (10.0 / 3.0) * _MW["CaO"],
    "Cc": _MW["CaO"] + _MW["CO2"],
}

# Output mineral columns in display order
NORM_COLS: list[str] = [
    "Q", "Or", "Ab", "An", "Ne", "Lc", "Kp",
    "Di", "Hy", "Ol", "ol_Fo",
    "Mt", "Il", "Hm", "Tn", "Pf", "Ru", "Ap", "Cc",
]


# ---------------------------------------------------------------------------
# Single-row CIPW calculation
# ---------------------------------------------------------------------------

def _cipw_row(oxides: dict[str, float]) -> dict[str, float]:
    """Compute CIPW norm for a single sample.

    Parameters
    ----------
    oxides : dict
        Mapping of canonical oxide name → wt% value.  Expected keys:
        ``SiO2 TiO2 Al2O3 Fe2O3_calc FeO_calc MnO MgO CaO Na2O K2O P2O5 CO2``.

    Returns
    -------
    dict
        Mineral wt% values plus ``ol_Fo``.  Keys match ``NORM_COLS``.
    """
    # Working moles (per 100 g sample)
    si = oxides.get("SiO2", 0.0) / _MW["SiO2"]
    ti = oxides.get("TiO2", 0.0) / _MW["TiO2"]
    al = oxides.get("Al2O3", 0.0) / _MW["Al2O3"]
    fm = oxides.get("Fe2O3_calc", 0.0) / _MW["Fe2O3"]   # moles of Fe2O3
    # Combine FeO + MnO (MnO behaves like FeO in CIPW allocations)
    f = (
        oxides.get("FeO_calc", 0.0) / _MW["FeO"]
        + oxides.get("MnO", 0.0) / _MW["MnO"]
    )
    mg = oxides.get("MgO", 0.0) / _MW["MgO"]
    ca = oxides.get("CaO", 0.0) / _MW["CaO"]
    na = oxides.get("Na2O", 0.0) / _MW["Na2O"]
    k = oxides.get("K2O", 0.0) / _MW["K2O"]
    p = oxides.get("P2O5", 0.0) / _MW["P2O5"]
    co2 = oxides.get("CO2", 0.0) / _MW["CO2"]

    # -----------------------------------------------------------------------
    # Step 1 — Apatite (Ap): consumes P2O5 and 10/3 CaO per mol P2O5
    # -----------------------------------------------------------------------
    ap = min(p, ca / (10.0 / 3.0)) if ca > 0 else 0.0
    ca -= ap * (10.0 / 3.0)
    p = 0.0

    # -----------------------------------------------------------------------
    # Step 2 — Ilmenite (Il): FeO·TiO2
    # -----------------------------------------------------------------------
    il = min(ti, f)
    ti -= il
    f -= il

    # -----------------------------------------------------------------------
    # Step 3 — Magnetite (Mt): FeO·Fe2O3
    # -----------------------------------------------------------------------
    mt = min(fm, f)
    fm -= mt
    f -= mt

    # -----------------------------------------------------------------------
    # Step 4 — Hematite (Hm): remaining Fe2O3
    # -----------------------------------------------------------------------
    hm = fm
    fm = 0.0

    # -----------------------------------------------------------------------
    # Step 5 — Titanite / Perovskite / Rutile from remaining TiO2
    # -----------------------------------------------------------------------
    tn = min(ti, ca, si) if ti > 0 else 0.0
    ti -= tn
    ca -= tn
    si -= tn

    pf = min(ti, ca) if ti > 0 else 0.0
    ti -= pf
    ca -= pf

    ru = ti
    ti = 0.0

    # -----------------------------------------------------------------------
    # Step 6 — Calcite (Cc): CaO·CO2
    # -----------------------------------------------------------------------
    cc = min(co2, ca)
    ca -= cc
    co2 -= cc

    # -----------------------------------------------------------------------
    # Step 7 — Orthoclase (Or): K2O·Al2O3·6SiO2
    # -----------------------------------------------------------------------
    or_ = min(k, al)
    al -= or_
    k -= or_
    si -= or_ * 6.0

    # -----------------------------------------------------------------------
    # Step 8 — Albite (Ab): Na2O·Al2O3·6SiO2
    # -----------------------------------------------------------------------
    ab = min(na, al)
    al -= ab
    na -= ab
    si -= ab * 6.0

    # -----------------------------------------------------------------------
    # Step 9 — Anorthite (An): CaO·Al2O3·2SiO2
    # -----------------------------------------------------------------------
    an = min(ca, al)
    al -= an
    ca -= an
    si -= an * 2.0

    # -----------------------------------------------------------------------
    # Undersaturation corrections — convert feldspars / leucite → feldspathoids
    # if Si < 0.  Run in priority order: Ab → Ne, Or → Lc, Lc → Kp.
    # -----------------------------------------------------------------------
    ne = 0.0
    lc = 0.0
    kp = 0.0

    if si < -1e-12:
        # Ab → Ne: each mol releases 4 SiO2
        ne_from_ab = min(ab, (-si) / 4.0)
        ab -= ne_from_ab
        ne += ne_from_ab
        si += ne_from_ab * 4.0

    if si < -1e-12:
        # Or → Lc: each mol releases 2 SiO2
        lc_from_or = min(or_, (-si) / 2.0)
        or_ -= lc_from_or
        lc += lc_from_or
        si += lc_from_or * 2.0

    if si < -1e-12:
        # Lc → Kp: each mol releases 2 SiO2
        kp_from_lc = min(lc, (-si) / 2.0)
        lc -= kp_from_lc
        kp += kp_from_lc
        si += kp_from_lc * 2.0

    # -----------------------------------------------------------------------
    # Step 10 — Diopside (Di): remaining CaO paired with Mg then Fe
    # -----------------------------------------------------------------------
    di_mg = min(mg, ca)
    mg -= di_mg
    ca -= di_mg
    si -= di_mg * 2.0

    di_fe = min(f, ca)
    f -= di_fe
    ca -= di_fe
    si -= di_fe * 2.0

    # -----------------------------------------------------------------------
    # Step 11 — Hypersthene (Hy): remaining Mg and Fe → orthopyroxene
    # -----------------------------------------------------------------------
    hy_mg = mg
    hy_fe = f
    mg = 0.0
    f = 0.0
    si -= hy_mg + hy_fe

    # -----------------------------------------------------------------------
    # Step 11b — Hy → Ol if Si still negative
    # 2 Hy → 1 Ol + 1 SiO2  ⟹  each Hy releases 0.5 SiO2
    # -----------------------------------------------------------------------
    mol_fo = 0.0
    mol_fa = 0.0

    if si < -1e-12:
        hy_total = hy_mg + hy_fe
        hy_needed = (-si) * 2.0  # mol Hy to convert to recover -si SiO2
        hy_conv = min(hy_total, hy_needed)

        if hy_total > 1e-15:
            frac_mg = hy_mg / hy_total
            frac_fe = hy_fe / hy_total
        else:
            frac_mg = 0.5
            frac_fe = 0.5

        hy_mg_conv = hy_conv * frac_mg
        hy_fe_conv = hy_conv * frac_fe

        hy_mg -= hy_mg_conv
        hy_fe -= hy_fe_conv

        mol_fo = hy_mg_conv / 2.0   # true moles of Fo (Mg2SiO4)
        mol_fa = hy_fe_conv / 2.0   # true moles of Fa (Fe2SiO4)

        si += hy_conv * 0.5

    # -----------------------------------------------------------------------
    # Step 12 — Quartz (Q): remaining Si
    # -----------------------------------------------------------------------
    q = max(0.0, si)

    # -----------------------------------------------------------------------
    # Convert molar amounts → wt%
    # -----------------------------------------------------------------------
    wt: dict[str, float] = {
        "Q": q * _MWM["Q"],
        "Or": or_ * _MWM["Or"],
        "Ab": ab * _MWM["Ab"],
        "An": an * _MWM["An"],
        "Ne": ne * _MWM["Ne"],
        "Lc": lc * _MWM["Lc"],
        "Kp": kp * _MWM["Kp"],
        "Di": di_mg * _MWM["Di_Mg"] + di_fe * _MWM["Di_Fe"],
        "Hy": hy_mg * _MWM["Hy_Mg"] + hy_fe * _MWM["Hy_Fe"],
        "Ol": mol_fo * _MWM["Fo"] + mol_fa * _MWM["Fa"],
        "Mt": mt * _MWM["Mt"],
        "Il": il * _MWM["Il"],
        "Hm": hm * _MWM["Hm"],
        "Tn": tn * _MWM["Tn"],
        "Pf": pf * _MWM["Pf"],
        "Ru": ru * _MWM["Ru"],
        "Ap": ap * _MWM["Ap"],
        "Cc": cc * _MWM["Cc"],
    }

    # Renormalise to 100 wt%
    total = sum(wt.values())
    if total > 0:
        wt = {k: v / total * 100.0 for k, v in wt.items()}

    # Forsterite fraction of normative olivine
    ol_fo_val = mol_fo / (mol_fo + mol_fa) if (mol_fo + mol_fa) > 1e-15 else np.nan
    wt["ol_Fo"] = ol_fo_val

    return wt


# ---------------------------------------------------------------------------
# DataFrame-level entry point
# ---------------------------------------------------------------------------

def compute_cipw(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CIPW normative mineralogy for every row in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Fe2O3_calc`` and ``FeO_calc`` (from
        :func:`src.fe_partition.partition_fe`) plus the standard oxide columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with CIPW mineral columns appended.  Mineral wt%
        columns are rounded to 4 decimal places; ``ol_Fo`` is unrounded
        (NaN when olivine is absent).
    """
    oxide_keys = [
        "SiO2", "TiO2", "Al2O3", "Fe2O3_calc", "FeO_calc",
        "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "CO2",
    ]

    def _apply_row(row: pd.Series) -> pd.Series:
        oxide_dict = {k: row.get(k, 0.0) for k in oxide_keys}
        result = _cipw_row(oxide_dict)
        return pd.Series(result)

    norm_df = df.apply(_apply_row, axis=1)

    # Round mineral wt% columns (not ol_Fo, which is a ratio)
    wt_cols = [c for c in NORM_COLS if c != "ol_Fo"]
    norm_df[wt_cols] = norm_df[wt_cols].round(4)

    return pd.concat([df.reset_index(drop=True), norm_df.reset_index(drop=True)], axis=1)
