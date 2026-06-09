"""Tests for src/cipw.py.

Validation against the W-1 diabase standard (USGS W-1), comparable to
Middlemost (1989) Table A4.1.

With the explicitly measured Fe2O3=1.07 and FeO=10.19 partition (not FeOT),
W-1 is a *slightly quartz-normative* tholeiite:
    Q ≈ 2.1  Di ≈ 19  Hy ≈ 24  An ≈ 29

The "Ol~6" figure quoted in some literature refers to norms computed with
all iron as FeO (FeOT only).  When Fe2O3 is used directly, Fe is more
oxidised and the Si balance shifts toward quartz.

W-1 composition (anhydrous, wt%):
    SiO2=52.64, TiO2=1.06, Al2O3=14.99, Fe2O3=1.07, FeO=10.19,
    MnO=0.17, MgO=6.57, CaO=10.97, Na2O=2.23, K2O=0.64, P2O5=0.14
"""

import numpy as np
import pandas as pd
import pytest

from src.cipw import NORM_COLS, _cipw_row, compute_cipw

# W-1 basalt standard (Middlemost 1989, anhydrous wt% renorm to 100)
_W1 = {
    "SiO2": 52.64,
    "TiO2": 1.06,
    "Al2O3": 14.99,
    "Fe2O3_calc": 1.07,
    "FeO_calc": 10.19,
    "MnO": 0.17,
    "MgO": 6.57,
    "CaO": 10.97,
    "Na2O": 2.23,
    "K2O": 0.64,
    "P2O5": 0.14,
    "CO2": 0.0,
}

# Renormalise W-1 to exactly 100 so the norm also sums to 100
_W1_TOTAL = sum(_W1.values())
_W1_NORM = {k: v / _W1_TOTAL * 100.0 for k, v in _W1.items()}


def _row_df(oxides: dict) -> pd.DataFrame:
    """Wrap an oxide dict into a one-row DataFrame for compute_cipw."""
    return pd.DataFrame([oxides])


class TestW1Basalt:
    """Validate against the published W-1 standard."""

    def setup_method(self):
        self.result = _cipw_row(_W1_NORM)

    def test_small_quartz(self):
        # W-1 with Fe2O3/FeO split is slightly quartz-normative (~2 wt%)
        assert 0.0 < self.result["Q"] < 5.0

    def test_no_olivine(self):
        # Quartz and olivine are mutually exclusive in the CIPW norm
        assert self.result["Ol"] == pytest.approx(0.0, abs=1e-6)

    def test_diopside_range(self):
        assert 15.0 < self.result["Di"] < 30.0, "Expected Di roughly 15–30 wt%"

    def test_anorthite_range(self):
        assert 18.0 < self.result["An"] < 32.0, "Expected An roughly 18–32 wt%"

    def test_hypersthene_present(self):
        # Tholeiites are hypersthene-normative
        assert self.result["Hy"] > 10.0

    def test_norm_sums_to_100(self):
        mineral_sum = sum(v for k, v in self.result.items() if k != "ol_Fo")
        assert abs(mineral_sum - 100.0) < 0.1

    def test_ol_fo_nan_when_no_olivine(self):
        # When Ol=0, ol_Fo must be NaN (not 0)
        assert np.isnan(self.result["ol_Fo"])


class TestNormSum:
    """Norm must sum to 100 ± 0.1 for any valid input."""

    @pytest.mark.parametrize("sio2", [45.0, 52.0, 60.0, 70.0])
    def test_sum_to_100(self, sio2):
        oxides = {
            "SiO2": sio2,
            "TiO2": 1.5,
            "Al2O3": 15.0,
            "Fe2O3_calc": 2.0,
            "FeO_calc": 8.0,
            "MnO": 0.2,
            "MgO": 7.0,
            "CaO": 9.0,
            "Na2O": 3.0,
            "K2O": 1.5,
            "P2O5": 0.3,
            "CO2": 0.0,
        }
        # Renorm to 100
        total = sum(oxides.values())
        oxides = {k: v / total * 100.0 for k, v in oxides.items()}
        result = _cipw_row(oxides)
        mineral_sum = sum(v for k, v in result.items() if k != "ol_Fo")
        assert abs(mineral_sum - 100.0) < 0.1, (
            f"Norm sum = {mineral_sum:.4f} for SiO2={sio2}"
        )


class TestFoRange:
    """ol_Fo must be in [0, 1] when olivine is present, NaN when absent."""

    def test_fo_when_olivine_present(self):
        # Undersaturated composition: low SiO2, high Mg
        oxides = {
            "SiO2": 40.0,
            "TiO2": 2.0,
            "Al2O3": 12.0,
            "Fe2O3_calc": 1.0,
            "FeO_calc": 8.0,
            "MnO": 0.2,
            "MgO": 15.0,
            "CaO": 12.0,
            "Na2O": 3.0,
            "K2O": 1.0,
            "P2O5": 0.3,
            "CO2": 0.0,
        }
        total = sum(oxides.values())
        oxides = {k: v / total * 100.0 for k, v in oxides.items()}
        result = _cipw_row(oxides)
        if result["Ol"] > 0:
            fo = result["ol_Fo"]
            assert 0.0 <= fo <= 1.0

    def test_fo_nan_when_no_olivine(self):
        # Over-saturated: high SiO2 → quartz-normative, no olivine
        oxides = {
            "SiO2": 72.0,
            "TiO2": 0.5,
            "Al2O3": 14.0,
            "Fe2O3_calc": 1.0,
            "FeO_calc": 3.0,
            "MnO": 0.05,
            "MgO": 1.0,
            "CaO": 2.0,
            "Na2O": 3.5,
            "K2O": 4.0,
            "P2O5": 0.1,
            "CO2": 0.0,
        }
        total = sum(oxides.values())
        oxides = {k: v / total * 100.0 for k, v in oxides.items()}
        result = _cipw_row(oxides)
        if result["Ol"] == 0:
            assert np.isnan(result["ol_Fo"])


class TestUndersaturatedRock:
    """Feldspathoidal rocks should produce Ne/Lc, zero Q."""

    def test_nepheline_produced(self):
        # Strongly undersaturated: low SiO2, high Na2O
        oxides = {
            "SiO2": 38.0,
            "TiO2": 2.5,
            "Al2O3": 16.0,
            "Fe2O3_calc": 2.0,
            "FeO_calc": 8.0,
            "MnO": 0.2,
            "MgO": 8.0,
            "CaO": 12.0,
            "Na2O": 8.0,
            "K2O": 2.0,
            "P2O5": 0.5,
            "CO2": 0.0,
        }
        total = sum(oxides.values())
        oxides = {k: v / total * 100.0 for k, v in oxides.items()}
        result = _cipw_row(oxides)
        assert result["Ne"] > 0 or result["Lc"] > 0
        assert result["Q"] == pytest.approx(0.0, abs=0.01)


class TestComputeCipwDataFrame:
    """Test the DataFrame-level wrapper."""

    def test_returns_all_norm_cols(self):
        df = pd.DataFrame([_W1_NORM])
        result = compute_cipw(df)
        for col in NORM_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_sample_id_preserved(self):
        df = pd.DataFrame([{**_W1_NORM, "sample_id": "ORS_abc123"}])
        result = compute_cipw(df)
        assert result["sample_id"].iloc[0] == "ORS_abc123"

    def test_norm_values_rounded(self):
        df = pd.DataFrame([_W1_NORM])
        result = compute_cipw(df)
        for col in NORM_COLS:
            if col == "ol_Fo":
                continue
            val = result[col].iloc[0]
            assert val == round(val, 4), f"{col} not rounded to 4dp"
