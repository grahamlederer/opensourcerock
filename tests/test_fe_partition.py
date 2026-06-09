"""Tests for src/fe_partition.py."""

import numpy as np
import pandas as pd
import pytest

from src.fe_partition import partition_fe

_MW_FEO = 71.844
_MW_FE2O3 = 159.688
_FE2O3_TO_FEO = 2 * _MW_FEO / _MW_FE2O3


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame for partition_fe testing."""
    defaults = {
        "SiO2": [50.0],
        "Fe2O3": [0.0],
        "FeO": [0.0],
        "FeOT": [10.0],
        "sample_id": ["ORS_test01"],
        "rock_type": ["BASALT"],
        "source_file": ["test.csv"],
        "oxide_total_raw": [97.0],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestPartitionFeErrors:
    def test_invalid_method(self):
        df = _make_df()
        with pytest.raises(ValueError, match="method must be"):
            partition_fe(df, method="invalid")


class TestPartitionFeFixedRatio:
    def test_fe2o3_fraction(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="fixed_ratio")
        fe2o3 = result["Fe2O3_calc"].iloc[0]
        feot = 10.0
        assert abs(fe2o3 / feot - 0.15) < 1e-10

    def test_feot_conserved(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="fixed_ratio")
        fe2o3 = result["Fe2O3_calc"].iloc[0]
        feo = result["FeO_calc"].iloc[0]
        reconstructed_feot = feo + fe2o3 * _FE2O3_TO_FEO
        assert abs(reconstructed_feot - 10.0) < 1e-8

    def test_originals_preserved(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="fixed_ratio")
        assert result["Fe2O3_original"].iloc[0] == 0.0
        assert result["FeO_original"].iloc[0] == 0.0


class TestPartitionFeMiddlemost:
    def test_ratio(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="middlemost")
        fe2o3 = result["Fe2O3_calc"].iloc[0]
        feo = result["FeO_calc"].iloc[0]
        assert abs(fe2o3 / feo - 0.15) < 1e-8

    def test_feot_conserved(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="middlemost")
        fe2o3 = result["Fe2O3_calc"].iloc[0]
        feo = result["FeO_calc"].iloc[0]
        reconstructed_feot = feo + fe2o3 * _FE2O3_TO_FEO
        assert abs(reconstructed_feot - 10.0) < 1e-8


class TestPartitionFeKressCarmichael:
    def test_feot_conserved(self):
        df = _make_df(FeOT=[10.0], SiO2=[55.0])
        result = partition_fe(df, method="kress_carmichael")
        fe2o3 = result["Fe2O3_calc"].iloc[0]
        feo = result["FeO_calc"].iloc[0]
        reconstructed_feot = feo + fe2o3 * _FE2O3_TO_FEO
        assert abs(reconstructed_feot - 10.0) < 1e-8

    def test_more_oxidised_at_high_sio2(self):
        df_lo = _make_df(FeOT=[10.0], SiO2=[45.0])
        df_hi = _make_df(FeOT=[10.0], SiO2=[70.0])
        result_lo = partition_fe(df_lo, method="kress_carmichael")
        result_hi = partition_fe(df_hi, method="kress_carmichael")
        assert result_hi["Fe2O3_calc"].iloc[0] > result_lo["Fe2O3_calc"].iloc[0]


class TestDirectIronPassthrough:
    def test_both_fe_columns_not_repartitioned(self):
        df = _make_df(Fe2O3=[2.5], FeO=[7.5], FeOT=[9.25])
        result = partition_fe(df, method="fixed_ratio")
        assert result["Fe2O3_calc"].iloc[0] == pytest.approx(2.5)
        assert result["FeO_calc"].iloc[0] == pytest.approx(7.5)
        assert result["fe_partition_method"].iloc[0] == "direct"

    def test_feot_only_uses_method(self):
        df = _make_df(Fe2O3=[0.0], FeO=[0.0], FeOT=[10.0])
        result = partition_fe(df, method="fixed_ratio")
        assert result["fe_partition_method"].iloc[0] == "fixed_ratio"


class TestFe2Fetotal:
    def test_range(self):
        df = _make_df(FeOT=[10.0])
        result = partition_fe(df, method="fixed_ratio")
        val = result["fe2_fetotal"].iloc[0]
        assert 0.0 <= val <= 1.0

    def test_nan_when_no_iron(self):
        df = _make_df(Fe2O3=[0.0], FeO=[0.0], FeOT=[0.0])
        result = partition_fe(df, method="fixed_ratio")
        assert np.isnan(result["fe2_fetotal"].iloc[0])
