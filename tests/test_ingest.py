"""Tests for src/ingest.py."""

import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    OXIDE_COLS,
    _make_sample_id,
    _rock_type_from_filename,
    _standardize_columns,
    clean_and_renorm,
    load_georoc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_raw() -> pd.DataFrame:
    """A minimal DataFrame mimicking a parsed GEOROC row."""
    return pd.DataFrame({
        "SAMPLE NAME": ["s_TEST [1]"],
        "ROCK NAME": ["Basalt"],
        "ROCK TYPE": ["VOL"],
        "SIO2(WT%)": [49.0],
        "TIO2(WT%)": [2.0],
        "AL2O3(WT%)": [14.0],
        "FE2O3(WT%)": [2.0],
        "FEO(WT%)": [7.0],
        "FEOT(WT%)": [np.nan],
        "MGO(WT%)": [8.0],
        "CAO(WT%)": [10.0],
        "NA2O(WT%)": [3.0],
        "K2O(WT%)": [1.5],
        "P2O5(WT%)": [0.3],
        "MNO(WT%)": [0.15],
        "CO2(WT%)": [0.0],
    })


@pytest.fixture()
def tmp_csv_dir(tmp_path: pathlib.Path, minimal_raw: pd.DataFrame) -> pathlib.Path:
    """Write a synthetic GEOROC CSV to a temp directory."""
    csv_path = tmp_path / "2025-12-2JETOA_BASALT_part1.csv"
    minimal_raw.to_csv(csv_path, index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestRockTypeFromFilename:
    def test_basic(self):
        p = pathlib.Path("2025-12-2JETOA_BASALT_part1.csv")
        assert _rock_type_from_filename(p) == "BASALT"

    def test_multiword(self):
        p = pathlib.Path("2025-12-2JETOA_ALKALI_BASALT.csv")
        assert _rock_type_from_filename(p) == "ALKALI_BASALT"

    def test_no_part_suffix(self):
        p = pathlib.Path("2025-12-2JETOA_ANKARAMITE.csv")
        assert _rock_type_from_filename(p) == "ANKARAMITE"


class TestStandardizeColumns:
    def test_renames_oxides(self, minimal_raw):
        result = _standardize_columns(minimal_raw)
        assert "SiO2" in result.columns
        assert "TiO2" in result.columns
        assert "Fe2O3" in result.columns
        assert "FeO" in result.columns
        assert "FeOT" in result.columns
        # Raw GEOROC names should be gone
        assert "SIO2(WT%)" not in result.columns

    def test_unknown_columns_dropped(self, minimal_raw):
        minimal_raw["UNKNOWN_COL"] = 99
        result = _standardize_columns(minimal_raw)
        assert "UNKNOWN_COL" not in result.columns

    def test_meta_cols_renamed(self, minimal_raw):
        result = _standardize_columns(minimal_raw)
        assert "sample_name" in result.columns
        assert "rock_name" in result.columns


class TestMakeSampleId:
    def test_prefix(self):
        sid = _make_sample_id("file.csv", "sample_A", 0)
        assert sid.startswith("ORS_")

    def test_stable(self):
        a = _make_sample_id("file.csv", "sample_A", 0)
        b = _make_sample_id("file.csv", "sample_A", 0)
        assert a == b

    def test_unique(self):
        a = _make_sample_id("file.csv", "sample_A", 0)
        b = _make_sample_id("file.csv", "sample_B", 0)
        assert a != b


class TestLoadGeoroc:
    def test_returns_dataframe(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        assert isinstance(df, pd.DataFrame)

    def test_rock_type_tagged(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        assert "rock_type" in df.columns
        assert df["rock_type"].iloc[0] == "BASALT"

    def test_sample_id_present(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        assert "sample_id" in df.columns
        assert df["sample_id"].iloc[0].startswith("ORS_")

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_georoc(tmp_path)


class TestCleanAndRenorm:
    def test_drops_zero_sio2(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        df_bad = df.copy()
        df_bad["SiO2"] = 0.0
        result = clean_and_renorm(df_bad)
        assert len(result) == 0

    def test_oxide_total_recorded(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        result = clean_and_renorm(df)
        assert "oxide_total_raw" in result.columns
        assert result["oxide_total_raw"].iloc[0] > 0

    def test_renorm_sums_to_100(self, tmp_csv_dir):
        from src.ingest import _RENORM_OXIDES
        df = load_georoc(tmp_csv_dir)
        result = clean_and_renorm(df)
        totals = result[_RENORM_OXIDES].sum(axis=1)
        assert (abs(totals - 100.0) < 1e-6).all()

    def test_missing_oxide_filled_zero(self, tmp_csv_dir):
        df = load_georoc(tmp_csv_dir)
        df["CO2"] = np.nan
        result = clean_and_renorm(df)
        assert (result["CO2"] == 0.0).all()
