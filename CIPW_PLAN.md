# CIPW normative mineralogy — project plan

## Goal

Calculate CIPW normative mineralogy from bulk geochemical data downloaded from
GEOROC. The primary outputs of interest are:

- **Normative olivine mode** (wt% of the CIPW norm)
- **Olivine composition** (forsterite fraction: Fo = Mg / (Mg + Fe) molar)
- **Fe²⁺/Fe_total** (molar, derived from the Fe redox partition step)
- **Magnesium number** (Mg# = Mg / (Mg + Fe²⁺) molar)
- A **flat output CSV** with one row per sample and one column per normative
  mineral, plus the derived metrics above

Input data lives in `dataverse_files/` as individual CSVs, one per rock type.

---

## Repository layout

```
opensourcerock/
├── dataverse_files/        # raw GEOROC downloads (do not modify)
├── data/
│   └── processed/          # intermediate and final outputs
├── src/
│   ├── __init__.py
│   ├── ingest.py           # load + standardise GEOROC CSVs
│   ├── fe_partition.py     # split FeOT → FeO + Fe2O3; compute Fe2/Fetotal
│   ├── cipw.py             # core CIPW norm algorithm
│   └── utils.py            # Mg#, renormalisation, ID generation helpers
├── tests/
│   ├── test_ingest.py
│   ├── test_fe_partition.py
│   └── test_cipw.py        # validate against published norm examples
├── notebooks/
│   └── 01_explore.ipynb    # EDA and spot-checks
├── run_norm.py             # top-level entry point (CLI)
├── requirements.txt
├── .gitignore
└── CIPW_PLAN.md            # this file
```

---

## Workflow — step by step

### Step 1 — Ingest (`src/ingest.py`)

- Glob all `*.csv` files in `dataverse_files/`
- For each file, tag rows with a `rock_type` column derived from the filename
- Rename GEOROC columns to canonical oxide names:
  `SiO2 TiO2 Al2O3 Fe2O3 FeO FeOT MnO MgO CaO Na2O K2O P2O5 CO2`
  (GEOROC uses inconsistent capitalisations and unit suffixes — handle all variants)
- Concatenate into one DataFrame
- Assign a stable `sample_id` (hash of source file + sample name + row index,
  prefixed `ORS_`)
- Drop rows where SiO2 is missing or zero

### Step 2 — Clean and renormalise (`src/ingest.py`)

- Coerce all oxide columns to numeric; fill missing oxides with 0
- Renormalise major oxides (anhydrous, volatile-free) to sum to 100 wt%
- Record original total before renormalisation as `oxide_total_raw`

### Step 3 — Fe redox partition (`src/fe_partition.py`)

GEOROC data often reports only total iron as `FeOT`. The CIPW norm requires
separate FeO and Fe2O3. Implement three methods and let the user choose via a
CLI flag:

| Method | Description |
|---|---|
| `fixed_ratio` | Fe2O3 / FeOT (wt) = 0.15 (default; conservative) |
| `middlemost` | Fe2O3 / FeO = 0.15 after Middlemost (1989) |
| `kress_carmichael` | Oxidation state approximated from SiO2 content |

After partitioning, compute:

- `fe2_fetotal` = Fe²⁺ / (Fe²⁺ + Fe³⁺) on a molar basis
- Store `Fe2O3_calc` and `FeO_calc` as the working columns

### Step 4 — Mg number (`src/utils.py`)

```
Mg# = Mg_mol / (Mg_mol + Fe2_mol)
```

where Fe2 comes from the partitioned FeO column.

### Step 5 — CIPW norm (`src/cipw.py`)

Implement the standard CIPW algorithm (Le Bas & Streckeisen 1991; Verma et al. 2002):

1. Convert oxide wt% → molar proportions using standard molecular weights
2. Allocate minerals in CIPW priority order:
   `Ap → Il → Mt → Hm → Tn/Pf/Ru → Cc → Or → Ab → An → Di → Hy → Ol → Q`
3. If Si goes negative after Or/Ab/An allocation, convert:
   - Ab → Ne (albite to nepheline; releases 4 Si/mol)
   - Or → Lc (orthoclase to leucite; releases 2 Si/mol)
   - Hy → Ol (hypersthene to olivine; releases 0.5 Si/mol per Hy)
4. Convert molar mineral amounts back to wt% using end-member molecular weights
5. Renormalise norm to 100 wt%

**Olivine-specific outputs:**

- `ol_mode` — olivine wt% of the norm (column `Ol`)
- `ol_Fo` — forsterite content of normative olivine = Mg_ol / (Mg_ol + Fe_ol) molar

Normative minerals to output as columns:

`Q Or Ab An Ne Lc Kp Di Hy Ol Mt Il Hm Tn Pf Ru Ap Cc`

### Step 6 — Validation (`tests/`)

- Test against the worked example in Middlemost (1989) Table A4.1
  (basalt standard W-1; expected: Q=0, Ol~6, Di~22, An~25)
- Test that norm totals = 100 ± 0.1 for all samples
- Test that Fo is in [0, 1]
- Test Fe2/Fetotal is in [0, 1]

### Step 7 — Export (`run_norm.py`)

Final flat file: `data/processed/cipw_norm_output.csv`

Column order:

```
sample_id  rock_type  source_file
SiO2 TiO2 Al2O3 Fe2O3_calc FeO_calc MnO MgO CaO Na2O K2O P2O5 CO2
oxide_total_raw  fe2_fetotal  mg_number
Q Or Ab An Ne Lc Kp Di Hy Ol ol_Fo Mt Il Hm Tn Pf Ru Ap Cc
```

---

## Key references

- Le Bas, M.J. & Streckeisen, A.L. (1991). The IUGS systematics of igneous rocks.
  *Journal of the Geological Society*, 148, 825–833.
- Middlemost, E.A.K. (1989). *Magmas and Magmatic Rocks*. Appendix 4.
- Verma, S.P. et al. (2002). SINCLAS computer program. *Computers & Geosciences*.
- Kress, V.C. & Carmichael, I.S.E. (1991). The compressibility of silicate liquids.
  *Contributions to Mineralogy and Petrology*, 108, 82–92.

---

## Molecular weights to use (IUPAC 2021)

| Oxide | MW (g/mol) |
|-------|-----------|
| SiO2  | 60.084 |
| TiO2  | 79.866 |
| Al2O3 | 101.961 |
| Fe2O3 | 159.688 |
| FeO   | 71.844 |
| MnO   | 70.937 |
| MgO   | 40.304 |
| CaO   | 56.077 |
| Na2O  | 61.979 |
| K2O   | 94.196 |
| P2O5  | 141.944 |

---

## CLI usage (target)

```bash
# default Fe partition method (fixed_ratio)
python run_norm.py --data dataverse_files/ --output data/processed/

# specify Fe partition method
python run_norm.py --data dataverse_files/ --fe-method kress_carmichael

# run tests
pytest tests/
```

---

## Notes for Claude Code

- Keep each source file under ~300 lines; split if larger
- All functions must have numpy-style docstrings
- Use `pandas` + `numpy` only (no external geochemistry libraries)
- `sample_id` is the primary key throughout — never drop it
- Preserve `rock_type` and `source_file` columns in every intermediate DataFrame
- The Fe partition step must be reversible / auditable: keep both
  `FeO_original`, `Fe2O3_original` and the computed `FeO_calc`, `Fe2O3_calc`
- Olivine Fo and Mg# should be `NaN` (not 0) when olivine is absent or iron
  is zero, respectively
- Do not round intermediate values; only round the final output to 4 decimal places
