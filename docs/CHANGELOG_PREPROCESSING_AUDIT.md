# Changelog: Dataset Preprocessing & Schema Audit

**Project:** Congestion-Aware Transformer-XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Date:** 2026-10-11  

---

## 1. Summary of Changes

### A. Dataset Builder (`src/build_processed_dataset.py`)
- **Schema Alignment:** Corrected the hardcoded PyArrow schema from the non-compliant float/int8 specification to the canonical project schema:
  - `sender_id`, `sender_alias`, `sender_driver_profile`, `receiver_driver_profile`: `pa.large_string()`
  - `message_id`, `send_time`, `receive_time`, `delay`, `label`: `pa.int64()`
  - Kinematic vehicle features and `distance_to_road_edge`: `pa.float64()`
- **Precision Preservation:** Replaced floating-point conversions for timestamps and message IDs with `to_int` helper, preserving full integer nanosecond precision and avoiding floating-point rounding.
- **Critical Overwrite Safety Guard:** Added explicit CLI argument parsing (`--condition`, `--split`, `--test-mode`, `--overwrite`). The builder now **strictly refuses to run** if the target output file exists, unless the `--overwrite` flag is explicitly provided.
- **Atomic Validation:** Parquet files are written to `.tmp` first, validated for row counts, schema matching, and label presence, and only then promoted to the final target path. Temporary files are safely cleaned up on failure.

### B. Schema & Split Validator (`src/check_processed_splits.py`)
- **Comprehensive Verification:** Updated the validator to check:
  - Exact column ordering and name matching (28 columns).
  - Exact data type matching across all six primary splits (`highway_2` and `highway_7` across train, validation, and test).
  - Delay integrity verification ($\text{delay} == \text{receive\_time} - \text{send\_time}$).
  - Non-zero row count validation.
  - Proper exit codes (`0` on PASS, `1` on FAIL) for CI/scripting integration.

### C. Data Quality Audit Script (`src/audit_data_quality.py`)
- **Batch-Based Architecture:** Implemented a memory-efficient, strictly read-only auditor that processes large datasets (such as the 7.3M-row `highway_7/test.parquet`) in 250k-row chunks without exceeding memory limits.
- **Multi-Faceted Audit:** Evaluated schema integrity, missingness breakdown by scenario, numeric distributions, timestamp precision, full row duplicates, multi-receiver broadcast reception multiplicity, and cross-split leakage.
- **Machine-Readable Outputs:** Saved structured results under `results/data_quality_audit/`:
  - `file_summary.csv`
  - `column_quality.csv`
  - `numeric_and_label_summary.csv`
  - `cross_split_leakage.csv`
  - `profile_values.json`
  - `audit_report.md`

### D. Documentation Deliverables
- Created `docs/preprocessing_and_feature_policy.md` detailing feature roles, leakage hazards, `distance_to_road_edge` quarantine, and Transformer vs. XGBoost requirements.
- Created `docs/CHANGELOG_PREPROCESSING_AUDIT.md` (this file).

---

## 2. Root Cause Analysis of Schema Mismatch

### Investigation Findings
1. **Initial Pipeline (`889ba09`):** Datasets were initially generated using pandas (`dataframe.to_parquet()`). Pandas serialized integers as `int64`, floats as `double`, and strings as `large_string` with pandas metadata.
2. **Streaming Builder Refactor (`268ebf8`):** In commit `268ebf8`, `src/build_processed_dataset.py` was rewritten to use PyArrow streaming to handle large datasets. However, the author manually declared a new PyArrow schema with `message_id: string`, `send_time: float64`, `receive_time: float64`, `delay: float64`, and `label: int8`.
3. **Emergency Fix via `fix_test_schema.py`:** When `highway_7/test.parquet` was generated with this mismatched schema, the author created `fix_test_schema.py` to cast `test.parquet` back to `highway_7/train.parquet`'s schema, saving the uncast version as `test_before_schema_fix.parquet`.
4. **Unresolved Gap:** Although the parquet file was cast, `build_processed_dataset.py` was left uncorrected and configured to overwrite `highway_7/test.parquet` without overwrite guards.
5. **Resolution:** The builder has now been updated to generate the canonical schema directly, with active protection against accidental overwrites.

---

## 3. Data Protection Confirmation

- **Zero Existing Datasets Modified:** No files in `data/raw/` or `processed_data/` were modified, overwritten, regenerated, renamed, or deleted.
- **Existing Backups Preserved:** `processed_data/highway_7/test_before_schema_fix.parquet` and `test_sample.parquet` remain untouched.
- **Audit Directory Isolation:** All new audit results were saved under `results/data_quality_audit/`, preserving previous results in `results/data_audit/`.

---

## 4. Unresolved Findings and Open Decisions

1. **Scenario Concatenation Overlap:**
   - In both `highway_2` and `highway_7`, the Normal scenario (`InTAS_highway_X`) and DoS scenario (`InTAS_highway_X_dosAttack`) were generated from identical simulation seeds.
   - Merging both files into the same split produced identical broadcast transmissions with conflicting labels: legitimate in Normal (`label = 0`), but malicious in DoS (`label = 1`) for attacking vehicles (~20% of traffic).
   - *Decision Needed:* Whether to train on the combined dataset as currently stored, or introduce a deduplication / split strategy separating pure baseline congestion from the attack simulation.
2. **`distance_to_road_edge` Exclusion:**
   - The feature is 100% missing from Normal scenario records and present in DoS records. As documented in `docs/preprocessing_and_feature_policy.md`, this feature must be completely excluded from the model feature matrix to prevent trivial scenario leakage.
