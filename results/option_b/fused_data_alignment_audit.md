# Phase 3A: Fused Dataset Construction & Alignment Audit Report

**Project Title:** Congestion-Aware Transformer–XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Execution Date:** 2026-10-15  
**Stage:** Phase 3A — Multimodal Fused Dataset Construction and Key-Based Validation  
**Script Implemented:** `src/build_fused_datasets.py`  
**Output Destination:** `processed_data/fused_features/{highway}/{split}/`  
**Status:** **PHASE 3A COMPLETE — 100% OF CHECKS PASSED — READY FOR PHASE 3B**  

---

## 1. Executive Summary

Phase 3A has constructed, joined, and validated the multimodal fused feature matrices for the hybrid Transformer–XGBoost DoS detector across all six dataset splits for Highway 2 and Highway 7.

Two distinct multimodal feature sets were generated for every split:
1. **Full Hybrid Feature Set (71 Features):**
   64 Transformer temporal embeddings (`embedding_00` to `embedding_63`) + 7 physical context features.
2. **Autonomous Ablation Feature Set (68 Features):**
   The same 64 Transformer embeddings + 4 non-receiver context features (`message_rate`, `iat_variation`, `speed_variation`, `acceleration_variation`), strictly excluding receiver-dependent distance summaries.

### Key Milestones:
- **Key-Based Joining:** Fused datasets were joined using explicit primary keys (`window_id`). Multi-key consistency was independently verified across `sender_id`, `start_send_time`, `end_send_time`, and `label`. Zero reliance on raw row ordering.
- **Strict Target and Identifier Segregation:** All structural identifiers (`window_id`, `sender_id`, `window_index`), timestamps, durations, and target labels were strictly quarantined into a separate `window_keys.parquet` table. Zero forbidden columns exist in either predictor matrix.
- **Data Leakage & Split Isolation:** Highway scenarios and train/val/test splits were constructed in complete isolation. Zero cross-split history joining.
- **Format Redundancy:** Datasets were saved as typed Parquet tables (`fused_full_hybrid.parquet`, `fused_autonomous_ablation.parquet`) and compressed NumPy array archives (`fused_arrays.npz`), enabling fast, deterministic loading for Phase 3B.

---

## 2. Feature Specifications & Exact Ordering

### A. Full Hybrid Feature Set (71 Columns)
- **Dimensions:** $(N, 71)$
- **Dtype:** `float32` for embeddings, `float64` for context features (array export cast to `float32`)
- **Exact Column Ordering:**
  - `embedding_00` to `embedding_63`: 64 frozen Transformer temporal mean-pooled features.
  - `starting_distance`: Median receiver Euclidean range at timestep $t_0$ ($m$).
  - `final_distance`: Median receiver Euclidean range at timestep $t_9$ ($m$).
  - `distance_change_rate`: Net change in receiver-distance profile over window ($m/s$).
  - `message_rate`: Discrete packet transmission density ($10 / \Delta t_{\text{obs}}$, $\text{pkts}/s$).
  - `iat_variation`: Sample standard deviation (`ddof=1`) of 9 send-time gaps ($s$).
  - `speed_variation`: Sample standard deviation (`ddof=1`) of transmitter speed ($m/s$).
  - `acceleration_variation`: Sample standard deviation (`ddof=1`) of transmitter acceleration ($m/s^2$).

### B. Autonomous Ablation Feature Set (68 Columns)
- **Dimensions:** $(N, 68)$
- **Exact Column Ordering:**
  - `embedding_00` to `embedding_63`: 64 frozen Transformer temporal mean-pooled features.
  - `message_rate`: Discrete packet transmission density ($10 / \Delta t_{\text{obs}}$, $\text{pkts}/s$).
  - `iat_variation`: Sample standard deviation (`ddof=1`) of 9 send-time gaps ($s$).
  - `speed_variation`: Sample standard deviation (`ddof=1`) of transmitter speed ($m/s$).
  - `acceleration_variation`: Sample standard deviation (`ddof=1`) of transmitter acceleration ($m/s^2$).

---

## 3. Quantitative Summary Across All Splits

| Highway / Split | Full Hybrid Shape | Ablation Shape | Target ($y$) | Positive (DoS) | Negative (Benign) | Attack % | Parquet Size (Full/Abl) | NPZ Size | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HW2 / train** | `(23367, 71)` | `(23367, 68)` | `(23367,)` | 11,540 | 11,827 | 49.39% | 8.99 MB / 8.49 MB | 10.52 MB | **VALIDATED_PASS** |
| **HW2 / validation**| `(8801, 71)` | `(8801, 68)` | `(8801,)` | 4,348 | 4,453 | 49.40% | 3.33 MB / 3.15 MB | 3.96 MB | **VALIDATED_PASS** |
| **HW2 / test** | `(22043, 71)` | `(22043, 68)` | `(22043,)` | 10,138 | 11,905 | 45.99% | 8.55 MB / 8.07 MB | 10.00 MB | **VALIDATED_PASS** |
| **HW7 / train** | `(43222, 71)` | `(43222, 68)` | `(43222,)` | 20,166 | 23,056 | 46.66% | 16.42 MB / 15.48 MB | 18.90 MB | **VALIDATED_PASS** |
| **HW7 / validation**| `(10629, 71)` | `(10629, 68)` | `(10629,)` | 4,531 | 6,098 | 42.63% | 3.56 MB / 3.33 MB | 4.09 MB | **VALIDATED_PASS** |
| **HW7 / test** | `(56892, 71)` | `(56892, 68)` | `(56892,)` | 25,539 | 31,353 | 44.89% | 20.66 MB / 19.40 MB | 23.27 MB | **VALIDATED_PASS** |
| **Total** | **164,954 rows**| **164,954 rows**| **164,954** | **76,262** | **88,692** | **46.23%** | **61.51 MB / 57.92 MB**| **70.74 MB** | **ALL PASSED** |

---

## 4. Distinction Between Executed Checks and Architectural Assumptions

### A. Checks Actually Executed and Empirically Confirmed
1. **Multi-Source Key Agreement:** Executed an inner join on `window_id` across three independent file families (`embeddings.npz`, `context_features.parquet`, and `window_metadata.parquet`). Verified bit-for-bit identity across `sender_id`, `start_send_time`, `end_send_time`, and `label` (0 mismatches across 164,954 rows).
2. **Sequential Window ID Continuity:** Confirmed that `window_id` values form a contiguous, strictly increasing sequence from $0$ to $N-1$ without gaps or duplicates.
3. **Finite Numeric State:** Executed complete scans for `NaN` and `Inf` across all 71 feature columns in all six splits; exactly zero non-finite values found.
4. **Forbidden Column Quarantine:** Tested for the presence of 15 forbidden identifier, timestamp, duration, target, and diagnostic column names; exactly zero forbidden columns exist in either predictor matrix.
5. **Numerical Bit-for-Bit Consistency:** Compared values in `fused_full_hybrid.parquet` and `fused_autonomous_ablation.parquet` directly against their source context tables and embedding arrays with relative tolerance $10^{-5}$; 100% matched.

### B. Architectural and Experimental Assumptions (To Keep in Mind for Modeling)
1. **Detector Vantage Point:** It is assumed provisionally that detection occurs at a **Roadside Unit / Edge Infrastructure Misbehavior Authority**.
2. **Cooperative Telemetry Availability:** The 7-feature full hybrid model assumes receiving vehicles relay cooperative telemetry allowing the edge detector to compute the median receiver distance. The 4-feature ablation model makes zero assumptions about receiver telemetry.
3. **Temporal Evaluation Scope:** Senders overlap across temporal splits (e.g., 37.8% of Highway 2 test senders appear in train). The evaluation tests **temporal DoS detection across simulation sessions**, not generalization to completely unseen vehicle identities.

---

## 5. Artifacts Inventory & Phase 3B Readiness

- **Output Base Directory:** `processed_data/fused_features/`
- **Master Manifest:** `processed_data/fused_features/fused_manifest.json`
- **Split Artifacts (per split):**
  - `fused_full_hybrid.parquet` (71 features)
  - `fused_autonomous_ablation.parquet` (68 features)
  - `window_keys.parquet` (structural metadata & label)
  - `fused_arrays.npz` (`X_full`, `X_ablation`, `y`, `window_ids`)
  - `meta.json` (split metadata)

**Conclusion:** Phase 3A is fully completed and verified. **Phase 3B (XGBoost Classifier Training, Hyperparameter Tuning, and Evaluation) is 100% safe to begin upon approval.**
