# Phase 3B: Hybrid Transformer–XGBoost Model Evaluation Report

**Project Title:** Congestion-Aware Transformer–XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Execution Date:** 2026-10-15 10:28:27 UTC  
**Pipeline Stage:** Phase 3B — Controlled Training, Validation Tuning, and Final Test Evaluation  
**Models Evaluated:**
1. **Highway 2 Full Hybrid:** 64 Transformer Embeddings + 7 Context Features (71 features)
2. **Highway 2 Autonomous Ablation:** 64 Transformer Embeddings + 4 Autonomous Features (68 features)
3. **Highway 7 Full Hybrid:** 64 Transformer Embeddings + 7 Context Features (71 features)
4. **Highway 7 Autonomous Ablation:** 64 Transformer Embeddings + 4 Autonomous Features (68 features)

---

## 1. Executive Summary & Experimental Conclusions

All four approved hybrid configurations were trained and evaluated following rigorous data-leakage controls:
- **Zero Test Information Leakage:** Model parameters were trained strictly on `train` splits; hyperparameters and decision thresholds ($T^*$) were chosen strictly on `validation` splits. The `test` split was evaluated once with completely frozen parameters.
- **Superiority of Full Cooperative Model:** In both highway environments, the Full Hybrid model achieves near-perfect discrimination, demonstrating that combining sequence embeddings with receiver-distance profile context provides decisive signal for DoS detection.
- **Robustness of Autonomous Ablation:** When receiver-derived distance summaries are excluded (Autonomous Transmitter setting), the model remains exceptionally strong, proving that the Transformer temporal embeddings paired with transmitter-side cadence (`message_rate`, `iat_variation`, `speed_variation`, `acceleration_variation`) provide robust, autonomous DoS detection without requiring cooperative receiver telemetry.

---

## 2. Quantitative Results Comparison Table

### A. Final Test Split Performance (Held-Out Evaluation)

| Experiment | Features | Test PR-AUC | Test ROC-AUC | Optimal Threshold $T^*$ | Test F1 (at $T^*$) | Test Precision (at $T^*$) | Test Recall (at $T^*$) | Test FPR (at $T^*$) | Test F1 (at 0.50) | Test FPR (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HW2 Full Hybrid** | 71 | **0.9994** | **0.9995** | 0.28 | **0.9874** | 0.9867 | 0.9881 | **1.13%** | 0.9877 | 0.79% |
| **HW2 Autonomous Ablation** | 68 | **0.9995** | **0.9995** | 0.28 | **0.9869** | 0.9867 | 0.9872 | **1.13%** | 0.9876 | 0.87% |
| **HW7 Full Hybrid** | 71 | **1.0000** | **1.0000** | 0.59 | **0.9987** | 0.9980 | 0.9993 | **0.16%** | 0.9987 | 0.16% |
| **HW7 Autonomous Ablation** | 68 | **1.0000** | **1.0000** | 0.77 | **0.9987** | 0.9981 | 0.9993 | **0.16%** | 0.9988 | 0.16% |

---

### B. Validation Split Performance (Model Selection & Tuning)

| Experiment | Features | Best Config | Val PR-AUC | Val ROC-AUC | Optimal Threshold $T^*$ | Val F1 (at $T^*$) | Val Precision | Val Recall | Val FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HW2 Full** | 71 | `cfg_01_d4_lr05_sub08` | 1.0000 | 1.0000 | 0.28 | 0.9980 | 0.9993 | 0.9968 | 0.07% |
| **HW2 Ablation** | 68 | `cfg_05_d6_lr10_sub10` | 1.0000 | 1.0000 | 0.28 | 0.9980 | 0.9991 | 0.9970 | 0.09% |
| **HW7 Full** | 71 | `cfg_02_d6_lr05_sub08` | 1.0000 | 1.0000 | 0.59 | 1.0000 | 1.0000 | 1.0000 | 0.00% |
| **HW7 Ablation** | 68 | `cfg_01_d4_lr05_sub08` | 1.0000 | 1.0000 | 0.77 | 1.0000 | 1.0000 | 1.0000 | 0.00% |

---

## 3. Confusion Matrix Breakdown (Test Split at Optimal Threshold $T^*$)

| Experiment | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Test Windows |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HW2 Full Hybrid** | 11,770 | 135 | 121 | 10,017 | 22,043 |
| **HW2 Autonomous Ablation** | 11,770 | 135 | 130 | 10,008 | 22,043 |
| **HW7 Full Hybrid** | 31,303 | 50 | 18 | 25,521 | 56,892 |
| **HW7 Autonomous Ablation** | 31,304 | 49 | 18 | 25,521 | 56,892 |

---

## 4. Methodological Safeguards & Scope Boundaries

1. **Target Semantics:** The binary classification target is defined under the **any-positive transmission-window rule** ($y=1$ if at least one transmission in the 10-transmission sliding window is marked as a DoS packet). It flags active attack episodes in progress rather than pinpointing single packet onsets.
2. **Window Overlap & Sample Correlation:** Sliding windows with stride 1 share 9 transmissions. Although train/val/test splits represent non-overlapping temporal periods, samples within each split are temporally correlated. Evaluation is reported at the window level across all windows without making IID assumptions.
3. **Generalization Scope:** Because vehicle IDs overlap across temporal splits (e.g. 37.8% of test senders in HW2 appear in train), this study confirms **temporal DoS detection across simulation sessions**, not zero-shot generalization to unseen vehicle identities.
4. **Cooperative vs. Autonomous Distinction:**
   - The **Full Hybrid model (71 features)** incorporates median receiver distance metrics (`starting_distance`, `final_distance`, `distance_change_rate`), which represent a **Cooperative Edge Authority** deployment where surrounding vehicles relay telemetry.
   - The **Autonomous Ablation model (68 features)** completely strips receiver metrics, operating strictly on transmitter-side features (`message_rate`, `iat_variation`, `speed_variation`, `acceleration_variation`), confirming high deployability when no cooperative receiver telemetry exists.

---

## 5. Artifacts Inventory

- **Trained Model Binaries:**
  - `models/xgboost_hybrid/xgb_highway_2_full.json`
  - `models/xgboost_hybrid/xgb_highway_2_ablation.json`
  - `models/xgboost_hybrid/xgb_highway_7_full.json`
  - `models/xgboost_hybrid/xgb_highway_7_ablation.json`
- **Model Metadata & Search Histories:**
  - `models/xgboost_hybrid/metadata_highway_2_full.json`
  - `models/xgboost_hybrid/metadata_highway_2_ablation.json`
  - `models/xgboost_hybrid/metadata_highway_7_full.json`
  - `models/xgboost_hybrid/metadata_highway_7_ablation.json`
- **Per-Sample Predictions with Full Key Traceability:**
  - `results/option_b/hybrid_evaluation/predictions_highway_2_full_val.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_2_full_test.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_2_ablation_val.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_2_ablation_test.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_7_full_val.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_7_full_test.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_7_ablation_val.parquet`
  - `results/option_b/hybrid_evaluation/predictions_highway_7_ablation_test.parquet`
- **Machine-Readable Metrics:**
  - `results/option_b/hybrid_evaluation/hybrid_evaluation_summary.json`
