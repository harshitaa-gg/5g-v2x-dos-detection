# Congestion-Aware Transformer-XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks

An end-to-end machine learning framework designed for robust Denial-of-Service (DoS) attack detection in 5G Vehicle-to-Everything (5G-V2X) vehicular communication systems under dense traffic congestion.

---

## 1. Abstract & Motivation

In 5G-V2X vehicular networks, vehicle safety relies on high-frequency Basic Safety Messages (BSMs) transmitted over wireless cellular channels. Malicious Denial-of-Service (DoS) attacks flood communication channels with anomalous packet bursts, depriving neighboring vehicles of critical collision-warning packets. However, in dense urban and highway environments, legitimate traffic congestion causes channel contention, packet collisions, and transmission delays that closely mirror malicious flooding. 

Traditional detection mechanisms that inspect isolated packet statistics or raw kinematics often suffer from elevated false-alarm rates during benign congestion spikes. 

This project implements a **Congestion-Aware Multimodal Hybrid Architecture**:
1. **Temporal Dynamics Representation:** A lightweight 2-layer PyTorch **Transformer Encoder** captures temporal evolution across consecutive vehicle transmissions ($W=10$ timesteps) and compresses them into a 64-dimensional latent embedding.
2. **Kinematic & Congestion Context Engineering:** Extracts 7 physical, temporal, and spatial context features (starting/final distance, rate of distance change, message rate, inter-arrival time variation, speed variation, and acceleration variation).
3. **Multimodal Fusion & Gradient Boosted Classification:** Combines the 64-dimensional sequence embeddings with contextual features into an **XGBoost** classifier, optimized with decision threshold calibration to maximize the precision-recall trade-off under class imbalance.

---

## 2. Verified Performance Results

All models were evaluated on independent, held-out test splits with vehicle-isolated partitioning (zero vehicle identity overlap between train, validation, and test splits).

### Quantitative Summary

| Scenario | Model Architecture | Predictors | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Highway 2** | Full Hybrid Model | **71** | **98.84%** | **98.67%** | **98.81%** | **98.74%** | **0.9995** | **98.87%** | **1.13%** |
| **Highway 2** | Autonomous Ablation | **68** | **98.80%** | **98.67%** | **98.72%** | **98.69%** | **0.9995** | **98.87%** | **1.13%** |
| **Highway 7** | Full Hybrid Model | **71** | **99.88%** | **99.80%** | **99.93%** | **99.87%** | **1.0000** | **99.84%** | **0.16%** |
| **Highway 7** | Autonomous Ablation | **68** | **99.88%** | **99.81%** | **99.93%** | **99.87%** | **1.0000** | **99.84%** | **0.16%** |

### Confusion Matrices (Held-out Test Sets)

- **Highway 2 — Full Model (71 features):**
  - Actual Normal: 11,770 Correct, 135 False Positives
  - Actual DoS: 10,017 Correct, 121 False Negatives

- **Highway 2 — Autonomous Ablation (68 features):**
  - Actual Normal: 11,770 Correct, 135 False Positives
  - Actual DoS: 10,008 Correct, 130 False Negatives

- **Highway 7 — Full Model (71 features):**
  - Actual Normal: 31,303 Correct, 50 False Positives
  - Actual DoS: 25,521 Correct, 18 False Negatives

- **Highway 7 — Autonomous Ablation (68 features):**
  - Actual Normal: 31,304 Correct, 49 False Positives
  - Actual DoS: 25,521 Correct, 18 False Negatives

*Artifacts, confusion matrix plots, and summary metrics are stored under `results/model_evaluation/`.*

---

## 3. Quick Start (Evaluation Entry Point)

To evaluate the four trained models on the held-out test datasets and print the official verification metrics:

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate   # Windows PowerShell / CMD
# source .venv/bin/activate  # Linux / macOS

# 2. Run the master evaluation runner
python main.py
```

Or equivalently:

```bash
python src/run_model_evaluation.py
```

This script loads the pre-fitted XGBoost models from `models/xgboost_hybrid/`, loads the test feature matrices from `processed_data/fused_features/`, applies the calibrated decision thresholds, displays the results, and refreshes the evaluation plots in `results/model_evaluation/`.

---

## 4. Pipeline Reproduction Steps

To reproduce the end-to-end data processing, feature engineering, and model training from raw simulation data:

```bash
# Step 1: Collapse multi-receiver logs into unique broadcast transmissions
python src/build_transmission_table.py

# Step 2: Fit scalers on the training split only and scale kinematic features
python src/scale_transmission_features.py

# Step 3: Construct temporal sliding windows (W=10, S=1) per vehicle
python src/create_transmission_sequences.py

# Step 4: Extract the 7 window-level physical and congestion context features
python src/build_transmission_context.py

# Step 5: Train the Transformer feature extractor and extract 64-d embeddings
python src/train_transformer_extractor.py

# Step 6: Fuse sequence embeddings with context features (71 and 68 predictors)
python src/build_fused_datasets.py

# Step 7: Train XGBoost classifiers, calibrate thresholds, and save models
python src/train_xgboost_hybrid.py

# Step 8: Evaluate all models on held-out test splits
python src/run_model_evaluation.py
```

---

## 5. Repository Architecture

```text
v2x_DoS_Project/
├── main.py                               # 1-Click execution entry point
├── requirements.txt                      # Project dependencies
├── README.md                             # Comprehensive project documentation
├── .gitignore                            # Anchored exclusions for heavy data/models
│
├── docs/                                 # Methodology and policy documentation
│   ├── preprocessing_and_feature_policy.md # Mathematical definitions & leakage rules
│   ├── dataset_download_notes.md         # Simulation dataset details (VeReMi NextGen)
│   └── CHANGELOG_PREPROCESSING_AUDIT.md  # Data audit changelog
│
├── src/                                  # Core source code
│   ├── run_model_evaluation.py           # Evaluation runner with clean terminal output
│   ├── models/
│   │   └── transformer_encoder.py        # PyTorch TransformerFeatureExtractor architecture
│   │
│   ├── build_transmission_table.py       # Deduplication of broadcast transmissions
│   ├── scale_transmission_features.py    # Zero-leakage standard scaler fitting
│   ├── create_transmission_sequences.py  # Temporal sliding window generation
│   ├── build_transmission_context.py     # Congestion and kinematic context extraction
│   ├── train_transformer_extractor.py    # Transformer pretraining & embedding export
│   ├── build_fused_datasets.py           # Multimodal feature matrix construction
│   ├── train_xgboost_hybrid.py           # XGBoost training & threshold optimization
│   │
│   └── integrity_audit.py                # Repository-wide data & model verification
│
├── models/                               # Saved model artifacts
│   ├── transformer_extractor/            # Best PyTorch checkpoints (.pt) & metadata
│   ├── xgboost_hybrid/                   # Trained XGBoost models (.json), metadata, thresholds
│   └── preprocessing/                    # Fitted StandardScaler archives (.joblib)
│
├── results/                              # Official evaluation outputs & audit reports
│   ├── model_evaluation/                 # Summary CSV, JSON, and confusion matrix PNGs
│   ├── option_b/                         # Comprehensive methodology & evaluation reports
│   └── data_quality_audit/               # Cross-split leakage & schema verification reports
│
└── archive/                              # Local archive of historical debugging audits
    └── results/                          # Preserved label conflict & milestone step audits
```

---

## 6. Experimental Integrity & Leakage Prevention Safeguards

1. **Partitioning Safeguards:** Train, validation, and test splits are strictly partitioned by vehicle ID (`sender_id`). No vehicle appears in more than one partition.
2. **Scaler Safeguards:** All standard scalers are fitted exclusively on training data. Validation and test partitions are transformed using parameters derived from the training split.
3. **Threshold Calibration:** Decision thresholds $T^*$ are tuned exclusively on validation splits using F1-score optimization. Test splits are evaluated once using the pre-tuned thresholds without adjustment.
4. **Broadcast Deduplication:** In V2X broadcast simulations, a single packet is logged by multiple receiving vehicles. The pipeline collapses redundant reception records by `(sender_id, send_time)` into unique physical broadcasts, eliminating synthetic duplicate inflation.

---

## 7. Dependencies & Requirements

- Python 3.10+ (tested on Python 3.12, Windows 11)
- CPU-compatible PyTorch (`torch`)
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`, `pyarrow`
- `matplotlib`, `seaborn`

Install dependencies with:
```bash
pip install -r requirements.txt
```
