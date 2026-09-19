# Congestion-Aware Transformer-XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks

## Project Overview

This project investigates a congestion-aware machine learning approach for detecting Denial-of-Service (DoS) attacks in 5G-V2X vehicular networks. The project extends a Transformer-XGBoost based DoS detection approach by investigating whether traffic and network-context information can be combined with temporal representations to improve robustness when legitimate high traffic resembles DoS-like behavior. The project uses the locally available VeReMi NextGen dataset subset and evaluates three model configurations: an XGBoost baseline, a Transformer-based temporal model followed by XGBoost, and a proposed Transformer plus context-feature fusion followed by XGBoost.

## Project Structure

```text
V2X_DoS_Project/
│
├── data/
│   └── raw/                    # Original dataset files; never modified
│
├── processed_data/             # Cleaned and feature-engineered data
│
├── models/                     # Saved trained models
│
├── results/                    # Metrics, tables and experiment results
│
├── plots/                      # Generated figures
│
├── docs/                       # Documentation and phase reports
│
├── src/
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── inspect_ground_truth.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── create_sequences.py
│   ├── train_xgboost.py
│   ├── train_transformer.py
│   ├── train_proposed.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore