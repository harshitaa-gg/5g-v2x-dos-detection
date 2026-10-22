# Phase 1: Repository Inspection & Multimodal Architecture Audit Report

**Project Title:** Congestion-Aware Transformer–XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Execution Date:** 2026-10-15  
**Stage:** Phase 1 — Repository Inspection and Architecture Verification Only  
**Target Architecture:**
$$\text{10-Step Transmission Sequences } (X) \xrightarrow{\text{Frozen Transformer}} \text{Embedding } (E) \xrightarrow{\text{Concatenate with Context } (C)} [E, C] \xrightarrow{\text{XGBoost}} \hat{y} \in \{0, 1\}$$

---

## 1. Executive Summary & Verification of Artifacts

All data artifacts, sequence archives, metadata tables, and environment dependencies were inspected without modifying files, running models, or installing packages.

### Confirmed Pipeline Structure & Artifacts:
- **Sequence Arrays ($X, y$):** Located at `processed_data/sequences_tx/{highway}/{split}/sequences.npz`.
  - Shape $X$: `(N, 10, 9)`, dtype `float32`.
  - Shape $y$: `(N,)`, dtype `int8`.
- **Window Metadata:** Located at `processed_data/sequences_tx/{highway}/{split}/window_metadata.parquet`.
  - Contains exact window bounds: `window_id`, `sender_id`, `start_send_time`, `end_send_time`, `observation_duration_s`, `label`.
- **Context Feature Matrices ($C$):** Located at `processed_data/transmission_context/{highway}/{split}/context_features.parquet`.
  - Shape $C$: `(N, 7)`, dtype `float64`.
  - Contains the 7 non-redundant context features plus segregated metadata.
- **Multimodal Manifest:** Located at `processed_data/fusion_manifest.json`.
- **Legacy Files Preserved:** All 6 legacy `.npz` files in `processed_data/sequences/` (202.04 MB) remain completely untouched.

---

## 2. Quantitative Summary Across Splits

| Highway / Split | Sequence Rows ($N$) | Context Rows ($N$) | 1-to-1 Match | Positive (DoS) | Negative (Benign) | DoS % | Duration Range (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HW2 / train** | **23,367** | **23,367** | 100% | 11,540 | 11,827 | 49.39% | 2.1250 – 192.0000 | **CONFIRMED** |
| **HW2 / validation** | **8,801** | **8,801** | 100% | 4,348 | 4,453 | 49.40% | 2.1250 – 77.0000 | **CONFIRMED** |
| **HW2 / test** | **22,043** | **22,043** | 100% | 10,138 | 11,905 | 45.99% | 2.1250 – 198.0000 | **CONFIRMED** |
| **HW7 / train** | **43,222** | **43,222** | 100% | 20,166 | 23,056 | 46.66% | 2.1250 – 132.0000 | **CONFIRMED** |
| **HW7 / validation** | **10,629** | **10,629** | 100% | 4,531 | 6,098 | 42.63% | 2.1250 – 33.0000 | **CONFIRMED** |
| **HW7 / test** | **56,892** | **56,892** | 100% | 25,539 | 31,353 | 44.89% | 2.1250 – 178.0000 | **CONFIRMED** |
| **Total** | **164,954** | **164,954** | **100%** | **76,262** | **88,692** | **46.23%** | **2.1250 – 198.0000** | **ALL PASSED** |

---

## 3. Environment & Dependency Audit

Inspection of python interpreters on the host system revealed two environments:

1. **Host Global Python (`C:\Users\Aishvarya Balaji\AppData\Local\Programs\Python\Python310`):**
   - Python: `3.10.0`
   - `torch`: `2.8.0+cpu`
   - `xgboost`: **NOT INSTALLED**
   - `sklearn`: **NOT INSTALLED**

2. **Project Virtual Environment (`d:\v2x_DoS_Project\.venv`):**
   - Executable: `d:\v2x_DoS_Project\.venv\Scripts\python.exe`
   - Python: `3.12.4`
   - `torch`: `2.14.0+cpu` (CPU with 12 threads available)
   - `xgboost`: `3.4.1` (Fully installed & operational)
   - `scikit-learn`: `1.9.1` (Fully installed & operational)
   - `scipy`: `1.18.1`
   - `numpy`: `2.5.3`
   - `pandas`: `3.0.6`
   - `pyarrow`: `25.0.1`
   - `joblib`: `1.6.0`

**Operational Requirement:** All subsequent model training, extraction, and evaluation scripts must explicitly invoke `d:\v2x_DoS_Project\.venv\Scripts\python.exe` to guarantee deterministic execution without installing any new packages.

---

## 4. Recommended CPU-Compatible Transformer Architecture

Given that training executes on CPU (12 threads) and sequences are compact ($W=10, D=9$):

### Architecture Specification
- **Input Dimension:** $D_{\text{in}} = 9$, Sequence Length: $T = 10$.
- **Linear Input Projection:** Projects 9 features to $d_{\text{model}} = 64$ dimensions.
- **Positional Encoding:** Standard learned positional embedding ($\mathbb{R}^{10 \times 64}$) or fixed sinusoidal encoding.
- **Transformer Encoder Stack:**
  - Number of Layers: $L = 2$ layers (`nn.TransformerEncoderLayer`)
  - Model Dimension: $d_{\text{model}} = 64$
  - Multi-Head Attention: $n_{\text{head}} = 4$ (head dimension $d_k = 16$)
  - Feed-Forward Dimension: $d_{\text{ff}} = 128$
  - Activation: GELU or ReLU
  - Dropout: $0.1$
  - Parameter Count: $\approx 38,000$ parameters (lightweight, zero CPU latency bottleneck).

### Embedding Extraction Strategy
When passing sequence $X \in \mathbb{R}^{B \times 10 \times 9}$ through the Transformer encoder, the output hidden tensor is $\mathbf{H} \in \mathbb{R}^{B \times 10 \times 64}$.
- **Recommended Extraction:** **Temporal Mean-Pooling** across the 10 timesteps:
  $$\mathbf{e}_i = \frac{1}{10} \sum_{t=0}^9 \mathbf{h}_{i, t} \in \mathbb{R}^{64}$$
  *Rationale:* Mean-pooling aggregates temporal momentum across the entire observation window, providing a stable representation that captures average transmission dynamics without extreme sensitivity to a single timestep.

---

## 5. Training, Validation & Checkpoint Strategy

### 1. Temporary Pretraining Classification Objective
To learn discriminative representations without building an independent baseline classifier:
- A temporary lightweight linear projection head `Linear(64, 1)` with `BCEWithLogitsLoss` is attached on top of $\mathbf{e}_i$.
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`.
- **Batch Size:** 128 (memory-efficient on CPU).
- **Epoch Budget:** 15–20 epochs with validation-based early stopping.

### 2. Validation-Based Early Stopping & Split Isolation
- Evaluated on the `validation` split after every epoch (measuring BCE loss and PR-AUC).
- Early stopping patience: 4 epochs without validation loss improvement.
- Checkpoint saving: The best model weights are saved strictly when validation loss reaches a new minimum:
  `models/transformer/transformer_feature_extractor_{highway}.pt`
- The checkpoint file stores:
  `{epoch, model_state_dict, optimizer_state_dict, val_loss, val_prauc, model_config, created_at}`.

---

## 6. Multimodal Alignment with Context & Leakage Prevention

Once the Transformer is trained:
1. **Model Freezing:** Set `model.eval()` with gradient tracking disabled (`torch.no_grad()`).
2. **Deterministic Embedding Extraction:**
   - For every sample $i \in \{0, \dots, N-1\}$, extract $\mathbf{e}_i \in \mathbb{R}^{64}$.
   - Output embedding matrix: $\mathbf{E} \in \mathbb{R}^{N \times 64}$.
3. **Multimodal Feature Concatenation:**
   - Combine $\mathbf{E}$ with the raw context matrix $C \in \mathbb{R}^{N \times 7}$ along feature dimension:
     $$\mathbf{Z} = [\mathbf{E}, C] \in \mathbb{R}^{N \times 71}$$
   - Row $i$ of $\mathbf{E}$ and row $i$ of $C$ share identical `window_id` $i$, identical `sender_id`, and identical timestamps.
4. **XGBoost Training Protocol:**
   - Trained strictly on $\mathbf{Z}_{\text{train}}$ and $y_{\text{train}}$.
   - Monitored on $\mathbf{Z}_{\text{val}}$ and $y_{\text{val}}$ using `early_stopping_rounds=20`.
   - Evaluated once on frozen $\mathbf{Z}_{\text{test}}$.
   - Zero test or validation information is used in fitting the Transformer or XGBoost.

---

## 7. Critical Risks & Pre-Implementation Controls

1. **Receiver-Derived Context Features:**
   - `starting_distance`, `final_distance`, and `distance_change_rate` depend on median distance across receiving vehicles.
   - *Operational Control:* Evaluated under a primary **Cooperative Edge Authority** setting. A secondary ablation under an **Autonomous Transmitter** setting (`[message_rate, iat_variation, speed_variation, acceleration_variation]`) must be conducted to assess performance without receiver feedback.
2. **Vehicle Overlap Across Splits:**
   - Empirical audit revealed that 37.8% of test senders in Highway 2 and 11.7% of test senders in Highway 7 appear in training data.
   - *Control:* The evaluation must be explicitly reported as **temporal DoS detection across simulation sessions**, avoiding unsupported claims of generalization to unseen vehicles.
3. **Autocorrelation from Overlapping Windows:**
   - Windows of length 10 with stride 1 share 9 transmissions.
   - *Control:* Windows never cross split boundaries. Evaluation must be reported at the window level with full metrics (Precision, Recall, F1, PR-AUC, FPR).
4. **Scope Boundaries:**
   - No standalone context-only XGBoost or standalone Transformer will be treated as primary project deliverables; the objective remains strictly the integrated multimodal architecture.
