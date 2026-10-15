# Preprocessing and Feature Policy

**Project Title:** Congestion-Aware Transformer-XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Document Version:** 1.0  
**Date:** 2026-10-11  

---

## 1. Scope and Objectives

This document establishes the official data classification, leakage mitigation, and preprocessing policy for the 5G-V2X DoS detection research pipeline.

The target system is a hybrid ML architecture comprising:
1. **Temporal Sequence Model (Transformer):** Captures multi-step kinematic dynamics and transmission patterns over sliding message windows.
2. **Context & Congestion Engine:** Derives network and local vehicle-density metrics to distinguish legitimate high-congestion traffic from DoS flooding.
3. **Classification Head (XGBoost):** Combines latent temporal embeddings from the Transformer with engineered context features to produce final binary predictions.

This policy defines which dataset attributes may be consumed as model inputs, which must be quarantined or transformed, and how transformations must be strictly fitted to prevent data leakage.

---

## 2. Comprehensive Column Classification

The processed Parquet dataset contains 28 attributes. These are classified below into functional roles:

| Column Name | PyArrow Dtype | Role / Category | Model Status | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `sender_id` | `large_string` | Identifier / Metadata | **Exclude** | High cardinality categorical identifier. Using it directly risks vehicle memorization rather than learning generalizable DoS behavior. |
| `sender_alias` | `large_string` | Identifier / Pseudonym | **Exclude** | Temporary vehicular pseudonym; uninformative for kinematic DoS detection. |
| `message_id` | `int64` | Message Sequence ID | **Exclude** | Arbitrary sequential packet counter. Not a stationary numeric feature. |
| `send_time` | `int64` | Raw Timestamp (ns) | **Exclude directly** (Derive temporal gaps) | Monotonically increasing simulation clock. Cannot be used as a raw numeric feature. Used strictly to compute inter-arrival times ($\Delta t$). |
| `receive_time` | `int64` | Raw Timestamp (ns) | **Exclude** | Redundant with `send_time` + `delay`. Raw simulation clock. |
| `delay` | `int64` | Network Latency (ns) | **Candidate Feature** (Scaled/Transformed) | Network latency $\Delta = t_{\text{rcv}} - t_{\text{send}}$. High packet delays often correlate with channel congestion and flooding attacks. |
| `sender_pos_x` | `double` | Kinematic Coordinate | **Exclude directly** (Derive relative distance) | Global coordinate; prone to spatial overfitting. Should only be used to compute relative distance to receiver or neighbor density. |
| `sender_pos_y` | `double` | Kinematic Coordinate | **Exclude directly** (Derive relative distance) | Global coordinate; prone to spatial overfitting. |
| `sender_pos_z` | `double` | Altitude | **Exclude** | Constant zero ($0.0$) across all 12.2M records; uninformative constant feature. |
| `sender_speed` | `double` | Kinematic Feature | **Include** | Core vehicle kinematic feature ($m/s$). |
| `sender_acceleration`| `double` | Kinematic Feature | **Include** | Core vehicle kinematic feature ($m/s^2$). |
| `sender_heading` | `double` | Kinematic Feature | **Include** (Trigonometric encoding) | Vehicle orientation angle ($0$ to $2\pi$ or $-180^\circ$ to $180^\circ$). Encode via $\sin(\theta), \cos(\theta)$ to preserve angular continuity. |
| `sender_speed_noise` | `double` | Measurement Error | **Include** | Simulated sensor noise feature. |
| `sender_acceleration_noise` | `double` | Measurement Error | **Include** | Simulated sensor noise feature. |
| `sender_heading_noise` | `double` | Measurement Error | **Include** | Simulated sensor noise feature. |
| `receiver_pos_x` | `double` | Receiver Coordinate | **Exclude directly** | Used strictly to compute transmitter-receiver euclidean distance: $\sqrt{(x_s - x_r)^2 + (y_s - y_r)^2}$. |
| `receiver_pos_y` | `double` | Receiver Coordinate | **Exclude directly** | Used strictly for relative distance derivation. |
| `receiver_pos_z` | `double` | Receiver Altitude | **Exclude** | Constant zero ($0.0$) across all records. |
| `receiver_speed` | `double` | Receiver Kinematic | **Include** | Ego-vehicle velocity context. |
| `receiver_acceleration` | `double` | Receiver Kinematic | **Include** | Ego-vehicle acceleration context. |
| `receiver_heading` | `double` | Receiver Kinematic | **Include** (Trigonometric encoding) | Ego-vehicle orientation angle ($\sin, \cos$). |
| `receiver_speed_noise` | `double` | Receiver Noise | **Include** | Ego-vehicle sensor noise context. |
| `receiver_acceleration_noise`| `double` | Receiver Noise | **Include** | Ego-vehicle sensor noise context. |
| `receiver_heading_noise` | `double` | Receiver Noise | **Include** | Ego-vehicle sensor noise context. |
| `sender_driver_profile` | `large_string` | Categorical Driver Style | **Include** (One-Hot Encoded) | Three discrete profiles: `['AGGRESSIVE', 'CAUTIOUS', 'NORMAL']`. |
| `receiver_driver_profile`| `large_string` | Categorical Driver Style | **Include** (One-Hot Encoded) | Three discrete profiles: `['AGGRESSIVE', 'CAUTIOUS', 'NORMAL']`. |
| `distance_to_road_edge` | `double` | Lateral Road Geometry | **CRITICAL EXCLUSION** | **100% missing in Normal scenario, present in DoS scenario.** Direct scenario leakage shortcut. Must NOT be fed to model. |
| `label` | `int64` | Target Variable | **Target Only** | Binary ground-truth: `0` (Honest), `1` (DoS Malicious). Never an input feature. |

---

## 3. Critical Leakage and Confounder Analysis

### 3.1 The `distance_to_road_edge` Confounder
- **Observation:** Our read-only audit established that `distance_to_road_edge` has an exact missingness rate of:
  - `highway_2/train`: 37,701 missing (100% of Normal records)
  - `highway_2/validation`: 17,977 missing (100% of Normal records)
  - `highway_2/test`: 44,259 missing (100% of Normal records)
  - `highway_7/train`: 1,380,604 missing (100% of Normal records)
  - `highway_7/validation`: 556,916 missing (100% of Normal records)
  - `highway_7/test`: 3,011,370 missing (100% of Normal records)
- **Leakage Mechanism:** If a tree-based model (e.g., XGBoost) or neural network is supplied with `distance_to_road_edge` (or a missingness indicator), the model will immediately split on `is_missing(distance_to_road_edge)`. Since all Normal scenario records are missing this feature, the model achieves a trivial, artificial shortcut that distinguishes the simulation scenario rather than identifying malicious DoS kinematics or packet traffic patterns.
- **Mandatory Policy:** `distance_to_road_edge` **must be completely excluded** from the active feature set for both the Transformer and XGBoost models.

### 3.2 The Raw `attacker` Attribute
- In the raw JSON records, the field `attacker` directly specifies whether a vehicle was assigned to transmit malicious DoS traffic.
- In the builder, `label` is derived from this field. The raw `attacker` field is omitted from the processed dataset. Under no circumstances may any raw attacker indicator enter the feature matrix.

### 3.3 Normal vs. DoS Simulation Replay Overlap
- **Observation:** In the processed datasets, records from the baseline simulation `InTAS_highway_X` and the attack simulation `InTAS_highway_X_dosAttack` were concatenated together into the same Parquet split.
- Because both simulations used identical mobility traces:
  - 100% of the broadcast messages present in the Normal files exist identically in the DoS files.
  - For the 20% of vehicles designated as attackers in the DoS scenario, their exact broadcast messages appear twice in the dataset: once under `label = 0` (from the Normal scenario) and once under `label = 1` (from the DoS scenario).
- **Impact on Preprocessing:**
  - When fitting preprocessing pipelines, downstream models must be protected against conflicting sample training.
  - Future data curation may deduplicate these transmissions or clearly separate the baseline congestion evaluation from the in-run attack evaluation.

---

## 4. Feature Engineering Specifications

Before feeding data to models, the following legitimate features should be derived:

1. **Relative Spatial Distance ($d_{\text{rel}}$):**
   $$d_{\text{rel}} = \sqrt{(x_{\text{sender}} - x_{\text{receiver}})^2 + (y_{\text{sender}} - y_{\text{receiver}})^2}$$
2. **Relative Speed ($\Delta v$):**
   $$\Delta v = v_{\text{sender}} - v_{\text{receiver}}$$
3. **Heading Trigonometric Encoding:**
   $$\sin(\theta) = \sin(\text{heading}), \quad \cos(\theta) = \cos(\text{heading})$$
4. **Inter-Arrival Message Gap ($\Delta t$):**
   For consecutive messages received from the same `sender_id`:
   $$\Delta t = t_{\text{send}, i} - t_{\text{send}, i-1}$$
   *(Computed strictly within historical sliding windows to prevent future-time leakage).*
5. **Local Congestion Context (Density Features):**
   - Count of unique active senders observed within a spatial radius (e.g., $r = 300\text{ m}$) over a preceding time window (e.g., $\tau = 1.0\text{ s}$).
   - Aggregate message arrival rate ($\text{pkts/sec}$) in the local neighborhood.
   - *Requirement:* All windowed aggregations must be computed looking strictly backward in simulation time ($t \le t_{\text{current}}$).

---

## 5. Model-Specific Pipeline Requirements

### 5.1 XGBoost Preprocessing Pipeline
- **Categorical Features:** One-Hot Encode `sender_driver_profile` and `receiver_driver_profile` using `handle_unknown='ignore'`.
- **Scaling:** Tree-based models are invariant to monotonic transformations. Do **not** apply standard scaling or min-max normalization to numeric features for XGBoost, preserving numeric precision and interpretability.
- **Missing Values:** XGBoost handles missing values natively, but our validated feature set has 0 missing values across all retained columns.

### 5.2 Transformer Preprocessing Pipeline
- **Input Representation:** 3D tensor of shape `(batch_size, sequence_length, num_features)`.
- **Sequencing Policy:** Sequences must be constructed per transmitter-receiver pair or local receiver observation stream, sorted monotonically by `send_time`.
- **Numeric Scaling:** Apply `StandardScaler` (or `RobustScaler` for heavy-tailed features such as `delay`).
- **Strict Fitting Rule:** Scalers must be fitted **exclusively on the training split** (`train.parquet`). The fitted parameters ($\mu, \sigma$) must be frozen and applied unchanged to `validation.parquet` and `test.parquet`.
- **Categorical Embeddings:** Driver profiles must be encoded to dense vectors or passed via the fitted one-hot representation.

---

## 6. Preprocessing Reproducibility & Artifact Storage

When the preprocessing script is implemented:
1. All fitted transformers (scalers, encoders) must be serialized to `models/preprocessing/` (e.g., via `joblib` or `pickle`) with clear versioning and training-split metadata.
2. An automated check must verify that validation and test transformations match the training schema exactly.
3. Random seeds must be pinned where applicable.
