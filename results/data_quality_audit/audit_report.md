# Comprehensive Data Quality and Integrity Audit Report

**Project Title:** Congestion-Aware Transformer-XGBoost Framework for Robust DoS Detection in 5G-V2X Vehicular Networks  
**Dataset:** VeReMi NextGen (Processed Parquet Splits)  
**Audit Execution Date:** 2026-10-11  
**Status:** Read-only verification completed across all six primary processed splits.  

---

## 1. Executive Summary

A rigorous, read-only data quality audit was conducted across all six Parquet splits in `processed_data/` 
(`highway_2` and `highway_7` across `train`, `validation`, and `test`).

### Key Findings:
1. **Schema Consistency:** All six files share an identical PyArrow schema with 28 columns, ordered consistently. All integer columns (`message_id`, `send_time`, `receive_time`, `delay`, `label`) are `int64`, floating features are `double`, and strings are `large_string` with pandas metadata.
2. **Timestamp and Delay Precision:** The formula `delay == receive_time - send_time` holds with **100.0% exact precision** across all 12,183,124 rows. Units are nanoseconds (ranging from 0.94 ms to 6.02 ms). There are **zero** negative delays and zero precision truncations.
3. **Coordinate Validity:** Altitude coordinates (`sender_pos_z` and `receiver_pos_z`) are uniformly `0.0` across all files, reflecting a 2D planar vehicular road network in the simulation. Lateral distances to road edge include valid negative numbers (-5.45 m to -0.01 m) representing vehicles centered past the inner road edge in SUMO.
4. **Missing Values & Scenario Artifact:** Missing values exist **only** in `distance_to_road_edge`. Exactly 100% of records from the Normal scenario (`InTAS_highway_X`) have `distance_to_road_edge = NaN`, whereas 100% of records from the DoS scenario (`InTAS_highway_X_dosAttack`) have finite floating-point values. No other column contains nulls or NaNs.
5. **Cross-Scenario Broadcast Overlap & Label Contradiction:** Because the builder concatenated the benign baseline run and the attack run from identical simulation seeds, **100% of all broadcast messages in the Normal scenario exist identically in the DoS scenario**. For vehicles designated as attackers in the DoS scenario, the identical message appears twice in the same split: once with `label = 0` (from Normal) and once with `label = 1` (from DoS). This creates label ambiguity and duplicates.
6. **Split Design & Leakage:** There is **zero (0) message overlap** between Train, Validation, and Test splits. Sender IDs naturally persist across temporal windows as vehicles traverse the highway, but individual transmission events are strictly partitioned across simulation time windows.

---

## 2. File and Schema Integrity

| Split Name | File Size (MB) | Rows | Columns | Row Groups | Label 0 Count (%) | Label 1 Count (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `highway_2/train` | 6.71 MB | 92,991 | 28 | 1 | 67,270 (72.34%) | 25,721 (27.66%) |
| `highway_2/validation` | 2.73 MB | 44,098 | 28 | 1 | 32,139 (72.88%) | 11,959 (27.12%) |
| `highway_2/test` | 7.26 MB | 104,977 | 28 | 1 | 80,325 (76.52%) | 24,652 (23.48%) |
| `highway_7/train` | 154.14 MB | 3,338,030 | 28 | 4 | 2,464,737 (73.84%) | 873,293 (26.16%) |
| `highway_7/validation` | 54.39 MB | 1,314,941 | 28 | 2 | 1,008,935 (76.73%) | 306,006 (23.27%) |
| `highway_7/test` | 485.58 MB | 7,288,087 | 28 | 73 | 5,446,078 (74.73%) | 1,842,009 (25.27%) |

All column names, types, and ordering match 100% across all splits.

---

## 3. Timestamp and Delay Integrity

- **Simulation Time Resolution:** Timestamps are stored in nanoseconds (`int64`).
- **Delay Equation:** $\text{delay} = \text{receive\_time} - \text{send\_time}$.
- **Verification Result:**
  - Total records audited: **12,183,124**
  - Formula mismatches: **0**
  - Negative delays: **0**
  - Zero delays: **0**
  - Maximum observed delay: **6,020,751 ns (6.02 ms)**
  - Minimum observed delay: **944,429 ns (0.94 ms)**

---

## 4. Scenario Breakdown and Label Distribution

| Split | Total Rows | Normal Scenario (label 0) | DoS Benign (label 0) | DoS Attacker (label 1) | Normal Broadcast Overlap (label 0) | Normal Attacker Overlap (label 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `highway_2/train` | 92,991 | 37,701 | 29,569 | 25,721 | 15,224 | 3,962 |
| `highway_2/validation` | 44,098 | 17,977 | 14,162 | 11,959 | 5,535 | 1,488 |
| `highway_2/test` | 104,977 | 44,259 | 36,066 | 24,652 | 15,293 | 3,599 |
| `highway_7/train` | 3,338,030 | 1,380,604 | 1,084,133 | 873,293 | 27,045 | 7,087 |
| `highway_7/validation` | 1,314,941 | 556,916 | 452,019 | 306,006 | 7,615 | 1,636 |
| `highway_7/test` | 7,288,087 | 3,011,370 | 2,434,708 | 1,842,009 | 34,919 | 8,472 |

> [!WARNING]
> In all splits, all broadcast messages originating from the Normal scenario are identical copies of messages in the DoS scenario. When a vehicle was assigned as an attacker in the DoS scenario, its exact kinematic messages exist in the dataset under both label 0 (from the Normal file) and label 1 (from the DoS file).

---

## 5. Duplicate and Leakage Audit

### Multi-Receiver Broadcast Reception
In vehicular ad-hoc networks (VANET), a single vehicle broadcast is received by multiple neighboring vehicles. Consequently, each unique broadcast message `(sender_id, message_id, send_time)` appears multiple times in the dataset—each entry corresponding to a distinct receiver at its respective location and receive time. This is standard vehicular network behavior and does not represent duplicate errors.

### Cross-Split Overlap (Train vs Validation vs Test)

| Condition | Train Broadcasts | Val Broadcasts | Test Broadcasts | Train-Val Msg Overlap | Train-Test Msg Overlap | Val-Test Msg Overlap | Sender Overlap (Train-Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `highway_2` | 27,625 | 10,153 | 26,286 | 0 | 0 | 0 | 180 |
| `highway_7` | 48,219 | 12,538 | 61,350 | 0 | 0 | 0 | 59 |

- **Message Leakage:** **Zero.** No transmission message is shared between train, validation, and test splits.
- **Sender Persistence:** Sender vehicles reappear across splits because the simulation partitions continuous vehicle trips into temporal simulation windows.

---

## 6. Categorical Profiles and Numeric Validity

### Categorical Driver Profiles
- **`sender_driver_profile`**: `['AGGRESSIVE', 'CAUTIOUS', 'NORMAL']`
- **`receiver_driver_profile`**: `['AGGRESSIVE', 'CAUTIOUS', 'NORMAL']`

### Elevation (`pos_z`)
- Both `sender_pos_z` and `receiver_pos_z` have **0 non-zero values** across all 12.2M records. The simulation is strictly 2-dimensional.

### `distance_to_road_edge`
- Negative values represent valid SUMO lateral coordinates where the vehicle center has crossed past the reference boundary lane.
- **Critical Hazard:** Because this feature is 100% missing from Normal scenario records, it leaks scenario identity and must not be used as an unmitigated model feature.

