"""
select_features.py
==================
Feature-selection step for the Congestion-Aware Transformer-XGBoost DoS detection pipeline.

Reads the deduplicated Parquet splits for highway_2 and highway_7, selects the agreed
model-input feature set (see CANDIDATE_FEATURES below), appends the target column
(``label``), and writes the result to:

    processed_data/feature_selected/<highway>/{train,validation,test}.parquet

No rows are added, removed, or reordered.  No encoding, scaling, or imputation is
applied.  The source files are never modified.

Column-exclusion rationale
--------------------------
sender_id              – High-cardinality vehicle identifier; risks memorisation.
sender_alias           – Temporary pseudonym; carries no kinematic signal.
message_id             – Arbitrary sequential packet counter; not a stationary feature.
send_time              – Raw monotonic simulation clock (ns).  Excluded directly; used
                          in a later feature-engineering step to derive inter-arrival
                          gaps (Δt) per sender.
receive_time           – Redundant with send_time + delay; same raw-clock issue.
sender_pos_x           – Global coordinate; excluded to prevent spatial overfitting.
                          Retained for computing relative transmitter-receiver distance
                          in the feature-engineering step, not as a raw model input.
sender_pos_y           – Same rationale as sender_pos_x.
sender_pos_z           – Confirmed constant 0.0 across all 6 splits (both highways,
                          all three data-splits).  Zero-variance; uninformative.
receiver_pos_x         – Global coordinate; same rationale as sender_pos_x.
receiver_pos_y         – Global coordinate; same rationale as sender_pos_y.
receiver_pos_z         – Confirmed constant 0.0 across all 6 splits.  Zero-variance.
distance_to_road_edge  – CRITICAL exclusion.  100 % missing for Normal-scenario records,
                          present for DoS-scenario records.  Feeding this column (or a
                          missingness indicator derived from it) to any model would
                          constitute direct scenario leakage, trivially separating
                          simulation conditions rather than detecting malicious behaviour.

Column-inclusion rationale
--------------------------
delay                          – Network latency (ns); elevated under channel flooding.
sender_speed                   – Core sender kinematic feature (m/s).
sender_acceleration            – Core sender kinematic feature (m/s²).
sender_heading                 – Sender orientation angle; retained as-is for now.
                                  Trigonometric encoding (sin/cos) is deferred to the
                                  feature-engineering step.
sender_speed_noise             – Simulated sensor noise; captures measurement fidelity.
sender_acceleration_noise      – Simulated sensor noise.
sender_heading_noise           – Simulated sensor noise.
receiver_speed                 – Ego-vehicle velocity context.
receiver_acceleration          – Ego-vehicle acceleration context.
receiver_heading               – Ego-vehicle orientation; deferred trig-encoding.
receiver_speed_noise           – Ego-vehicle sensor noise context.
receiver_acceleration_noise    – Ego-vehicle sensor noise context.
receiver_heading_noise         – Ego-vehicle sensor noise context.
sender_driver_profile          – Categorical driver style (AGGRESSIVE / CAUTIOUS /
                                  NORMAL).  Retained as text; one-hot encoding is
                                  deferred to the model-specific preprocessing step.
receiver_driver_profile        – Same rationale as sender_driver_profile.
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "preprocessed_deduplicated")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "feature_selected")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS   = ["train", "validation", "test"]

TARGET_COLUMN = "label"

# ---------------------------------------------------------------------------
# Candidate feature list  (ordered; does NOT include the target column)
# ---------------------------------------------------------------------------
# Every column listed here must exist in the source files.  If a column is
# absent the script raises a clear KeyError rather than silently omitting it.

CANDIDATE_FEATURES: list[str] = [
    # --- Network / transmission ---
    "delay",                        # Network latency (ns): elevated under DoS flooding

    # --- Sender kinematics ---
    "sender_speed",                 # Sender velocity (m/s)
    "sender_acceleration",          # Sender acceleration (m/s²)
    "sender_heading",               # Sender orientation angle (rad or deg; raw)

    # --- Sender sensor noise ---
    "sender_speed_noise",           # Measurement noise on sender speed
    "sender_acceleration_noise",    # Measurement noise on sender acceleration
    "sender_heading_noise",         # Measurement noise on sender heading

    # --- Receiver kinematics ---
    "receiver_speed",               # Receiver (ego-vehicle) velocity (m/s)
    "receiver_acceleration",        # Receiver acceleration (m/s²)
    "receiver_heading",             # Receiver orientation angle (raw)

    # --- Receiver sensor noise ---
    "receiver_speed_noise",         # Measurement noise on receiver speed
    "receiver_acceleration_noise",  # Measurement noise on receiver acceleration
    "receiver_heading_noise",       # Measurement noise on receiver heading

    # --- Categorical driver profiles (retained as text; not encoded here) ---
    "sender_driver_profile",        # Sender driving style: AGGRESSIVE / CAUTIOUS / NORMAL
    "receiver_driver_profile",      # Receiver driving style: AGGRESSIVE / CAUTIOUS / NORMAL
]

# Final output columns = features + target (target appended last)
OUTPUT_COLUMNS: list[str] = CANDIDATE_FEATURES + [TARGET_COLUMN]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, expected: list[str], path: str) -> None:
    """Raise a KeyError listing every missing column, rather than silently skipping."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following expected columns are absent in '{path}':\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def _select_and_save(src_path: str, dst_path: str) -> int:
    """
    Load *src_path*, select OUTPUT_COLUMNS, write to *dst_path*.

    Returns the number of rows written.
    """
    df = pd.read_parquet(src_path)

    # Guard: every expected column must be present
    _validate_columns(df, OUTPUT_COLUMNS, src_path)

    selected = df[OUTPUT_COLUMNS]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    selected.to_parquet(dst_path, index=False)

    return len(selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Feature Selection")
    print(f"  Input base : {INPUT_BASE}")
    print(f"  Output base: {OUTPUT_BASE}")
    print(f"  Highways   : {HIGHWAYS}")
    print(f"  Splits     : {SPLITS}")
    print()
    print(f"  Target column        : {TARGET_COLUMN!r}")
    print(f"  Candidate features   : {len(CANDIDATE_FEATURES)}")
    for i, col in enumerate(CANDIDATE_FEATURES, 1):
        print(f"    {i:2d}. {col}")
    print(f"  Total output columns : {len(OUTPUT_COLUMNS)}  (features + target)")
    print("=" * 70)
    print()

    written: list[dict] = []

    for highway in HIGHWAYS:
        for split in SPLITS:
            src_path = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
            dst_path = os.path.join(OUTPUT_BASE, highway, f"{split}.parquet")

            if not os.path.isfile(src_path):
                print(f"[ERROR] Source file not found: {src_path}", file=sys.stderr)
                sys.exit(1)

            n_rows = _select_and_save(src_path, dst_path)

            written.append({
                "highway": highway,
                "split":   split,
                "rows":    n_rows,
                "path":    dst_path,
            })
            print(f"  [{highway}/{split:10s}]  {n_rows:>8,} rows  ->  {dst_path}")

    print()
    print("Feature selection complete.  Output summary:")
    print("-" * 70)
    for entry in written:
        print(f"  {entry['highway']}/{entry['split']:<10s}  {entry['rows']:>8,} rows  {entry['path']}")
    print("-" * 70)
    print(f"  Selected features ({len(CANDIDATE_FEATURES)}):")
    for col in CANDIDATE_FEATURES:
        print(f"    {col}")
    print(f"  Target column: {TARGET_COLUMN}")
    print()
    print("Next step: feature engineering (derive relative distance, delta-t, trig-encoding, etc.)")


if __name__ == "__main__":
    main()
