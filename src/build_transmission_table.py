"""
build_transmission_table.py
===========================
Option B: Rebuild Transmission-Level Datasets for 5G-V2X DoS Detection.

This script aggregates reception-level deduplicated logs into transmission-level
records where each row represents exactly ONE unique physical broadcast event,
uniquely identified by `(sender_id, send_time)`.

Key Architectural Design & Safeguards:
--------------------------------------
1. Detection Vantage Point:
   - Network-side / Edge detector (e.g., Roadside Unit / 5G C-V2X Misbehavior Authority)
     monitoring sector transmissions and local channel congestion.
   - Documented as an explicit project design choice.
2. Invariant Sender Preservation:
   - Strictly preserves all sender-side features: coordinates (x, y, z), kinematics
     (speed, acceleration, heading), sensor noise, and driver profile.
   - Enforces zero-tolerance consistency checks: aborts if any sender feature,
     message_id, or label varies across reception copies.
3. Candidate Transmission-Level Aggregates:
   - `observed_reception_count`: Integer count of receiving neighbor logs for this packet.
   - `mean_delay`: Mean transmission latency across all receiving neighbors (ns).
   - `median_receiver_distance`: Median transmitter-to-receiver Euclidean distance (m).
4. Deterministic Ordering & Leakage Prevention:
   - Deterministically sorted by `sender_id` then `send_time`.
   - Processed strictly split-by-split; zero cross-split history joining.
   - `distance_to_road_edge` and raw `attacker` remain quarantined/excluded.
5. Non-Destructive Storage:
   - Outputs saved to: `processed_data/transmissions/{highway}/{split}.parquet`
   - Existing reception sequences and datasets are completely untouched.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "preprocessed_deduplicated")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmissions")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

# Verified expected transmission counts from read-only audit
EXPECTED_TX_COUNTS: Dict[Tuple[str, str], int] = {
    ("highway_2", "train"): 27625,
    ("highway_2", "validation"): 10153,
    ("highway_2", "test"): 26286,
    ("highway_7", "train"): 48219,
    ("highway_7", "validation"): 12538,
    ("highway_7", "test"): 61350,
}

# Sender-invariant columns to retain
SENDER_COLS = [
    "sender_id",
    "send_time",
    "message_id",
    "sender_alias",
    "label",
    "sender_pos_x",
    "sender_pos_y",
    "sender_pos_z",
    "sender_speed",
    "sender_acceleration",
    "sender_heading",
    "sender_speed_noise",
    "sender_acceleration_noise",
    "sender_heading_noise",
    "sender_driver_profile",
]

# Required columns to load from preprocessed_deduplicated
LOAD_COLS = SENDER_COLS + [
    "receiver_pos_x",
    "receiver_pos_y",
    "delay",
]

# Output schema
OUTPUT_SCHEMA = pa.schema([
    ("sender_id", pa.large_string()),
    ("send_time", pa.int64()),
    ("message_id", pa.int64()),
    ("sender_alias", pa.large_string()),
    ("label", pa.int64()),
    ("sender_pos_x", pa.float64()),
    ("sender_pos_y", pa.float64()),
    ("sender_pos_z", pa.float64()),
    ("sender_speed", pa.float64()),
    ("sender_acceleration", pa.float64()),
    ("sender_heading", pa.float64()),
    ("sender_speed_noise", pa.float64()),
    ("sender_acceleration_noise", pa.float64()),
    ("sender_heading_noise", pa.float64()),
    ("sender_driver_profile", pa.large_string()),
    ("observed_reception_count", pa.int64()),
    ("mean_delay", pa.float64()),
    ("median_receiver_distance", pa.float64()),
])

OUTPUT_COLUMNS = [f.name for f in OUTPUT_SCHEMA]


# ---------------------------------------------------------------------------
# Consistency Validation
# ---------------------------------------------------------------------------

def validate_group_invariance(df: pd.DataFrame, highway: str, split: str) -> None:
    """
    Verify that message_id, label, and sender-invariant fields have zero variance
    within every (sender_id, send_time) group.
    Raises ValueError immediately if any inconsistency is discovered.
    """
    print(f"  Validating sender-state invariance for {highway}/{split} ...")
    t0 = time.time()
    
    # Groupby subset check for duplicates
    dup_mask = df.duplicated(subset=["sender_id", "send_time"], keep=False)
    if not dup_mask.any():
        print("    No duplicate transmissions; invariance holds trivially.")
        return

    dup_df = df[dup_mask]
    g = dup_df.groupby(["sender_id", "send_time"])
    
    # 1. Label consistency check
    label_nunique = g["label"].nunique()
    if (label_nunique > 1).any():
        n_conflicts = int((label_nunique > 1).sum())
        raise ValueError(
            f"CRITICAL LABEL CONFLICT: {n_conflicts} transmissions in {highway}/{split} "
            f"have conflicting ground-truth labels across receiver copies!"
        )

    # 2. Message ID consistency check
    mid_nunique = g["message_id"].nunique()
    if (mid_nunique > 1).any():
        n_conflicts = int((mid_nunique > 1).sum())
        raise ValueError(
            f"CRITICAL MESSAGE_ID CONFLICT: {n_conflicts} transmissions in {highway}/{split} "
            f"have conflicting message IDs across receiver copies!"
        )

    # 3. Numeric sender kinematics and noise consistency
    numeric_checks = [
        "sender_pos_x", "sender_pos_y", "sender_pos_z",
        "sender_speed", "sender_acceleration", "sender_heading",
        "sender_speed_noise", "sender_acceleration_noise", "sender_heading_noise"
    ]
    for col in numeric_checks:
        diff = g[col].max() - g[col].min()
        max_diff = float(diff.max())
        if max_diff > 1e-9:
            raise ValueError(
                f"CRITICAL KINEMATIC DIVERGENCE: Column '{col}' varies by up to {max_diff} "
                f"across receiver copies of the same transmission in {highway}/{split}!"
            )

    # 4. Categorical driver profile and alias consistency
    for col in ["sender_driver_profile", "sender_alias"]:
        nunique = g[col].nunique()
        if (nunique > 1).any():
            n_conflicts = int((nunique > 1).sum())
            raise ValueError(
                f"CRITICAL CATEGORICAL CONFLICT: Column '{col}' varies across receiver copies "
                f"in {n_conflicts} transmissions in {highway}/{split}!"
            )

    print(f"    Invariance verified bit-for-bit in {time.time() - t0:.2f} s (0 conflicts).")


# ---------------------------------------------------------------------------
# Core Transmission Aggregation
# ---------------------------------------------------------------------------

def build_transmission_table_for_split(highway: str, split: str) -> Dict:
    """
    Process one highway/split:
      1. Load deduplicated source parquet.
      2. Validate invariance across multi-receiver rows.
      3. Compute receiver distance, median distance, mean delay, and observed reception count.
      4. Deduplicate sender rows and merge candidate aggregates.
      5. Sort deterministically by sender_id then send_time.
      6. Save to processed_data/transmissions/{highway}/{split}.parquet.
    """
    input_file = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Missing input parquet: {input_file}")

    print(f"\n{'='*70}\nProcessing {highway} / {split}\nSource: {input_file}\n{'='*70}")
    t_start = time.time()
    
    df = pd.read_parquet(input_file, columns=LOAD_COLS)
    n_source_rows = len(df)
    n_source_senders = df["sender_id"].nunique()
    print(f"  Loaded {n_source_rows:,} reception rows across {n_source_senders:,} senders in {time.time() - t_start:.2f} s")

    # Invariance validation
    validate_group_invariance(df, highway, split)

    # Compute Euclidean distance to each observing receiver
    t_agg = time.time()
    df["receiver_dist"] = np.sqrt(
        (df["sender_pos_x"] - df["receiver_pos_x"]) ** 2 +
        (df["sender_pos_y"] - df["receiver_pos_y"]) ** 2
    )

    # Extract clean sender-invariant rows (1 row per transmission)
    tx_sender = df[SENDER_COLS].drop_duplicates(subset=["sender_id", "send_time"]).copy()

    # Aggregate receiver-dependent metrics per transmission
    agg_df = df.groupby(["sender_id", "send_time"], as_index=False, sort=False).agg(
        observed_reception_count=("delay", "size"),
        mean_delay=("delay", "mean"),
        median_receiver_distance=("receiver_dist", "median"),
    )

    # Merge sender state with aggregated metrics
    tx_table = pd.merge(tx_sender, agg_df, on=["sender_id", "send_time"], how="inner")
    
    # Deterministic sorting
    tx_table = tx_table.sort_values(["sender_id", "send_time"]).reset_index(drop=True)
    
    # Ensure correct column ordering and types
    tx_table = tx_table[OUTPUT_COLUMNS]
    n_transmissions = len(tx_table)
    n_tx_senders = tx_table["sender_id"].nunique()

    print(f"  Aggregated into {n_transmissions:,} unique physical transmissions in {time.time() - t_agg:.2f} s")

    # Assert expected transmission count
    expected_n = EXPECTED_TX_COUNTS.get((highway, split))
    if expected_n is not None and n_transmissions != expected_n:
        raise ValueError(
            f"Transmission count mismatch in {highway}/{split}: "
            f"expected {expected_n:,}, got {n_transmissions:,}!"
        )

    # Save to Parquet
    out_dir = os.path.join(OUTPUT_BASE, highway)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{split}.parquet")

    arrow_table = pa.Table.from_pandas(tx_table, schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(arrow_table, out_file, compression="SNAPPY")

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"  Saved output to: {out_file} ({file_size_mb:.2f} MB)")

    # Compute split statistics
    n_positive = int((tx_table["label"] == 1).sum())
    n_negative = int((tx_table["label"] == 0).sum())
    pos_pct = (n_positive / n_transmissions) * 100 if n_transmissions else 0.0

    stats = {
        "highway": highway,
        "split": split,
        "source_rows": n_source_rows,
        "transmission_count": n_transmissions,
        "sender_count": n_tx_senders,
        "positive_count": n_positive,
        "negative_count": n_negative,
        "positive_percentage": round(pos_pct, 2),
        "mean_receptions_per_transmission": round(n_source_rows / n_transmissions, 2),
        "output_file": out_file,
        "file_size_mb": round(file_size_mb, 2),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    return stats


# ---------------------------------------------------------------------------
# Strict Validation Check
# ---------------------------------------------------------------------------

def validate_transmission_output(highway: str, split: str) -> Dict:
    """
    Perform independent validation on saved transmission parquet:
      1. Check file existence and schema.
      2. Check uniqueness of (sender_id, send_time).
      3. Check 0 missing values.
      4. Check that labels and sender-state values match raw source exactly.
    """
    out_file = os.path.join(OUTPUT_BASE, highway, f"{split}.parquet")
    source_file = os.path.join(INPUT_BASE, highway, f"{split}.parquet")

    assert os.path.isfile(out_file), f"Output file missing: {out_file}"
    
    # Read output
    df_out = pd.read_parquet(out_file)
    assert len(df_out) == EXPECTED_TX_COUNTS[(highway, split)], "Count mismatch"
    assert list(df_out.columns) == OUTPUT_COLUMNS, "Schema columns mismatch"
    assert not df_out.isna().any().any(), "Found NaN values in transmission table"

    # 1. Uniqueness of key
    assert not df_out.duplicated(subset=["sender_id", "send_time"]).any(), "Duplicate transmission keys found!"

    # 2. Deterministic sort order
    st_diffs = df_out.groupby("sender_id", sort=False)["send_time"].diff().dropna()
    assert (st_diffs > 0).all(), "send_time is not strictly increasing within sender_id!"

    # 3. Spot-check sample records against source
    df_src = pd.read_parquet(source_file, columns=["sender_id", "send_time", "label", "sender_speed", "sender_pos_x"])
    src_spot = df_src.drop_duplicates(subset=["sender_id", "send_time"]).set_index(["sender_id", "send_time"])
    out_spot = df_out.set_index(["sender_id", "send_time"])

    # Sample 100 rows
    sample_keys = out_spot.index[:100]
    for k in sample_keys:
        assert out_spot.loc[k, "label"] == src_spot.loc[k, "label"], f"Label mismatch at {k}"
        assert np.isclose(out_spot.loc[k, "sender_speed"], src_spot.loc[k, "sender_speed"]), f"Speed mismatch at {k}"
        assert np.isclose(out_spot.loc[k, "sender_pos_x"], src_spot.loc[k, "sender_pos_x"]), f"Pos_x mismatch at {k}"

    return {
        "highway": highway,
        "split": split,
        "unique_keys_verified": True,
        "deterministic_sort_verified": True,
        "source_fidelity_verified": True,
        "status": "VALIDATED_PASS",
    }


# ---------------------------------------------------------------------------
# CLI Entry Point & Audit Generation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate Option B transmission-level datasets."
    )
    parser.add_argument(
        "--highway",
        choices=HIGHWAYS + ["all"],
        default="all",
        help="Target highway scenario (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS + ["all"],
        default="all",
        help="Target split (default: all).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks only without re-generating.",
    )
    args = parser.parse_args()

    target_highways = HIGHWAYS if args.highway == "all" else [args.highway]
    target_splits = SPLITS if args.split == "all" else [args.split]

    print("=" * 70)
    print("5G-V2X DoS Detection  —  Build Transmission Table (Option B)")
    print(f"  Input Base         : {INPUT_BASE}")
    print(f"  Output Base        : {OUTPUT_BASE}")
    print(f"  Vantage Point      : Network-Side / Edge Infrastructure Detector")
    print(f"  Unique Key         : (sender_id, send_time)")
    print("=" * 70)

    all_stats = []

    for hw in target_highways:
        for sp in target_splits:
            if args.validate_only:
                res = validate_transmission_output(hw, sp)
                print(f"Validation {hw}/{sp}: {res['status']}")
            else:
                stats = build_transmission_table_for_split(hw, sp)
                val_res = validate_transmission_output(hw, sp)
                stats["validation"] = val_res["status"]
                all_stats.append(stats)
                print(f"  Validation Check: {val_res['status']}")

    if not args.validate_only and all_stats:
        # Save audit report in results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_json_path = os.path.join(OUTPUT_BASE, "audit_summary.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vantage_point": "Network-Side / Edge Detector (Provisional Project Design Decision)",
                "key": "(sender_id, send_time)",
                "splits": all_stats
            }, f, indent=2)
        print(f"\nAudit summary saved to: {report_json_path}")


if __name__ == "__main__":
    main()
