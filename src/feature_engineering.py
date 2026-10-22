import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 10
STRIDE = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEDUP_BASE = os.path.join(PROJECT_ROOT, "processed_data", "preprocessed_deduplicated")
SEQ_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "context_features")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

# Required source columns in deduplicated parquet files
REQUIRED_SOURCE_COLUMNS = [
    "sender_id",
    "send_time",
    "receive_time",
    "sender_pos_x",
    "sender_pos_y",
    "receiver_pos_x",
    "receiver_pos_y",
    "sender_speed",
    "sender_acceleration",
]

# The agreed 7 context feature column names in exact order
CONTEXT_FEATURE_NAMES = [
    "starting_distance",
    "final_distance",
    "distance_change_rate",
    "message_rate",
    "mean_inter_arrival_time",
    "speed_variation",
    "acceleration_variation",
]

# Obsolete / redundant columns to strictly prohibit from outputs
OBSOLETE_REDUNDANT_COLUMNS = [
    "sender_message_rate",
    "max_sender_rate",
    "mean_communication_delay",
    "inter_arrival_time",
]

# Comprehensive feature catalog documenting retained, removed, and pending features
FEATURE_CATALOG: Dict[str, Dict[str, str]] = {
    "starting_distance": {
        "status": "RETAINED",
        "formula": "sqrt((sender_pos_x[0] - receiver_pos_x[0])^2 + (sender_pos_y[0] - receiver_pos_y[0])^2)",
        "source_columns": "sender_pos_x, receiver_pos_x, sender_pos_y, receiver_pos_y",
        "units": "meters",
        "notes": "Euclidean distance at the initial step of the 10-message window."
    },
    "final_distance": {
        "status": "RETAINED",
        "formula": "sqrt((sender_pos_x[-1] - receiver_pos_x[-1])^2 + (sender_pos_y[-1] - receiver_pos_y[-1])^2)",
        "source_columns": "sender_pos_x, receiver_pos_x, sender_pos_y, receiver_pos_y",
        "units": "meters",
        "notes": "Euclidean distance at the final step of the 10-message window."
    },
    "distance_change_rate": {
        "status": "RETAINED",
        "formula": "(final_distance - starting_distance) / observation_duration_seconds",
        "source_columns": "send_time, sender_pos_x, receiver_pos_x, sender_pos_y, receiver_pos_y",
        "units": "meters/second",
        "notes": "Rate of distance change over window. Safe policy: returns 0.0 if observation duration <= 0 (multi-receiver same-timestamp burst)."
    },
    "message_rate": {
        "status": "RETAINED",
        "formula": "number_of_valid_message_events / observation_window_duration_seconds",
        "source_columns": "send_time",
        "units": "messages/second",
        "notes": "Unique message events deduplicated using unique send_time to prevent inflating count from multi-receiver broadcast copies. Safe policy: returns 0.0 if duration <= 0."
    },
    "mean_inter_arrival_time": {
        "status": "RETAINED",
        "formula": "mean(diff(sort(receive_time))) across window in seconds",
        "source_columns": "receive_time",
        "units": "seconds",
        "notes": "Mean consecutive message reception interval sorted chronologically within the single-sender window."
    },
    "speed_variation": {
        "status": "RETAINED",
        "formula": "std(sender_speed within context window)",
        "source_columns": "sender_speed",
        "units": "meters/second",
        "notes": "Population standard deviation (ddof=0) of sender_speed across the 10 messages of the window."
    },
    "acceleration_variation": {
        "status": "RETAINED",
        "formula": "std(sender_acceleration within context window)",
        "source_columns": "sender_acceleration",
        "units": "meters/second²",
        "notes": "Population standard deviation (ddof=0) of sender_acceleration across the 10 messages of the window."
    },
    "sender_message_rate": {
        "status": "REMOVED_REDUNDANT",
        "formula": "message_count_for_sender / observation_window_duration_seconds",
        "source_columns": "sender_id, send_time",
        "units": "messages/second",
        "notes": "Removed: Identical to message_rate because every context window contains only one sender."
    },
    "max_sender_rate": {
        "status": "REMOVED_REDUNDANT",
        "formula": "maximum(sender_message_rate across senders in window)",
        "source_columns": "sender_id, send_time",
        "units": "messages/second",
        "notes": "Removed: Identical to message_rate under the single-sender window design."
    },
    "mean_communication_delay": {
        "status": "REMOVED_REDUNDANT",
        "formula": "mean(receive_time - send_time) in seconds",
        "source_columns": "receive_time, send_time, delay",
        "units": "seconds",
        "notes": "Removed: Deliberate redundancy-removal decision because it is derived from the same send/receive timing information represented by the existing temporal delay feature in the Transformer input."
    },
    "active_vehicle_count": {
        "status": "PENDING",
        "formula": "number_of_senders_whose_message_count >= minimum_message_threshold",
        "source_columns": "sender_id",
        "units": "count",
        "notes": "Pending: minimum_message_threshold is not specified in project documentation or codebase. Prohibited from selecting an arbitrary threshold."
    },
    "message_load_per_active_vehicle": {
        "status": "PENDING",
        "formula": "number_of_valid_message_events / active_vehicle_count",
        "source_columns": "send_time, sender_id",
        "units": "messages/vehicle",
        "notes": "Pending: Dependent on active_vehicle_count. Safe policy for zero active vehicles: 0.0."
    }
}


# ---------------------------------------------------------------------------
# Schema Inspection & Verification
# ---------------------------------------------------------------------------

def inspect_schema(df: pd.DataFrame) -> Dict[str, str]:
    """
    Verify presence and data types of required columns in the source DataFrame.
    Returns mapping of verified column names.
    Raises KeyError if any mandatory column is missing.
    """
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in source dataset: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return {col: str(df[col].dtype) for col in REQUIRED_SOURCE_COLUMNS}


# ---------------------------------------------------------------------------
# Core Vectorized Feature Extraction per Sender Group
# ---------------------------------------------------------------------------

def compute_context_features_for_group(
    xs: np.ndarray,      # sender_pos_x
    ys: np.ndarray,      # sender_pos_y
    xr: np.ndarray,      # receiver_pos_x
    yr: np.ndarray,      # receiver_pos_y
    st: np.ndarray,      # send_time (ns)
    rt: np.ndarray,      # receive_time (ns)
    spd: np.ndarray,     # sender_speed (m/s)
    acl: np.ndarray,     # sender_acceleration (m/s²)
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Optional[np.ndarray]:
    """
    Compute context feature vectors for a single sender group using fully
    vectorized operations.

    Input arrays must already be sorted chronologically by send_time.
    Returns an array of shape (n_windows, 7),
    or None if the group has fewer messages than window_size.
    """
    n_rows = len(st)
    if n_rows < window_size:
        return None

    k_windows = (n_rows - window_size) // stride + 1

    # 1. Starting distance (meters): Euclidean distance at window start (index i)
    xs_start = xs[:k_windows]
    ys_start = ys[:k_windows]
    xr_start = xr[:k_windows]
    yr_start = yr[:k_windows]
    start_dist = np.sqrt((xs_start - xr_start) ** 2 + (ys_start - yr_start) ** 2)

    # 2. Final distance (meters): Euclidean distance at window end (index i + 9)
    end_offset = window_size - 1
    xs_end = xs[end_offset : end_offset + k_windows]
    ys_end = ys[end_offset : end_offset + k_windows]
    xr_end = xr[end_offset : end_offset + k_windows]
    yr_end = yr[end_offset : end_offset + k_windows]
    end_dist = np.sqrt((xs_end - xr_end) ** 2 + (ys_end - yr_end) ** 2)

    # Observation duration (seconds) derived from send_time timestamps
    st_start = st[:k_windows]
    st_end = st[end_offset : end_offset + k_windows]
    durations = (st_end - st_start) / 1e9

    # 3. Rate of distance change (m/s)
    dist_change_rate = np.where(
        durations > 0.0,
        (end_dist - start_dist) / np.maximum(durations, 1e-9),
        0.0,
    )

    # 4. Overall message rate (unique message events / duration in seconds)
    is_new_msg = (st[1:] != st[:-1]).astype(np.int32)
    cumsum_new = np.concatenate([[0], np.cumsum(is_new_msg)])
    n_valid_msgs = 1 + cumsum_new[end_offset : end_offset + k_windows] - cumsum_new[:k_windows]
    msg_rate = np.where(
        durations > 0.0,
        n_valid_msgs / np.maximum(durations, 1e-9),
        0.0,
    )

    # 5. Mean Inter-arrival time (mean sorted receive_time diffs in seconds)
    rt_wv = sliding_window_view(rt, window_size)
    sorted_rt = np.sort(rt_wv, axis=1)
    diffs_rt = np.diff(sorted_rt, axis=1) / 1e9
    mean_iat = np.mean(diffs_rt, axis=1)

    # 6. Speed variation (population standard deviation in m/s)
    spd_wv = sliding_window_view(spd, window_size)
    spd_std = np.std(spd_wv, axis=1)

    # 7. Acceleration variation (population standard deviation in m/s²)
    acl_wv = sliding_window_view(acl, window_size)
    acl_std = np.std(acl_wv, axis=1)

    # Stack the 7 retained features in exact specified order
    return np.column_stack([
        start_dist,
        end_dist,
        dist_change_rate,
        msg_rate,
        mean_iat,
        spd_std,
        acl_std,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Split Processor
# ---------------------------------------------------------------------------

def process_split(
    highway: str,
    split: str,
    output_base: str = OUTPUT_BASE,
) -> Dict:
    """
    Process a single split (highway, split):
      1. Load deduplicated source parquet with required columns only.
      2. Group by sender_id and stable-sort by send_time (exact create_sequences alignment).
      3. Compute vectorized context features per window for the 7 retained features.
      4. Verify alignment against existing sequences.npz and meta.json.
      5. Save context_features.npz, context_features.parquet, and meta.json.
    """
    dedup_path = os.path.join(DEDUP_BASE, highway, f"{split}.parquet")
    seq_meta_path = os.path.join(SEQ_BASE, highway, split, "meta.json")

    if not os.path.isfile(dedup_path):
        raise FileNotFoundError(f"Deduplicated source file missing: {dedup_path}")

    print(f"\n[{highway} / {split}] Loading deduplicated source ...")
    t0 = time.time()
    df = pd.read_parquet(dedup_path, columns=REQUIRED_SOURCE_COLUMNS)
    load_time = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {load_time:.2f} s")

    # Inspect schema
    inspect_schema(df)

    # Extract raw numpy arrays
    sender_all = df["sender_id"].to_numpy()
    send_time_all = df["send_time"].to_numpy()
    rt_all = df["receive_time"].to_numpy()
    xs_all = df["sender_pos_x"].to_numpy()
    ys_all = df["sender_pos_y"].to_numpy()
    xr_all = df["receiver_pos_x"].to_numpy()
    yr_all = df["receiver_pos_y"].to_numpy()
    spd_all = df["sender_speed"].to_numpy()
    acl_all = df["sender_acceleration"].to_numpy()

    # Free DataFrame memory
    del df

    # Unique senders in exact same order as create_sequences.py
    unique_senders = np.unique(sender_all)
    n_groups = len(unique_senders)
    skipped_small = 0

    print(f"  Computing 7 context features for {n_groups:,} sender streams ...")
    t1 = time.time()

    all_context_blocks: List[np.ndarray] = []

    for sid in unique_senders:
        mask = sender_all == sid
        idxs = np.where(mask)[0]

        # Stable sort by send_time within group (strictly mirrors create_sequences.py)
        times = send_time_all[idxs]
        order = np.argsort(times, kind="stable")
        idxs = idxs[order]

        grp_feats = compute_context_features_for_group(
            xs=xs_all[idxs],
            ys=ys_all[idxs],
            xr=xr_all[idxs],
            yr=yr_all[idxs],
            st=send_time_all[idxs],
            rt=rt_all[idxs],
            spd=spd_all[idxs],
            acl=acl_all[idxs],
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        if grp_feats is None:
            skipped_small += 1
        else:
            all_context_blocks.append(grp_feats)

    # Concatenate all blocks
    if all_context_blocks:
        C_out = np.concatenate(all_context_blocks, axis=0)
    else:
        C_out = np.empty((0, len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)

    compute_time = time.time() - t1
    print(
        f"  Computed {len(C_out):,} context vectors (skipped {skipped_small} small groups) "
        f"in {compute_time:.2f} s"
    )

    # Verify alignment against existing sequences.npz and meta.json
    expected_sequences = None
    if os.path.isfile(seq_meta_path):
        with open(seq_meta_path, "r", encoding="utf-8") as f:
            seq_meta = json.load(f)
        expected_sequences = seq_meta.get("n_sequences")
        if expected_sequences is not None:
            if len(C_out) != expected_sequences:
                raise ValueError(
                    f"Alignment Error: Generated {len(C_out)} context vectors, but "
                    f"existing sequence metadata specifies {expected_sequences} sequences."
                )
            print(f"  [VERIFIED] Exact sequence alignment match: {len(C_out):,} vectors.")

    # Integrity verification: check for NaN / Inf
    n_nans = int(np.isnan(C_out).sum())
    n_infs = int(np.isinf(C_out).sum())
    if n_nans > 0 or n_infs > 0:
        raise ValueError(
            f"Numerical corruption detected: {n_nans} NaNs and {n_infs} Infs in context features."
        )

    # Save outputs under processed_data/context_features/{highway}/{split}/
    out_dir = os.path.join(output_base, highway, split)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "context_features.npz")
    parquet_path = os.path.join(out_dir, "context_features.parquet")
    meta_path = os.path.join(out_dir, "meta.json")

    # 1. Save compressed NPZ
    np.savez_compressed(npz_path, context_features=C_out)

    # 2. Save Parquet
    df_context = pd.DataFrame(C_out, columns=CONTEXT_FEATURE_NAMES)
    df_context.to_parquet(parquet_path, index=False)
    del df_context

    # 3. Save comprehensive metadata
    meta = {
        "highway": highway,
        "split": split,
        "n_sequences": len(C_out),
        "n_context_features": len(CONTEXT_FEATURE_NAMES),
        "feature_names": CONTEXT_FEATURE_NAMES,
        "context_shape": list(C_out.shape),
        "context_dtype": str(C_out.dtype),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "n_sender_groups": n_groups,
        "n_skipped_small": skipped_small,
        "alignment_verified_against_sequences": expected_sequences is not None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deliberate_redundancy_removal_decisions": [
            {
                "feature": "sender_message_rate",
                "reason": "Identical to message_rate because every context window contains only one sender."
            },
            {
                "feature": "max_sender_rate",
                "reason": "Identical to message_rate under the single-sender window design."
            },
            {
                "feature": "mean_communication_delay",
                "reason": "Derived from the same send/receive timing information represented by the existing temporal delay feature in the Transformer input."
            }
        ],
        "pending_features": [
            {
                "feature": "active_vehicle_count",
                "reason": "Active-vehicle threshold (minimum_message_threshold) has not been established; arbitrary threshold prohibited."
            },
            {
                "feature": "message_load_per_active_vehicle",
                "reason": "Directly dependent on active_vehicle_count."
            }
        ],
        "catalog": FEATURE_CATALOG,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Outputs saved to: {out_dir}")
    print(f"    - {os.path.basename(npz_path)} ({os.path.getsize(npz_path) / (1024*1024):.2f} MB)")
    print(f"    - {os.path.basename(parquet_path)} ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")
    print(f"    - {os.path.basename(meta_path)}")

    return meta


# ---------------------------------------------------------------------------
# Validation Routine
# ---------------------------------------------------------------------------

def validate_context_features(highway: str, split: str, output_base: str = OUTPUT_BASE) -> Dict:
    """
    Perform lightweight validation on generated context features:
      1. Check file existence (.npz, .parquet, meta.json).
      2. Verify array shapes (must be exactly (N, 7)) and float32 dtype.
      3. Verify exact feature names match CONTEXT_FEATURE_NAMES.
      4. Verify obsolete redundant columns are completely absent.
      5. Verify exact alignment with sequences.npz.
      6. Verify statistical sanity (non-negative distances, finite values).
    """
    out_dir = os.path.join(output_base, highway, split)
    npz_path = os.path.join(out_dir, "context_features.npz")
    parquet_path = os.path.join(out_dir, "context_features.parquet")
    meta_path = os.path.join(out_dir, "meta.json")
    seq_npz_path = os.path.join(SEQ_BASE, highway, split, "sequences.npz")

    assert os.path.isfile(npz_path), f"Missing {npz_path}"
    assert os.path.isfile(parquet_path), f"Missing {parquet_path}"
    assert os.path.isfile(meta_path), f"Missing {meta_path}"

    with np.load(npz_path) as data:
        C = data["context_features"]

    assert C.ndim == 2, f"Expected 2D context array, got ndim={C.ndim}"
    assert C.shape[1] == len(CONTEXT_FEATURE_NAMES), (
        f"Feature dimension mismatch: expected {len(CONTEXT_FEATURE_NAMES)}, got {C.shape[1]}"
    )
    assert not np.isnan(C).any(), "Found NaNs in context features"
    assert not np.isinf(C).any(), "Found Infs in context features"

    # Verify distances, message rate, IAT, variations are non-negative
    start_dists = C[:, CONTEXT_FEATURE_NAMES.index("starting_distance")]
    end_dists = C[:, CONTEXT_FEATURE_NAMES.index("final_distance")]
    msg_rate = C[:, CONTEXT_FEATURE_NAMES.index("message_rate")]
    mean_iat = C[:, CONTEXT_FEATURE_NAMES.index("mean_inter_arrival_time")]
    spd_var = C[:, CONTEXT_FEATURE_NAMES.index("speed_variation")]
    acl_var = C[:, CONTEXT_FEATURE_NAMES.index("acceleration_variation")]

    assert (start_dists >= 0.0).all(), "Negative starting distance found"
    assert (end_dists >= 0.0).all(), "Negative final distance found"
    assert (msg_rate >= 0.0).all(), "Negative message rate found"
    assert (mean_iat >= 0.0).all(), "Negative mean IAT found"
    assert (spd_var >= 0.0).all(), "Negative speed variation found"
    assert (acl_var >= 0.0).all(), "Negative acceleration variation found"

    # Verify Parquet columns match exact 7 names and have no obsolete columns
    df_p = pd.read_parquet(parquet_path)
    assert len(df_p) == len(C), "Parquet row count does not match NPZ"
    assert list(df_p.columns) == CONTEXT_FEATURE_NAMES, (
        f"Parquet columns mismatch: {list(df_p.columns)} != {CONTEXT_FEATURE_NAMES}"
    )

    for obsolete_col in OBSOLETE_REDUNDANT_COLUMNS:
        assert obsolete_col not in df_p.columns, (
            f"Obsolete redundant column '{obsolete_col}' found in regenerated output!"
        )

    # Verify alignment with sequences.npz if present
    seq_aligned = False
    if os.path.isfile(seq_npz_path):
        with np.load(seq_npz_path) as sdata:
            n_seq = len(sdata["y"])
        assert len(C) == n_seq, (
            f"Shape mismatch: {len(C)} context vectors vs {n_seq} sequences"
        )
        seq_aligned = True

    stats = {
        "highway": highway,
        "split": split,
        "n_samples": len(C),
        "n_features": C.shape[1],
        "feature_names": CONTEXT_FEATURE_NAMES,
        "dtype": str(C.dtype),
        "sequence_aligned": seq_aligned,
        "obsolete_columns_absent": True,
        "status": "VALIDATED_PASS",
    }
    return stats


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and validate 7 refined context features for 5G-V2X DoS detection."
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
        help="Run validation on already-generated context features without re-extracting.",
    )
    args = parser.parse_args()

    target_highways = HIGHWAYS if args.highway == "all" else [args.highway]
    target_splits = SPLITS if args.split == "all" else [args.split]

    print("=" * 70)
    print("5G-V2X DoS Detection  —  Refined Context Feature Engineering")
    print(f"  Window Size        : {WINDOW_SIZE}")
    print(f"  Stride             : {STRIDE}")
    print(f"  Retained Features  : {len(CONTEXT_FEATURE_NAMES)}")
    print(f"  Feature Names      : {CONTEXT_FEATURE_NAMES}")
    print(f"  Output Directory   : {OUTPUT_BASE}")
    print("=" * 70)

    for hw in target_highways:
        for sp in target_splits:
            if args.validate_only:
                print(f"\nValidating context features for {hw} / {sp} ...")
                res = validate_context_features(hw, sp)
                print(f"  Result: {res}")
            else:
                meta = process_split(hw, sp)
                val_res = validate_context_features(hw, sp)
                print(f"  Validation: {val_res['status']} ({val_res['n_samples']:,} samples, {val_res['n_features']} features)")

    print("\n" + "=" * 70)
    print("Refined context feature extraction and validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
