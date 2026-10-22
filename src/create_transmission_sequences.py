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
# Configuration & Paths
# ---------------------------------------------------------------------------
WINDOW_SIZE = 10
STRIDE = 1

TARGET_COLUMN = "label"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmissions_scaled")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences_tx")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

# The 9 confirmed sender-invariant Transformer input features
TRANSFORMER_FEATURES = [
    "sender_speed",
    "sender_acceleration",
    "sender_heading",
    "sender_speed_noise",
    "sender_acceleration_noise",
    "sender_heading_noise",
    "sender_driver_profile_AGGRESSIVE",
    "sender_driver_profile_CAUTIOUS",
    "sender_driver_profile_NORMAL",
]


def process_split(highway: str, split: str) -> dict:
    """
    Builds sliding window sequences for a single highway/split from scaled transmissions.
    """
    print(f"\n-------------------------------------------------------------")
    print(f"Building Sequences: {highway} / {split}")
    print(f"-------------------------------------------------------------")
    t0 = time.time()

    split_path = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Missing scaled transmission input: {split_path}")

    # Read the required columns
    load_cols = [
        "sender_id",
        "send_time",
        "label",
        "sender_pos_x",
        "sender_pos_y",
        "median_receiver_distance",
    ] + TRANSFORMER_FEATURES

    df = pq.read_table(split_path, columns=load_cols).to_pandas()
    n_rows = len(df)
    print(f"  Loaded {n_rows:,} transmissions from {split_path}")

    # Extract numpy arrays for fast vectorised operations
    sender_arr = df["sender_id"].to_numpy()
    send_time_arr = df["send_time"].to_numpy(dtype=np.int64)
    label_arr = df["label"].to_numpy(dtype=np.int8)
    pos_x_arr = df["sender_pos_x"].to_numpy(dtype=np.float64)
    pos_y_arr = df["sender_pos_y"].to_numpy(dtype=np.float64)
    dist_arr = df["median_receiver_distance"].to_numpy(dtype=np.float64)
    feat_arr = df[TRANSFORMER_FEATURES].to_numpy(dtype=np.float32)

    del df  # Free DataFrame memory

    # Group by sender
    unique_senders, sender_first_idx = np.unique(sender_arr, return_index=True)
    n_senders_total = len(unique_senders)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    window_counter = 0
    senders_with_seq = 0
    senders_skipped_lt10 = 0
    total_violations = 0

    for sid in unique_senders:
        idxs = np.where(sender_arr == sid)[0]
        n_pts = len(idxs)

        if n_pts < WINDOW_SIZE:
            senders_skipped_lt10 += 1
            continue

        # Check send_time sorting
        times = send_time_arr[idxs]
        diffs = np.diff(times)
        if (diffs <= 0).any():
            total_violations += 1
            # Sort chronologically if not strictly monotonic
            order = np.argsort(times, kind="stable")
            idxs = idxs[order]
            times = times[order]
            diffs = np.diff(times)
            if (diffs <= 0).any():
                raise ValueError(
                    f"Duplicate send_time encountered within sender {sid} in {highway}/{split}!"
                )

        # Slice sender arrays
        feat_g = feat_arr[idxs]
        label_g = label_arr[idxs]
        times_g = send_time_arr[idxs]
        pos_x_g = pos_x_arr[idxs]
        pos_y_g = pos_y_arr[idxs]
        dist_g = dist_arr[idxs]

        n_windows = (n_pts - WINDOW_SIZE) // STRIDE + 1
        senders_with_seq += 1

        X_g = np.empty((n_windows, WINDOW_SIZE, len(TRANSFORMER_FEATURES)), dtype=np.float32)
        y_g = np.empty(n_windows, dtype=np.int8)

        for w in range(n_windows):
            start = w * STRIDE
            end = start + WINDOW_SIZE

            X_g[w] = feat_g[start:end]
            # Any-positive rule: 1 if any transmission in window is labeled 1
            w_label = 1 if (label_g[start:end] == 1).any() else 0
            y_g[w] = w_label

            duration_ns = int(times_g[end - 1] - times_g[start])
            if duration_ns <= 0:
                raise ValueError(f"Non-positive window duration in window {window_counter}: {duration_ns}")

            metadata_rows.append({
                "window_id": window_counter,
                "sender_id": sid,
                "window_index": w,
                "start_send_time": int(times_g[start]),
                "end_send_time": int(times_g[end - 1]),
                "observation_duration_ns": duration_ns,
                "observation_duration_s": float(duration_ns * 1e-9),
                "start_pos_x": float(pos_x_g[start]),
                "start_pos_y": float(pos_y_g[start]),
                "end_pos_x": float(pos_x_g[end - 1]),
                "end_pos_y": float(pos_y_g[end - 1]),
                "start_median_receiver_dist": float(dist_g[start]),
                "end_median_receiver_dist": float(dist_g[end - 1]),
                "label": int(w_label),
            })
            window_counter += 1

        all_X.append(X_g)
        all_y.append(y_g)

    if total_violations > 0:
        print(f"  [WARN] {total_violations} senders required send_time re-sorting.")

    # Concatenate all sequences
    if all_X:
        X_out = np.concatenate(all_X, axis=0)  # shape: (N, 10, 9)
        y_out = np.concatenate(all_y, axis=0)  # shape: (N,)
    else:
        X_out = np.empty((0, WINDOW_SIZE, len(TRANSFORMER_FEATURES)), dtype=np.float32)
        y_out = np.empty(0, dtype=np.int8)

    n_sequences = len(y_out)
    n_positive = int((y_out == 1).sum())
    n_negative = int((y_out == 0).sum())
    pos_ratio = float(n_positive / n_sequences) if n_sequences > 0 else 0.0

    print(f"  Constructed {n_sequences:,} sequences:")
    print(f"    Positive (DoS): {n_positive:,} ({pos_ratio * 100:.2f}%)")
    print(f"    Negative (Benign): {n_negative:,} ({(1 - pos_ratio) * 100:.2f}%)")
    print(f"    Tensor shape X: {X_out.shape}, dtype: {X_out.dtype}")
    print(f"    Array shape y: {y_out.shape}, dtype: {y_out.dtype}")

    # Verify no NaN or Inf in X
    if np.isnan(X_out).any():
        raise ValueError(f"NaN detected in sequence tensor X for {highway}/{split}!")
    if np.isinf(X_out).any():
        raise ValueError(f"Inf detected in sequence tensor X for {highway}/{split}!")

    # Save directory
    out_dir = os.path.join(OUTPUT_BASE, highway, split)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save sequences.npz
    npz_path = os.path.join(out_dir, "sequences.npz")
    window_ids_arr = np.array([r["window_id"] for r in metadata_rows], dtype=np.int64)
    np.savez_compressed(npz_path, X=X_out, y=y_out, window_ids=window_ids_arr)
    npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"  Saved sequences archive -> {npz_path} ({npz_size_mb:.2f} MB)")

    # 2. Save window_metadata.parquet
    meta_df = pd.DataFrame(metadata_rows)
    meta_parquet_path = os.path.join(out_dir, "window_metadata.parquet")
    meta_table = pa.Table.from_pandas(meta_df, preserve_index=False)
    pq.write_table(meta_table, meta_parquet_path, compression="snappy")
    meta_pq_size_mb = os.path.getsize(meta_parquet_path) / (1024 * 1024)
    print(f"  Saved window metadata -> {meta_parquet_path} ({meta_pq_size_mb:.2f} MB)")

    # 3. Save meta.json
    meta_json_path = os.path.join(out_dir, "meta.json")
    meta_info = {
        "highway": highway,
        "split": split,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "n_transmissions": n_rows,
        "n_sequences": n_sequences,
        "n_features": len(TRANSFORMER_FEATURES),
        "feature_names": TRANSFORMER_FEATURES,
        "tensor_shape": list(X_out.shape),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_ratio": round(pos_ratio, 6),
        "n_senders_total": n_senders_total,
        "n_senders_with_sequences": senders_with_seq,
        "n_senders_skipped_lt10": senders_skipped_lt10,
        "label_rule": "any_positive",
        "timestamp_violations_found": total_violations,
        "duration_min_s": round(float(meta_df["observation_duration_s"].min()), 4),
        "duration_mean_s": round(float(meta_df["observation_duration_s"].mean()), 4),
        "duration_max_s": round(float(meta_df["observation_duration_s"].max()), 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_json_path, "w") as fp:
        json.dump(meta_info, fp, indent=2)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f} s.")
    return meta_info


def main():
    start_time = time.time()
    print("=================================================================")
    print("Option B: Generating Transformer Sequences for 5G-V2X DoS Detection")
    print("=================================================================")

    summary = {
        "pipeline_stage": "Option B - Transmission Sequence Generation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "transformer_features": TRANSFORMER_FEATURES,
        "splits": {},
    }

    total_seq = 0
    total_pos = 0
    total_neg = 0

    for highway in HIGHWAYS:
        for split in SPLITS:
            meta = process_split(highway, split)
            summary["splits"][f"{highway}/{split}"] = meta
            total_seq += meta["n_sequences"]
            total_pos += meta["n_positive"]
            total_neg += meta["n_negative"]

    summary["total_sequences"] = total_seq
    summary["total_positive"] = total_pos
    summary["total_negative"] = total_neg
    summary["overall_positive_ratio"] = round(total_pos / total_seq, 6) if total_seq > 0 else 0.0

    summary_path = os.path.join(OUTPUT_BASE, "sequence_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"Sequence Generation Complete across all 6 splits in {elapsed:.2f} s.")
    print(f"Total Sequences: {total_seq:,} (Positive: {total_pos:,}, Negative: {total_neg:,})")
    print(f"Summary written to: {summary_path}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
