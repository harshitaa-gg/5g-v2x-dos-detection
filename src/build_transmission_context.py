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
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TX_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmissions")
SEQ_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences_tx")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmission_context")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

WINDOW_SIZE = 10
STRIDE = 1

PRIMARY_CONTEXT_FEATURES = [
    "starting_distance",
    "final_distance",
    "distance_change_rate",
    "message_rate",
    "iat_variation",
    "speed_variation",
    "acceleration_variation",
]


def process_split(highway: str, split: str) -> Tuple[pd.DataFrame, dict]:
    """
    Computes context features for a single highway/split and verifies
    1-to-1 alignment against sequence metadata.
    """
    print(f"\n=============================================================")
    print(f"Computing Context Features: {highway} / {split}")
    print(f"=============================================================")
    t0 = time.time()

    tx_path = os.path.join(TX_BASE, highway, f"{split}.parquet")
    meta_path = os.path.join(SEQ_BASE, highway, split, "window_metadata.parquet")
    npz_path = os.path.join(SEQ_BASE, highway, split, "sequences.npz")

    if not os.path.isfile(tx_path):
        raise FileNotFoundError(f"Missing transmission table: {tx_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing sequence metadata: {meta_path}")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing sequence npz: {npz_path}")

    # Load sequence metadata and npz for verification
    meta_df = pq.read_table(meta_path).to_pandas()
    n_expected_windows = len(meta_df)
    npz = np.load(npz_path)
    X_seq = npz["X"]
    y_seq = npz["y"]
    window_ids_npz = npz["window_ids"]

    assert len(X_seq) == n_expected_windows, f"Sequence count mismatch: {len(X_seq)} vs {n_expected_windows}"
    assert (window_ids_npz == meta_df["window_id"].to_numpy()).all(), "Window ID mismatch between NPZ and parquet!"

    # Load raw, unscaled transmission data
    cols = ["sender_id", "send_time", "sender_speed", "sender_acceleration", "median_receiver_distance", "label"]
    tx_df = pq.read_table(tx_path, columns=cols).to_pandas()
    n_transmissions = len(tx_df)
    print(f"  Loaded {n_transmissions:,} transmissions from {tx_path}")
    print(f"  Loaded {n_expected_windows:,} expected windows from {meta_path}")

    sender_arr = tx_df["sender_id"].to_numpy()
    time_arr = tx_df["send_time"].to_numpy(dtype=np.int64)
    spd_arr = tx_df["sender_speed"].to_numpy(dtype=np.float64)
    acl_arr = tx_df["sender_acceleration"].to_numpy(dtype=np.float64)
    dist_arr = tx_df["median_receiver_distance"].to_numpy(dtype=np.float64)
    label_arr = tx_df["label"].to_numpy(dtype=np.int8)

    del tx_df

    unique_senders = np.unique(sender_arr)

    context_rows: List[dict] = []
    window_id_counter = 0

    for sid in unique_senders:
        idxs = np.where(sender_arr == sid)[0]
        n_pts = len(idxs)
        if n_pts < WINDOW_SIZE:
            continue

        # Extract sender arrays
        times = time_arr[idxs]
        speeds = spd_arr[idxs]
        acls = acl_arr[idxs]
        dists = dist_arr[idxs]
        labels = label_arr[idxs]

        # Enforce strict timestamp monotonicity
        diffs_check = np.diff(times)
        if (diffs_check <= 0).any():
            order = np.argsort(times, kind="stable")
            idxs = idxs[order]
            times = times[order]
            speeds = speeds[order]
            acls = acls[order]
            dists = dists[order]
            labels = labels[order]
            if (np.diff(times) <= 0).any():
                raise ValueError(f"Duplicate timestamp in sender {sid}")

        n_windows = (n_pts - WINDOW_SIZE) // STRIDE + 1

        # Vectorized sliding window slices
        time_wv = sliding_window_view(times, WINDOW_SIZE)
        spd_wv = sliding_window_view(speeds, WINDOW_SIZE)
        acl_wv = sliding_window_view(acls, WINDOW_SIZE)
        dist_wv = sliding_window_view(dists, WINDOW_SIZE)
        lbl_wv = sliding_window_view(labels, WINDOW_SIZE)

        for w in range(n_windows):
            t_win = time_wv[w]
            s_win = spd_wv[w]
            a_win = acl_wv[w]
            d_win = dist_wv[w]
            l_win = lbl_wv[w]

            start_t = int(t_win[0])
            end_t = int(t_win[-1])
            duration_ns = end_t - start_t
            if duration_ns <= 0:
                raise ValueError(f"Invalid non-positive duration: {duration_ns} ns")
            duration_s = float(duration_ns * 1e-9)

            # Feature 1 & 2: Starting and Final median receiver distance
            start_dist = float(d_win[0])
            final_dist = float(d_win[-1])

            # Feature 3: Distance change rate (m/s)
            dist_change_rate = float((final_dist - start_dist) / duration_s)

            # Feature 4: Message rate (pkts/s) - 10 discrete packet transmissions / duration
            message_rate = float(WINDOW_SIZE / duration_s)

            # Auxiliary: Interval frequency (Hz) - 9 intervals / duration = 1 / mean_iat
            interval_rate = float((WINDOW_SIZE - 1) / duration_s)

            # Feature 5: IAT variation (s) - sample std of the 9 consecutive send gaps (ddof=1)
            iat_gaps = np.diff(t_win) * 1e-9
            iat_variation = float(np.std(iat_gaps, ddof=1))

            # Feature 6: Speed variation (m/s) - sample std (ddof=1) of raw speed across 10 pkts
            speed_variation = float(np.std(s_win, ddof=1))

            # Feature 7: Acceleration variation (m/s²) - sample std (ddof=1) of raw accel across 10 pkts
            acceleration_variation = float(np.std(a_win, ddof=1))

            w_label = int(1 if (l_win == 1).any() else 0)

            context_rows.append({
                "window_id": window_id_counter,
                "sender_id": sid,
                "window_index": w,
                "start_send_time": start_t,
                "end_send_time": end_t,
                "observation_duration_s": duration_s,
                "label": w_label,
                "starting_distance": start_dist,
                "final_distance": final_dist,
                "distance_change_rate": dist_change_rate,
                "message_rate": message_rate,
                "message_rate_interval": interval_rate,
                "iat_variation": iat_variation,
                "speed_variation": speed_variation,
                "acceleration_variation": acceleration_variation,
            })
            window_id_counter += 1

    df_context = pd.DataFrame(context_rows)

    # -----------------------------------------------------------------------
    # Comprehensive Alignment & Integrity Verification
    # -----------------------------------------------------------------------
    print(f"  Verifying 1-to-1 alignment against sequence metadata ...")
    assert len(df_context) == n_expected_windows, (
        f"Row count mismatch: generated {len(df_context)} vs expected {n_expected_windows}"
    )

    # Check key alignment bit-for-bit
    assert (df_context["window_id"].to_numpy() == meta_df["window_id"].to_numpy()).all(), "window_id mismatch!"
    assert (df_context["sender_id"].to_numpy() == meta_df["sender_id"].to_numpy()).all(), "sender_id mismatch!"
    assert (df_context["start_send_time"].to_numpy() == meta_df["start_send_time"].to_numpy()).all(), "start_send_time mismatch!"
    assert (df_context["end_send_time"].to_numpy() == meta_df["end_send_time"].to_numpy()).all(), "end_send_time mismatch!"
    assert (df_context["label"].to_numpy() == meta_df["label"].to_numpy()).all(), "label mismatch against window_metadata!"
    assert (df_context["label"].to_numpy() == y_seq).all(), "label mismatch against sequences.npz y!"

    # Starting and ending distances must match window_metadata exactly
    np.testing.assert_allclose(
        df_context["starting_distance"].to_numpy(),
        meta_df["start_median_receiver_dist"].to_numpy(),
        rtol=1e-5, atol=1e-5, err_msg="starting_distance mismatch with window_metadata!"
    )
    np.testing.assert_allclose(
        df_context["final_distance"].to_numpy(),
        meta_df["end_median_receiver_dist"].to_numpy(),
        rtol=1e-5, atol=1e-5, err_msg="final_distance mismatch with window_metadata!"
    )

    # Null, NaN, Inf, and non-positive duration checks
    assert not df_context.isnull().any().any(), "Found NULLs in context features!"
    for feat in PRIMARY_CONTEXT_FEATURES:
        vals = df_context[feat].to_numpy()
        assert not np.isnan(vals).any(), f"Found NaNs in {feat}!"
        assert not np.isinf(vals).any(), f"Found Infs in {feat}!"

    assert (df_context["observation_duration_s"] > 0).all(), "Found non-positive durations!"
    assert (df_context["starting_distance"] >= 0).all(), "Found negative starting distance!"
    assert (df_context["final_distance"] >= 0).all(), "Found negative final distance!"
    assert (df_context["message_rate"] > 0).all(), "Found non-positive message rate!"
    assert (df_context["iat_variation"] >= 0).all(), "Found negative iat_variation!"
    assert (df_context["speed_variation"] >= 0).all(), "Found negative speed_variation!"
    assert (df_context["acceleration_variation"] >= 0).all(), "Found negative acceleration_variation!"

    print("  1-to-1 alignment and mathematical validity confirmed 100%.")

    # -----------------------------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------------------------
    out_dir = os.path.join(OUTPUT_BASE, highway, split)
    os.makedirs(out_dir, exist_ok=True)

    out_parquet_path = os.path.join(out_dir, "context_features.parquet")
    table = pa.Table.from_pandas(df_context, preserve_index=False)
    pq.write_table(table, out_parquet_path, compression="snappy")
    file_size_mb = os.path.getsize(out_parquet_path) / (1024 * 1024)
    print(f"  Saved context features -> {out_parquet_path} ({len(df_context):,} rows, {file_size_mb:.2f} MB)")

    # Save metadata summary for split
    meta_json_path = os.path.join(out_dir, "context_metadata.json")
    stats = {}
    for feat in PRIMARY_CONTEXT_FEATURES + ["message_rate_interval"]:
        c = df_context[feat]
        stats[feat] = {
            "min": round(float(c.min()), 4),
            "mean": round(float(c.mean()), 4),
            "median": round(float(c.median()), 4),
            "max": round(float(c.max()), 4),
            "std": round(float(c.std()), 4),
        }

    meta_info = {
        "highway": highway,
        "split": split,
        "n_windows": len(df_context),
        "primary_features": PRIMARY_CONTEXT_FEATURES,
        "message_rate_definition": "10 / observation_duration_s (discrete event count rate)",
        "message_rate_interval_definition": "9 / observation_duration_s = 1 / mean_iat (interval frequency)",
        "feature_statistics": stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_json_path, "w") as fp:
        json.dump(meta_info, fp, indent=2)

    elapsed = time.time() - t0
    print(f"  Split completed in {elapsed:.2f} s.")
    return df_context, meta_info


def main():
    start_time = time.time()
    print("=================================================================")
    print("Option B — Stage 3: Context Feature Engineering and Alignment")
    print("=================================================================")

    master_summary = {
        "pipeline_stage": "Option B — Stage 3: Context Feature Engineering",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_features": PRIMARY_CONTEXT_FEATURES,
        "splits": {},
    }

    total_windows = 0

    for highway in HIGHWAYS:
        for split in SPLITS:
            _, meta = process_split(highway, split)
            master_summary["splits"][f"{highway}/{split}"] = meta
            total_windows += meta["n_windows"]

    master_summary["total_context_windows"] = total_windows

    summary_path = os.path.join(OUTPUT_BASE, "context_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(master_summary, fp, indent=2)

    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"Context Feature Engineering Complete in {elapsed:.2f} s.")
    print(f"Total Context Rows: {total_windows:,} (1-to-1 match with sequences)")
    print(f"Master summary saved to: {summary_path}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
