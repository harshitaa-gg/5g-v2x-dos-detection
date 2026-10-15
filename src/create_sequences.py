"""
create_sequences.py
===================
Converts preprocessed tabular data into fixed-length sliding-window sequences
for the Transformer model in the Congestion-Aware Transformer-XGBoost DoS
detection pipeline.

Window configuration (base-paper settings)
-------------------------------------------
  Window size : 10 consecutive messages
  Stride      : 1 message

Grouping and ordering
---------------------
Messages must be grouped by sender vehicle (sender_id) and ordered
chronologically (send_time) before windowing.

  - sender_id and send_time are NOT present in the scaled input files
    (they were excluded during feature selection as identifiers).
  - They ARE present in the deduplicated source files at the same row
    positions (the full preprocessing pipeline preserves row order and
    row count at every stage — verified: label columns match exactly).
  - This script reads sender_id and send_time from
    processed_data/preprocessed_deduplicated/<highway>/<split>.parquet
    and aligns them positionally with the scaled feature matrix.
  - Sorting by send_time within each sender group is mandatory: 417/482
    senders in highway_2/train have non-monotonic send_time in file order
    due to the normal+DoS simulation merge. After sorting, all groups are
    monotonically increasing.

Mixed-label rule
----------------
A window is assigned label = 1 if ANY of its 10 messages carries label = 1;
otherwise label = 0.  Rationale: a window containing even one malicious
packet represents an active DoS event visible to the receiver at that point
in time.

Sender-boundary rule
--------------------
Windows never cross sender boundaries. A sender group with fewer than 10
messages after sorting produces zero sequences and is skipped with a warning.

Output format
-------------
NumPy compressed archives (.npz) — one file per highway/split:

    processed_data/sequences/<highway>/<split>/sequences.npz

Each archive contains two arrays:
    X  – float32, shape (N_sequences, WINDOW_SIZE, N_FEATURES)
    y  – int8,    shape (N_sequences,)

  N_FEATURES = 19  (all 20 columns from the scaled file, minus label)
  Using float32 (not float64) to halve memory usage during loading.

A companion metadata file is written alongside:
    processed_data/sequences/<highway>/<split>/meta.json

It records: highway, split, window_size, stride, n_sequences, n_features,
feature_columns, label_rule, n_positive, n_negative, created_at.

Memory strategy (Highway 7: ~1.96M rows in train)
--------------------------------------------------
  1. Load the full split into a DataFrame once.
  2. Process one sender group at a time, appending resulting windows to
     a pre-allocated list of small NumPy arrays.
  3. Concatenate all group arrays once at the end — no intermediate
     copies of the full feature matrix are held in RAM.
  4. Cast the final feature array to float32 before saving.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE = 10
STRIDE      = 1

TARGET_COLUMN = "label"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SCALED_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transformer_scaled")
DEDUP_BASE  = os.path.join(PROJECT_ROOT, "processed_data", "preprocessed_deduplicated")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS   = ["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_split(highway: str, split: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the scaled Parquet file and join sender_id + send_time from the
    deduplicated source by positional alignment.

    Returns (merged_df, feature_columns) where feature_columns excludes
    label, sender_id, and send_time.
    """
    scaled_path = os.path.join(SCALED_BASE, highway, f"{split}.parquet")
    dedup_path  = os.path.join(DEDUP_BASE,  highway, f"{split}.parquet")

    if not os.path.isfile(scaled_path):
        print(f"[ERROR] Missing scaled file: {scaled_path}", file=sys.stderr)
        sys.exit(1)

    scaled_df = pd.read_parquet(scaled_path)

    # sender_id / send_time availability
    sender_id_col  = "sender_id"
    send_time_col  = "send_time"
    has_group_cols = False

    if os.path.isfile(dedup_path):
        dedup_df = pd.read_parquet(
            dedup_path,
            columns=[sender_id_col, send_time_col, TARGET_COLUMN],
        )
        if len(dedup_df) == len(scaled_df):
            # Verify row alignment via label column
            if (dedup_df[TARGET_COLUMN].values == scaled_df[TARGET_COLUMN].values).all():
                scaled_df = scaled_df.copy()
                scaled_df[sender_id_col] = dedup_df[sender_id_col].values
                scaled_df[send_time_col] = dedup_df[send_time_col].values
                has_group_cols = True
            else:
                print(
                    f"[WARN] {highway}/{split}: label mismatch between scaled and "
                    f"deduplicated files — cannot safely align sender_id/send_time. "
                    f"Sequences will be built in file order without per-sender grouping.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[WARN] {highway}/{split}: row count mismatch between scaled "
                f"({len(scaled_df)}) and deduplicated ({len(dedup_df)}) files. "
                f"Sequences will be built in file order without per-sender grouping.",
                file=sys.stderr,
            )
    else:
        print(
            f"[WARN] {highway}/{split}: deduplicated source not found at "
            f"'{dedup_path}'. Sequences will be built in file order without "
            f"per-sender grouping. This may produce spurious cross-sender windows.",
            file=sys.stderr,
        )

    # Feature columns = all scaled columns except label (and grouping cols)
    non_feature = {TARGET_COLUMN, sender_id_col, send_time_col}
    feature_cols = [c for c in scaled_df.columns if c not in non_feature]

    return scaled_df, feature_cols, has_group_cols


def _build_sequences_for_group(
    feat_arr: np.ndarray,   # shape (n_rows, n_features), float64
    label_arr: np.ndarray,  # shape (n_rows,), int
    window_size: int,
    stride: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Slide a window over one sender's sorted messages.

    Returns (X_group, y_group) arrays or (None, None) if group is too small.

    X_group shape: (n_windows, window_size, n_features)
    y_group shape: (n_windows,)

    Label assignment: any-positive rule — window label = 1 if any row label = 1.
    """
    n = len(feat_arr)
    if n < window_size:
        return None, None

    n_windows = (n - window_size) // stride + 1
    n_features = feat_arr.shape[1]

    X = np.empty((n_windows, window_size, n_features), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int8)

    for i in range(n_windows):
        start = i * stride
        end   = start + window_size
        X[i]  = feat_arr[start:end]
        # any-positive rule
        y[i]  = 1 if label_arr[start:end].any() else 0

    return X, y


def _process_split(highway: str, split: str) -> Dict:
    """
    Process a single highway/split: load data, group by sender, build
    windows, save output.  Returns a summary dict.
    """
    print(f"\n  Loading {highway}/{split} ...")
    df, feature_cols, has_group_cols = _load_split(highway, split)

    n_input_rows = len(df)
    n_features   = len(feature_cols)

    # ---- Determine grouping ----
    if has_group_cols:
        group_col     = "sender_id"
        time_col      = "send_time"
        group_ids     = df[group_col].unique()
        n_groups      = len(group_ids)
        skipped_small = 0
        grouping_note = "per-sender, sorted by send_time"
    else:
        # Fallback: treat entire split as one group in file order
        group_col     = None
        n_groups      = 1
        skipped_small = 0
        grouping_note = (
            "WARNING: whole split treated as single group in file order "
            "(sender_id/send_time unavailable)"
        )
        print(f"    [LIMITATION] {grouping_note}", file=sys.stderr)

    # ---- Pre-extract arrays ----
    feat_all  = df[feature_cols].to_numpy(dtype=np.float64)
    label_all = df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    if has_group_cols:
        sender_all    = df["sender_id"].to_numpy()
        send_time_all = df["send_time"].to_numpy()
    else:
        sender_all    = None
        send_time_all = None

    # Free the DataFrame to save RAM before the windowing loop
    del df

    # ---- Build windows group by group ----
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    if has_group_cols:
        # Group indices without creating a full copy of data
        # Build a dict: sender_id -> sorted positional indices
        unique_senders = np.unique(sender_all)
        for sid in unique_senders:
            mask = sender_all == sid
            idxs = np.where(mask)[0]

            # Sort by send_time within group
            times = send_time_all[idxs]
            order = np.argsort(times, kind="stable")
            idxs  = idxs[order]

            feat_g  = feat_all[idxs]
            label_g = label_all[idxs]

            X_g, y_g = _build_sequences_for_group(
                feat_g, label_g, WINDOW_SIZE, STRIDE
            )
            if X_g is None:
                skipped_small += 1
            else:
                all_X.append(X_g)
                all_y.append(y_g)
    else:
        X_g, y_g = _build_sequences_for_group(
            feat_all, label_all, WINDOW_SIZE, STRIDE
        )
        if X_g is not None:
            all_X.append(X_g)
            all_y.append(y_g)

    # ---- Concatenate ----
    if all_X:
        X_out = np.concatenate(all_X, axis=0)   # (N_seq, 10, 19)
        y_out = np.concatenate(all_y, axis=0)    # (N_seq,)
    else:
        X_out = np.empty((0, WINDOW_SIZE, n_features), dtype=np.float32)
        y_out = np.empty(0, dtype=np.int8)

    n_sequences = len(y_out)
    n_positive  = int((y_out == 1).sum())
    n_negative  = int((y_out == 0).sum())

    # ---- Save ----
    out_dir = os.path.join(OUTPUT_BASE, highway, split)
    os.makedirs(out_dir, exist_ok=True)

    npz_path  = os.path.join(out_dir, "sequences.npz")
    meta_path = os.path.join(out_dir, "meta.json")

    np.savez_compressed(npz_path, X=X_out, y=y_out)

    meta = {
        "highway":         highway,
        "split":           split,
        "window_size":     WINDOW_SIZE,
        "stride":          STRIDE,
        "n_input_rows":    n_input_rows,
        "n_sender_groups": n_groups,
        "n_skipped_small": skipped_small,
        "n_sequences":     n_sequences,
        "n_features":      n_features,
        "feature_columns": feature_cols,
        "label_rule":      "any-positive: window label=1 if any row label=1, else 0",
        "grouping":        grouping_note,
        "n_positive":      n_positive,
        "n_negative":      n_negative,
        "X_shape":         list(X_out.shape),
        "y_shape":         list(y_out.shape),
        "X_dtype":         str(X_out.dtype),
        "y_dtype":         str(y_out.dtype),
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Sequence Creation  —  Transformer Pipeline")
    print(f"  Window size : {WINDOW_SIZE}")
    print(f"  Stride      : {STRIDE}")
    print(f"  Label rule  : any-positive (window=1 if any row label=1)")
    print(f"  Output base : {OUTPUT_BASE}")
    print(f"  Output fmt  : NumPy .npz  (X: float32, y: int8)")
    print("=" * 70)

    all_metas = []

    for highway in HIGHWAYS:
        print(f"\n{'='*35} {highway} {'='*35}")
        for split in SPLITS:
            meta = _process_split(highway, split)
            all_metas.append(meta)

            skipped_note = (
                f"  ({meta['n_skipped_small']} groups skipped, < {WINDOW_SIZE} msgs)"
                if meta["n_skipped_small"] > 0 else ""
            )
            print(
                f"  [{split:10s}] "
                f"rows={meta['n_input_rows']:>9,}  "
                f"senders={meta['n_sender_groups']:>5}  "
                f"sequences={meta['n_sequences']:>9,}  "
                f"pos={meta['n_positive']:>8,}  neg={meta['n_negative']:>9,}"
                + skipped_note
            )

    # ---- Final output table ----
    print()
    print("=" * 70)
    print("Output files")
    print("-" * 70)
    for m in all_metas:
        npz  = os.path.join(OUTPUT_BASE, m["highway"], m["split"], "sequences.npz")
        meta = os.path.join(OUTPUT_BASE, m["highway"], m["split"], "meta.json")
        size_mb = os.path.getsize(npz) / 1_048_576
        print(
            f"  {m['highway']}/{m['split']:<10s}  "
            f"X{str(m['X_shape']):<20s}  "
            f"{size_mb:6.1f} MB  {npz}"
        )
        print(f"    meta : {meta}")

    print()
    print("Loading example (in model training script):")
    print("  data = np.load('processed_data/sequences/highway_2/train/sequences.npz')")
    print("  X, y = data['X'], data['y']   # X: (N,10,19) float32  y: (N,) int8")
    print()
    print("Sequence creation complete.")


if __name__ == "__main__":
    main()
