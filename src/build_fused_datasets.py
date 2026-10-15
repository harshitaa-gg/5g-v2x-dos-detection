"""
src/build_fused_datasets.py
===========================
Phase 3A: Fused Dataset Construction and Multi-Key Alignment Validation.

This script constructs the multimodal fused datasets combining:
  1. 64-dimensional Transformer embeddings (Phase 2)
  2. 7 engineered context features (Phase 3)

Two feature sets are produced for each highway and each split:
  - Full Hybrid (71 features): 64 embeddings + 7 context features
  - Autonomous Ablation (68 features): 64 embeddings + 4 autonomous context features

Key Safeguards:
---------------
1. Explicit Key-Based Joining:
   - Merges on 'window_id' and validates sender_id, start_send_time, end_send_time,
     and label across all sources. Never relies on row order alone.
2. Exact Feature Ordering & Deterministic Naming:
   - Embedding columns: embedding_00 through embedding_63
   - Context columns: strictly preserves the documented ordering.
3. Strict Separation of Target and Identifiers:
   - Predictor matrices strictly exclude window_id, sender_id, timestamps, durations,
     labels, and auxiliary diagnostics.
4. Highway and Split Isolation:
   - Processed split-by-split; zero cross-split history joining.
5. Comprehensive Integrity Checks:
   - Exact row count, uniqueness, finite values (no NaNs/Infs), absence of forbidden columns.
"""

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

SEQ_BASE = os.path.join(PROJECT_ROOT, "processed_data", "sequences_tx")
CTX_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmission_context")
EMB_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transformer_embeddings")
OUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "fused_features")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "option_b")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

# 64 deterministic embedding column names
EMBEDDING_COLS = [f"embedding_{i:02d}" for i in range(64)]

# 7 Primary Context Features (Exact verified order)
CONTEXT_7_COLS = [
    "starting_distance",
    "final_distance",
    "distance_change_rate",
    "message_rate",
    "iat_variation",
    "speed_variation",
    "acceleration_variation",
]

# 4 Autonomous Ablation Context Features (Excludes receiver-derived distance metrics)
CONTEXT_4_COLS = [
    "message_rate",
    "iat_variation",
    "speed_variation",
    "acceleration_variation",
]

# Full 71-feature and 68-feature lists
FULL_HYBRID_FEATURES = EMBEDDING_COLS + CONTEXT_7_COLS
AUTONOMOUS_FEATURES = EMBEDDING_COLS + CONTEXT_4_COLS

# Forbidden columns that must NEVER appear in the predictor matrix
FORBIDDEN_COLUMNS = {
    "window_id",
    "sender_id",
    "window_index",
    "start_send_time",
    "end_send_time",
    "observation_duration_ns",
    "observation_duration_s",
    "label",
    "message_rate_interval",
    "start_pos_x",
    "start_pos_y",
    "end_pos_x",
    "end_pos_y",
    "start_median_receiver_dist",
    "end_median_receiver_dist",
}


def process_split(highway: str, split: str) -> Dict:
    """
    Loads, key-joins, validates, and saves fused datasets for one highway/split.
    """
    print(f"\n=============================================================")
    print(f"Constructing Fused Datasets: {highway} / {split}")
    print(f"=============================================================")
    t0 = time.time()

    emb_npz_path = os.path.join(EMB_BASE, highway, split, "embeddings.npz")
    ctx_pq_path = os.path.join(CTX_BASE, highway, split, "context_features.parquet")
    meta_pq_path = os.path.join(SEQ_BASE, highway, split, "window_metadata.parquet")

    # 1. Existence check
    for p in [emb_npz_path, ctx_pq_path, meta_pq_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required input file: {p}")

    # 2. Load Embeddings
    emb_data = np.load(emb_npz_path)
    emb_matrix = emb_data["embeddings"]  # (N, 64), float32
    emb_wids = emb_data["window_ids"]  # (N,), int64
    emb_labels = emb_data["labels"]  # (N,), int8
    n_samples = len(emb_wids)

    assert emb_matrix.shape == (n_samples, 64), f"Unexpected embedding shape: {emb_matrix.shape}"
    assert not np.isnan(emb_matrix).any(), "NaN detected in embeddings"
    assert not np.isinf(emb_matrix).any(), "Inf detected in embeddings"

    emb_df = pd.DataFrame(emb_matrix, columns=EMBEDDING_COLS)
    emb_df["window_id"] = emb_wids
    emb_df["label_emb"] = emb_labels

    # 3. Load Context Features
    ctx_df = pq.read_table(ctx_pq_path).to_pandas()
    assert len(ctx_df) == n_samples, f"Count mismatch in ctx_df: {len(ctx_df)} vs {n_samples}"

    # 4. Load Window Metadata
    meta_df = pq.read_table(meta_pq_path).to_pandas()
    assert len(meta_df) == n_samples, f"Count mismatch in meta_df: {len(meta_df)} vs {n_samples}"

    # -----------------------------------------------------------------------
    # 5. Explicit Key-Based Joining & Multi-Key Validation
    # -----------------------------------------------------------------------
    print(f"  Performing key-based join on window_id and validating integrity ...")

    # Merge meta and context on window_id
    merged = pd.merge(meta_df, ctx_df, on="window_id", suffixes=("_meta", "_ctx"))
    assert len(merged) == n_samples, f"Merge dropped/duplicated rows: {len(merged)} vs {n_samples}"

    # Merge with embeddings on window_id
    merged = pd.merge(merged, emb_df, on="window_id")
    assert len(merged) == n_samples, f"Embedding merge dropped/duplicated rows: {len(merged)} vs {n_samples}"

    # Multi-key agreement checks
    assert (merged["sender_id_meta"] == merged["sender_id_ctx"]).all(), "sender_id mismatch!"
    assert (merged["start_send_time_meta"] == merged["start_send_time_ctx"]).all(), "start_send_time mismatch!"
    assert (merged["end_send_time_meta"] == merged["end_send_time_ctx"]).all(), "end_send_time mismatch!"
    assert (merged["label_meta"] == merged["label_ctx"]).all(), "label mismatch between meta and ctx!"
    assert (merged["label_meta"] == merged["label_emb"]).all(), "label mismatch between meta and emb!"

    # Window ID uniqueness and ordering
    wids = merged["window_id"].to_numpy()
    assert len(np.unique(wids)) == n_samples, "Duplicate window_ids detected!"
    assert (np.diff(wids) == 1).all(), "window_ids not strictly sequential 0..N-1!"

    print(f"  Key-based join validated: 100% agreement on sender_id, timestamps, and labels.")

    # -----------------------------------------------------------------------
    # 6. Extract Predictor Matrices & Separate Target / Identifiers
    # -----------------------------------------------------------------------
    # Full Hybrid Feature Matrix (71 features)
    df_full = merged[FULL_HYBRID_FEATURES].copy()
    assert df_full.shape == (n_samples, 71), f"Unexpected shape for df_full: {df_full.shape}"

    # Autonomous Ablation Feature Matrix (68 features)
    df_ablation = merged[AUTONOMOUS_FEATURES].copy()
    assert df_ablation.shape == (n_samples, 68), f"Unexpected shape for df_ablation: {df_ablation.shape}"

    # Target Label and Keys
    target_labels = merged["label_meta"].to_numpy(dtype=np.int8)
    df_keys = merged[[
        "window_id",
        "sender_id_meta",
        "start_send_time_meta",
        "end_send_time_meta",
        "observation_duration_s_meta",
        "label_meta",
    ]].copy()
    df_keys.columns = [
        "window_id",
        "sender_id",
        "start_send_time",
        "end_send_time",
        "observation_duration_s",
        "label",
    ]

    # -----------------------------------------------------------------------
    # 7. Absence of Forbidden Columns & Finite Numeric Validation
    # -----------------------------------------------------------------------
    for col in df_full.columns:
        if col in FORBIDDEN_COLUMNS:
            raise ValueError(f"Forbidden column '{col}' detected in Full Hybrid feature matrix!")
    for col in df_ablation.columns:
        if col in FORBIDDEN_COLUMNS:
            raise ValueError(f"Forbidden column '{col}' detected in Autonomous feature matrix!")

    assert not df_full.isnull().any().any(), "Nulls detected in df_full!"
    assert not df_ablation.isnull().any().any(), "Nulls detected in df_ablation!"

    X_full_arr = df_full.to_numpy(dtype=np.float32)
    X_ablation_arr = df_ablation.to_numpy(dtype=np.float32)

    assert not np.isnan(X_full_arr).any(), "NaN detected in X_full array!"
    assert not np.isinf(X_full_arr).any(), "Inf detected in X_full array!"
    assert not np.isnan(X_ablation_arr).any(), "NaN detected in X_ablation array!"
    assert not np.isinf(X_ablation_arr).any(), "Inf detected in X_ablation array!"

    # Class balance
    n_pos = int((target_labels == 1).sum())
    n_neg = int((target_labels == 0).sum())
    pos_pct = round(n_pos / n_samples * 100, 2)

    print(f"  Validated {n_samples:,} samples | Positive: {n_pos:,} ({pos_pct}%), Negative: {n_neg:,} ({100-pos_pct:.2f}%)")

    # -----------------------------------------------------------------------
    # 8. Save Fused Datasets
    # -----------------------------------------------------------------------
    out_dir = os.path.join(OUT_BASE, highway, split)
    os.makedirs(out_dir, exist_ok=True)

    # Save Parquet tables
    full_pq_path = os.path.join(out_dir, "fused_full_hybrid.parquet")
    ablation_pq_path = os.path.join(out_dir, "fused_autonomous_ablation.parquet")
    keys_pq_path = os.path.join(out_dir, "window_keys.parquet")

    pq.write_table(pa.Table.from_pandas(df_full, preserve_index=False), full_pq_path, compression="snappy")
    pq.write_table(pa.Table.from_pandas(df_ablation, preserve_index=False), ablation_pq_path, compression="snappy")
    pq.write_table(pa.Table.from_pandas(df_keys, preserve_index=False), keys_pq_path, compression="snappy")

    # Save Compressed NumPy Array Archive (for instant training loading)
    npz_path = os.path.join(out_dir, "fused_arrays.npz")
    np.savez_compressed(
        npz_path,
        X_full=X_full_arr,
        X_ablation=X_ablation_arr,
        y=target_labels,
        window_ids=wids,
    )

    full_sz_mb = os.path.getsize(full_pq_path) / (1024 * 1024)
    ablation_sz_mb = os.path.getsize(ablation_pq_path) / (1024 * 1024)
    npz_sz_mb = os.path.getsize(npz_path) / (1024 * 1024)

    print(f"  Saved -> {full_pq_path} ({full_sz_mb:.2f} MB)")
    print(f"  Saved -> {ablation_pq_path} ({ablation_sz_mb:.2f} MB)")
    print(f"  Saved -> {npz_path} ({npz_sz_mb:.2f} MB)")

    # Save split meta.json
    meta_json_path = os.path.join(out_dir, "meta.json")
    split_meta = {
        "highway": highway,
        "split": split,
        "n_samples": n_samples,
        "full_hybrid": {
            "n_features": 71,
            "feature_names": FULL_HYBRID_FEATURES,
            "tensor_shape": list(X_full_arr.shape),
            "parquet_file": "fused_full_hybrid.parquet",
        },
        "autonomous_ablation": {
            "n_features": 68,
            "feature_names": AUTONOMOUS_FEATURES,
            "tensor_shape": list(X_ablation_arr.shape),
            "parquet_file": "fused_autonomous_ablation.parquet",
        },
        "target": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_ratio": round(n_pos / n_samples, 6),
            "attack_percentage": pos_pct,
        },
        "join_validation": {
            "key": "window_id",
            "sender_id_match": True,
            "timestamps_match": True,
            "labels_match": True,
            "unique_window_ids": True,
            "forbidden_columns_absent": True,
            "finite_numeric_values": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_json_path, "w") as fp:
        json.dump(split_meta, fp, indent=2)

    elapsed = time.time() - t0
    print(f"  Split completed in {elapsed:.2f} s.")
    return split_meta


def main():
    start_total = time.time()
    print("=================================================================")
    print("Phase 3A: Fused Dataset Construction and Key-Based Validation")
    print("=================================================================")

    manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 3A — Fused Dataset Construction & Validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_specifications": {
            "full_hybrid_features": {
                "n_features": 71,
                "n_embeddings": 64,
                "n_context": 7,
                "feature_names": FULL_HYBRID_FEATURES,
            },
            "autonomous_ablation_features": {
                "n_features": 68,
                "n_embeddings": 64,
                "n_context": 4,
                "feature_names": AUTONOMOUS_FEATURES,
            },
        },
        "splits": {},
    }

    total_samples = 0
    total_pos = 0
    total_neg = 0

    for highway in HIGHWAYS:
        for split in SPLITS:
            meta = process_split(highway, split)
            manifest["splits"][f"{highway}/{split}"] = meta
            total_samples += meta["n_samples"]
            total_pos += meta["target"]["n_positive"]
            total_neg += meta["target"]["n_negative"]

    manifest["total_samples"] = total_samples
    manifest["total_positive"] = total_pos
    manifest["total_negative"] = total_neg
    manifest["overall_positive_ratio"] = round(total_pos / total_samples, 6)

    os.makedirs(OUT_BASE, exist_ok=True)
    manifest_path = os.path.join(OUT_BASE, "fused_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    elapsed_total = time.time() - start_total
    print(f"\n=================================================================")
    print(f"Phase 3A Complete in {elapsed_total:.2f} s across all 6 splits.")
    print(f"Total Fused Samples: {total_samples:,} (Positive: {total_pos:,}, Negative: {total_neg:,})")
    print(f"Master manifest saved to: {manifest_path}")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
