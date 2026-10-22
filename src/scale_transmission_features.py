import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmissions")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transmissions_scaled")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "preprocessing")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS = ["train", "validation", "test"]

# The 6 continuous sender features to scale
CONTINUOUS_FEATURES = [
    "sender_speed",
    "sender_acceleration",
    "sender_heading",
    "sender_speed_noise",
    "sender_acceleration_noise",
    "sender_heading_noise",
]

# Categorical column and its fixed category ordering
CATEGORICAL_COL = "sender_driver_profile"
DRIVER_PROFILE_CATEGORIES = ["AGGRESSIVE", "CAUTIOUS", "NORMAL"]
OHE_COLUMNS = [f"{CATEGORICAL_COL}_{cat}" for cat in DRIVER_PROFILE_CATEGORIES]

# Passthrough columns to keep in the scaled Parquet files
PASSTHROUGH_COLS = [
    "sender_id",
    "send_time",
    "message_id",
    "sender_alias",
    "label",
    "sender_pos_x",
    "sender_pos_y",
    "sender_pos_z",
    "observed_reception_count",
    "mean_delay",
    "median_receiver_distance",
]


class TransmissionStandardScaler:
    """
    StandardScaler implementation that computes mean and std on train data (ddof=0)
    and handles constant columns (std == 0) safely without division by zero.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.constant_cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "TransmissionStandardScaler":
        for col in self.feature_names:
            vals = df[col].to_numpy(dtype=np.float64)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=0))
            self.means[col] = mean_val
            if std_val == 0.0 or np.isnan(std_val):
                self.stds[col] = 1.0
                self.constant_cols.append(col)
            else:
                self.stds[col] = std_val
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled_dict = {}
        for col in self.feature_names:
            mean_val = self.means[col]
            std_val = self.stds[col]
            scaled_dict[col] = (df[col].to_numpy(dtype=np.float64) - mean_val) / std_val
        return pd.DataFrame(scaled_dict, index=df.index)

    def to_dict(self) -> dict:
        return {
            "method": "numpy_standard_scaler_ddof0",
            "scaled_columns": self.feature_names,
            "constant_columns": self.constant_cols,
            "means": self.means,
            "stds": self.stds,
        }


def encode_driver_profile(series: pd.Series) -> pd.DataFrame:
    """
    One-hot encodes driver profile into fixed columns with zero for unseen categories.
    """
    ohe_data = {}
    for cat in DRIVER_PROFILE_CATEGORIES:
        col_name = f"{CATEGORICAL_COL}_{cat}"
        ohe_data[col_name] = (series == cat).astype(np.int64)
    return pd.DataFrame(ohe_data, index=series.index)


def process_highway(highway: str) -> dict:
    """
    Processes all splits for a given highway:
    1. Fits scaler on training split.
    2. Saves fitted scaler parameters to models/preprocessing.
    3. Transforms train, validation, and test splits.
    4. Writes scaled Parquet files to processed_data/transmissions_scaled/{highway}/.
    """
    print(f"\n==========================================")
    print(f"Processing Highway: {highway}")
    print(f"==========================================")
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_dir = os.path.join(OUTPUT_BASE, highway)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load train split to fit scaler
    train_path = os.path.join(INPUT_BASE, highway, "train.parquet")
    print(f"Loading training split from: {train_path}")
    train_df = pq.read_table(train_path).to_pandas()

    scaler = TransmissionStandardScaler(CONTINUOUS_FEATURES)
    scaler.fit(train_df)
    print("Fitted StandardScaler on training split:")
    for col in CONTINUOUS_FEATURES:
        print(f"  {col:28s}: mean={scaler.means[col]:12.4f}, std={scaler.stds[col]:12.4f}")

    if scaler.constant_cols:
        print(f"  [WARN] Constant columns detected: {scaler.constant_cols}")

    # Save scaler artifact
    scaler_artifact_path = os.path.join(MODELS_DIR, f"scaler_transmission_{highway}.joblib")
    scaler_payload = {
        "scaler": scaler.to_dict(),
        "driver_profile_categories": DRIVER_PROFILE_CATEGORIES,
        "driver_profile_columns": OHE_COLUMNS,
        "fitted_on": f"{highway}/train",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(scaler_payload, scaler_artifact_path)
    print(f"Saved scaler artifact to: {scaler_artifact_path}")

    split_stats = {}

    # 2. Transform all splits using frozen scaler
    for split in SPLITS:
        split_path = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
        df = pq.read_table(split_path).to_pandas()
        n_rows = len(df)

        # Scale continuous features
        scaled_cont_df = scaler.transform(df)

        # One-hot encode driver profile
        ohe_df = encode_driver_profile(df[CATEGORICAL_COL])

        # Assemble output DataFrame with fixed column ordering
        # Ordering: passthrough identifiers & targets, then scaled features, then OHE
        final_df = pd.concat([
            df[PASSTHROUGH_COLS],
            scaled_cont_df,
            ohe_df
        ], axis=1)

        # Verify no nulls or Infs in scaled columns
        for col in CONTINUOUS_FEATURES + OHE_COLUMNS:
            vals = final_df[col].to_numpy()
            if np.isnan(vals).any():
                raise ValueError(f"NaN detected in {highway}/{split} column {col}")
            if np.isinf(vals).any():
                raise ValueError(f"Inf detected in {highway}/{split} column {col}")

        # Save to Parquet
        out_parquet_path = os.path.join(out_dir, f"{split}.parquet")
        table = pa.Table.from_pandas(final_df, preserve_index=False)
        pq.write_table(table, out_parquet_path, compression="snappy")
        file_size_mb = os.path.getsize(out_parquet_path) / (1024 * 1024)

        print(f"  Saved {split:10s} -> {out_parquet_path} ({n_rows:,} rows, {file_size_mb:.2f} MB)")

        # Record split stats
        split_stats[split] = {
            "n_rows": n_rows,
            "file_size_mb": round(file_size_mb, 2),
            "means": {c: round(float(final_df[c].mean()), 4) for c in CONTINUOUS_FEATURES},
            "stds": {c: round(float(final_df[c].std(ddof=0)), 4) for c in CONTINUOUS_FEATURES},
            "ohe_sums": {c: int(final_df[c].sum()) for c in OHE_COLUMNS},
        }

    return {
        "scaler_artifact": scaler_artifact_path,
        "scaler_params": scaler.to_dict(),
        "splits": split_stats,
    }


def main():
    start_time = time.time()
    print("=================================================================")
    print("Option B: Scaling Transmission Features for 5G-V2X DoS Detection")
    print("=================================================================")

    summary = {
        "pipeline_stage": "Option B - Transmission Feature Scaling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "continuous_features": CONTINUOUS_FEATURES,
        "driver_profile_categories": DRIVER_PROFILE_CATEGORIES,
        "driver_profile_columns": OHE_COLUMNS,
        "passthrough_columns": PASSTHROUGH_COLS,
        "highways": {},
    }

    for highway in HIGHWAYS:
        highway_summary = process_highway(highway)
        summary["highways"][highway] = highway_summary

    summary_path = os.path.join(OUTPUT_BASE, "scaling_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    elapsed = time.time() - start_time
    print(f"\nScaling completed successfully in {elapsed:.2f} s.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
