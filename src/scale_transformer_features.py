import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE  = os.path.join(PROJECT_ROOT, "processed_data", "categorical_encoded")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "transformer_scaled")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "preprocessing")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS   = ["train", "validation", "test"]

TARGET_COLUMN = "label"

# Column prefixes that identify OHE indicator columns (never scaled)
OHE_PREFIXES = ("sender_driver_profile_", "receiver_driver_profile_")


# ---------------------------------------------------------------------------
# Detect sklearn
# ---------------------------------------------------------------------------
try:
    from sklearn.preprocessing import StandardScaler as _SklearnScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (scale_cols, passthrough_cols).

    scale_cols       – continuous numeric columns to standardise
    passthrough_cols – OHE indicators + label; left as-is
    """
    scale_cols: List[str] = []
    passthrough_cols: List[str] = []

    for col in df.columns:
        if col == TARGET_COLUMN:
            passthrough_cols.append(col)
        elif col.startswith(OHE_PREFIXES):
            passthrough_cols.append(col)
        elif df[col].dtype in (np.float64, np.int64, np.float32, np.int32):
            scale_cols.append(col)
        else:
            # Non-numeric columns unexpected at this stage; pass through safely
            passthrough_cols.append(col)

    return scale_cols, passthrough_cols


# ---------------------------------------------------------------------------
# sklearn path
# ---------------------------------------------------------------------------

def fit_sklearn(
    train_df: pd.DataFrame,
    scale_cols: List[str],
) -> Tuple[object, Dict]:
    """Fit a sklearn StandardScaler; return (fitted_scaler, params_dict)."""
    scaler = _SklearnScaler()
    scaler.fit(train_df[scale_cols])

    means = dict(zip(scale_cols, scaler.mean_.tolist()))
    stds  = dict(zip(scale_cols, scaler.scale_.tolist()))
    constant_cols = [c for c, s in stds.items() if s == 0.0]

    params = {
        "method":           "sklearn_StandardScaler",
        "scaled_columns":   scale_cols,
        "constant_columns": constant_cols,
        "means":            means,
        "stds":             stds,
    }
    return scaler, params


def apply_sklearn(df: pd.DataFrame, scaler: object, scale_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[scale_cols] = scaler.transform(out[scale_cols]).astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# NumPy fallback path
# ---------------------------------------------------------------------------

def fit_numpy(
    train_df: pd.DataFrame,
    scale_cols: List[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """
    Compute training-split mean and std (ddof=0) for each column.

    Constant features (std == 0) are recorded; they will not be divided.
    Returns (means, stds, params_dict).
    """
    means: Dict[str, float] = {}
    stds: Dict[str, float]  = {}

    for col in scale_cols:
        arr = train_df[col].to_numpy(dtype=np.float64)
        means[col] = float(np.mean(arr))
        stds[col]  = float(np.std(arr, ddof=0))   # ddof=0 matches sklearn default

    constant_cols = [c for c, s in stds.items() if s == 0.0]
    if constant_cols:
        print(f"  [WARN] Constant features detected (std=0); will be left unchanged: {constant_cols}")

    params = {
        "method":           "numpy_standard",
        "scaled_columns":   scale_cols,
        "constant_columns": constant_cols,
        "means":            means,
        "stds":             stds,
    }
    return means, stds, params


def apply_numpy(
    df: pd.DataFrame,
    scale_cols: List[str],
    means: Dict[str, float],
    stds: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply z = (x - mean) / std for each column.

    Constant features (std == 0) are centred (x - mean) but not divided,
    which results in all-zero values — equivalent to sklearn behaviour.
    """
    out = df.copy()
    for col in scale_cols:
        arr = out[col].to_numpy(dtype=np.float64)
        arr = arr - means[col]
        if stds[col] != 0.0:
            arr = arr / stds[col]
        out[col] = arr
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    method_label = "sklearn_StandardScaler" if SKLEARN_AVAILABLE else "numpy_standard (sklearn absent)"
    print("=" * 70)
    print("Feature Scaling  —  Transformer Pipeline")
    print(f"  Method      : {method_label}")
    print(f"  Input base  : {INPUT_BASE}")
    print(f"  Output base : {OUTPUT_BASE}")
    print(f"  Models dir  : {MODELS_DIR}")
    print(f"  Highways    : {HIGHWAYS}")
    print("=" * 70)

    all_results = []

    for highway in HIGHWAYS:
        print(f"\n--- {highway} ---")

        # ---- Load training split ----
        train_src = os.path.join(INPUT_BASE, highway, "train.parquet")
        if not os.path.isfile(train_src):
            print(f"[ERROR] Missing: {train_src}", file=sys.stderr)
            sys.exit(1)

        train_df = pd.read_parquet(train_src)
        scale_cols, passthrough_cols = classify_columns(train_df)

        print(f"  Columns to scale        ({len(scale_cols):2d}): {scale_cols}")
        print(f"  Columns passed through  ({len(passthrough_cols):2d}): {passthrough_cols}")

        # ---- Fit scaler on training data ----
        if SKLEARN_AVAILABLE:
            scaler_obj, params = fit_sklearn(train_df, scale_cols)
            _means = params["means"]
            _stds  = params["stds"]
        else:
            _means, _stds, params = fit_numpy(train_df, scale_cols)
            scaler_obj = None   # not used in numpy path

        # Report fitted statistics
        print(f"  Fitted statistics (mean / std) — training split:")
        for col in scale_cols:
            print(f"    {col:35s}  mean={_means[col]:+.6g}  std={_stds[col]:.6g}")
        if params["constant_columns"]:
            print(f"  [WARN] Constant columns: {params['constant_columns']}")

        # ---- Persist scaler artefact ----
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{highway}.joblib")
        save_obj = {"params": params}
        if SKLEARN_AVAILABLE:
            save_obj["sklearn_scaler"] = scaler_obj
        joblib.dump(save_obj, scaler_path)
        print(f"  Scaler saved : {scaler_path}")

        # ---- Apply to each split ----
        for split in SPLITS:
            src_path = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
            dst_path = os.path.join(OUTPUT_BASE, highway, f"{split}.parquet")

            if not os.path.isfile(src_path):
                print(f"[ERROR] Missing source: {src_path}", file=sys.stderr)
                sys.exit(1)

            df = pd.read_parquet(src_path)
            n_in  = len(df)
            cols_in = list(df.columns)

            # Validate expected columns exist
            missing = [c for c in scale_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Expected scale columns absent in '{src_path}': {missing}"
                )

            # Apply scaler
            if SKLEARN_AVAILABLE:
                encoded = apply_sklearn(df, scaler_obj, scale_cols)
            else:
                encoded = apply_numpy(df, scale_cols, _means, _stds)

            # Sanity checks
            if len(encoded) != n_in:
                raise RuntimeError(
                    f"Row count changed: {n_in} -> {len(encoded)} in '{src_path}'"
                )
            if list(encoded.columns) != cols_in:
                raise RuntimeError(
                    f"Column layout changed after scaling in '{src_path}'.\n"
                    f"  Before: {cols_in}\n  After : {list(encoded.columns)}"
                )
            if TARGET_COLUMN not in encoded.columns:
                raise RuntimeError(f"'{TARGET_COLUMN}' was lost during scaling.")

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            encoded.to_parquet(dst_path, index=False)

            all_results.append({
                "highway": highway,
                "split":   split,
                "rows":    len(encoded),
                "cols":    len(encoded.columns),
                "path":    dst_path,
            })
            print(f"  [{split:10s}]  {len(encoded):>9,} rows  {len(encoded.columns)} cols  -> {dst_path}")

    # ---- Final summary ----
    print()
    print("=" * 70)
    print("Summary")
    print("-" * 70)
    for r in all_results:
        print(
            f"  {r['highway']}/{r['split']:<10s}  "
            f"{r['rows']:>9,} rows  {r['cols']:>2} cols  {r['path']}"
        )

    print()
    print("Scaler artefacts:")
    for highway in HIGHWAYS:
        sp = os.path.join(MODELS_DIR, f"scaler_{highway}.joblib")
        print(f"  {sp}")

    # Show scaled vs pass-through column distinction from last run
    sample = pd.read_parquet(all_results[-1]["path"])
    scale_cols_final, passthrough_final = classify_columns(sample)
    print()
    print(f"Scaled columns        ({len(scale_cols_final)}): {scale_cols_final}")
    print(f"Pass-through columns  ({len(passthrough_final)}): {passthrough_final}")
    print(f"Method used: {method_label}")
    print()
    print("Feature scaling complete.  Next step: feature engineering.")


if __name__ == "__main__":
    main()
