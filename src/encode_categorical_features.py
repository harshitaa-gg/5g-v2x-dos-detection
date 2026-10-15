"""
encode_categorical_features.py
================================
One-hot encodes the categorical driver-profile columns for each highway.

Input
-----
  processed_data/feature_selected/<highway>/{train,validation,test}.parquet

Output
------
  processed_data/categorical_encoded/<highway>/{train,validation,test}.parquet
  models/preprocessing/ohe_encoder_<highway>.joblib

Encoding strategy
-----------------
One-Hot Encoding (OHE) is applied to:
  - sender_driver_profile
  - receiver_driver_profile

Both columns hold exactly three categories in the training data:
  AGGRESSIVE, CAUTIOUS, NORMAL

The vocabulary is fitted (sorted) from the training split of each highway.
Validation and test splits are encoded with the *frozen* training vocabulary.
Unseen categories in validation/test produce all-zero indicator rows for that
column; no crash, no NaN pollution.

The two text columns are replaced by binary integer indicator columns named:
  sender_driver_profile_AGGRESSIVE
  sender_driver_profile_CAUTIOUS
  sender_driver_profile_NORMAL
  receiver_driver_profile_AGGRESSIVE
  receiver_driver_profile_CAUTIOUS
  receiver_driver_profile_NORMAL

Implementation uses only pandas (pd.get_dummies with reindex) and joblib
(serialisation).  scikit-learn is NOT required.

Constraints respected
---------------------
- Encoder fitted on training split only; frozen for validation/test.
- label preserved in every output; row order and row count unchanged.
- Source files never modified.
- No scaling, PCA, row removal, or model training.
"""

import os
import sys
from typing import Dict, List

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_BASE  = os.path.join(PROJECT_ROOT, "processed_data", "feature_selected")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "processed_data", "categorical_encoded")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "preprocessing")

HIGHWAYS = ["highway_2", "highway_7"]
SPLITS   = ["train", "validation", "test"]

TARGET_COLUMN     = "label"
CAT_COLUMNS: List[str] = ["sender_driver_profile", "receiver_driver_profile"]


# ---------------------------------------------------------------------------
# Encoder definition
# ---------------------------------------------------------------------------
# The "encoder" is a plain Python dict:
#   {
#     "sender_driver_profile":   ["AGGRESSIVE", "CAUTIOUS", "NORMAL"],
#     "receiver_driver_profile": ["AGGRESSIVE", "CAUTIOUS", "NORMAL"],
#   }
# where each value is the sorted list of categories observed in the training
# split.  This dict is all that's needed to reproduce the exact column layout
# on any future split.

EncoderVocab = Dict[str, List[str]]  # column -> sorted category list


def fit_encoder(train_df: pd.DataFrame, cat_cols: List[str]) -> EncoderVocab:
    """
    Derive the vocabulary from *train_df* for each column in *cat_cols*.

    Returns a dict mapping column name -> sorted list of unique string values.
    """
    vocab: EncoderVocab = {}
    for col in cat_cols:
        if col not in train_df.columns:
            raise KeyError(f"Expected categorical column '{col}' is absent in the training split.")
        unique_vals = sorted(train_df[col].dropna().unique().tolist())
        if not unique_vals:
            raise ValueError(f"Column '{col}' has no non-null values in the training split.")
        vocab[col] = unique_vals
    return vocab


def apply_encoder(df: pd.DataFrame, vocab: EncoderVocab) -> pd.DataFrame:
    """
    One-hot encode the categorical columns in *df* using the frozen *vocab*.

    Steps
    -----
    1. For each categorical column, call pd.get_dummies to produce indicator
       columns named  <col>_<category>.
    2. Reindex the indicator columns to exactly the vocabulary columns:
         - Categories present in the split but absent from vocab → column added
           (should not occur in practice, but handled safely).
         - Categories in vocab absent from this split → column filled with 0.
         - Unseen values produce a 0 in every indicator column for that row.
    3. Drop the original text column.
    4. Concatenate all indicator frames with the remaining numeric columns.
    5. Return the result with label as the last column.

    Row order and row count are preserved throughout.
    """
    result = df.copy()

    for col, categories in vocab.items():
        if col not in result.columns:
            raise KeyError(
                f"Expected column '{col}' is absent in a validation/test split.\n"
                f"Source file may be incompatible with the selected feature set."
            )

        # Expected output column names for this categorical column
        expected_cols = [f"{col}_{cat}" for cat in categories]

        # get_dummies produces indicator columns for whatever values exist in
        # this split.  We then reindex to the exact vocabulary columns; any
        # column not produced (unseen category or split missing a category) is
        # filled with 0.
        dummies = pd.get_dummies(result[col], prefix=col, dtype=int)
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)

        # Drop the original text column and insert the indicator columns in its
        # place (preserving relative column order).
        insert_pos = result.columns.get_loc(col)
        result = result.drop(columns=[col])

        for offset, indicator_col in enumerate(expected_cols):
            result.insert(insert_pos + offset, indicator_col, dummies[indicator_col])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("Categorical Encoding")
    print(f"  Input base  : {INPUT_BASE}")
    print(f"  Output base : {OUTPUT_BASE}")
    print(f"  Models dir  : {MODELS_DIR}")
    print(f"  Highways    : {HIGHWAYS}")
    print(f"  Cat columns : {CAT_COLUMNS}")
    print("=" * 70)

    all_results = []

    for highway in HIGHWAYS:
        print(f"\n--- {highway} ---")

        # ---- Step 1: load training split and fit encoder ----
        train_src = os.path.join(INPUT_BASE, highway, "train.parquet")
        if not os.path.isfile(train_src):
            print(f"[ERROR] Missing: {train_src}", file=sys.stderr)
            sys.exit(1)

        train_df = pd.read_parquet(train_src)
        vocab = fit_encoder(train_df, CAT_COLUMNS)

        print(f"  Fitted vocabulary:")
        for col, cats in vocab.items():
            print(f"    {col}: {cats}")

        # ---- Step 2: persist the fitted encoder ----
        encoder_path = os.path.join(MODELS_DIR, f"ohe_encoder_{highway}.joblib")
        joblib.dump({"cat_columns": CAT_COLUMNS, "vocabulary": vocab}, encoder_path)
        print(f"  Encoder saved: {encoder_path}")

        # ---- Step 3: encode each split ----
        for split in SPLITS:
            src_path = os.path.join(INPUT_BASE, highway, f"{split}.parquet")
            dst_path = os.path.join(OUTPUT_BASE, highway, f"{split}.parquet")

            if not os.path.isfile(src_path):
                print(f"[ERROR] Missing source: {src_path}", file=sys.stderr)
                sys.exit(1)

            df = pd.read_parquet(src_path)
            n_in = len(df)

            # Validate target column present
            if TARGET_COLUMN not in df.columns:
                raise KeyError(
                    f"Target column '{TARGET_COLUMN}' is absent in '{src_path}'."
                )

            encoded = apply_encoder(df, vocab)
            n_out = len(encoded)

            # Sanity checks
            if n_out != n_in:
                raise RuntimeError(
                    f"Row count changed during encoding: {n_in} -> {n_out} in '{src_path}'."
                )
            if TARGET_COLUMN not in encoded.columns:
                raise RuntimeError(f"'{TARGET_COLUMN}' was lost during encoding.")

            # Ensure label is the last column
            cols = [c for c in encoded.columns if c != TARGET_COLUMN] + [TARGET_COLUMN]
            encoded = encoded[cols]

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            encoded.to_parquet(dst_path, index=False)

            all_results.append({
                "highway": highway,
                "split":   split,
                "rows":    n_out,
                "cols":    len(encoded.columns),
                "path":    dst_path,
            })
            print(f"  [{split:10s}]  {n_out:>8,} rows  {len(encoded.columns)} cols  -> {dst_path}")

    # ---- Final summary ----
    print()
    print("=" * 70)
    print("Summary")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['highway']}/{r['split']:<10s}  {r['rows']:>9,} rows  {r['cols']:>2} cols  {r['path']}")
    print()

    # Show final column layout using the last encoded file as reference
    sample = pd.read_parquet(all_results[-1]["path"])
    print(f"Output columns ({len(sample.columns)}):")
    for col in sample.columns:
        print(f"  {col}  [{sample[col].dtype}]")
    print()
    print("Encoder files:")
    for highway in HIGHWAYS:
        ep = os.path.join(MODELS_DIR, f"ohe_encoder_{highway}.joblib")
        print(f"  {ep}")
    print()
    print("Categorical encoding complete.  Next step: feature engineering.")


if __name__ == "__main__":
    main()
