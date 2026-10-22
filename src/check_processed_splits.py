import sys
from pathlib import Path
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "highway_2/train": ROOT / "processed_data/highway_2/train.parquet",
    "highway_2/validation": ROOT / "processed_data/highway_2/validation.parquet",
    "highway_2/test": ROOT / "processed_data/highway_2/test.parquet",
    "highway_7/train": ROOT / "processed_data/highway_7/train.parquet",
    "highway_7/validation": ROOT / "processed_data/highway_7/validation.parquet",
    "highway_7/test": ROOT / "processed_data/highway_7/test.parquet",
}

EXPECTED_COLUMNS = [
    "sender_id",
    "sender_alias",
    "message_id",
    "send_time",
    "receive_time",
    "delay",
    "sender_pos_x",
    "sender_pos_y",
    "sender_pos_z",
    "sender_speed",
    "sender_acceleration",
    "sender_heading",
    "sender_speed_noise",
    "sender_acceleration_noise",
    "sender_heading_noise",
    "receiver_pos_x",
    "receiver_pos_y",
    "receiver_pos_z",
    "receiver_speed",
    "receiver_acceleration",
    "receiver_heading",
    "receiver_speed_noise",
    "receiver_acceleration_noise",
    "receiver_heading_noise",
    "sender_driver_profile",
    "receiver_driver_profile",
    "label",
    "distance_to_road_edge",
]


def main():
    schemas = {}
    overall_ok = True

    print("=" * 65)
    print("PROCESSED DATASET SPLIT & SCHEMA VALIDATOR")
    print("=" * 65)

    for name, path in FILES.items():
        print(f"\nChecking split: {name}")
        print("-" * 50)

        if not path.exists():
            print(f"FAIL: File not found: {path}")
            overall_ok = False
            continue

        try:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            schemas[name] = schema

            columns = schema.names
            num_rows = parquet.metadata.num_rows

            if num_rows == 0:
                print("FAIL: File is empty (0 rows).")
                overall_ok = False

            if columns != EXPECTED_COLUMNS:
                print("FAIL: Column names or ordering do not match expected schema.")
                print("  Expected:", EXPECTED_COLUMNS)
                print("  Actual:  ", columns)
                overall_ok = False

            if len(columns) != len(set(columns)):
                print("FAIL: Duplicate column names detected.")
                overall_ok = False

            if "label" not in columns:
                print("FAIL: Missing 'label' column.")
                overall_ok = False
                continue

            null_counts = Counter()
            label_counts = Counter()
            delay_mismatches = 0

            for batch in parquet.iter_batches(batch_size=100_000):
                for column_name, column in zip(batch.schema.names, batch.columns):
                    null_counts[column_name] += column.null_count

                labels = batch.column("label").to_pylist()
                label_counts.update(labels)

                # Delay verification
                st = batch.column("send_time").to_numpy()
                rt = batch.column("receive_time").to_numpy()
                dl = batch.column("delay").to_numpy()
                mismatches = ((rt - st) != dl).sum()
                delay_mismatches += int(mismatches)

            invalid_labels = {
                value: count
                for value, count in label_counts.items()
                if value not in (0, 1)
            }

            missing_columns = {
                col: count
                for col, count in null_counts.items()
                if count > 0
            }

            print(f"  Rows:         {num_rows:,}")
            print(f"  Columns:      {len(columns)}")
            print(f"  Label counts: {dict(label_counts)}")

            if delay_mismatches > 0:
                print(f"  FAIL: Delay mismatches (delay != rcvTime - sendTime): {delay_mismatches:,}")
                overall_ok = False
            else:
                print("  Delay check:  PASS (delay == receive_time - send_time holds)")

            if missing_columns:
                print(f"  Missing vals: {missing_columns}")
                # Note: distance_to_road_edge is missing from Normal scenario records
                unexpected_missing = {k: v for k, v in missing_columns.items() if k != "distance_to_road_edge"}
                if unexpected_missing:
                    print(f"  FAIL: Unexpected missing values in: {unexpected_missing}")
                    overall_ok = False

            if invalid_labels:
                print(f"  FAIL: Unexpected labels found: {invalid_labels}")
                overall_ok = False
            elif label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
                print("  WARNING: One class is absent in this split.")
            else:
                print("  Label check:  PASS (Both 0 and 1 classes present and valid)")

        except Exception as error:
            print(f"FAIL: Could not read or check file: {error}")
            overall_ok = False

    print("\n" + "=" * 65)
    print("CROSS-SPLIT COMPATIBILITY CHECK")
    print("=" * 65)

    if len(schemas) != len(FILES):
        print("FAIL: One or more files could not be checked.")
        overall_ok = False
    elif schemas:
        reference_name = next(iter(schemas))
        reference = schemas[reference_name]

        for name, schema in schemas.items():
            same_columns = schema.names == reference.names
            same_types = schema.types == reference.types

            status = "MATCH" if (same_columns and same_types) else "MISMATCH"
            print(f"  {name:<25}: columns={'MATCH' if same_columns else 'MISMATCH'}, types={'MATCH' if same_types else 'MISMATCH'} -> {status}")

            if not same_columns or not same_types:
                overall_ok = False
                print(f"    Mismatch in {name}:")
                if not same_columns:
                    print("    Columns:", schema.names)
                if not same_types:
                    print("    Types:  ", schema.types)

    print("\n" + "=" * 65)
    result_str = "PASS - ALL SPLITS COMPLIANT" if overall_ok else "FAIL - ISSUES DETECTED"
    print(f"FINAL RESULT: {result_str}")
    print("Note: This script is strictly read-only and does not modify any dataset.")
    print("=" * 65)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
