
from pathlib import Path
from collections import Counter

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


def main():
    schemas = {}
    overall_ok = True

    for name, path in FILES.items():
        print(f"\n{'=' * 60}")
        print(f"Checking: {name}")

        if not path.exists():
            print(f"FAIL: File not found: {path}")
            overall_ok = False
            continue

        try:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            schemas[name] = schema

            columns = schema.names

            if len(columns) != len(set(columns)):
                print("FAIL: Duplicate column names")
                overall_ok = False

            if "label" not in columns:
                print("FAIL: Missing label column")
                overall_ok = False
                continue

            null_counts = Counter()
            label_counts = Counter()

            for batch in parquet.iter_batches(batch_size=100_000):
                for column_name, column in zip(
                    batch.schema.names, batch.columns
                ):
                    null_counts[column_name] += column.null_count

                labels = batch.column("label").to_pylist()
                label_counts.update(labels)

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

            print(f"Rows: {parquet.metadata.num_rows:,}")
            print(f"Columns: {len(columns)}")
            print(f"Label counts: {dict(label_counts)}")
            print(f"Schema: {schema}")

            print(
                "Missing values:",
                f"{sum(missing_columns.values()):,}",
            )

            if missing_columns:
                print("Columns with missing values:", missing_columns)

            if invalid_labels:
                print("FAIL: Unexpected labels:", invalid_labels)
                overall_ok = False
            elif label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
                print("WARNING: One class is absent in this split.")
            else:
                print("PASS: Both classes present; labels are valid.")

        except Exception as error:
            print(f"FAIL: Could not read/check file: {error}")
            overall_ok = False

    print(f"\n{'=' * 60}")
    print("CROSS-SPLIT COMPATIBILITY")

    if len(schemas) != len(FILES):
        print("FAIL: One or more files could not be checked.")
        overall_ok = False
    elif schemas:
        reference_name = next(iter(schemas))
        reference = schemas[reference_name]

        for name, schema in schemas.items():
            same_columns = schema.names == reference.names
            same_types = schema.types == reference.types

            print(
                f"{name}: "
                f"columns={'MATCH' if same_columns else 'MISMATCH'}, "
                f"types={'MATCH' if same_types else 'MISMATCH'}"
            )

            if not same_columns or not same_types:
                overall_ok = False
                print("  Actual columns:", schema.names)
                print("  Actual types:", schema.types)

    print(f"\nFINAL RESULT: {'PASS' if overall_ok else 'ISSUES FOUND'}")
    print("Note: This script does not modify any dataset.")


if __name__ == "__main__":
    main()
