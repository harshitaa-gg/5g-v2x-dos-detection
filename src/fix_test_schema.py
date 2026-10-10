
from pathlib import Path
import shutil
import gc

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "processed_data" / "highway_7"

reference_path = folder / "train.parquet"
target_path = folder / "test.parquet"
temp_path = folder / "test_schema_fixed.tmp"
backup_path = folder / "test_before_schema_fix.parquet"

reference_schema = pq.read_schema(reference_path)

if reference_schema.names != pq.read_schema(target_path).names:
    raise ValueError("Column names/order differ. Stopping safely.")

if backup_path.exists():
    print(f"Existing backup found: {backup_path}")
else:
    shutil.copy2(target_path, backup_path)
    print(f"Backup created: {backup_path}")

original_rows = pq.ParquetFile(target_path).metadata.num_rows

try:
    # Write and close the temporary Parquet file.
    with pq.ParquetFile(target_path) as reader:
        with pq.ParquetWriter(temp_path, reference_schema) as writer:
            for batch in reader.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])

                arrays = []
                for field in reference_schema:
                    column = table[field.name]
                    if column.type != field.type:
                        column = column.cast(field.type, safe=True)
                    arrays.append(column)

                normalized = pa.Table.from_arrays(
                    arrays, schema=reference_schema
                )
                writer.write_table(normalized)

    # Release file handles before validating/replacing.
    gc.collect()

    with pq.ParquetFile(temp_path) as check:
        if check.metadata.num_rows != original_rows:
            raise ValueError("Row count changed; original preserved.")

        if not check.schema_arrow.equals(
            reference_schema, check_metadata=False
        ):
            raise ValueError("Schema mismatch; original preserved.")

    gc.collect()

    # Replace only after all handles are closed.
    temp_path.replace(target_path)

    print("SUCCESS: Highway 7 Test schema fixed.")
    print(f"Rows preserved: {original_rows:,}")
    print(f"Backup: {backup_path}")

except Exception:
    gc.collect()
    print("Fix did not complete. Original target was not intentionally changed.")
    print(f"Keep your backup at: {backup_path}")
    print(f"If present, temporary file: {temp_path}")
    raise
