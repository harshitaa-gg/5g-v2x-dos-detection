from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data"
OUT_DIR = ROOT / "results" / "data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    DATA_DIR / highway / f"{split}.parquet"
    for highway in ("highway_2", "highway_7")
    for split in ("train", "validation", "test")
]

HIGH_CARDINALITY = {
    "sender_id", "sender_alias", "message_id",
    "send_time", "receive_time"
}
PROFILE_COLUMNS = {"sender_driver_profile", "receiver_driver_profile"}
NUMERIC_COLUMNS = {
    "send_time", "receive_time", "delay", "sender_pos_x", "sender_pos_y",
    "sender_pos_z", "sender_speed", "sender_acceleration", "sender_heading",
    "sender_speed_noise", "sender_acceleration_noise", "sender_heading_noise",
    "receiver_pos_x", "receiver_pos_y", "receiver_pos_z", "receiver_speed",
    "receiver_acceleration", "receiver_heading", "receiver_speed_noise",
    "receiver_acceleration_noise", "receiver_heading_noise",
    "distance_to_road_edge", "label"
}

file_rows = []
column_rows = []
label_rows = []
profile_values = {}

for path in FILES:
    if not path.exists():
        print(f"MISSING FILE: {path}")
        file_rows.append({
            "file": str(path.relative_to(ROOT)),
            "status": "MISSING",
            "rows": None,
            "columns": None,
            "schema": None,
        })
        continue

    print(f"Auditing {path.relative_to(ROOT)} ...")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    file_id = f"{path.parent.name}/{path.stem}"
    file_rows.append({
        "file": str(path.relative_to(ROOT)),
        "status": "FOUND",
        "rows": parquet.metadata.num_rows,
        "columns": len(schema.names),
        "schema": " | ".join(f"{field.name}:{field.type}" for field in schema),
    })

    stats = {}
    for field in schema:
        stats[field.name] = {
            "dtype": str(field.type),
            "null_count": 0,
            "nan_count": 0,
            "infinite_count": 0,
            "min": None,
            "max": None,
            "category_values": set(),
        }

    label_counts = {}
    for batch in parquet.iter_batches(batch_size=200_000):
        df = batch.to_pandas()

        if "label" in df.columns:
            counts = df["label"].value_counts(dropna=False)
            for value, count in counts.items():
                key = str(value)
                label_counts[key] = label_counts.get(key, 0) + int(count)

        for col in df.columns:
            s = df[col]
            entry = stats[col]
            entry["null_count"] += int(s.isna().sum())

            if pd.api.types.is_numeric_dtype(s.dtype):
                arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
                entry["nan_count"] += int(np.isnan(arr).sum())
                entry["infinite_count"] += int(np.isinf(arr).sum())
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    batch_min = float(finite.min())
                    batch_max = float(finite.max())
                    entry["min"] = batch_min if entry["min"] is None else min(entry["min"], batch_min)
                    entry["max"] = batch_max if entry["max"] is None else max(entry["max"], batch_max)

            if col in PROFILE_COLUMNS:
                vals = s.dropna().astype(str).unique().tolist()
                seen = profile_values.setdefault(col, set())
                for value in vals:
                    if len(seen) < 100:
                        seen.add(value)

    for col, entry in stats.items():
        values = sorted(entry["category_values"])
        column_rows.append({
            "file": str(path.relative_to(ROOT)),
            "column": col,
            "dtype": entry["dtype"],
            "missing_count": entry["null_count"],
            "nan_count": entry["nan_count"],
            "infinite_count": entry["infinite_count"],
            "finite_min": entry["min"],
            "finite_max": entry["max"],
        })

    for label, count in sorted(label_counts.items()):
        label_rows.append({
            "file": str(path.relative_to(ROOT)),
            "label_value": label,
            "count": count,
        })

pd.DataFrame(file_rows).to_csv(OUT_DIR / "file_summary.csv", index=False)
pd.DataFrame(column_rows).to_csv(OUT_DIR / "column_quality.csv", index=False)
pd.DataFrame(label_rows).to_csv(OUT_DIR / "label_counts.csv", index=False)

with (OUT_DIR / "profile_values.json").open("w", encoding="utf-8") as f:
    json.dump({k: sorted(v) for k, v in profile_values.items()}, f, indent=2)

print("\nAUDIT COMPLETE")
print(f"Reports saved in: {OUT_DIR}")
print("- file_summary.csv: row counts and schemas")
print("- column_quality.csv: missing values, NaN/infinity, numeric ranges")
print("- label_counts.csv: label distributions")
print("- profile_values.json: observed driver-profile categories")
print("\nThis is a read-only audit. It does not modify or delete your datasets.")
