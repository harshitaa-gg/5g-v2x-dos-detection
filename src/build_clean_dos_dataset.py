"""
build_clean_dos_dataset.py

Builds the clean, non-conflicting 5G-V2X DoS detection dataset exclusively from
the raw DoS attack simulation records (InTAS_highway_2_dosAttack and
InTAS_highway_7_dosAttack).

Preserves:
  - Original Train, Validation, and Test partition boundaries.
  - Documented `attacker` label semantics (0: benign background traffic, 1: DoS attack).
  - Exact integer nanosecond timestamps and integer message sequence IDs.
  - Full floating-point kinematics and noise features.

Outputs:
  processed_data/clean_dos_scenario/
    ├── highway_2/
    │   ├── train.parquet
    │   ├── validation.parquet
    │   └── test.parquet
    └── highway_7/
        ├── train.parquet
        ├── validation.parquet
        └── test.parquet

SAFETY:
  - Never touches or modifies data/raw/ or processed_data/highway_*/.
  - Writes to temporary files and validates row counts and schema before promotion.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
CLEAN_OUTPUT_ROOT = PROJECT_ROOT / "processed_data" / "clean_dos_scenario"

BATCH_SIZE = 50_000
PROGRESS_EVERY = 50

COLUMNS = [
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

SCHEMA = pa.schema([
    ("sender_id", pa.large_string()),
    ("sender_alias", pa.large_string()),
    ("message_id", pa.int64()),
    ("send_time", pa.int64()),
    ("receive_time", pa.int64()),
    ("delay", pa.int64()),
    ("sender_pos_x", pa.float64()),
    ("sender_pos_y", pa.float64()),
    ("sender_pos_z", pa.float64()),
    ("sender_speed", pa.float64()),
    ("sender_acceleration", pa.float64()),
    ("sender_heading", pa.float64()),
    ("sender_speed_noise", pa.float64()),
    ("sender_acceleration_noise", pa.float64()),
    ("sender_heading_noise", pa.float64()),
    ("receiver_pos_x", pa.float64()),
    ("receiver_pos_y", pa.float64()),
    ("receiver_pos_z", pa.float64()),
    ("receiver_speed", pa.float64()),
    ("receiver_acceleration", pa.float64()),
    ("receiver_heading", pa.float64()),
    ("receiver_speed_noise", pa.float64()),
    ("receiver_acceleration_noise", pa.float64()),
    ("receiver_heading_noise", pa.float64()),
    ("sender_driver_profile", pa.large_string()),
    ("receiver_driver_profile", pa.large_string()),
    ("label", pa.int64()),
    ("distance_to_road_edge", pa.float64()),
])


def first_present(dictionary, *keys):
    if not isinstance(dictionary, dict):
        return None
    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]
    return None


def to_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        res = float(value)
        if res != res or abs(res) == float("inf"):
            return None
        return res
    except (TypeError, ValueError, OverflowError):
        return None


def to_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def to_string(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_position(vehicle):
    pos = first_present(vehicle, "pos", "position")
    if isinstance(pos, str):
        parts = [p.strip() for p in pos.split(",")]
        if len(parts) < 2:
            return None, None, None
        x = to_float(parts[0])
        y = to_float(parts[1])
        z = to_float(parts[2]) if len(parts) > 2 else None
        return x, y, z
    if isinstance(pos, (list, tuple)):
        x = to_float(pos[0]) if len(pos) > 0 else None
        y = to_float(pos[1]) if len(pos) > 1 else None
        z = to_float(pos[2]) if len(pos) > 2 else None
        return x, y, z
    if isinstance(pos, dict):
        x = to_float(first_present(pos, "x", "X", "pos_x"))
        y = to_float(first_present(pos, "y", "Y", "pos_y"))
        z = to_float(first_present(pos, "z", "Z", "pos_z"))
        return x, y, z
    return None, None, None


def parse_vehicle(vehicle, prefix):
    if not isinstance(vehicle, dict):
        vehicle = {}
    pos_x, pos_y, pos_z = parse_position(vehicle)
    return {
        f"{prefix}_pos_x": pos_x,
        f"{prefix}_pos_y": pos_y,
        f"{prefix}_pos_z": pos_z,
        f"{prefix}_speed": to_float(first_present(vehicle, "spd", "speed")),
        f"{prefix}_acceleration": to_float(first_present(vehicle, "acl", "acceleration", "acc")),
        f"{prefix}_heading": to_float(first_present(vehicle, "hed", "heading")),
        f"{prefix}_speed_noise": to_float(first_present(vehicle, "spd_noise", "speed_noise")),
        f"{prefix}_acceleration_noise": to_float(first_present(vehicle, "acl_noise", "acceleration_noise", "acc_noise")),
        f"{prefix}_heading_noise": to_float(first_present(vehicle, "hed_noise", "heading_noise")),
        f"{prefix}_driver_profile": to_string(first_present(vehicle, "driversProfile", "driver_profile", "driverProfile")),
    }


def parse_dos_record(record):
    """
    Parse a single message record from the raw DoS simulation JSON.
    Directly extracts the ground-truth attacker field (0 or 1).
    """
    sender = record.get("sender", {})
    receiver = record.get("receiver", {})

    send_time = to_int(first_present(record, "sendTime", "send_time"))
    receive_time = to_int(first_present(record, "rcvTime", "receive_time", "receiveTime"))
    delay = to_int(first_present(record, "delay"))

    if delay is None and send_time is not None and receive_time is not None:
        delay = receive_time - send_time

    attacker_val = first_present(record, "attacker")
    if attacker_val not in (0, 1, "0", "1", False, True):
        raise ValueError(f"Invalid or missing 'attacker' label: {attacker_val!r}")
    label = int(attacker_val)

    message_id = to_int(first_present(record, "messageID", "message_id", "messageId"))

    row = {
        "sender_id": to_string(first_present(record, "sender_id", "senderID")),
        "sender_alias": to_string(first_present(record, "sender_alias", "senderAlias")),
        "message_id": message_id,
        "send_time": send_time,
        "receive_time": receive_time,
        "delay": delay,
        "label": label,
    }

    row.update(parse_vehicle(sender, "sender"))
    row.update(parse_vehicle(receiver, "receiver"))

    distance = to_float(first_present(record, "distance_to_road_edge"))
    if distance is None and isinstance(sender, dict):
        distance = to_float(first_present(sender, "distance_to_road_edge"))
    row["distance_to_road_edge"] = distance

    return {col: row.get(col) for col in COLUMNS}


def build_dos_split(condition, split, overwrite=False):
    split_cap = split.capitalize()
    split_lower = split.lower()

    out_dir = CLEAN_OUTPUT_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split_lower}.parquet"
    tmp_file = out_dir / f"{split_lower}.parquet.tmp"

    print("\n" + "=" * 65)
    print(f"BUILDING CLEAN DoS DATASET: {condition} - {split_cap}")
    print("=" * 65)
    print(f"Target Output: {out_file}")

    if out_file.exists() and not overwrite:
        print(f"File already exists: {out_file}")
        print("Skipping (pass --overwrite to recreate).")
        return out_file

    raw_scenario = f"InTAS_{condition}_dosAttack"
    raw_folder = RAW_ROOT / raw_scenario / split_cap

    if not raw_folder.exists():
        raise FileNotFoundError(f"Raw DoS directory not found: {raw_folder}")

    files = sorted(raw_folder.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {raw_folder}")

    print(f"Source JSON files: {len(files):,}")

    if tmp_file.exists():
        tmp_file.unlink()

    batch = []
    stats = {
        "records": 0,
        "label_counts": {0: 0, 1: 0},
    }

    start_time = time.time()
    writer = None

    try:
        writer = pq.ParquetWriter(str(tmp_file), SCHEMA, compression="snappy")

        for idx, path in enumerate(files, start=1):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]

            for rec in records:
                row = parse_dos_record(rec)
                batch.append(row)
                stats["records"] += 1
                stats["label_counts"][row["label"]] += 1

                if len(batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=SCHEMA)
                    writer.write_table(table, row_group_size=BATCH_SIZE)
                    batch.clear()

            if idx % PROGRESS_EVERY == 0 or idx == len(files):
                elapsed = time.time() - start_time
                print(
                    f"  Processed {idx:,}/{len(files):,} files | "
                    f"Records: {stats['records']:,} | "
                    f"Labels: {stats['label_counts']} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        if batch:
            table = pa.Table.from_pylist(batch, schema=SCHEMA)
            writer.write_table(table, row_group_size=BATCH_SIZE)
            batch.clear()

        writer.close()
        writer = None

        # Validate temporary parquet before promotion
        with pq.ParquetFile(tmp_file) as pf:
            if pf.metadata.num_rows != stats["records"]:
                raise ValueError(
                    f"Row mismatch: expected {stats['records']:,}, got {pf.metadata.num_rows:,}"
                )
            if pf.schema_arrow.names != COLUMNS:
                raise ValueError("Column name mismatch in temporary file.")
            if pf.schema_arrow.types != SCHEMA.types:
                raise ValueError("Data type mismatch in temporary file.")

        del pf
        gc.collect()

        tmp_file.replace(out_file)
        print(f"\nSUCCESS: Published {out_file}")
        print(f"Total Rows: {stats['records']:,} | Class 0: {stats['label_counts'][0]:,}, Class 1: {stats['label_counts'][1]:,}")

    except Exception:
        if writer is not None:
            writer.close()
        gc.collect()
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
        raise
    finally:
        gc.collect()

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Build clean DoS-scenario dataset splits.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clean dataset files.")
    parser.add_argument("--condition", choices=["highway_2", "highway_7", "all"], default="all")
    args = parser.parse_args()

    conditions = ["highway_2", "highway_7"] if args.condition == "all" else [args.condition]
    splits = ["Train", "Validation", "Test"]

    for cond in conditions:
        for s in splits:
            build_dos_split(cond, s, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
