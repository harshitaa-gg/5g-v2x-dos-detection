import argparse
import gc
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "processed_data"

BATCH_SIZE = 10_000
PROGRESS_EVERY = 25
MAX_TEST_FILES = 10


# ============================================================
# CANONICAL OUTPUT SCHEMA
#
# Must match the reference schema of all six primary splits:
#   - Identifiers: large_string for vehicle IDs, int64 for message_id
#   - Timestamps & Delay: int64 (nanoseconds)
#   - Kinematic & Noise features: float64 (double)
#   - Driver profiles: large_string
#   - Target: int64 (0: Legitimate, 1: DoS Malicious)
#   - Optional road edge: float64 (double, NaN for Normal scenario)
# ============================================================

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


# ============================================================
# PARSING HELPERS
# ============================================================

def first_present(dictionary, *keys):
    """Return the first non-None value among keys in dictionary."""
    if not isinstance(dictionary, dict):
        return None
    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]
    return None


def to_float(value):
    """Safely convert value to finite float or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        if result != result or abs(result) == float("inf"):
            return None
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def to_int(value):
    """Safely convert string or numeric value to integer or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def to_string(value):
    """Safely convert value to string or None."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_records(data, path):
    """
    Extract a list of message records from raw JSON.
    Handles lists, wrapper dicts {'messages': [...]}, and single records.
    """
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "messages" in data:
        records = data["messages"]
        if not isinstance(records, list):
            raise ValueError(f"'messages' must contain a list in {path}")
    elif isinstance(data, dict):
        if "sender" in data or "receiver" in data or "attacker" in data:
            records = [data]
        else:
            raise ValueError(
                f"Unexpected JSON dictionary structure in {path}. "
                f"Keys: {list(data.keys())}"
            )
    else:
        raise ValueError(
            f"Unsupported JSON structure in {path}: {type(data).__name__}"
        )

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {path} is not a JSON object.")

    return records


def parse_position(vehicle):
    """Parse [x, y, z] coordinate representation."""
    position = first_present(vehicle, "pos", "position")

    if isinstance(position, str):
        parts = [part.strip() for part in position.split(",")]
        if len(parts) < 2:
            return None, None, None
        x = to_float(parts[0])
        y = to_float(parts[1])
        z = to_float(parts[2]) if len(parts) > 2 else None
        return x, y, z

    if isinstance(position, (list, tuple)):
        x = to_float(position[0]) if len(position) > 0 else None
        y = to_float(position[1]) if len(position) > 1 else None
        z = to_float(position[2]) if len(position) > 2 else None
        return x, y, z

    if isinstance(position, dict):
        x = to_float(first_present(position, "x", "X", "pos_x"))
        y = to_float(first_present(position, "y", "Y", "pos_y"))
        z = to_float(first_present(position, "z", "Z", "pos_z"))
        return x, y, z

    return None, None, None


def parse_vehicle(vehicle, prefix):
    """Parse vehicle kinematics, noise, and driver profile."""
    if not isinstance(vehicle, dict):
        vehicle = {}

    pos_x, pos_y, pos_z = parse_position(vehicle)

    speed = to_float(first_present(vehicle, "spd", "speed"))
    acceleration = to_float(first_present(vehicle, "acl", "acceleration", "acc"))
    heading = to_float(first_present(vehicle, "hed", "heading"))

    speed_noise = to_float(first_present(vehicle, "spd_noise", "speed_noise"))
    acceleration_noise = to_float(first_present(vehicle, "acl_noise", "acceleration_noise", "acc_noise"))
    heading_noise = to_float(first_present(vehicle, "hed_noise", "heading_noise"))

    driver_profile = to_string(first_present(vehicle, "driversProfile", "driver_profile", "driverProfile"))

    return {
        f"{prefix}_pos_x": pos_x,
        f"{prefix}_pos_y": pos_y,
        f"{prefix}_pos_z": pos_z,
        f"{prefix}_speed": speed,
        f"{prefix}_acceleration": acceleration,
        f"{prefix}_heading": heading,
        f"{prefix}_speed_noise": speed_noise,
        f"{prefix}_acceleration_noise": acceleration_noise,
        f"{prefix}_heading_noise": heading_noise,
        f"{prefix}_driver_profile": driver_profile,
    }


def parse_record(record, scenario_label):
    """
    Parse a single raw JSON message into a flat schema dictionary.

    Label Policy:
      - Normal scenario (scenario_label = 0) -> label = 0
      - DoS scenario (scenario_label = 1) -> label = int(record['attacker'])
    """
    sender = record.get("sender", {})
    receiver = record.get("receiver", {})

    send_time = to_int(first_present(record, "sendTime", "send_time"))
    receive_time = to_int(first_present(record, "rcvTime", "receive_time", "receiveTime"))
    delay = to_int(first_present(record, "delay"))

    if delay is None and send_time is not None and receive_time is not None:
        delay = receive_time - send_time

    if scenario_label == 0:
        label = 0
    else:
        attacker_val = first_present(record, "attacker")
        if attacker_val not in (0, 1, "0", "1", False, True):
            raise ValueError(
                f"Missing or invalid 'attacker' field in DoS scenario: {attacker_val!r}"
            )
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


# ============================================================
# BATCH WRITING & VALIDATION
# ============================================================

def flush_batch(batch, writer):
    if not batch:
        return
    table = pa.Table.from_pylist(batch, schema=SCHEMA)
    writer.write_table(table, row_group_size=BATCH_SIZE)
    batch.clear()


def process_json_file(path, scenario_label, batch, writer, stats):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = extract_records(data, path)

    for record in records:
        row = parse_record(record, scenario_label)
        batch.append(row)
        stats["records"] += 1
        stats["label_counts"][row["label"]] += 1

        if len(batch) >= BATCH_SIZE:
            flush_batch(batch, writer)
            stats["batches_written"] += 1


def validate_parquet(path, expected_rows, require_both_classes):
    parquet_file = pq.ParquetFile(path)
    actual_rows = parquet_file.metadata.num_rows
    actual_names = parquet_file.schema_arrow.names
    actual_types = parquet_file.schema_arrow.types

    if actual_rows != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows:,}, found {actual_rows:,}"
        )

    if actual_names != COLUMNS:
        raise ValueError(
            f"Column names mismatch: expected {COLUMNS}, found {actual_names}"
        )

    if actual_types != SCHEMA.types:
        raise ValueError(
            f"Column types mismatch:\nExpected: {SCHEMA.types}\nActual:   {actual_types}"
        )

    label_counts = {0: 0, 1: 0}
    for batch in parquet_file.iter_batches(columns=["label"], batch_size=BATCH_SIZE):
        labels = batch.column(0).to_pylist()
        for lbl in labels:
            if lbl not in (0, 1):
                raise ValueError(f"Unexpected label {lbl} in temporary file.")
            label_counts[lbl] += 1

    if require_both_classes and (label_counts[0] == 0 or label_counts[1] == 0):
        raise ValueError(f"Missing class in dataset output: {label_counts}")

    print("\nTemporary Parquet validation passed:")
    print(f"  Rows:    {actual_rows:,}")
    print(f"  Columns: {len(actual_names)}")
    print(f"  Label 0: {label_counts[0]:,}")
    print(f"  Label 1: {label_counts[1]:,}")

    return label_counts


# ============================================================
# MAIN BUILD EXECUTION
# ============================================================

def build_split(condition, split, test_mode=False, overwrite=False):
    split_cap = split.capitalize()
    split_lower = split.lower()

    output_dir = OUTPUT_ROOT / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        output_file = output_dir / f"{split_lower}_sample.parquet"
    else:
        output_file = output_dir / f"{split_lower}.parquet"

    temp_file = output_dir / f"{output_file.name}.tmp"

    print("=" * 65)
    print("INCREMENTAL 5G-V2X DATASET BUILDER")
    print("=" * 65)
    print(f"Condition:  {condition}")
    print(f"Split:      {split_cap}")
    print(f"Test mode:  {test_mode}")
    print(f"Output:     {output_file}")
    print(f"Overwrite:  {overwrite}")

    # Critical Safety Check: Prevent accidental destruction of existing datasets
    if output_file.exists() and not overwrite:
        print("\n" + "!" * 65)
        print("CRITICAL SAFETY GUARD TRIGGERED:")
        print(f"Target file already exists: {output_file}")
        print("To protect existing data from accidental overwriting,")
        print("the builder will NOT proceed without the explicit '--overwrite' flag.")
        print("!" * 65)
        print("\nAborting safely. No files were modified.")
        sys.exit(0)

    normal_scenario = f"InTAS_{condition}"
    dos_scenario = f"InTAS_{condition}_dosAttack"

    normal_dir = RAW_ROOT / normal_scenario / split_cap
    dos_dir = RAW_ROOT / dos_scenario / split_cap

    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal raw directory not found: {normal_dir}")
    if not dos_dir.exists():
        raise FileNotFoundError(f"DoS raw directory not found: {dos_dir}")

    normal_files = sorted(normal_dir.rglob("*.json"))
    dos_files = sorted(dos_dir.rglob("*.json"))

    if not normal_files:
        raise FileNotFoundError(f"No Normal JSON files found in {normal_dir}")
    if not dos_files:
        raise FileNotFoundError(f"No DoS JSON files found in {dos_dir}")

    if test_mode:
        normal_files = normal_files[:MAX_TEST_FILES]
        dos_files = dos_files[:MAX_TEST_FILES]

    print(f"\nFiles to process:")
    print(f"  Normal: {len(normal_files):,}")
    print(f"  DoS:    {len(dos_files):,}")

    if temp_file.exists():
        temp_file.unlink()

    stats = {
        "records": 0,
        "files_processed": 0,
        "batches_written": 0,
        "label_counts": {0: 0, 1: 0},
    }

    batch = []
    writer = None

    try:
        writer = pq.ParquetWriter(str(temp_file), SCHEMA, compression="snappy")

        scenarios = [
            ("NORMAL", normal_files, 0),
            ("DOS", dos_files, 1),
        ]

        for scen_name, files, scen_label in scenarios:
            print(f"\nProcessing {scen_name} scenario ({len(files)} files)...")
            for idx, path in enumerate(files, start=1):
                process_json_file(path, scen_label, batch, writer, stats)
                stats["files_processed"] += 1

                if idx % PROGRESS_EVERY == 0 or idx == len(files):
                    print(
                        f"  {scen_name}: {idx:,}/{len(files):,} files | "
                        f"Records: {stats['records']:,} | Labels: {stats['label_counts']}"
                    )

        flush_batch(batch, writer)
        writer.close()
        writer = None

        # Validate temporary parquet before replacing destination
        validate_parquet(
            temp_file,
            expected_rows=stats["records"],
            require_both_classes=not test_mode,
        )

        # Atomic replacement only after validation
        temp_file.replace(output_file)

        print("\n" + "=" * 65)
        print("BUILD SUCCESSFUL")
        print("=" * 65)
        print(f"Published file: {output_file}")
        print(f"Total records:  {stats['records']:,}")
        print(f"Labels:         {stats['label_counts']}")

    except Exception:
        if writer is not None:
            writer.close()
        if temp_file.exists():
            temp_file.unlink()
        raise

    finally:
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental 5G-V2X Dataset Builder with Safety Guards."
    )
    parser.add_argument(
        "--condition",
        choices=["highway_2", "highway_7"],
        default="highway_2",
        help="Highway condition to process (default: highway_2).",
    )
    parser.add_argument(
        "--split",
        choices=["Train", "Validation", "Test", "train", "validation", "test"],
        default="Validation",
        help="Split to process (default: Validation).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run on a small sample of files into a *_sample.parquet file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit overwriting an existing output dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_split(
        condition=args.condition,
        split=args.split,
        test_mode=args.test_mode,
        overwrite=args.overwrite,
    )
