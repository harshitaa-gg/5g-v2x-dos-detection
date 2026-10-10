
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# Start with a small sample. Switch TEST_MODE to False only
# after the sample output passes validation.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "processed_data"

CONDITION = "highway_7"
SPLIT = "Test"

NORMAL_SCENARIO = "InTAS_highway_7"
DOS_SCENARIO = "InTAS_highway_7_dosAttack"

TEST_MODE = False
MAX_FILES_PER_SCENARIO = 10

OUTPUT_DIR = OUTPUT_ROOT / CONDITION

if TEST_MODE:
    OUTPUT_FILE = OUTPUT_DIR / "test_sample.parquet"
else:
    OUTPUT_FILE = OUTPUT_DIR / "test.parquet"

TEMP_FILE = OUTPUT_DIR / (OUTPUT_FILE.name + ".tmp")

BATCH_SIZE = 10_000
PROGRESS_EVERY = 5


# ============================================================
# OUTPUT SCHEMA
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
    ("sender_id", pa.string()),
    ("sender_alias", pa.string()),
    ("message_id", pa.string()),
    ("send_time", pa.float64()),
    ("receive_time", pa.float64()),
    ("delay", pa.float64()),
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
    ("sender_driver_profile", pa.string()),
    ("receiver_driver_profile", pa.string()),
    ("label", pa.int8()),
    ("distance_to_road_edge", pa.float64()),
])


# ============================================================
# GENERAL HELPERS
# ============================================================

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
        result = float(value)

        if result != result or abs(result) == float("inf"):
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def to_string(value):
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


# ============================================================
# JSON STRUCTURE HANDLING
#
# Supports:
#   1. A list of records
#   2. A single record
#   3. A wrapper dictionary containing {"messages": [...]}
#
# Unexpected structures raise an error instead of silently
# producing a malformed dataset.
# ============================================================

def extract_records(data, path):
    if isinstance(data, list):
        records = data

    elif isinstance(data, dict) and "messages" in data:
        records = data["messages"]

        if not isinstance(records, list):
            raise ValueError(
                f"'messages' must contain a list in {path}"
            )

    elif isinstance(data, dict):
        # A single record must look like a vehicle-message record.
        if (
            "sender" in data
            or "receiver" in data
            or "attacker" in data
        ):
            records = [data]
        else:
            raise ValueError(
                f"Unexpected JSON dictionary structure in {path}. "
                f"Top-level keys: {list(data.keys())}"
            )

    else:
        raise ValueError(
            f"Unsupported JSON structure in {path}: "
            f"{type(data).__name__}"
        )

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {index} in {path} is not a JSON object."
            )

    return records


# ============================================================
# POSITION PARSING
# Supports lists, dictionaries, and comma-separated strings.
# Examples:
#   [12.4, 35.6, 0.0]
#   {"x": 12.4, "y": 35.6, "z": 0.0}
#   "12.4,35.6,0.0"
# ============================================================

def parse_position(vehicle):
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


# ============================================================
# VEHICLE FEATURE PARSING
# ============================================================

def parse_vehicle(vehicle, prefix):
    if not isinstance(vehicle, dict):
        vehicle = {}

    pos_x, pos_y, pos_z = parse_position(vehicle)

    speed = to_float(
        first_present(vehicle, "spd", "speed")
    )

    acceleration = to_float(
        first_present(vehicle, "acl", "acceleration", "acc")
    )

    heading = to_float(
        first_present(vehicle, "hed", "heading")
    )

    speed_noise = to_float(
        first_present(vehicle, "spd_noise", "speed_noise")
    )

    acceleration_noise = to_float(
        first_present(
            vehicle, "acl_noise", "acceleration_noise", "acc_noise"
        )
    )

    heading_noise = to_float(
        first_present(vehicle, "hed_noise", "heading_noise")
    )

    driver_profile = to_string(
        first_present(
            vehicle,
            "driversProfile",
            "driver_profile",
            "driverProfile",
        )
    )

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


# ============================================================
# RECORD PARSING AND LABELS
#
# Normal scenario records -> label 0
# DoS scenario records -> use the dataset's attacker field
#
# This is the project's intended binary-label convention.
# Invalid or missing DoS labels are rejected.
# ============================================================

def parse_record(record, scenario_label):
    sender = record.get("sender", {})
    receiver = record.get("receiver", {})

    if not isinstance(sender, dict):
        sender = {}

    if not isinstance(receiver, dict):
        receiver = {}

    send_time = to_float(
        first_present(record, "sendTime", "send_time")
    )

    receive_time = to_float(
        first_present(record, "rcvTime", "receive_time", "receiveTime")
    )

    delay = to_float(first_present(record, "delay"))

    if delay is None and send_time is not None and receive_time is not None:
        delay = receive_time - send_time

    if scenario_label == 0:
        label = 0
    else:
        attacker_value = first_present(record, "attacker")

        if attacker_value not in (0, 1, False, True):
            raise ValueError(
                "Invalid or missing attacker label in DoS scenario: "
                f"{attacker_value!r}"
            )

        label = int(attacker_value)

    row = {
        "sender_id": to_string(
            first_present(record, "sender_id", "senderID")
        ),
        "sender_alias": to_string(
            first_present(record, "sender_alias", "senderAlias")
        ),
        "message_id": to_string(
            first_present(record, "messageID", "message_id", "messageId")
        ),
        "send_time": send_time,
        "receive_time": receive_time,
        "delay": delay,
        "label": label,
    }

    row.update(parse_vehicle(sender, "sender"))
    row.update(parse_vehicle(receiver, "receiver"))

    distance = to_float(
        first_present(record, "distance_to_road_edge")
    )

    if distance is None:
        distance = to_float(
            first_present(sender, "distance_to_road_edge")
        )

    row["distance_to_road_edge"] = distance

    return {column: row.get(column) for column in COLUMNS}


# ============================================================
# PARQUET BATCH WRITING
# ============================================================

def flush_batch(batch, writer):
    if not batch:
        return

    table = pa.Table.from_pylist(batch, schema=SCHEMA)
    writer.write_table(table, row_group_size=BATCH_SIZE)
    batch.clear()


def process_json_file(path, scenario_label, batch, writer, stats):
    # Note: json.load reads one complete source file into memory.
    # Batch writing limits the additional processed-row buffer.
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


# ============================================================
# FILE DISCOVERY
# ============================================================

def get_json_files(scenario_name):
    folder = RAW_ROOT / scenario_name / SPLIT

    if not folder.exists():
        raise FileNotFoundError(f"Scenario folder not found: {folder}")

    files = sorted(folder.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found in: {folder}")

    if TEST_MODE:
        files = files[:MAX_FILES_PER_SCENARIO]

    return files


# ============================================================
# VALIDATE TEMPORARY PARQUET BEFORE PUBLISHING
# ============================================================

def validate_parquet(path, expected_rows, require_both_classes):
    parquet_file = pq.ParquetFile(path)

    actual_rows = parquet_file.metadata.num_rows
    actual_names = parquet_file.schema_arrow.names
    actual_types = parquet_file.schema_arrow.types

    if actual_rows != expected_rows:
        raise ValueError(
            f"Row-count mismatch: expected {expected_rows:,}, "
            f"found {actual_rows:,}"
        )

    if actual_names != COLUMNS:
        raise ValueError(
            "Column names or order do not match the expected schema."
        )

    if actual_types != SCHEMA.types:
        raise ValueError(
            "Column data types do not match the expected schema.\n"
            f"Expected: {SCHEMA.types}\n"
            f"Actual:   {actual_types}"
        )

    # Count labels in batches without loading the full dataset into memory.
    label_counts = {0: 0, 1: 0}

    for record_batch in parquet_file.iter_batches(
        columns=["label"],
        batch_size=BATCH_SIZE,
    ):
        labels = record_batch.column(0).to_pylist()

        for label in labels:
            if label not in (0, 1):
                raise ValueError(f"Unexpected label value: {label!r}")

            label_counts[label] += 1

    if sum(label_counts.values()) != actual_rows:
        raise ValueError("Validated label counts do not match row count.")

    if require_both_classes and (
        label_counts[0] == 0 or label_counts[1] == 0
    ):
        raise ValueError(
            f"Missing class in full dataset: {label_counts}"
        )

    print("\nTemporary Parquet validation passed.")
    print(f"Rows:       {actual_rows:,}")
    print(f"Columns:    {len(actual_names)}")
    print(f"Label 0:    {label_counts[0]:,}")
    print(f"Label 1:    {label_counts[1]:,}")

    return label_counts


# ============================================================
# MAIN BUILDER
# ============================================================

def build_split():
    print("=" * 65)
    print("INCREMENTAL V2X DATASET BUILDER")
    print("=" * 65)
    print(f"Condition: {CONDITION}")
    print(f"Split:     {SPLIT}")
    print(f"Test mode: {TEST_MODE}")
    print(f"Output:    {OUTPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal_files = get_json_files(NORMAL_SCENARIO)
    dos_files = get_json_files(DOS_SCENARIO)

    print(f"\nNormal files selected: {len(normal_files):,}")
    print(f"DoS files selected:    {len(dos_files):,}")

    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    stats = {
        "records": 0,
        "files_processed": 0,
        "batches_written": 0,
        "label_counts": {0: 0, 1: 0},
    }

    batch = []
    writer = None

    try:
        writer = pq.ParquetWriter(
            str(TEMP_FILE),
            SCHEMA,
            compression="snappy",
        )

        scenarios = [
            ("NORMAL", normal_files, 0),
            ("DOS", dos_files, 1),
        ]

        for scenario_name, files, scenario_label in scenarios:
            print(f"\nProcessing {scenario_name} scenario...")

            for index, path in enumerate(files, start=1):
                process_json_file(
                    path,
                    scenario_label,
                    batch,
                    writer,
                    stats,
                )

                stats["files_processed"] += 1

                if index % PROGRESS_EVERY == 0 or index == len(files):
                    print(
                        f"{scenario_name}: {index:,}/{len(files):,} files | "
                        f"Records: {stats['records']:,} | "
                        f"Labels: {stats['label_counts']}"
                    )

        flush_batch(batch, writer)
        writer.close()
        writer = None

        # Validate the temporary output BEFORE replacing any existing file.
        validate_parquet(
            TEMP_FILE,
            expected_rows=stats["records"],
            require_both_classes=not TEST_MODE,
        )

        # Publish only after successful validation.
        TEMP_FILE.replace(OUTPUT_FILE)

    except Exception:
        if writer is not None:
            writer.close()

        if TEMP_FILE.exists():
            TEMP_FILE.unlink()

        raise

    print("\n" + "=" * 65)
    print("BUILD COMPLETE")
    print("=" * 65)
    print(f"Files processed: {stats['files_processed']:,}")
    print(f"Records:         {stats['records']:,}")
    print(f"Label 0:         {stats['label_counts'][0]:,}")
    print(f"Label 1:         {stats['label_counts'][1]:,}")
    print(f"Saved to:        {OUTPUT_FILE}")
    print("Final output was published only after validation.")


if __name__ == "__main__":
    build_split()
