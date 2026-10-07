import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "processed_data"


def parse_position(position_string):
    """
    Convert 'x,y,z' position string into three floats.
    """

    if position_string is None:
        return None, None, None

    try:
        x, y, z = position_string.split(",")

        return float(x), float(y), float(z)

    except (ValueError, AttributeError):
        return None, None, None


def parse_record(record, scenario_label):
    """
    Convert one raw V2X message into a flat processed record.

    Normal scenario:
        scenario_label = 0

    DoS scenario:
        scenario_label = 1

        If the record contains 'attacker', the actual
        attacker value is always used.

        If a DoS record is missing 'attacker', raise an
        error instead of silently creating a wrong label.
    """

    sender = record.get("sender", {})
    receiver = record.get("receiver", {})

    # ---------------------------------------------------------
    # Parse sender position
    # ---------------------------------------------------------

    sender_x, sender_y, sender_z = parse_position(
        sender.get("pos")
    )

    # ---------------------------------------------------------
    # Parse receiver position
    # ---------------------------------------------------------

    receiver_x, receiver_y, receiver_z = parse_position(
        receiver.get("pos")
    )

    # ---------------------------------------------------------
    # Convert timestamps
    # ---------------------------------------------------------

    send_time = int(record["sendTime"])
    receive_time = int(record["rcvTime"])

    delay = receive_time - send_time

    # ---------------------------------------------------------
    # Determine label
    # ---------------------------------------------------------

    if "attacker" in record:
        label = int(record["attacker"])

    elif scenario_label == 1:
        raise ValueError(
            "DoS record is missing the 'attacker' field. "
            "Processing stopped to prevent incorrect labels."
        )

    else:
        label = 0

    # ---------------------------------------------------------
    # Build flat processed record
    # ---------------------------------------------------------

    processed = {
        # Identification
        "sender_id": record.get("sender_id"),
        "sender_alias": str(record.get("sender_alias")),
        "message_id": int(record["messageID"]),

        # Time
        "send_time": send_time,
        "receive_time": receive_time,
        "delay": delay,

        # Sender position
        "sender_pos_x": sender_x,
        "sender_pos_y": sender_y,
        "sender_pos_z": sender_z,

        # Sender mobility
        "sender_speed": float(sender["spd"]),
        "sender_acceleration": float(sender["acl"]),
        "sender_heading": float(sender["hed"]),

        # Sender noise
        "sender_speed_noise": float(sender["spd_noise"]),
        "sender_acceleration_noise": float(sender["acl_noise"]),
        "sender_heading_noise": float(sender["hed_noise"]),

        # Receiver position
        "receiver_pos_x": receiver_x,
        "receiver_pos_y": receiver_y,
        "receiver_pos_z": receiver_z,

        # Receiver mobility
        "receiver_speed": float(receiver["spd"]),
        "receiver_acceleration": float(receiver["acl"]),
        "receiver_heading": float(receiver["hed"]),

        # Receiver noise
        "receiver_speed_noise": float(receiver["spd_noise"]),
        "receiver_acceleration_noise": float(
            receiver["acl_noise"]
        ),
        "receiver_heading_noise": float(
            receiver["hed_noise"]
        ),

        # Driver profiles
        "sender_driver_profile": sender.get(
            "driversProfile"
        ),
        "receiver_driver_profile": receiver.get(
            "driversProfile"
        ),

        # Binary target
        "label": label,

        # Optional DoS-related field
        "distance_to_road_edge": (
            float(sender["distance_to_road_edge"])
            if "distance_to_road_edge" in sender
            else None
        ),
    }

    return processed


def process_json_file(json_file, scenario_label):
    """
    Read one JSON file and convert all messages.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    processed_records = []

    for record in records:
        processed_records.append(
            parse_record(
                record,
                scenario_label=scenario_label
            )
        )

    return processed_records


def build_split(
    condition,
    split,
    normal_scenario,
    dos_scenario,
):
    """
    Build one processed split by combining the corresponding
    Normal and DoS files.

    The original Train/Validation/Test boundary is preserved.
    """

    print("\n" + "=" * 90)
    print(f"BUILDING: {condition} - {split}")
    print("=" * 90)

    normal_root = DATASET_ROOT / normal_scenario / split
    dos_root = DATASET_ROOT / dos_scenario / split

    normal_files = sorted(
        normal_root.rglob("*.json")
    )

    dos_files = sorted(
        dos_root.rglob("*.json")
    )

    print(f"Normal JSON files : {len(normal_files)}")
    print(f"DoS JSON files    : {len(dos_files)}")

    if not normal_files:
        raise FileNotFoundError(
            f"No Normal JSON files found in {normal_root}"
        )

    if not dos_files:
        raise FileNotFoundError(
            f"No DoS JSON files found in {dos_root}"
        )

    all_records = []

    # =========================================================
    # NORMAL FILES
    # =========================================================

    print("\nProcessing Normal files...")

    for index, json_file in enumerate(
        normal_files,
        start=1
    ):

        records = process_json_file(
            json_file,
            scenario_label=0
        )

        all_records.extend(records)

        if index % 50 == 0 or index == len(normal_files):
            print(
                f"  Normal: "
                f"{index}/{len(normal_files)} files"
                f" | records={len(all_records):,}"
            )

    normal_count = len(all_records)

    # =========================================================
    # DOS FILES
    # =========================================================

    print("\nProcessing DoS files...")

    dos_start_count = len(all_records)

    for index, json_file in enumerate(
        dos_files,
        start=1
    ):

        records = process_json_file(
            json_file,
            scenario_label=1
        )

        all_records.extend(records)

        if index % 50 == 0 or index == len(dos_files):
            print(
                f"  DoS: "
                f"{index}/{len(dos_files)} files"
                f" | total records={len(all_records):,}"
            )

    dos_count = len(all_records) - dos_start_count

    # =========================================================
    # CREATE DATAFRAME
    # =========================================================

    dataframe = pd.DataFrame(all_records)

    # =========================================================
    # BASIC SAFETY CHECKS
    # =========================================================

    expected_columns = [
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

    if list(dataframe.columns) != expected_columns:
        raise ValueError(
            "Processed columns do not match the expected schema."
        )

    # Labels must only be 0 or 1
    invalid_labels = set(
        dataframe["label"].dropna().unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}"
        )

    # =========================================================
    # OUTPUT
    # =========================================================

    output_directory = OUTPUT_ROOT / condition

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory /
        f"{split.lower()}.parquet"
    )

    dataframe.to_parquet(
        output_file,
        index=False
    )

    # =========================================================
    # RESULTS
    # =========================================================

    print("\n" + "-" * 90)
    print("SPLIT COMPLETE")
    print("-" * 90)

    print(f"Output file    : {output_file}")
    print(f"Total records  : {len(dataframe):,}")
    print(f"Normal records : {normal_count:,}")
    print(f"DoS records    : {dos_count:,}")

    print("\nFinal label distribution:")

    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nNumber of columns:")
    print(f"  {len(dataframe.columns)}")

    return output_file


if __name__ == "__main__":

    print("=" * 90)
    print("PHASE 4.2 - PROCESSED DATASET BUILDER")
    print("=" * 90)

    print("\nSAFETY TEST:")
    print("Only Highway 2 Train will be processed.")
    print("Validation and Test will NOT be processed.")
    print("Highway 7 will NOT be processed.")
    print("Raw data will NOT be modified.")

    build_split(
        condition="highway_7",
        split="Train",
        normal_scenario="InTAS_highway_7",
        dos_scenario="InTAS_highway_7_dosAttack",
    )

    print("\n" + "=" * 90)
    print("HIGHWAY 2 TRAIN TEST COMPLETE")
    print("=" * 90)