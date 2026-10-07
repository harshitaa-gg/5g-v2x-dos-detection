import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"


def parse_position(position_string):
    """
    Convert a position string:

        'x,y,z'

    into three floating-point values.
    """

    if position_string is None:
        return None, None, None

    try:
        x, y, z = position_string.split(",")

        return float(x), float(y), float(z)

    except (ValueError, AttributeError):
        return None, None, None


def parse_record(record, label):
    """
    Convert one raw JSON message into a flat dictionary.

    label:
        - Normal scenario: 0
        - DoS scenario: use record["attacker"]
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
    # Determine final label
    # ---------------------------------------------------------

    if "attacker" in record:
        final_label = int(record["attacker"])
    else:
        final_label = int(label)

    # ---------------------------------------------------------
    # Build flat record
    # ---------------------------------------------------------

    processed = {
        # Identification
        "sender_id": record.get("sender_id"),
        "sender_alias": record.get("sender_alias"),
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
        "receiver_acceleration_noise": float(receiver["acl_noise"]),
        "receiver_heading_noise": float(receiver["hed_noise"]),

        # Driver profile
        "sender_driver_profile": sender.get("driversProfile"),
        "receiver_driver_profile": receiver.get("driversProfile"),

        # Binary target
        "label": final_label,
    }

    # ---------------------------------------------------------
    # Optional DoS field
    # ---------------------------------------------------------

    if "distance_to_road_edge" in sender:
        processed["distance_to_road_edge"] = float(
            sender["distance_to_road_edge"]
        )
    else:
        processed["distance_to_road_edge"] = None

    return processed
    


def process_file(json_file, label):
    """
    Read one JSON file and convert all messages.
    """

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    processed_records = []

    for record in records:
        processed_records.append(
            parse_record(record, label)
        )

    return processed_records


if __name__ == "__main__":

    print("=" * 80)
    print("PHASE 4 - PREPROCESSING MODULE")
    print("=" * 80)

    print("\nPreprocessing functions loaded successfully.")

    print("\nThis module currently provides:")
    print("  - Position parsing")
    print("  - Timestamp conversion")
    print("  - Delay calculation")
    print("  - Nested JSON flattening")
    print("  - Binary label support")

    print("\nNo dataset files have been modified.")