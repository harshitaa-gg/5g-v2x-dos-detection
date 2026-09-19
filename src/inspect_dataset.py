from pathlib import Path
import json
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("data/raw")

SCENARIOS = [
    "InTAS_highway_2",
    "InTAS_highway_2_dosAttack",
    "InTAS_highway_7",
    "InTAS_highway_7_dosAttack",
]

SPLITS = [
    "Train",
    "Validation",
    "Test",
]

SAMPLE_FILES_PER_SPLIT = 1
SAMPLE_RECORDS = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_vehicle_files(folder):
    """Find vehicle JSON files recursively."""
    return sorted(folder.rglob("veh_*.json"))


def value_type(value):
    """Return a readable description of a Python value."""
    if value is None:
        return "None"

    if isinstance(value, dict):
        return "dict"

    if isinstance(value, list):
        return "list"

    return type(value).__name__


def inspect_record(record, prefix=""):
    """Print fields recursively."""
    if isinstance(record, dict):
        for key, value in record.items():
            field_name = f"{prefix}.{key}" if prefix else key

            print(
                f"    {field_name:<35} "
                f"type={value_type(value):<10} "
                f"sample={str(value)[:120]}"
            )

            if isinstance(value, dict):
                inspect_record(value, field_name)


def inspect_file(file_path):
    """Inspect one vehicle JSON file."""

    print("\n" + "=" * 90)
    print(f"FILE: {file_path}")
    print(f"SIZE: {file_path.stat().st_size / 1024:.2f} KB")
    print("=" * 90)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    print(f"Top-level Python type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"Number of records in this file: {len(data)}")
        records = data[:SAMPLE_RECORDS]

    elif isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        records = [data]

    else:
        print("Unexpected JSON structure.")
        return

    print("\n--- SAMPLE RECORD STRUCTURE ---")

    for index, record in enumerate(records):
        print(f"\nRecord {index + 1}:")

        if isinstance(record, dict):
            inspect_record(record)
        else:
            print(f"    {record}")


# ============================================================
# MAIN INSPECTION
# ============================================================

def main():

    print("=" * 90)
    print("VeReMi NextGen - RAW DATASET INSPECTION")
    print("=" * 90)

    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset folder does not exist:")
        print(DATASET_ROOT)
        return

    for scenario in SCENARIOS:

        print("\n\n")
        print("#" * 90)
        print(f"SCENARIO: {scenario}")
        print("#" * 90)

        scenario_path = DATASET_ROOT / scenario

        if not scenario_path.exists():
            print(f"WARNING: Scenario folder not found: {scenario_path}")
            continue

        for split in SPLITS:

            split_path = scenario_path / split

            print("\n" + "-" * 90)
            print(f"SPLIT: {split}")
            print(f"PATH:  {split_path}")
            print("-" * 90)

            if not split_path.exists():
                print("WARNING: Split folder not found.")
                continue

            vehicle_files = find_vehicle_files(split_path)

            print(f"Vehicle JSON files found: {len(vehicle_files)}")

            if not vehicle_files:
                print("No veh_*.json files found.")
                continue

            # Show a few filenames
            print("\nFirst few vehicle files:")

            for file_path in vehicle_files[:5]:
                size_mb = file_path.stat().st_size / (1024 * 1024)

                print(
                    f"  {file_path.name:<30} "
                    f"{size_mb:.3f} MB"
                )

            # Inspect one file
            for file_path in vehicle_files[:SAMPLE_FILES_PER_SPLIT]:
                inspect_file(file_path)

    print("\n")
    print("=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()