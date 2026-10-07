"""
Phase 2 - Raw Dataset Inspection

Purpose:
    Inspect the actual VeReMi NextGen files stored under data/raw/.

This script does NOT modify the raw dataset.
It only reads files and reports their structure.
"""

from pathlib import Path
import json


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"

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
# HELPERS
# ============================================================

def find_vehicle_files(folder: Path):
    """Find vehicle JSON files recursively."""
    return sorted(folder.rglob("veh_*.json"))


def describe_value(value):
    """Return a short description of a JSON value."""

    if value is None:
        return "None"

    if isinstance(value, dict):
        return f"dict ({len(value)} keys)"

    if isinstance(value, list):
        return f"list ({len(value)} items)"

    return type(value).__name__


def print_structure(value, prefix="", depth=0, max_depth=4):
    """
    Recursively print the structure of a JSON object.
    """

    if depth > max_depth:
        return

    if isinstance(value, dict):

        for key, child in value.items():

            field_name = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            if isinstance(child, dict):

                print(
                    f"    {field_name:<40} "
                    f"dict ({len(child)} keys)"
                )

                print_structure(
                    child,
                    field_name,
                    depth + 1,
                    max_depth,
                )

            elif isinstance(child, list):

                print(
                    f"    {field_name:<40} "
                    f"list ({len(child)} items)"
                )

                if child:

                    print_structure(
                        child[0],
                        f"{field_name}[0]",
                        depth + 1,
                        max_depth,
                    )

            else:

                sample = str(child)

                if len(sample) > 120:
                    sample = sample[:120] + "..."

                print(
                    f"    {field_name:<40} "
                    f"{type(child).__name__:<12} "
                    f"sample={sample}"
                )

    elif isinstance(value, list):

        for index, child in enumerate(value[:3]):

            print_structure(
                child,
                f"{prefix}[{index}]",
                depth + 1,
                max_depth,
            )


def inspect_file(file_path: Path):
    """Inspect one JSON file."""

    print("\n" + "=" * 100)
    print(f"FILE: {file_path}")
    print(
        f"SIZE: "
        f"{file_path.stat().st_size / (1024 * 1024):.3f} MB"
    )
    print("=" * 100)

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except Exception as error:

        print(f"ERROR: Could not read file.")
        print(f"Reason: {error}")
        return

    print(
        f"Top-level type: "
        f"{type(data).__name__}"
    )

    if isinstance(data, list):

        print(
            f"Number of records: "
            f"{len(data)}"
        )

        records = data[:SAMPLE_RECORDS]

    elif isinstance(data, dict):

        print(
            f"Top-level keys: "
            f"{list(data.keys())}"
        )

        records = [data]

    else:

        print("Unexpected JSON structure.")
        return

    print("\n--- SAMPLE RECORD STRUCTURE ---")

    for index, record in enumerate(records):

        print(f"\nRecord {index + 1}")

        if isinstance(record, dict):

            print_structure(record)

        else:

            print(
                f"    Value: "
                f"{str(record)[:500]}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("PHASE 2 - VeReMi NextGen RAW DATASET INSPECTION")
    print("=" * 100)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nDataset root:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():

        print(
            "\nERROR: data/raw/ does not exist."
        )

        return

    for scenario in SCENARIOS:

        scenario_path = DATASET_ROOT / scenario

        print("\n\n")
        print("#" * 100)
        print(f"SCENARIO: {scenario}")
        print("#" * 100)

        if not scenario_path.exists():

            print(
                f"WARNING: Scenario not found:"
                f"\n{scenario_path}"
            )

            continue

        for split in SPLITS:

            split_path = scenario_path / split

            print("\n" + "-" * 100)
            print(f"SPLIT: {split}")
            print(f"PATH:  {split_path}")
            print("-" * 100)

            if not split_path.exists():

                print("WARNING: Split folder not found.")
                continue

            vehicle_files = find_vehicle_files(
                split_path
            )

            print(
                f"Vehicle JSON files found: "
                f"{len(vehicle_files)}"
            )

            if not vehicle_files:

                print("No veh_*.json files found.")
                continue

            print("\nFirst five vehicle files:")

            for file_path in vehicle_files[:5]:

                size_mb = (
                    file_path.stat().st_size
                    / (1024 * 1024)
                )

                print(
                    f"  {file_path.name:<35}"
                    f"{size_mb:>10.3f} MB"
                )

            # Inspect only a small sample.
            for file_path in vehicle_files[
                :SAMPLE_FILES_PER_SPLIT
            ]:

                inspect_file(file_path)

    print("\n")
    print("=" * 100)
    print("PHASE 2 INSPECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()