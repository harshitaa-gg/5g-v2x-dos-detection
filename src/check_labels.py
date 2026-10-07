"""
Check whether the actual VeReMi NextGen JSON messages
contain an attacker/label field.

This script only reads the dataset.
It does not modify anything.
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"


def inspect_file(file_path):

    print("\n" + "=" * 90)
    print(f"FILE: {file_path}")
    print("=" * 90)

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            records = json.load(f)

    except Exception as error:

        print(f"ERROR: {error}")
        return

    print(f"Records: {len(records)}")

    if not records:
        return

    # Collect all fields appearing in records.
    all_keys = set()

    for record in records:

        if isinstance(record, dict):
            all_keys.update(record.keys())

    print("\nTop-level fields found:")
    for key in sorted(all_keys):
        print(f"  {key}")

    # Check specifically for attacker.
    print("\n" + "-" * 90)
    print("ATTACKER FIELD CHECK")
    print("-" * 90)

    attacker_values = []

    for record in records:

        if isinstance(record, dict):

            if "attacker" in record:

                attacker_values.append(
                    record["attacker"]
                )

    if attacker_values:

        print(
            f"attacker field FOUND "
            f"in {len(attacker_values)} records."
        )

        unique_values = sorted(
            set(
                str(value)
                for value in attacker_values
            )
        )

        print("Unique attacker values:")

        for value in unique_values:
            print(f"  {value}")

    else:

        print(
            "No 'attacker' field found "
            "in this file."
        )

    # Print one record containing attacker, if available.
    for record in records:

        if (
            isinstance(record, dict)
            and "attacker" in record
        ):

            print("\nExample record containing attacker:")
            print(
                json.dumps(
                    record,
                    indent=2
                )[:5000]
            )

            break


def main():

    print("=" * 90)
    print("PHASE 2.5 - LABEL FIELD INSPECTION")
    print("=" * 90)

    # Find one normal scenario file.
    normal_files = sorted(
        (
            RAW_DATA / "InTAS_highway_2"
        ).rglob("*.json")
    )

    # Find one DoS scenario file.
    dos_files = sorted(
        (
            RAW_DATA / "InTAS_highway_2_dosAttack"
        ).rglob("*.json")
    )

    print(
        f"\nNormal JSON files found: "
        f"{len(normal_files)}"
    )

    print(
        f"DoS JSON files found: "
        f"{len(dos_files)}"
    )

    # Inspect one normal file.
    if normal_files:

        print("\n\n>>> NORMAL SCENARIO <<<")

        inspect_file(
            normal_files[0]
        )

    # Inspect one DoS file.
    if dos_files:

        print("\n\n>>> DOS SCENARIO <<<")

        inspect_file(
            dos_files[0]
        )

    print("\n")
    print("=" * 90)
    print("LABEL INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()