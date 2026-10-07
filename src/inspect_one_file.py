"""
Inspect one actual VeReMi NextGen vehicle/message JSON file.

This script only reads the data.
It does not modify anything.
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"


def main():

    # Find the first JSON file anywhere inside data/raw
    json_files = sorted(RAW_DATA.rglob("*.json"))

    print("=" * 90)
    print("INSPECTING ONE RAW JSON FILE")
    print("=" * 90)

    print(f"\nJSON files found: {len(json_files)}")

    if not json_files:
        print("ERROR: No JSON files found.")
        return

    file_path = json_files[0]

    print(f"\nSelected file:")
    print(file_path)

    print(
        f"\nFile size: "
        f"{file_path.stat().st_size / 1024:.2f} KB"
    )

    print("\nLoading file...")

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as error:

        print("ERROR loading JSON:")
        print(error)
        return

    print("\n" + "-" * 90)
    print("TOP-LEVEL INFORMATION")
    print("-" * 90)

    print(f"Python type: {type(data).__name__}")

    if isinstance(data, list):

        print(f"Number of records: {len(data)}")

        if not data:
            print("List is empty.")
            return

        sample = data[0]

    elif isinstance(data, dict):

        print(
            f"Top-level keys: "
            f"{list(data.keys())}"
        )

        sample = data

    else:

        print("Unexpected JSON format.")
        return

    print("\n" + "-" * 90)
    print("FIRST RECORD")
    print("-" * 90)

    print(
        json.dumps(
            sample,
            indent=2
        )[:10000]
    )

    print("\n" + "-" * 90)
    print("TOP-LEVEL FIELD INFORMATION")
    print("-" * 90)

    if isinstance(sample, dict):

        for key, value in sample.items():

            print(
                f"{key:<30} "
                f"type={type(value).__name__:<12} "
                f"value={str(value)[:150]}"
            )

    print("\n" + "=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()