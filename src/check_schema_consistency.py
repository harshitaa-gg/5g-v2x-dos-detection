import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"

SCENARIOS = [
    "InTAS_highway_2",
    "InTAS_highway_2_dosAttack",
    "InTAS_highway_7",
    "InTAS_highway_7_dosAttack",
]

SAMPLE_LIMIT = 100


def value_type(value):
    """Return the Python type name of a value."""
    return type(value).__name__


def inspect_scenario(scenario):
    scenario_root = DATASET_ROOT / scenario
    json_files = list(scenario_root.rglob("*.json"))

    print("\n" + "=" * 90)
    print(f"SCENARIO: {scenario}")
    print("=" * 90)
    print(f"JSON files found: {len(json_files)}")

    if not json_files:
        print("No JSON files found.")
        return

    # Store observed types for each field
    field_types = {}

    files_checked = 0
    records_checked = 0

    for json_file in json_files:
        if files_checked >= SAMPLE_LIMIT:
            break

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for record in data:

                if not isinstance(record, dict):
                    continue

                records_checked += 1

                # Top-level fields
                for field, value in record.items():

                    if field not in field_types:
                        field_types[field] = Counter()

                    field_types[field][value_type(value)] += 1

                    # Nested sender/receiver fields
                    if field in ["sender", "receiver"] and isinstance(value, dict):

                        for nested_field, nested_value in value.items():

                            full_name = f"{field}.{nested_field}"

                            if full_name not in field_types:
                                field_types[full_name] = Counter()

                            field_types[full_name][
                                value_type(nested_value)
                            ] += 1

            files_checked += 1

        except Exception as e:
            print(f"Could not read {json_file}: {e}")

    print(f"\nFiles checked   : {files_checked}")
    print(f"Records checked : {records_checked}")

    print("\nFIELD TYPE SUMMARY")
    print("-" * 90)

    for field in sorted(field_types):

        types = field_types[field]

        type_summary = ", ".join(
            f"{name}={count}"
            for name, count in types.items()
        )

        print(f"{field:<35} {type_summary}")

    # Identify fields with multiple observed types
    inconsistent_fields = {
        field: types
        for field, types in field_types.items()
        if len(types) > 1
    }

    print("\n" + "-" * 90)
    print("TYPE INCONSISTENCIES")
    print("-" * 90)

    if inconsistent_fields:
        print("Fields with multiple Python types found:")

        for field, types in sorted(inconsistent_fields.items()):

            type_summary = ", ".join(
                f"{name}={count}"
                for name, count in types.items()
            )

            print(f"  {field}: {type_summary}")

    else:
        print("No type inconsistencies found.")


print("=" * 90)
print("PHASE 3.5 - SCHEMA CONSISTENCY CHECK")
print("=" * 90)

for scenario in SCENARIOS:
    inspect_scenario(scenario)

print("\n" + "=" * 90)
print("SCHEMA CHECK COMPLETE")
print("=" * 90)