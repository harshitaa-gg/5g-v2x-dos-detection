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

SPLITS = ["Train", "Validation", "Test"]


def analyze_folder(folder):
    json_files = list(folder.rglob("*.json"))

    total_records = 0
    attacker_values = Counter()
    attacker_found_records = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for record in data:
                total_records += 1

                if "attacker" in record:
                    attacker_found_records += 1
                    attacker_values[record["attacker"]] += 1

        except Exception as e:
            print(f"Could not read {json_file}: {e}")

    return (
        len(json_files),
        total_records,
        attacker_found_records,
        attacker_values,
    )


print("=" * 90)
print("PHASE 3 - LABEL STATISTICS")
print("=" * 90)


for scenario in SCENARIOS:

    print("\n" + "=" * 90)
    print(f"SCENARIO: {scenario}")
    print("=" * 90)

    scenario_root = DATASET_ROOT / scenario

    if not scenario_root.exists():
        print("Scenario folder not found.")
        continue

    scenario_files = 0
    scenario_records = 0
    scenario_attacker_records = 0
    scenario_attacker_values = Counter()

    for split in SPLITS:

        split_folder = scenario_root / split

        if not split_folder.exists():
            print(f"\n{split}: folder not found")
            continue

        (
            num_files,
            num_records,
            attacker_records,
            attacker_values,
        ) = analyze_folder(split_folder)

        scenario_files += num_files
        scenario_records += num_records
        scenario_attacker_records += attacker_records
        scenario_attacker_values.update(attacker_values)

        print(f"\n{split}")
        print("-" * 50)
        print(f"JSON files       : {num_files}")
        print(f"Total records    : {num_records}")
        print(f"Attacker records : {attacker_records}")

        if attacker_values:
            print("Attacker values:")
            for value, count in sorted(attacker_values.items()):
                print(f"  {value}: {count}")

    print("\nSCENARIO TOTAL")
    print("-" * 50)
    print(f"JSON files       : {scenario_files}")
    print(f"Total records    : {scenario_records}")
    print(f"Attacker records : {scenario_attacker_records}")

    if scenario_attacker_values:
        print("Attacker values:")
        for value, count in sorted(scenario_attacker_values.items()):
            print(f"  {value}: {count}")


print("\n" + "=" * 90)
print("PHASE 3 COMPLETE")
print("=" * 90)
