from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"

PAIRS = {
    "highway_2": (
        "InTAS_highway_2",
        "InTAS_highway_2_dosAttack",
    ),
    "highway_7": (
        "InTAS_highway_7",
        "InTAS_highway_7_dosAttack",
    ),
}

SPLITS = ["Train", "Validation", "Test"]


print("=" * 90)
print("SPLIT VERIFICATION")
print("=" * 90)

all_ok = True

for condition, (normal_name, dos_name) in PAIRS.items():

    print("\n" + "=" * 90)
    print(f"CONDITION: {condition}")
    print("=" * 90)

    normal_root = DATASET_ROOT / normal_name
    dos_root = DATASET_ROOT / dos_name

    for split in SPLITS:

        normal_split = normal_root / split
        dos_split = dos_root / split

        normal_files = list(normal_split.rglob("*.json"))
        dos_files = list(dos_split.rglob("*.json"))

        print(f"\n{split}")
        print("-" * 60)

        print(f"Normal folder exists : {normal_split.exists()}")
        print(f"DoS folder exists    : {dos_split.exists()}")

        print(f"Normal JSON files    : {len(normal_files)}")
        print(f"DoS JSON files       : {len(dos_files)}")

        if not normal_split.exists() or not dos_split.exists():
            all_ok = False

        if len(normal_files) == 0 or len(dos_files) == 0:
            all_ok = False


print("\n" + "=" * 90)
print("VERIFICATION RESULT")
print("=" * 90)

if all_ok:
    print("All expected Train / Validation / Test splits are present.")
    print("No files were modified.")
else:
    print("WARNING: One or more expected splits are missing.")

print("=" * 90)