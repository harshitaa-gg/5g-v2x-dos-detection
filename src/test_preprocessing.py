import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow importing from src/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import process_file


DATASET_ROOT = PROJECT_ROOT / "data" / "raw"


NORMAL_FILE = next(
    (DATASET_ROOT / "InTAS_highway_2").rglob("*.json")
)

DOS_FILE = next(
    (DATASET_ROOT / "InTAS_highway_2_dosAttack").rglob("*.json")
)


print("=" * 90)
print("PHASE 4.1 - PREPROCESSING TEST")
print("=" * 90)


# ============================================================
# NORMAL FILE
# ============================================================

print("\n" + "=" * 90)
print("TEST 1: NORMAL FILE")
print("=" * 90)

print(f"File: {NORMAL_FILE}")

normal_records = process_file(
    NORMAL_FILE,
    label=0
)

print(f"\nRecords processed: {len(normal_records)}")

print("\nFirst processed record:")
print("-" * 90)

for key, value in normal_records[0].items():
    print(f"{key:<35}: {value}")


# ============================================================
# DOS FILE
# ============================================================

print("\n" + "=" * 90)
print("TEST 2: DOS FILE")
print("=" * 90)

print(f"File: {DOS_FILE}")

dos_records = process_file(
    DOS_FILE,
    label=1
)

print(f"\nRecords processed: {len(dos_records)}")

# Check actual DoS labels
dos_labels = {}

for record in dos_records:
    label = record["label"]
    dos_labels[label] = dos_labels.get(label, 0) + 1

print("\nDoS label distribution:")

for label, count in sorted(dos_labels.items()):
    print(f"  label {label}: {count}")

print("\nFirst processed record:")
print("-" * 90)

for key, value in dos_records[0].items():
    print(f"{key:<35}: {value}")


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 90)
print("PREPROCESSING TEST COMPLETE")
print("=" * 90)

print("\nNormal label:", normal_records[0]["label"])
print("DoS label   :", dos_records[0]["label"])

print("\nNormal record fields:", len(normal_records[0]))
print("DoS record fields   :", len(dos_records[0]))

print("\nNo raw files were modified.")