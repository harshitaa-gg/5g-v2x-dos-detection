
import json
from pathlib import Path

folder = (
    Path("data/raw")
    / "InTAS_highway_7_dosAttack"
    / "Test"
)

found = {0: None, 1: None}

for path in folder.rglob("*.json"):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = data if isinstance(data, list) else [data]

    for record in records:
        if not isinstance(record, dict):
            continue

        label = record.get("attacker")

        if label in (0, 1) and found[label] is None:
            found[label] = (path, record)

    if all(value is not None for value in found.values()):
        break

for label in (0, 1):
    print(f"\n{'=' * 50}")
    print(f"attacker = {label}")

    result = found[label]

    if result is None:
        print("No example found.")
        continue

    path, record = result
    print("File:", path)
    print("Sender ID:", record.get("sender_id"))
    print("Sender profile:", record.get("sender", {}).get("driversProfile"))
    print("Message ID:", record.get("messageID"))
    print("Attacker field:", record.get("attacker"))
