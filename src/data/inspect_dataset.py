import csv
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "plant_images"
METADATA = DATA_DIR / "metadata.csv"

print("=" * 55)
print("InvaScope AI - Dataset Inspection")
print("=" * 55)

if not METADATA.exists():
    print(f"Metadata not found: {METADATA}")
    print("Run fetch_plant_images.py first.")
    raise SystemExit(1)

with METADATA.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Metadata records: {len(rows)}")

valid = 0
invalid = 0
licenses = Counter()
species = Counter()
seen_files = set()

for row in rows:
    filename = row.get("filename", "")
    path = DATA_DIR / filename

    species[row.get("species", "Unknown")] += 1
    licenses[row.get("license", "Unknown")] += 1

    if not path.exists() or path.name in seen_files:
        invalid += 1
        continue

    seen_files.add(path.name)

    try:
        with Image.open(path) as image:
            image.verify()
        valid += 1
    except Exception:
        invalid += 1

print(f"Valid image files: {valid}")
print(f"Missing, duplicate or unreadable files: {invalid}")

print("\nSpecies counts:")
for name, count in species.items():
    print(f"  {name}: {count}")

print("\nLicense counts:")
for name, count in licenses.items():
    print(f"  {name}: {count}")

print("\nInspection completed.")
