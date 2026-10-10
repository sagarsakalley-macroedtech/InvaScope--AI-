import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "plant_images"
METADATA = SOURCE / "metadata_all_species.csv"
OUTPUT = ROOT / "data" / "processed" / "image_dataset"

SEED = 42
random.seed(SEED)

if not METADATA.exists():
    raise SystemExit(
        f"Metadata not found: {METADATA}\n"
        "Run fetch_all_species.py first."
    )

with METADATA.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Validate files and group them by species.
by_species = defaultdict(list)
seen_photos = set()

for row in rows:
    species = row.get("species", "").strip()
    filename = row.get("filename", "").strip()
    photo_id = row.get("photo_id", "").strip()
    license_code = row.get("license", "").strip().lower()

    if not species or not filename:
        continue

    # Only include licenses permitted by the collection script.
    if license_code not in {"cc0", "cc-by", "cc-by-sa"}:
        continue

    if photo_id and photo_id in seen_photos:
        continue

    image_path = ROOT / filename
    if not image_path.is_file():
        continue

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception:
        print(f"Skipping unreadable image: {filename}")
        continue

    if photo_id:
        seen_photos.add(photo_id)

    by_species[species].append(image_path)

# Clear only the generated image_dataset folder.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ("train", "val", "test"):
    (OUTPUT / split).mkdir(parents=True, exist_ok=True)

summary = Counter()
manifest = []

for species, paths in sorted(by_species.items()):
    # A class needs at least three images for this basic split.
    if len(paths) < 3:
        print(
            f"Skipping {species}: only {len(paths)} usable images. "
            "Collect more before training."
        )
        continue

    random.shuffle(paths)
    n = len(paths)

    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))

    # Keep at least one image for the test set.
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    partitions = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }

    class_name = species.replace(" ", "_")

    for split, images in partitions.items():
        class_dir = OUTPUT / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            destination = class_dir / image_path.name
            shutil.copy2(image_path, destination)
            summary[(species, split)] += 1

            manifest.append({
                "species": species,
                "split": split,
                "source_file": str(image_path.relative_to(ROOT)),
                "prepared_file": str(destination.relative_to(ROOT)),
            })

manifest_path = OUTPUT / "split_manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "species", "split", "source_file", "prepared_file"
        ],
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nDATASET SPLIT SUMMARY")
for species in sorted(by_species):
    counts = [
        summary[(species, split)]
        for split in ("train", "val", "test")
    ]
    if sum(counts):
        print(
            f"{species}: train={counts[0]}, "
            f"val={counts[1]}, test={counts[2]}"
        )

print(f"\nPrepared dataset: {OUTPUT}")
print(f"Split manifest: {manifest_path}")
print("Review class counts and possible related-image leakage before training.")
