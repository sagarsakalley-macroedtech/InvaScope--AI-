import csv
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "plant_images"
METADATA = DATA / "metadata_all_species.csv"
REPORT = ROOT / "reports" / "model_metrics" / "dataset_quality_report.csv"

if not METADATA.exists():
    print("Metadata file not found:", METADATA)
    print("Run fetch_all_species.py first.")
    raise SystemExit(1)

with METADATA.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

summary = Counter()
report_rows = []

for row in rows:
    species = row.get("species", "Unknown")
    image_path = ROOT / row.get("filename", "")
    issues = []

    if not image_path.is_file():
        issues.append("missing_file")
    else:
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                if img.width < 64 or img.height < 64:
                    issues.append("image_too_small")
        except Exception:
            issues.append("unreadable_image")

    status = "PASS" if not issues else "REVIEW"
    summary[(species, status)] += 1

    report_rows.append({
        "species": species,
        "filename": row.get("filename", ""),
        "photo_id": row.get("photo_id", ""),
        "license": row.get("license", ""),
        "status": status,
        "issues": ";".join(issues),
    })

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "species", "filename", "photo_id",
            "license", "status", "issues"
        ],
    )
    writer.writeheader()
    writer.writerows(report_rows)

print("\nIN VASCOPE AI - DATASET QUALITY REPORT")
print("Metadata records:", len(rows))

for species in sorted({key[0] for key in summary}):
    passed = summary[(species, "PASS")]
    review = summary[(species, "REVIEW")]
    print(f"{species}: {passed} passed, {review} need review")

print("\nReport saved to:", REPORT)
