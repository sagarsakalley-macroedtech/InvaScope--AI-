import csv
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "plant_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "Lantana camara"
MAX_IMAGES = 30
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}

API_URL = "https://api.inaturalist.org/v1/observations"
session = requests.Session()
session.headers.update({
    "User-Agent": "InvaScopeAI-research-prototype/1.0"
})

params = {
    "taxon_name": SPECIES,
    "photos": "true",
    "quality_grade": "research",
    "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
    "per_page": min(MAX_IMAGES, 200),
    "page": 1,
}

print(f"Searching for {SPECIES}...")
response = session.get(API_URL, params=params, timeout=30)
response.raise_for_status()
observations = response.json().get("results", [])

rows = []
seen_photos = set()

for observation in observations:
    for photo in observation.get("photos", []):
        if len(rows) >= MAX_IMAGES:
            break

        photo_id = photo.get("id")
        image_url = photo.get("url")
        license_code = photo.get("license_code")

        if not photo_id or not image_url:
            continue
        if license_code not in ALLOWED_LICENSES:
            continue
        if photo_id in seen_photos:
            continue

        try:
            image_response = session.get(image_url, timeout=30)
            image_response.raise_for_status()

            content_type = image_response.headers.get(
                "Content-Type", ""
            ).lower()

            if not content_type.startswith("image/"):
                continue

            extension = (
                ".jpg"
                if "jpeg" in content_type or "jpg" in content_type
                else ".img"
            )
            filename = f"inat_{photo_id}{extension}"
            (OUTPUT_DIR / filename).write_bytes(
                image_response.content
            )

            coordinates = observation.get("geojson", {}).get(
                "coordinates"
            )
            longitude = coordinates[0] if coordinates else ""
            latitude = coordinates[1] if coordinates else ""

            rows.append({
                "filename": filename,
                "species": SPECIES,
                "observation_id": observation.get("id", ""),
                "photo_id": photo_id,
                "photo_url": image_url,
                "license": license_code,
                "observation_url": (
                    "https://www.inaturalist.org/observations/"
                    f"{observation.get('id', '')}"
                ),
                "observed_on": observation.get("observed_on", ""),
                "latitude": latitude,
                "longitude": longitude,
            })

            seen_photos.add(photo_id)
            print(f"Downloaded {len(rows)}: {filename}")
            time.sleep(0.3)

        except requests.RequestException as exc:
            print(f"Skipped photo {photo_id}: {exc}")

    if len(rows) >= MAX_IMAGES:
        break

metadata_path = OUTPUT_DIR / "metadata.csv"
fields = [
    "filename", "species", "observation_id", "photo_id",
    "photo_url", "license", "observation_url", "observed_on",
    "latitude", "longitude",
]

with metadata_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCompleted. Images downloaded: {len(rows)}")
print(f"Images and metadata saved in: {OUTPUT_DIR}")
