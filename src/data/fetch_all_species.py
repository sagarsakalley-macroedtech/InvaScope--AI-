import csv
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "raw" / "plant_images"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "Lantana camara",
    "Parthenium hysterophorus",
    "Prosopis juliflora",
    "Pueraria montana var. lobata",
    "Ailanthus altissima",
]

IMAGES_PER_SPECIES = 20
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
API_URL = "https://api.inaturalist.org/v1/observations"

session = requests.Session()
session.headers.update({
    "User-Agent": "InvaScopeAI-research-prototype/1.0"
})

all_rows = []

for species in SPECIES:
    slug = re.sub(r"[^a-z0-9]+", "_", species.lower()).strip("_")
    species_dir = OUTPUT / slug
    species_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSearching: {species}")

    params = {
        "taxon_name": species,
        "photos": "true",
        "quality_grade": "research",
        "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
        "per_page": 100,
        "page": 1,
    }

    try:
        response = session.get(API_URL, params=params, timeout=40)
        response.raise_for_status()
        observations = response.json().get("results", [])
    except requests.RequestException as exc:
        print(f"Search failed for {species}: {exc}")
        continue

    seen_photo_ids = set()
    saved = 0

    for observation in observations:
        for photo in observation.get("photos", []):
            if saved >= IMAGES_PER_SPECIES:
                break

            photo_id = photo.get("id")
            image_url = photo.get("url")
            license_code = photo.get("license_code")

            if not photo_id or not image_url:
                continue
            if license_code not in ALLOWED_LICENSES:
                continue
            if photo_id in seen_photo_ids:
                continue

            # Keep a consistent filename and reuse files on reruns.
            destination = species_dir / f"inat_{photo_id}.jpg"

            try:
                if not destination.exists():
                    image_response = session.get(
                        image_url, timeout=40
                    )
                    image_response.raise_for_status()

                    content_type = image_response.headers.get(
                        "Content-Type", ""
                    ).lower()

                    if not content_type.startswith("image/"):
                        continue

                    # Do not label non-JPEG content as JPEG.
                    extension = (
                        ".jpg"
                        if "jpeg" in content_type or "jpg" in content_type
                        else ".img"
                    )
                    destination = species_dir / f"inat_{photo_id}{extension}"

                    if not destination.exists():
                        destination.write_bytes(image_response.content)

                geo = observation.get("geojson") or {}
                coordinates = geo.get("coordinates") or []

                all_rows.append({
                    "filename": str(destination.relative_to(ROOT)),
                    "species": species,
                    "observation_id": observation.get("id", ""),
                    "photo_id": photo_id,
                    "license": license_code,
                    "photo_url": image_url,
                    "observation_url": (
                        "https://www.inaturalist.org/observations/"
                        f"{observation.get('id', '')}"
                    ),
                    "observed_on": observation.get("observed_on", ""),
                    "longitude": coordinates[0] if len(coordinates) >= 2 else "",
                    "latitude": coordinates[1] if len(coordinates) >= 2 else "",
                })

                seen_photo_ids.add(photo_id)
                saved += 1
                print(f"  {saved}/{IMAGES_PER_SPECIES} images")
                time.sleep(0.3)

            except requests.RequestException as exc:
                print(f"  Skipping photo {photo_id}: {exc}")

        if saved >= IMAGES_PER_SPECIES:
            break

    print(f"Collected {saved} images for {species}")

columns = [
    "filename", "species", "observation_id", "photo_id",
    "license", "photo_url", "observation_url", "observed_on",
    "longitude", "latitude",
]

metadata_path = OUTPUT / "metadata_all_species.csv"

with metadata_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(all_rows)

print("\nCollection finished.")
print(f"Total metadata records: {len(all_rows)}")
print(f"Master metadata: {metadata_path}")
print("Check image labels and licenses before model training.")
