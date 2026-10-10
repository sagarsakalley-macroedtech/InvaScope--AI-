import csv
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "raw" / "occurrences" / "gbif_occurrences.csv"

SPECIES = [
    "Lantana camara",
    "Parthenium hysterophorus",
    "Prosopis juliflora",
    "Pueraria montana var. lobata",
    "Ailanthus altissima",
]

COUNTRIES = {
    "IN": "India",
    "US": "United States",
}

API_URL = "https://api.gbif.org/v1/occurrence/search"
LIMIT = 100

FIELDS = [
    "gbifID",
    "requested_species",
    "scientificName",
    "acceptedScientificName",
    "country",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "basisOfRecord",
    "coordinateUncertaintyInMeters",
    "datasetKey",
]

session = requests.Session()
session.headers.update({
    "User-Agent": "InvaScopeAI-research-prototype/1.0"
})

all_records = []
seen_ids = set()

for species in SPECIES:
    for country_code, country_name in COUNTRIES.items():
        print(f"\nSearching {species} in {country_name}...")

        params = {
            "scientificName": species,
            "country": country_code,
            "hasCoordinate": "true",
            "limit": LIMIT,
            "offset": 0,
        }

        try:
            response = session.get(
                API_URL, params=params, timeout=60
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"Request failed: {exc}")
            continue

        results = payload.get("results", [])
        print(f"API returned {len(results)} records in this sample.")

        for record in results:
            gbif_id = str(record.get("gbifID", ""))

            if not gbif_id or gbif_id in seen_ids:
                continue

            lat = record.get("decimalLatitude")
            lon = record.get("decimalLongitude")

            if lat is None or lon is None:
                continue

            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            # Preserve the original reported identification.
            # Do not assume every record is taxonomically verified.
            all_records.append({
                "gbifID": gbif_id,
                "requested_species": species,
                "scientificName": record.get("scientificName", ""),
                "acceptedScientificName": record.get(
                    "acceptedScientificName", ""
                ),
                "country": record.get("country", country_name),
                "decimalLatitude": lat,
                "decimalLongitude": lon,
                "eventDate": record.get("eventDate", ""),
                "basisOfRecord": record.get("basisOfRecord", ""),
                "coordinateUncertaintyInMeters": record.get(
                    "coordinateUncertaintyInMeters", ""
                ),
                "datasetKey": record.get("datasetKey", ""),
            })
            seen_ids.add(gbif_id)

        time.sleep(0.3)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(all_records)

print("\nGBIF collection finished.")
print("Total unique records:", len(all_records))
print("Output file:", OUTPUT)
print(
    "Important: this is a limited sample, not an exhaustive "
    "distribution dataset."
)
