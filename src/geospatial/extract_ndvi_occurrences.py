
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import ee


# ============================================================
# InvaScope AI — Multi-Location Sentinel-2 NDVI Extraction
# ============================================================

PROJECT_ID = "invascope-ai"

INPUT_CSV = Path("data/raw/occurrences/gbif_occurrences.csv")
OUTPUT_CSV = Path("data/processed/gbif_occurrences_ndvi.csv")

START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

BUFFER_METERS = 500
MAX_RECORDS = 200
PER_GROUP_LIMIT = 10
RANDOM_SEED = 42
CLOUD_PERCENT_LIMIT = 30


def find_column(fieldnames, candidates):
    """Find a CSV column without depending on capitalization."""
    lookup = {
        name.lower().strip(): name
        for name in fieldnames
        if name
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_occurrences():
    """Load valid coordinate records from the GBIF CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV.resolve()}"
        )

    with INPUT_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("The input CSV has no column headers.")

        lat_col = find_column(
            reader.fieldnames,
            ["decimalLatitude", "latitude", "lat"],
        )

        lon_col = find_column(
            reader.fieldnames,
            ["decimalLongitude", "longitude", "lon", "lng"],
        )

        species_col = find_column(
            reader.fieldnames,
            [
                "acceptedScientificName",
                "scientificName",
                "species",
                "requested_species",
            ],
        )

        country_col = find_column(
            reader.fieldnames,
            ["country", "countryCode"],
        )

        if not lat_col or not lon_col:
            raise ValueError(
                "Could not identify coordinate columns. "
                f"Available columns: {reader.fieldnames}"
            )

        all_records = []

        for row in reader:
            try:
                lat = float(row.get(lat_col, ""))
                lon = float(row.get(lon_col, ""))
            except (TypeError, ValueError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            species = (
                row.get(species_col, "").strip()
                if species_col else ""
            )

            country = (
                row.get(country_col, "").strip()
                if country_col else ""
            )

            all_records.append({
                "latitude": lat,
                "longitude": lon,
                "species": species or "Unknown",
                "country": country or "Unknown",
            })

    return all_records


def balanced_sample(all_records):
    """Sample records within country/species groups."""
    groups = defaultdict(list)

    for record in all_records:
        key = (record["country"], record["species"])
        groups[key].append(record)

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in sorted(groups):
        group = groups[key].copy()
        rng.shuffle(group)
        selected.extend(group[:PER_GROUP_LIMIT])

    # Shuffle across groups so the overall limit is not dominated
    # by alphabetically early country/species groups.
    rng.shuffle(selected)
    selected = selected[:MAX_RECORDS]

    print(f"Valid source locations: {len(all_records)}")
    print(f"Country/species groups: {len(groups)}")
    print(f"Selected for NDVI: {len(selected)}")

    print("\nAvailable groups:")
    for country, species in sorted(groups):
        print(f"  {country} | {species}: {len(groups[(country, species)])}")

    print("\nSelected records by group:")
    selected_counts = defaultdict(int)
    for record in selected:
        selected_counts[(record["country"], record["species"])] += 1

    for (country, species), count in sorted(selected_counts.items()):
        print(f"  {country} | {species}: {count}")

    return selected


def mask_clouds(image):
    """Mask cloud and shadow classes using Sentinel-2 SCL."""
    scl = image.select("SCL")

    clear_pixels = (
        scl.neq(3)       # Cloud shadow
        .And(scl.neq(8))  # Medium-probability cloud
        .And(scl.neq(9))  # High-probability cloud
        .And(scl.neq(10)) # Cirrus
        .And(scl.neq(11)) # Snow or ice
    )

    return image.updateMask(clear_pixels)


def extract_ndvi(records):
    """Extract median-composite NDVI for each occurrence location."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lte(
            "CLOUDY_PIXEL_PERCENTAGE",
            CLOUD_PERCENT_LIMIT,
        ))
        .select(["B4", "B8", "SCL"])
        .map(mask_clouds)
    )

    scene_count = collection.size().getInfo()
    print(f"\nMatching Sentinel-2 scenes in date range: {scene_count}")

    if scene_count == 0:
        raise RuntimeError(
            "No Sentinel-2 scenes match the date and cloud filters."
        )

    composite = collection.median()

    ndvi_image = (
        composite.normalizedDifference(["B8", "B4"])
        .rename("NDVI")
    )

    results = []

    for index, record in enumerate(records, start=1):
        result_record = {
            **record,
            "ndvi": "",
            "status": "",
        }

        try:
            point = ee.Geometry.Point([
                record["longitude"],
                record["latitude"],
            ])

            region = point.buffer(BUFFER_METERS)

            result = ndvi_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10,
                maxPixels=1000000,
                bestEffort=True,
            ).getInfo()

            value = result.get("NDVI") if result else None

            if value is not None:
                result_record["ndvi"] = round(float(value), 6)
                result_record["status"] = "ok"
            else:
                result_record["status"] = "no_valid_pixels"

        except Exception as error:
            result_record["status"] = (
                f"error: {str(error)[:200]}"
            )

        results.append(result_record)
        print(
            f"Processed {index}/{len(records)}: "
            f"{record['species']} | {record['country']} | "
            f"{result_record['status']}"
        )

        # Keep requests spaced out.
        time.sleep(0.2)

    return results


def save_results(results):
    """Save extracted NDVI measurements to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "latitude",
        "longitude",
        "species",
        "country",
        "ndvi",
        "status",
    ]

    with OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV.resolve()}")

    valid_count = sum(
        row["status"] == "ok" and row["ndvi"] != ""
        for row in results
    )

    print(f"Records with NDVI values: {valid_count}/{len(results)}")


def main():
    print("InvaScope AI — Sentinel-2 NDVI extraction")
    print(f"Project: {PROJECT_ID}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    # Uses the Earth Engine authentication already saved locally.
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized successfully.")

    all_records = load_occurrences()

    if not all_records:
        print("No valid coordinate records were found.")
        return

    selected_records = balanced_sample(all_records)

    if not selected_records:
        print("No records were selected for processing.")
        return

    results = extract_ndvi(selected_records)
    save_results(results)


if __name__ == "__main__":
    main()
