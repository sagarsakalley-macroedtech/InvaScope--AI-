import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import ee

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

PROJECT_ID = "invascope-ai"

CSV_PATH = (
    ROOT / "data" / "raw" / "occurrences"
    / "gbif_occurrences.csv"
)

OUTPUT_DIR = ROOT / "reports" / "satellite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "sentinel2_ndvi_sample.json"

# Analysis period: end date is exclusive.
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

BUFFER_METERS = 500
MAX_CLOUD_PERCENT = 30

# Set to "India" or "United States" to filter the CSV.
# Use None to select the first valid observation.
PREFERRED_COUNTRY = "India"


# --------------------------------------------------
# 2. EARTH ENGINE CONNECTION
# --------------------------------------------------
def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine initialized:", PROJECT_ID)

    except Exception as exc:
        raise SystemExit(
            "\nEarth Engine initialization failed.\n"
            "Check that your account has Earth Engine access "
            "and that this Cloud project is correctly configured.\n"
            "Authenticate with:\n"
            "python -c \"import ee; "
            "ee.Authenticate(auth_mode='notebook')\"\n\n"
            f"Original error: {exc}"
        )


# --------------------------------------------------
# 3. READ A REAL GBIF OBSERVATION
# --------------------------------------------------
def select_observation():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"GBIF CSV not found: {CSV_PATH}\n"
            "Run src\\geospatial\\fetch_gbif_occurrences.py first."
        )

    with CSV_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            country = (row.get("country") or "").strip()

            if PREFERRED_COUNTRY:
                aliases = {
                    "India": {"India", "IN"},
                    "United States": {
                        "United States", "United States of America", "US"
                    },
                }
                allowed = aliases.get(
                    PREFERRED_COUNTRY, {PREFERRED_COUNTRY}
                )
                if country not in allowed:
                    continue

            try:
                lat = float(row["decimalLatitude"])
                lon = float(row["decimalLongitude"])
            except (ValueError, TypeError, KeyError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            return {
                "latitude": lat,
                "longitude": lon,
                "country": country,
                "species": row.get("requested_species", ""),
                "reported_name": row.get("scientificName", ""),
                "gbif_id": row.get("gbifID", ""),
            }

    raise SystemExit(
        f"No valid observation found for {PREFERRED_COUNTRY!r}. "
        "Check your CSV country values or set "
        "PREFERRED_COUNTRY = None."
    )


# --------------------------------------------------
# 4. CLOUD AND CLOUD-SHADOW MASK
# --------------------------------------------------
def mask_clouds(image):
    scl = image.select("SCL")

    # Exclude defective pixels, cloud shadows, clouds,
    # cirrus and snow/ice.
    valid = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(valid)


# --------------------------------------------------
# 5. SENTINEL-2 NDVI ANALYSIS
# --------------------------------------------------
def main():
    observation = select_observation()

    print("\nSelected GBIF observation:")
    print(json.dumps(observation, indent=2))

    initialize_earth_engine()

    point = ee.Geometry.Point([
        observation["longitude"],
        observation["latitude"],
    ])

    region = point.buffer(BUFFER_METERS)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(START_DATE, END_DATE)
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUD_PERCENT,
            )
        )
    )

    scene_count = collection.size().getInfo()

    print("\nMatching Sentinel-2 scenes:", scene_count)

    if scene_count == 0:
        raise SystemExit(
            "No Sentinel-2 scenes found. Try a different date "
            "range or another GBIF observation."
        )

    composite = collection.map(mask_clouds).median()

    red = composite.select("B4")
    nir = composite.select("B8")

    denominator = nir.add(red)

    # Avoid division by zero.
    valid_denominator = denominator.neq(0)

    ndvi = (
        nir.subtract(red)
        .divide(denominator)
        .updateMask(valid_denominator)
        .rename("NDVI")
    )

    statistics = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1000000,
        bestEffort=True,
    ).getInfo()

    mean_ndvi = statistics.get("NDVI")

    if mean_ndvi is not None:
        mean_ndvi = float(mean_ndvi)

    result = {
        "project_id": PROJECT_ID,
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "analysis_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "observation": observation,
        "analysis": {
            "start_date": START_DATE,
            "end_date_exclusive": END_DATE,
            "buffer_meters": BUFFER_METERS,
            "maximum_scene_cloud_percentage": MAX_CLOUD_PERCENT,
            "matching_scenes": scene_count,
            "mean_ndvi": mean_ndvi,
        },
        "interpretation": (
            "NDVI is a vegetation greenness indicator, not a "
            "species identification or proof of invasive plant "
            "presence. Scene-level cloud filtering does not "
            "guarantee that every pixel is cloud-free."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nAnalysis completed.")
    print("Mean NDVI:", mean_ndvi)
    print("Results saved to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
