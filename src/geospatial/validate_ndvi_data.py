
import csv
import json
from pathlib import Path
from collections import Counter

INPUT_CSV = Path("data/processed/gbif_occurrences_ndvi.csv")
OUTPUT_JSON = Path("reports/satellite/ndvi_data_quality.json")


def main():
    if not INPUT_CSV.exists():
        print(f"Missing file: {INPUT_CSV}")
        print("Run the NDVI extraction step first.")
        return

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    statuses = Counter(row.get("status", "") for row in rows)
    ndvi_values = []

    for row in rows:
        try:
            value = float(row.get("ndvi", ""))
            if -1 <= value <= 1:
                ndvi_values.append(value)
        except (TypeError, ValueError):
            pass

    countries = Counter(
        row.get("country", "").strip() or "Unknown"
        for row in rows
    )

    species = Counter(
        row.get("species", "").strip() or "Unknown"
        for row in rows
    )

    report = {
        "input_file": str(INPUT_CSV),
        "total_records": len(rows),
        "records_with_valid_ndvi": len(ndvi_values),
        "records_without_valid_ndvi": len(rows) - len(ndvi_values),
        "ndvi_coverage_percent": round(
            100 * len(ndvi_values) / len(rows), 2
        ) if rows else 0,
        "mean_ndvi": round(sum(ndvi_values) / len(ndvi_values), 4)
        if ndvi_values else None,
        "minimum_ndvi": round(min(ndvi_values), 4)
        if ndvi_values else None,
        "maximum_ndvi": round(max(ndvi_values), 4)
        if ndvi_values else None,
        "status_counts": dict(statuses),
        "country_counts": dict(countries),
        "species_counts": dict(species),
        "quality_note": (
            "NDVI indicates vegetation greenness, not species identity. "
            "Review coordinate uncertainty, observation dates, cloud "
            "masking and missing values before scientific conclusions."
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
