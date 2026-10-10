
import csv
from pathlib import Path
import html

INPUT_CSV = Path("data/processed/gbif_occurrences_ndvi.csv")
OUTPUT_HTML = Path("reports/invascope_ndvi_map.html")


def main():
    if not INPUT_CSV.exists():
        print(f"File not found: {INPUT_CSV}")
        print("Run the NDVI extraction script first.")
        return

    points = []

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (ValueError, KeyError, TypeError):
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            try:
                ndvi = float(row["ndvi"])
            except (ValueError, TypeError):
                ndvi = None

            points.append({
                "lat": lat,
                "lon": lon,
                "species": row.get("species", "Unknown"),
                "country": row.get("country", "Unknown"),
                "ndvi": ndvi,
                "status": row.get("status", ""),
            })

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    # Use an SVG coordinate plot so the map works offline.
    width, height = 1000, 520
    margin = 45

    valid_ndvi = [p["ndvi"] for p in points if p["ndvi"] is not None]

    def project(lon, lat):
        x = margin + (lon + 180) / 360 * (width - 2 * margin)
        y = margin + (90 - lat) / 180 * (height - 2 * margin)
        return x, y

    circles = []
    for p in points:
        x, y = project(p["lon"], p["lat"])
        if p["ndvi"] is None:
            fill = "#777777"
        elif p["ndvi"] < 0.2:
            fill = "#b44b3b"
        elif p["ndvi"] < 0.5:
            fill = "#e2b93b"
        else:
            fill = "#32864a"

        title = html.escape(
            f"Species: {p['species']} | Country: {p['country']} | "
            f"NDVI: {p['ndvi']} | Status: {p['status']}"
        )
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
            f'fill="{fill}" stroke="#222" stroke-width="0.7">'
            f'<title>{title}</title></circle>'
        )

    # Simple longitude/latitude grid.
    grid = []
    for lon in range(-180, 181, 30):
        x, _ = project(lon, 0)
        grid.append(
            f'<line x1="{x:.1f}" y1="{margin}" '
            f'x2="{x:.1f}" y2="{height-margin}" '
            f'stroke="#dddddd" stroke-width="0.7"/>'
        )
    for lat in range(-60, 91, 30):
        _, y = project(0, lat)
        grid.append(
            f'<line x1="{margin}" y1="{y:.1f}" '
            f'x2="{width-margin}" y2="{y:.1f}" '
            f'stroke="#dddddd" stroke-width="0.7"/>'
        )

    average = sum(valid_ndvi) / len(valid_ndvi) if valid_ndvi else None

    summary = (
        f"Locations plotted: {len(points)} | "
        f"Valid NDVI values: {len(valid_ndvi)} | "
        f"Mean NDVI: {average:.3f}" if average is not None
        else f"Locations plotted: {len(points)} | Valid NDVI values: 0"
    )

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>InvaScope AI — NDVI Occurrence Map</title>
<style>
body {{font-family:Arial,sans-serif;margin:20px;background:#f7f8fa;color:#222}}
main {{max-width:1100px;margin:auto;background:white;padding:20px;border-radius:12px}}
svg {{width:100%;height:auto;border:1px solid #ddd;background:#f4f8fb}}
.note {{color:#555;font-size:14px}}
</style>
</head>
<body><main>
<h1>InvaScope AI — NDVI Occurrence Map</h1>
<p>{html.escape(summary)}</p>
<svg viewBox="0 0 {width} {height}" role="img"
aria-label="Occurrence points plotted by longitude and latitude">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f4f8fb"/>
{''.join(grid)}
{''.join(circles)}
</svg>
<p>
Legend:
<span style="color:#b44b3b">● NDVI &lt; 0.2</span> &nbsp;
<span style="color:#e2b93b">● NDVI 0.2–0.5</span> &nbsp;
<span style="color:#32864a">● NDVI &gt; 0.5</span> &nbsp;
<span style="color:#777">● Missing NDVI</span>
</p>
<p class="note">
This is a longitude/latitude plot, not a satellite basemap.
NDVI represents vegetation greenness, not invasive-species identity.
Hover over a point to see its record.
</p>
</main></body></html>"""

    OUTPUT_HTML.write_text(document, encoding="utf-8")
    print(f"Map saved to: {OUTPUT_HTML.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
