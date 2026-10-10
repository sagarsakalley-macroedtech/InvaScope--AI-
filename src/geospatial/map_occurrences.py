import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "occurrences" / "gbif_occurrences.csv"
MAP_PATH = ROOT / "reports" / "gbif_occurrence_map.html"

if not CSV_PATH.exists():
    raise SystemExit(f"CSV not found: {CSV_PATH}")

records = []

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        try:
            lat = float(row["decimalLatitude"])
            lon = float(row["decimalLongitude"])
        except (ValueError, TypeError, KeyError):
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        records.append({
            "lat": lat,
            "lon": lon,
            "species": row.get("requested_species", "Unknown"),
            "country": row.get("country", ""),
            "name": row.get("scientificName", ""),
            "id": row.get("gbifID", ""),
        })

if not records:
    raise SystemExit("No valid occurrence coordinates found.")

points = json.dumps(records).replace("</", "<\\/")

page = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InvaScope AI - Occurrence Explorer</title>
<style>
body { font-family: Arial; margin: 0; color: #18332b; }
header { padding: 16px 22px; }
#map { width: 100%; height: 72vh; background: #edf3e9; }
svg { width: 100%; height: 100%; }
circle { cursor: pointer; stroke: #173d32; stroke-width: .5; }
#details { padding: 12px 22px; }
select { padding: 8px; margin: 4px; }
</style>
</head>
<body>
<header>
<h2>🌿 InvaScope AI — Occurrence Explorer</h2>
<p id="summary"></p>
<label>Species:
<select id="species"><option value="ALL">All species</option></select>
</label>
<label>Country:
<select id="country"><option value="ALL">All countries</option></select>
</label>
</header>
<div id="map">
<svg id="plot" viewBox="0 0 1000 500"
     preserveAspectRatio="xMidYMid meet"></svg>
</div>
<div id="details">Select a point to inspect its occurrence record.</div>
<script>
const data = POINTS_DATA;
const NS = "http://www.w3.org/2000/svg";
const plot = document.getElementById("plot");
const speciesSelect = document.getElementById("species");
const countrySelect = document.getElementById("country");
const details = document.getElementById("details");

[...new Set(data.map(d => d.species))].sort().forEach(s => {
  const o = document.createElement("option");
  o.value = s; o.textContent = s; speciesSelect.appendChild(o);
});
[...new Set(data.map(d => d.country))].sort().forEach(c => {
  const o = document.createElement("option");
  o.value = c; o.textContent = c; countrySelect.appendChild(o);
});

function draw() {
  plot.replaceChildren();

  // Simple equirectangular projection, with no external map tiles.
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = (lon + 180) / 360 * 1000;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", x); line.setAttribute("x2", x);
    line.setAttribute("y1", 0); line.setAttribute("y2", 500);
    line.setAttribute("stroke", "#d4dfd0");
    plot.appendChild(line);
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = (90 - lat) / 180 * 500;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("x2", 1000);
    line.setAttribute("y1", y); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#d4dfd0");
    plot.appendChild(line);
  }

  const filtered = data.filter(d =>
    (speciesSelect.value === "ALL" ||
     d.species === speciesSelect.value) &&
    (countrySelect.value === "ALL" ||
     d.country === countrySelect.value)
  );

  filtered.forEach(d => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", (d.lon + 180) / 360 * 1000);
    c.setAttribute("cy", (90 - d.lat) / 180 * 500);
    c.setAttribute("r", 3.5);
    c.setAttribute("fill",
      d.country === "IN" || d.country === "India" ? "#218c55" :
      d.country === "US" || d.country === "United States" ? "#2878b5" :
      "#bc6c25");
    c.addEventListener("click", () => {
      details.textContent =
        "Species: " + d.species + " | Reported name: " + d.name +
        " | Country: " + d.country + " | GBIF ID: " + d.id +
        " | Coordinates: " + d.lat + ", " + d.lon;
    });
    plot.appendChild(c);
  });

  document.getElementById("summary").textContent =
    "Showing " + filtered.length + " of " + data.length +
    " occurrence records. Green = India; blue = USA; brown = other/unknown.";
}

speciesSelect.addEventListener("change", draw);
countrySelect.addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""

page = page.replace("POINTS_DATA", points)

MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
MAP_PATH.write_text(page, encoding="utf-8")

print("Standalone map created:", MAP_PATH)
print("Occurrence points:", len(records))
print("No API key or external map tiles required.")
