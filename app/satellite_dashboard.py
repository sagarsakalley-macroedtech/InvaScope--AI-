
import csv
from pathlib import Path
from collections import Counter, defaultdict

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "gbif_occurrences_ndvi.csv"

st.set_page_config(
    page_title="InvaScope AI Satellite Dashboard",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 InvaScope AI")
st.subheader("Invasive Plant Occurrences and Satellite Analysis")
st.caption("GBIF occurrence records • Sentinel-2 • NDVI")

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Run the NDVI extraction script first.")
    st.stop()


def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

required = {
    "latitude", "longitude", "species",
    "country", "ndvi", "status",
}

if not rows or not required.issubset(rows[0].keys()):
    st.error("The CSV is empty or its columns do not match the expected format.")
    st.stop()

records = []

for row in rows:
    lat = to_float(row.get("latitude"))
    lon = to_float(row.get("longitude"))
    ndvi = to_float(row.get("ndvi"))

    if lat is None or lon is None:
        continue
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        continue

    records.append({
        "latitude": lat,
        "longitude": lon,
        "species": row.get("species", "Unknown") or "Unknown",
        "country": row.get("country", "Unknown") or "Unknown",
        "ndvi": ndvi,
        "status": row.get("status", ""),
    })

st.sidebar.header("Filters")

countries = sorted({r["country"] for r in records})
selected_countries = st.sidebar.multiselect(
    "Country", countries, default=countries
)

country_rows = [
    r for r in records if r["country"] in selected_countries
]

species_options = sorted({r["species"] for r in country_rows})
selected_species = st.sidebar.multiselect(
    "Species", species_options, default=species_options
)

view = [
    r for r in country_rows if r["species"] in selected_species
]
valid = [r for r in view if r["ndvi"] is not None]

coverage = 100 * len(valid) / len(view) if view else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Occurrence records", len(view))
c2.metric("Valid NDVI values", len(valid))
c3.metric("NDVI coverage", f"{coverage:.1f}%")
c4.metric("Species", len({r["species"] for r in view}))

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### Occurrence locations")
    if view:
        st.map([
            {"lat": r["latitude"], "lon": r["longitude"]}
            for r in view
        ])
    else:
        st.info("No locations match the selected filters.")

with right:
    st.markdown("### Records by country")
    country_counts = Counter(r["country"] for r in view)
    if country_counts:
        st.bar_chart(dict(country_counts))

st.markdown("### Average NDVI by species")

species_values = defaultdict(list)
for record in valid:
    species_values[record["species"]].append(record["ndvi"])

species_means = {
    species: sum(values) / len(values)
    for species, values in species_values.items()
}

if species_means:
    st.bar_chart(species_means)
else:
    st.info("No valid NDVI measurements for this selection.")

st.markdown("### Occurrence and NDVI records")
st.dataframe(view, use_container_width=True, hide_index=True)

csv_content = (
    ["latitude,longitude,species,country,ndvi,status"]
    + [
        ",".join(
            '"' + str(r[k]).replace('"', '""') + '"'
            for k in [
                "latitude", "longitude", "species",
                "country", "ndvi", "status",
            ]
        )
        for r in view
    ]
)

st.download_button(
    "Download filtered CSV",
    data="\n".join(csv_content).encode("utf-8"),
    file_name="invascope_ndvi_filtered.csv",
    mime="text/csv",
)

st.warning(
    "NDVI measures vegetation greenness; it does not identify invasive "
    "species. Interpret measurements with observation dates, coordinate "
    "uncertainty, seasonality and satellite image quality in mind."
)
