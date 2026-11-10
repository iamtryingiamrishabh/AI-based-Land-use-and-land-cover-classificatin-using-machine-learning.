import csv
import numpy as np
import rasterio
from rasterio.transform import xy

# Read existing training samples
with open("training_samples.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Remove incorrectly labelled water samples
rows = [r for r in rows if r["label"] != "Water"]

# Read SCL and find actual water pixels
with rasterio.open(
    "T43RGM_20260910T052641_SCL_20m.jp2"
) as scl_src:

    scl = scl_src.read(1)
    water_locations = np.argwhere(scl == 6)

    print("Available SCL water pixels:", len(water_locations))

    # Select 500 random water pixels
    rng = np.random.default_rng(42)

    selected = rng.choice(
        len(water_locations),
        size=min(500, len(water_locations)),
        replace=False
    )

    coordinates = []

    for index in selected:
        row, col = water_locations[index]

        x, y = xy(
            scl_src.transform,
            int(row),
            int(col),
            offset="center"
        )

        coordinates.append((x, y))

# Extract satellite features at those locations
with rasterio.open("masked_feature_stack.tif") as src:

    feature_values = list(src.sample(coordinates))

    for i, values in enumerate(feature_values):

        if not np.all(np.isfinite(values)):
            continue

        if np.any(values == src.nodata):
            continue

        row, col = water_locations[selected[i]]

        rows.append({
            "row": str(row),
            "col": str(col),
            "label": "Water",
            "Blue": str(values[0]),
            "Green": str(values[1]),
            "Red": str(values[2]),
            "NIR": str(values[3]),
            "NDVI": str(values[4]),
            "NDWI": str(values[5])
        })

# Save corrected training data
columns = [
    "row", "col", "label",
    "Blue", "Green", "Red",
    "NIR", "NDVI", "NDWI"
]

with open("training_samples.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\nCorrected dataset saved!")

for name in [
    "Built-up", "Agriculture", "Vegetation", "Water", "Barren"
]:
    count = sum(1 for r in rows if r["label"] == name)
    print(name, ":", count)