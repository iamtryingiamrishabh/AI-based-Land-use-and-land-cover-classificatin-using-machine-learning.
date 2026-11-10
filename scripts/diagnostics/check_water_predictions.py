import rasterio
import numpy as np
import csv

# Open classification and SCL files
with rasterio.open("lulc_classification.tif") as pred_src:
    prediction = pred_src.read(1)

with rasterio.open(
    "T43RGM_20260910T052641_SCL_20m.jp2"
) as scl_src:
    scl = scl_src.read(1)

# Compare predicted water with SCL water
valid = prediction != 0

predicted_water = prediction == 4
scl_water = scl == 6

print("\nWATER CLASS COMPARISON")
print("----------------------")

print(
    "Predicted water:",
    np.count_nonzero(predicted_water & valid)
)

print(
    "SCL water:",
    np.count_nonzero(scl_water & valid)
)

print(
    "Predicted water percentage:",
    round(np.count_nonzero(predicted_water & valid) /
          np.count_nonzero(valid) * 100, 2),
    "%"
)

print(
    "SCL water percentage:",
    round(np.count_nonzero(scl_water & valid) /
          np.count_nonzero(valid) * 100, 2),
    "%"
)

print(
    "Overlap:",
    np.count_nonzero(predicted_water & scl_water & valid)
)

# Check manually selected water samples
print("\nMANUALLY SELECTED WATER SAMPLES")
print("--------------------------------")

with open("training_samples.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["label"] == "Water":
            r = int(row["row"])
            c = int(row["col"])

            print(
                f"Row: {r}, Col: {c}, SCL class: {scl[r, c]}"
            )