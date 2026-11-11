import rasterio
import numpy as np
import joblib

# 1. Load trained Random Forest model
model = joblib.load("models/lulc_random_forest.pkl")

# 2. Define input and output paths
feature_file = "outputs/rasters/masked_feature_stack.tif"

scl_file = ( "outputs/rasters/SCL_corrected.tif")


output_file = "outputs/rasters/lulc_classification_hybrid.tif"

# 3. Open input rasters
with rasterio.open(feature_file) as src, \
     rasterio.open(scl_file) as scl_src:

    # Check raster alignment
    if (
        src.width != scl_src.width
        or src.height != scl_src.height
        or src.transform != scl_src.transform
        or src.crs != scl_src.crs
    ):
        raise ValueError("Feature stack and SCL rasters are not aligned!")

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="LZW"
    )

    # 4. Create output classification raster
    with rasterio.open(output_file, "w", **profile) as dst:

        for _, window in src.block_windows(1):

            data = src.read(window=window)
            scl = scl_src.read(1, window=window)

            # Identify valid pixels
            valid = np.all(np.isfinite(data), axis=0)

            if src.nodata is not None:
                valid &= np.all(data != src.nodata, axis=0)

            result = np.zeros(
                (data.shape[1], data.shape[2]),
                dtype=np.uint8
            )

            if np.any(valid):

                pixels = data[:, valid].T

                # Get Random Forest probabilities
                probabilities = model.predict_proba(pixels)

                # Exclude RF water class
                non_water = model.classes_ != 4

                non_water_classes = model.classes_[non_water]

                predictions = non_water_classes[
                    np.argmax(probabilities[:, non_water], axis=1)
                ]

                result[valid] = predictions.astype(np.uint8)

                # Apply SCL water correction
                water_pixels = valid & (scl == 6)

                result[water_pixels] = 4

            dst.write(result, 1, window=window)

            print(
                "Processed rows:",
                int(window.row_off + window.height),
                "/",
                src.height
            )

print("\nHybrid classification completed!")
print("Output:", output_file)
