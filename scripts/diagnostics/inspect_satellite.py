
import rasterio
import matplotlib.pyplot as plt

# Path to the Sentinel-2 Red band (20m)
file_path = "T43RGM_20260910T052641_B04_20m.jp2"

# Open the satellite image
with rasterio.open(file_path) as src:

    print("Image Width:", src.width)
    print("Image Height:", src.height)
    print("Coordinate System:", src.crs)
    print("Pixel Resolution:", src.res)
    print("Data Type:", src.dtypes)

    # Read pixel values
    image = src.read(1)

# Display the satellite image
plt.figure(figsize=(10, 8))
plt.imshow(image, cmap="gray")
plt.title("Sentinel-2 Red Band - Greater Noida")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.colorbar(label="Pixel Intensity")
plt.show()
