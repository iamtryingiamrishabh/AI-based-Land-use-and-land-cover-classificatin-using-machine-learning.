import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.enums import Resampling

class_names = {
    1: "Built-up",
    2: "Agriculture",
    3: "Vegetation",
    4: "Water",
    5: "Barren"
}

class_colors = [
    "#e63946",  # Built-up
    "#f4a261",  # Agriculture
    "#2a9d50",  # Vegetation
    "#2196f3",  # Water
    "#c2a477"   # Barren
]

with rasterio.open("outputs/rasters/lulc_classification_hybrid.tif") as src:

    # Read the full classification to calculate class areas
    classification = src.read(1)

    total_valid = np.count_nonzero(classification)

    print("\nCLASSIFICATION SUMMARY")
    print("----------------------")
    print("Valid pixels:", total_valid)

    for class_id, name in class_names.items():

        count = np.count_nonzero(classification == class_id)

        area_km2 = count * 400 / 1_000_000

        percentage = (
            count / total_valid * 100
            if total_valid else 0
        )

        print(
            f"{name}: {count:,} pixels | "
            f"{area_km2:.2f} km² | "
            f"{percentage:.2f}%"
        )

    # Read a smaller version for visualisation
    preview = src.read(
        1,
        out_shape=(1200, 1200),
        resampling=Resampling.nearest
    )

# Create discrete colour map
cmap = ListedColormap(["black"] + class_colors)
norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    cmap.N
)

plt.figure(figsize=(12, 10))

plt.imshow(preview, cmap=cmap, norm=norm)

legend_items = [
    Patch(facecolor=class_colors[i - 1], label=name)
    for i, name in class_names.items()
]

plt.legend(
    handles=legend_items,
    loc="lower right",
    title="Land Cover Classes"
)

plt.title("Random Forest Land Use / Land Cover Classification")
plt.axis("off")

plt.tight_layout()
plt.savefig("lulc_preview.png", dpi=200)
plt.show()

print("\nPreview saved as lulc_preview.png")