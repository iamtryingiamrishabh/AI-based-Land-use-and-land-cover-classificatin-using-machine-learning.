import rasterio
import numpy as np
import matplotlib.pyplot as plt
import csv

from rasterio.enums import Resampling
from rasterio.windows import Window

image_file = "masked_feature_stack.tif"
mask_file = "cloud_mask.tif"

classes = {
    "1": "Built-up",
    "2": "Agriculture",
    "3": "Vegetation",
    "4": "Water",
    "5": "Barren"
}

colors = {
    "Built-up": "blue",
    "Agriculture": "orange",
    "Vegetation": "green",
    "Water": "cyan",
    "Barren": "brown"
}

samples = []
current_class = None

with rasterio.open(image_file) as src, rasterio.open(mask_file) as mask_src:

    height = src.height
    width = src.width

    # Create a smaller RGB image for display
    rgb = src.read(
        [3, 2, 1],
        out_shape=(3, 1200, 1200),
        resampling=Resampling.average,
        masked=True
    )

    display = np.zeros((1200, 1200, 3))

    for i in range(3):
        values = rgb[i].compressed()
        low, high = np.percentile(values, [2, 98])

        display[:, :, i] = np.clip(
            (rgb[i].filled(low) - low) / (high - low),
            0, 1
        )

        display[:, :, i][np.ma.getmaskarray(rgb[i])] = 0

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(display)

    ax.set_title(
        "Press 1-5 to select class | Click to add sample | Q to save and quit"
    )

    ax.axis("off")

    def on_key(event):
        global current_class

        if event.key in classes:
            current_class = classes[event.key]
            print("\nSelected:", current_class)

        elif event.key == "q":
            save_samples()
            plt.close(fig)

    def on_click(event):
        if event.inaxes != ax or current_class is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata * width / 1200)
        row = int(event.ydata * height / 1200)

        col = min(col, width - 1)
        row = min(row, height - 1)

        # Check whether this pixel is usable
        valid = mask_src.read(
            1, window=Window(col, row, 1, 1)
        )[0, 0]

        if valid != 1:
            print("Excluded pixel. Choose another location.")
            return

        # Extract six features at this location
        values = src.read(
            window=Window(col, row, 1, 1)
        )[:, 0, 0]

        if not np.all(np.isfinite(values)):
            print("Invalid pixel. Choose another location.")
            return

        samples.append({
            "row": row,
            "col": col,
            "label": current_class,
            "features": values.tolist()
        })

        ax.scatter(
            event.xdata,
            event.ydata,
            color=colors[current_class],
            s=25
        )

        fig.canvas.draw_idle()

        count = sum(
            1 for s in samples if s["label"] == current_class
        )

        print(
            current_class,
            "| Samples:", count,
            "| Total:", len(samples)
        )

    def save_samples():
        if not samples:
            print("No samples collected.")
            return

        with open("training_samples.csv", "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                "row", "col", "label",
                "Blue", "Green", "Red",
                "NIR", "NDVI", "NDWI"
            ])

            for s in samples:
                writer.writerow([
                    s["row"], s["col"], s["label"],
                    *s["features"]
                ])

        print("\nSamples saved to training_samples.csv")

        for name in classes.values():
            count = sum(1 for s in samples if s["label"] == name)
            print(name, ":", count)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()