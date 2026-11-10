import rasterio
import numpy as np
import glob
import os

files = sorted(glob.glob("training_data/worldcover_2021/*.tif"))

for file in files:
    counts = {}

    with rasterio.open(file) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            values, frequencies = np.unique(data, return_counts=True)

            for value, frequency in zip(values, frequencies):
                counts[int(value)] = counts.get(int(value), 0) + int(frequency)

    print("\n" + os.path.basename(file))
    for class_id, count in sorted(counts.items()):
        print(f"{class_id}: {count:,}")
