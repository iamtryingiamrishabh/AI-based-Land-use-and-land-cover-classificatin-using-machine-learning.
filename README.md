# Land Use / Land Cover Classification (Greater Noida and surroundings)

Classifies land into **forest**, **agricultural**, and **built-up** using Sentinel-2 imagery
and ESA WorldCover as ground-truth labels.

## Data (not stored in this repo)

Put these under `data/` (the scripts search the project folder by file name):

| Data | Source |
|------|--------|
| Sentinel-2 L2A, tile T43RGM, 2026-09-10: B02, B03, B04, B08 (10 m); optional B11, B12, SCL (20 m) | Copernicus Data Space |
| ESA WorldCover 2021 v200, tile N27E075 | https://esa-worldcover.org |
| Area-of-interest boundary (optional GeoJSON/shapefile) | own file |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python scripts/lulc_baseline.py
```

This aligns WorldCover labels to the Sentinel-2 grid, samples pixels with a
spatial block train/test split, trains a Random Forest, prints a per-class
report and confusion matrix, and writes `lulc_prediction.tif`.

## Status

- [x] Baseline pipeline: Sentinel-2 + WorldCover labels
- [ ] 20 m bands (B11, B12) and SCL cloud mask
- [ ] Second date (Feb-Mar) to separate forest from cropland
- [ ] Image-patch classifier and upload app

## Notes

- WorldCover labels are model-derived, not field surveys.
- WorldCover tile N27E075 ends at 78 E; the east edge of the Sentinel tile lies outside it.