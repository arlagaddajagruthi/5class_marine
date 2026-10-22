# Marine Debris Mapper

An interactive website that finds **floating marine debris, Sargassum and other surface features** in Sentinel-2 satellite images.

Upload an image, or pick a MARIDA / MADOS test patch. Every pixel is classified as **Water, Marine Debris, Algae/Sargassum or Other**. The site then shows:

- the satellite image with every pixel coloured by class, plus ground-truth and confidence layers;
- a plain-language summary of what is in the image and how much area each class covers;
- a list of **detected patches**: class, size in pixels and m², confidence and location (lat/lon with a map link), downloadable as CSV;
- a **map view** on satellite basemap tiles for georeferenced images (MARIDA patches and Sentinel-2 GeoTIFFs);
- for test-set samples, accuracy against the ground-truth labels.

This branch contains only the website. The experiments it is based on are on the other branches of this repository.

---

## Models

The classifier is a **Random Forest** (100 trees, log class weights w = 1 / ln(1.03 + class frequency)), trained on every annotated pixel of the official training split, exactly as in the staged feature study.

| Model key | Trained on | Features | Test macro F1 | Test debris F1 |
|---|---|---|---|---|
| `marida_s8` (default for MARIDA and Sentinel-2 uploads) | MARIDA | stage 8: 11 bands + NDWI + 4 band ratios + 6 GLCM texture | 0.917 | 0.831 |
| `mados_s8` (default for MADOS) | MADOS | stage 8 | 0.761 | 0.708 |
| `marida_s5` | MARIDA | stage 5: 11 bands + NDWI + 4 band ratios | 0.904 | 0.813 |
| `mados_s5` | MADOS | stage 5 | 0.719 | 0.717 |

Scores are on every annotated pixel of the official test splits, over 4 classes (`models/report.json`).

**Classes**

| ID | Class | Includes |
|---|---|---|
| 1 | Water | marine, turbid, shallow, sediment-laden and mixed water; waves; wakes |
| 2 | Marine Debris | floating litter and plastics |
| 3 | Algae / Sargassum | dense and sparse Sargassum, floating algae |
| 4 | Other | natural organic material, ships, foam (MADOS: also oil spills, oil platforms, jellyfish, sea snot) |
| 5 | Uncertain | top-class probability below the threshold set on the page |
| 6 | Cloud / land | large bright or land-like areas, outside the ocean-only model's scope |

---

## Supported inputs

| Input | Notes |
|---|---|
| MARIDA patch | one 11-band GeoTIFF; add its `_cl.tif` to see ground truth |
| MADOS patch | the 11 `*_L2R_rhorc_*` band files, or a `.zip` of them; add `*_L2R_cl_*.tif` for ground truth. MADOS has no location or date, so positions are shown in pixels. |
| Sentinel-2 multi-band GeoTIFF | 11 bands (B1–B8, B8A, B11, B12), 12 bands (L2A, with B9) or 13 bands (L1C) in standard order; digital numbers are scaled to reflectance automatically |
| Sentinel-2 single-band files | B01–B12 and B8A as `.tif` / `.jp2` (e.g. from Copernicus Browser or a `.SAFE` folder), or a `.zip` of them |

Ordinary photos (JPG/PNG, Google Maps screenshots) cannot be analysed: they have no infrared bands, and infrared is where floating material shows up.

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### 1. Datasets

The datasets are **not** in this repository. Download `MARIDA_5Class` and `MADOS_5Class` (the 4-class versions of MARIDA and MADOS, available on Kaggle) and place them as:

```
data/
├── MARIDA_5Class/patches/S2_<scene>/S2_<patch>.tif, S2_<patch>_cl.tif
└── MADOS_5Class/Scene_<n>/{10,20,60}/..., MADOS_5Class/splits/{train,test,val}_X.txt
```

Or point to another location with the `MARIDA_ROOT` and `MADOS_ROOT` environment variables. The MARIDA train/test patch lists are included in `splits/marida/`.

### 2. Train the models

```bash
python train_models.py                      # MARIDA and MADOS (about 25 min on a laptop CPU)
python train_models.py --datasets MARIDA    # MARIDA only (about 8 min)
```

This writes `models/rf_<key>.joblib`, updates `models/report.json` with the test scores, and writes `models/samples.json` (the sample gallery). The model files are not committed because they are rebuilt here.

Optional: `python eval_guard.py` checks how many annotated (ocean) test pixels the cloud/land mask hides by mistake.

### 3. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**. On Windows you can also double-click `run.bat`.

The first analysis after start-up takes about 20 s while the GLCM texture code is compiled; after that each 256×256 patch takes under a second.

---

## How it works

1. **Read** the 11 Sentinel-2 bands as surface reflectance, resampling 20 m and 60 m bands to the 10 m grid.
2. **Compute features** per pixel: the bands; NDWI; the red/green, NIR/red, NIR/green and SWIR1/NIR ratios; and, for stage 8, six GLCM texture measures (contrast, dissimilarity, homogeneity, energy, correlation, ASM) over a 13×13 window.
3. **Classify** each pixel with the Random Forest, which gives a probability for each of the 4 classes.
4. **Post-process**: mark low-probability pixels as *Uncertain*, mask large cloud- or land-like areas, then group pixels of the same class that lie within about 20 m of each other into **patches**, measuring each patch's area, mean confidence and centre.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask server and JSON API (`/api/meta`, `/api/sample`, `/api/upload`, `/api/reanalyse`) |
| `marine/readers.py` | reads MARIDA/MADOS patches and Sentinel-2 files into the 11-band stack |
| `marine/features.py`, `marine/glcm.py` | stage-5 and stage-8 features |
| `marine/inference.py` | prediction, uncertainty and cloud/land masks, patch detection, ground-truth scores |
| `marine/render.py` | image layers (true colour, prediction, ground truth, confidence) |
| `static/` | web page (HTML, CSS, JavaScript, Leaflet map) |
| `train_models.py`, `eval_guard.py` | model training and the cloud/land mask check |
| `splits/marida/` | MARIDA train/test patch lists |
| `models/report.json`, `models/samples.json` | test scores and sample gallery |

## Limitations

- **Sentinel-2 only**, because the model needs the infrared bands.
- **Processing differences:** MARIDA and MADOS were atmospherically corrected in a particular way. Standard Sentinel-2 L1C/L2A products differ, so expect lower accuracy on them than the test scores above. For L2A products from 2022 onwards, tick *Subtract L2A offset* on the page.
- **Ocean only:** the model never saw land or clouds. The mask catches large areas, but coastlines can still be mislabelled.
- **10 m pixels:** only aggregated debris (patches and lines tens of metres long) is visible, not single items.

## Data credits

- MARIDA: Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., Karantzalos, K. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE 17(1): e0262247.
- MADOS: Kikaki, K., Kakogeorgiou, I., Hoteit, I., Karantzalos, K. (2024). *Detecting marine pollutants and sea surface features with deep learning in Sentinel-2 imagery.* ISPRS Journal of Photogrammetry and Remote Sensing 210: 39–54.
- Basemap imagery: Esri World Imagery.
