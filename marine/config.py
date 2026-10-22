import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset locations: data/MARIDA_5Class and data/MADOS_5Class next to app.py,
# or set the MARIDA_ROOT / MADOS_ROOT environment variables.
MARIDA_ROOT = os.environ.get("MARIDA_ROOT", os.path.join(HERE, "data", "MARIDA_5Class"))
MADOS_ROOT = os.environ.get("MADOS_ROOT", os.path.join(HERE, "data", "MADOS_5Class"))

MODEL_DIR = os.path.join(HERE, "models")
SAMPLES_FILE = os.path.join(MODEL_DIR, "samples.json")

CLASSES = {
    0: {"name": "Not analysed", "color": [0, 0, 0, 0]},
    1: {"name": "Water", "color": [30, 90, 200, 35]},
    2: {"name": "Marine Debris", "color": [255, 40, 40, 255]},
    3: {"name": "Algae / Sargassum", "color": [40, 220, 70, 235]},
    4: {"name": "Other (foam, ship, organic, oil*)", "color": [255, 190, 0, 230]},
    5: {"name": "Uncertain", "color": [200, 0, 255, 200]},
    6: {"name": "Cloud / land (outside model scope)", "color": [160, 160, 160, 170]},
}

DESCRIPTIONS = {
    1: "Open, turbid, shallow or sediment-laden water, including waves and wakes.",
    2: "Floating marine debris: plastics, litter and mixed anthropogenic material aggregated on the surface.",
    3: "Floating algae: dense or sparse Sargassum mats and other floating vegetation.",
    4: "Other floating features: natural organic material, ships, sea foam; in MADOS also oil spills, oil platforms, jellyfish and sea snakes.",
    5: "The model's top class probability was below the uncertainty threshold, so no label is given.",
    6: "Very bright (cloud) or dry, vegetated (land) pixels. The model was trained on ocean pixels only, so these are not classified.",
}

# Official MARIDA train/test patch lists (patches with annotated pixels).
MARIDA_SPLITS = os.environ.get("MARIDA_SPLITS", os.path.join(HERE, "splits", "marida"))

# key -> (training dataset, feature stage)
MODELS = {
    "marida_s8": {"dataset": "MARIDA", "stage": "stage8",
                  "label": "RF stage 8 · trained on MARIDA (bands + NDWI + ratios + GLCM)"},
    "mados_s8": {"dataset": "MADOS", "stage": "stage8",
                 "label": "RF stage 8 · trained on MADOS (bands + NDWI + ratios + GLCM)"},
    "marida_s5": {"dataset": "MARIDA", "stage": "stage5",
                  "label": "RF stage 5 · trained on MARIDA (bands + NDWI + ratios)"},
    "mados_s5": {"dataset": "MADOS", "stage": "stage5",
                 "label": "RF stage 5 · trained on MADOS (bands + NDWI + ratios)"},
}
for _k, _m in MODELS.items():
    _m["file"] = f"rf_{_k}.joblib"
DEFAULT_MODEL = {"marida": "marida_s8", "mados": "mados_s8", "sentinel2": "marida_s8"}

MAX_PIXELS = 1024 * 1024        # larger uploads are downsampled before prediction
