"""
Marine debris mapper: local web app.

    python app.py            -> http://127.0.0.1:5000
"""
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from marine import config, inference, render
from marine.readers import InputError, load_files, load_sample

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024          # 1 GB uploads

# Recently analysed scenes, so changing a setting doesn't need a re-upload.
_cache = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", 8))


def _remember(scene):
    token = uuid.uuid4().hex
    with _cache_lock:
        _cache[token] = {"scene": scene, "probs": {}}
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return token


def _get(token):
    with _cache_lock:
        if token not in _cache:
            raise InputError("This image has expired from the server cache; please load it again.")
        _cache.move_to_end(token)
        return _cache[token]


def _samples():
    if not os.path.exists(config.SAMPLES_FILE):
        return []
    return json.load(open(config.SAMPLES_FILE))


def _available_models():
    return {k: m for k, m in config.MODELS.items()
            if os.path.exists(os.path.join(config.MODEL_DIR, m["file"]))}


def _downsample(scene):
    c, h, w = scene.img.shape
    if h * w <= config.MAX_PIXELS:
        return
    f = int(np.ceil(np.sqrt(h * w / config.MAX_PIXELS)))
    scene.img = scene.img[:, ::f, ::f].copy()
    if scene.gt is not None:
        scene.gt = scene.gt[::f, ::f].copy()
    if scene.transform is not None:
        scene.transform = scene.transform * scene.transform.scale(f, f)
    scene.notes.append(f"Image was {h}×{w} px; analysed every {f}th pixel "
                       f"({scene.img.shape[1]}×{scene.img.shape[2]}) to keep the site responsive.")


def analyse(token, model_key, threshold, use_scope):
    entry = _get(token)
    scene = entry["scene"]
    models = _available_models()
    if model_key not in models:
        model_key = config.DEFAULT_MODEL.get(scene.source, "marida_s8")
        if model_key not in models:
            model_key = next(iter(models))
    t0 = time.time()
    if model_key not in entry["probs"]:
        entry["probs"][model_key] = inference.predict_proba(scene.img, model_key)
    probs, valid = entry["probs"][model_key]
    t_pred = time.time() - t0

    labels, conf = inference.label_map(probs, valid, scene.img, threshold, use_scope)
    regions = inference.find_regions(labels, conf, scene)
    h, w = labels.shape
    px_w, px_h = scene.pixel_size_m

    out = {
        "token": token,
        "name": scene.name,
        "source": scene.source,
        "model": model_key,
        "model_label": config.MODELS[model_key]["label"],
        "width": w, "height": h,
        "pixel_size_m": [px_w, px_h],
        "georeferenced": scene.crs is not None,
        "bounds": scene.bounds_latlon(),
        "notes": scene.notes,
        "seconds": round(t_pred, 2),
        "layers": {
            "rgb": render.true_colour(scene.img, valid),
            "pred": render.class_layer(labels),
            "conf": render.confidence_layer(conf, valid),
        },
        "labels": render.raw_b64(labels),
        "conf": render.raw_b64((conf * 255).astype(np.uint8)),
        "summary": inference.class_summary(labels, scene),
        "regions": regions,
        "gt_metrics": None,
    }
    if scene.gt is not None:
        out["layers"]["gt"] = render.ground_truth_layer(scene.gt)
        out["gt_labels"] = render.raw_b64(scene.gt.astype(np.uint8))
        out["gt_metrics"] = inference.gt_metrics(scene.gt, probs, valid)
    return out


def _opts(src):
    return (src.get("model") or "",
            float(src.get("threshold", 0.5)),
            str(src.get("scope", "true")).lower() in ("1", "true", "on", "yes"))


# ------------------------------------------------------------------ routes

@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/meta")
def meta():
    report = {}
    rpath = os.path.join(config.MODEL_DIR, "report.json")
    if os.path.exists(rpath):
        report = json.load(open(rpath))
    return jsonify({
        "classes": {k: {**v, "description": config.DESCRIPTIONS.get(k, "")} for k, v in config.CLASSES.items()},
        "models": {k: {"label": m["label"], "dataset": m["dataset"], "stage": m["stage"],
                       "scores": report.get(k)} for k, m in _available_models().items()},
        "default_model": config.DEFAULT_MODEL,
        "samples": _samples(),
        "scope_guard": {k: v for k, v in report.items() if k.endswith("scope_guard")},
    })


@app.get("/api/thumb/<dataset>/<sid>.png")
def thumb(dataset, sid):
    cache_dir = os.path.join(config.MODEL_DIR, "thumbs")
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{dataset}_{sid}.png"
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        sc = load_sample({"dataset": dataset, "id": sid}, config.MARIDA_ROOT, config.MADOS_ROOT)
        import base64, io
        data = render.true_colour(sc.img, inference.valid_mask(sc.img)).split(",", 1)[1]
        Image.open(io.BytesIO(base64.b64decode(data))).resize((128, 128), Image.NEAREST).save(path)
    return send_from_directory(cache_dir, fname)


@app.post("/api/sample")
def run_sample():
    body = request.get_json(force=True)
    sample = next((s for s in _samples() if s["id"] == body.get("id") and s["dataset"] == body.get("dataset")), None)
    if sample is None:
        return jsonify({"error": "Unknown sample."}), 404
    scene = load_sample(sample, config.MARIDA_ROOT, config.MADOS_ROOT)
    token = _remember(scene)
    return jsonify(analyse(token, *_opts(body)))


@app.post("/api/upload")
def run_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No file received."}), 400
    tmp = tempfile.mkdtemp(prefix="marine_up_")
    try:
        paths = []
        for f in files:
            name = os.path.basename(f.filename or "upload.tif")
            p = os.path.join(tmp, name)
            f.save(p)
            paths.append(p)
        scene = load_files(paths, boa_offset=request.form.get("boa_offset") == "true")
        _downsample(scene)
        token = _remember(scene)
        return jsonify(analyse(token, *_opts(request.form)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/reanalyse")
def reanalyse():
    body = request.get_json(force=True)
    return jsonify(analyse(body["token"], *_opts(body)))


@app.errorhandler(InputError)
def input_error(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(FileNotFoundError)
def missing(e):
    return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (limit 1 GB)."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Marine debris mapper on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
