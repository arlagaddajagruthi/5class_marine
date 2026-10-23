"""
26-feature study (MARIDA or MADOS, `--dataset`): train, test and document U-Net,
SegNeXt-T and the MariNeXt architecture on the 26 input channels of features26.py.

Single seed 0. Each dataset uses its paper's recipe (RECIPES below):
  MARIDA  the MARIDA U-Net code: 45 epochs, Adam 2e-4, batch 5, LR / 10 at epoch 40,
          weighted CE 1/ln(1.03 + p_c), rotation + flip augmentation, no VSCP
  MADOS   the MADOS/MariNeXt code: 80 epochs, Adam 2e-4, batch 5 (drop_last), LR / 10
          at epochs 45 and 65, the same weighted CE, flip + rotation augmentation,
          VSCP copy-paste (batch // 2 extra samples per batch), EMA of the weights
          (decay 0.999; the EMA model is validated, selected and tested)
Runs:
  R1   --model unet     --init ssl4eo  --bands raw
  R2   --model unet     --init ssl4eo  --bands std
  R3   --model unet     --init scratch --bands std
  R4   --model segnext  --init scratch --bands std
  R5   --model marinext --init scratch --bands std
  G2b  --model unet     --init ssl4eo  --bands raw --classes 11   (MARIDA paper's 11 classes)

Checkpoint: best validation macro F1 with soft 8-view test-time augmentation (the
softmax averaged over 4 rotations x 2 flips), evaluated every epoch. Test: the
same soft TTA; class probabilities are stored for every pixel. MADOS runs: R3, R4, R5.

Commands (repo root):
  python scripts/marida_f26/train.py arch  --run R1 --model unet --init ssl4eo --bands raw
  python scripts/marida_f26/train.py check --run R1 --model unet --init ssl4eo --bands raw
  python scripts/marida_f26/train.py train --run R1 --model unet --init ssl4eo --bands raw [--resume]
  python scripts/marida_f26/train.py test  --run R1 --model unet --init ssl4eo --bands raw
  (add --dataset MADOS for MADOS)
Outputs: outputs/<dataset>_f26/runs/<run>/
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "scripts", "segnext")]
import features26 as F26  # noqa: E402

DATASET = F26.DATASET
RUNS = os.path.join(F26.STUDY, "runs")
ORIG_MARIDA = "C:/PROJECT/marine_pollution_detection/MARIDA"
_TTA = "soft, 8 views (4 rotations x 2 flips), softmax averaged"
RECIPES = {
    "MARIDA": dict(epochs=45, batch=5, lr=2e-4, decay=0.0, lr_steps=[40], gamma=0.1, weight_param=1.03, seed=0,
                   augmentation="rotation -90/0/90/180 + horizontal flip (MARIDA U-Net)", vscp=False, ema=False,
                   drop_last=False, tta=_TTA, select="best validation macro F1 with soft TTA"),
    "MADOS": dict(epochs=80, batch=5, lr=2e-4, decay=0.0, lr_steps=[45, 65], gamma=0.1, weight_param=1.03, seed=0,
                  augmentation="h / v / both / no flip, then p = 0.8 rotation by k x 90 (MADOS code)", vscp=True,
                  ema=True, ema_decay=0.999, drop_last=True, tta=_TTA,
                  select="best validation macro F1 of the EMA model with soft TTA"),
}
RECIPE = RECIPES[DATASET]
SSL4EO_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]     # B1-B8, B8A, B11, B12 of the 13 pretrained bands

CLASSES4 = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
# MARIDA paper, 11 classes: Waves, Cloud Shadows, Wakes and Mixed Water merged into Marine Water
CLASSES11 = ["Marine Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic Material", "Ship",
             "Clouds", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water"]
ORIG_TO_11 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 6, 13: 6, 14: 6, 15: 6}


def vscp(image, target):
    """Very Simple Copy-Paste, gkakogeorgiou/mados utils/vscp.py (as scripts/deeplabv3_resnet50/deeplab.py):
    sample i + n's annotated pixels are pasted onto sample i, giving n = batch // 2 extra samples."""
    n = image.shape[0] // 2
    image_temp, target_temp = image[:n * 2].copy(), target[:n * 2].copy()
    ia, ta = [], []
    for i in range(n):
        m = target_temp[i + n] != -1
        image_temp[i][:, m] = image_temp[i + n][:, m]
        ia.append(image_temp[i].copy())
        target_temp[i][m] = target_temp[i + n][m]
        ta.append(target_temp[i].copy())
    return np.stack(ia), np.stack(ta)


def with_vscp(x, t):
    if not RECIPE["vscp"]:
        return x, t
    ia, ta = vscp(x.numpy(), t.numpy())
    return torch.cat([x, torch.from_numpy(ia)]), torch.cat([t, torch.from_numpy(ta)])


def class_names(n):
    return CLASSES4 if n == 4 else CLASSES11


def seed_all(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################
# Data
###############################################################
class RunDataset(F26.MaridaF26):
    """features26.MaridaF26, optionally with the MARIDA paper's 11-class labels (original masks)."""

    def __init__(self, split, train, bands, n_classes):
        super().__init__(split, train, bands)
        self.n_classes = n_classes

    def unscaled(self, i):
        x, mask = super().unscaled(i)
        if self.n_classes == 11:
            pid = self.ids[i]
            scene = pid.rsplit("_", 1)[0]
            import tifffile
            m = tifffile.imread(os.path.join(ORIG_MARIDA, "patches", f"S2_{scene}", f"S2_{pid}_cl.tif"))
            m = m.astype(np.int64)
            out = np.zeros_like(m)                       # 0 = unlabelled
            for k, v in ORIG_TO_11.items():
                out[m == k] = v + 1
            mask = out
        return x, mask


def train_counts(n_classes):
    ds = RunDataset("train", False, "raw", n_classes)
    c = np.zeros(n_classes, np.int64)
    for i in range(len(ds)):
        _, m = ds.unscaled(i)
        c += np.bincount(m[m > 0].ravel() - 1, minlength=n_classes)
    return c


###############################################################
# Models
###############################################################
def build_model(model, init, n_in=26, n_classes=4):
    if model == "unet":
        import segmentation_models_pytorch as smp
        net = smp.Unet("resnet50", encoder_weights=None, in_channels=n_in, classes=n_classes)
        if init == "ssl4eo":
            from torchgeo.models import ResNet50_Weights
            sd = ResNet50_Weights.SENTINEL2_ALL_MOCO.get_state_dict(progress=False)
            w = sd["conv1.weight"]                                    # (64, 13, 7, 7)
            new = torch.zeros(w.shape[0], n_in, *w.shape[2:])
            new[:, :11] = w[:, SSL4EO_KEEP]                           # the 11 bands keep their filters
            sd["conv1.weight"] = new                                  # the other 15 channels start at zero
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            missing, unexpected = nn.Module.load_state_dict(net.encoder, sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"encoder weights do not match: missing {missing}, unexpected {unexpected}")
        return net
    if init != "scratch":
        raise SystemExit("SegNeXt-T / MariNeXt have no Sentinel-2 pretrained weights: use --init scratch")
    from segnext_model import SegNeXt
    return SegNeXt(n_in, n_classes, first_stage=(model == "marinext"))


MODEL_NAME = {"unet": "U-Net (ResNet-50)", "segnext": "SegNeXt-T", "marinext": "MariNeXt architecture"}
MODEL_TEXT = {
    "unet": "U-Net (segmentation_models_pytorch) with a ResNet-50 encoder: 5 encoder stages (strides 2-32), "
            "a decoder of 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections from every "
            "encoder stage, and a 3x3 convolution to the class scores at full resolution.",
    "segnext": "SegNeXt-T: MSCAN-T encoder (4 stages, 32/64/160/256 channels, multi-scale convolutional attention "
               "with depth-wise strip convolutions) and a LightHamHead decoder (matrix-decomposition 'Hamburger' "
               "module) on stages 2-4; prediction at 1/8 resolution, bilinearly upsampled.",
    "marinext": "MariNeXt architecture: SegNeXt-T with the stage-1 features added to the LightHamHead (stages 1-4), "
                "so the prediction is made at 1/4 resolution, then bilinearly upsampled (as in the MADOS paper).",
}


###############################################################
# Soft test-time augmentation
###############################################################
def tta_probs(model, x):
    """x (1, C, H, W) -> (1, K, H, W) softmax averaged over 4 rotations x 2 horizontal flips."""
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        views += [r, torch.flip(r, dims=(-1,))]
    p = F.softmax(model(torch.cat(views)), 1)
    back = []
    for k in range(4):
        back.append(torch.rot90(p[2 * k], -k, dims=(-2, -1)))
        back.append(torch.rot90(torch.flip(p[2 * k + 1], dims=(-1,)), -k, dims=(-2, -1)))
    return torch.stack(back).mean(0, keepdim=True)


def predict(model, ds, device, tta=True, keep_maps=False):
    """Annotated-pixel labels, predictions and probabilities (+ per-patch index, + full probability maps)."""
    model.eval()
    yt, pr, pid = [], [], []
    maps = [] if keep_maps else None
    with torch.no_grad():
        for i in range(len(ds)):
            x, t = ds[i]
            x = x[None].to(device)
            p = tta_probs(model, x) if tta else F.softmax(model(x), 1)
            p = p[0].cpu().numpy()
            t = t.numpy()
            keep = t >= 0
            yt.append(t[keep])
            pr.append(p[:, keep].T)
            pid.append(np.full(keep.sum(), i, np.int32))
            if keep_maps:
                maps.append(p.astype(np.float16))
    return np.concatenate(yt), np.vstack(pr), np.concatenate(pid), (np.stack(maps) if keep_maps else None)


def quick_scores(yt, yp, n):
    from sklearn.metrics import f1_score
    f1 = f1_score(yt, yp, labels=list(range(n)), average=None, zero_division=0)
    return float(f1.mean()), f1


###############################################################
# Commands
###############################################################
def setup(a):
    out = os.path.join(RUNS, a.run)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out, "log.txt"), filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cfg_path = os.path.join(out, "run_config.json")
    names = class_names(a.classes)
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        for k in ("model", "init", "bands", "classes"):
            if cfg[k] != getattr(a, k):
                sys.exit(f"{a.run} was set up with {k}={cfg[k]}, not {getattr(a, k)}")
    else:
        counts = train_counts(a.classes)
        p = counts / counts.sum()
        w = 1 / np.log(RECIPE["weight_param"] + p)
        title = (f"{MODEL_NAME[a.model]}, {'SSL4EO-S12 pretrained' if a.init == 'ssl4eo' else 'from scratch'}, "
                 f"{'raw-reflectance' if a.bands == 'raw' else 'standardised'} bands")
        if a.classes == 11 and DATASET != "MARIDA":
            sys.exit("--classes 11 is the MARIDA paper's class scheme")
        cfg = {"run": a.run, "title": title, "dataset": DATASET, "model": a.model, "init": a.init, "bands": a.bands,
               "classes": a.classes, "class_names": names, "channels": F26.CHANNELS,
               "groups": F26.GROUPS, "recipe": RECIPE,
               "train_annotated_pixels": dict(zip(names, counts.tolist())),
               "ce_weight": dict(zip(names, np.round(w, 4).tolist()))}
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    return out, cfg, torch.tensor(list(cfg["ce_weight"].values()), dtype=torch.float32)


def cmd_arch(a):
    """architecture.md: layer summary, parameters, input channels, first-layer initialisation, data flow."""
    from torchinfo import summary
    out, cfg, _ = setup(a)
    net = build_model(a.model, a.init, F26.N_CH, a.classes)
    P = F26.PATCH
    s = summary(net, input_size=(1, F26.N_CH, P, P), depth=4, verbose=0,
                col_names=("output_size", "num_params"), device="cpu")
    open(os.path.join(out, "model_summary.txt"), "w", encoding="utf-8").write(str(s))
    st = json.load(open(F26.stats_path()))
    band_txt = ("reflectance unchanged (the SSL4EO pretraining scale)" if a.bands == "raw"
                else f"(x − mean) / std with the {DATASET} paper's band statistics")
    if a.model == "unet" and a.init == "ssl4eo":
        init_txt = ("Encoder initialised from SSL4EO-S12 MoCo ResNet-50 weights (torchgeo "
                    "`ResNet50_Weights.SENTINEL2_ALL_MOCO`, pretrained on 13-band Sentinel-2 L1C / 10000). "
                    "First convolution (64 filters, 7×7, stride 2): the pretrained filters of B1–B8, B8A, B11, "
                    "B12 are copied to input channels 1–11 (B9 and B10 dropped); the filters of channels 12–26 "
                    "start at zero, so at the start the network sees only the bands and learns how to use the 15 extra channels. The decoder is randomly initialised.")
    else:
        init_txt = "All weights randomly initialised (PyTorch / timm defaults); trained from scratch."
    lines = [f"# Architecture and input pipeline: run {a.run}", "",
             f"**Model**: {MODEL_TEXT[a.model]}", "",
             f"**Initialisation**: {init_txt}", "",
             f"**Parameters**: {s.total_params:,} ({s.trainable_params:,} trainable).", "",
             f"**Output**: {a.classes} class scores per pixel ({', '.join(cfg['class_names'])}); "
             "softmax → probabilities; prediction = argmax. Unlabelled pixels are ignored by the loss and the "
             "scores.", "",
             "## How the features are fed in", "",
             "```",
             f"{DATASET} patch (11 bands, {P}×{P}, Rayleigh-corrected reflectance, NaN → 0)",
             "   ├─ 20 spectral features per pixel (src/features/pixel_features.py, as the staged ML)",
             "   │     11 bands │ NDVI NDWI FAI FDI NDMI │ red/green nir/red nir/green swir1/nir",
             "   └─ 6 GLCM maps (glcm_maps.py, 13×13 window, as the staged ML)",
             "         contrast dissimilarity homogeneity energy correlation ASM",
             f"        ↓  stacked in that order → tensor 26 × {P} × {P}",
             f"   bands (ch 1–11): {band_txt}",
             "   other 15 channels: clipped to the train 1st–99th percentile, standardised",
             f"        ↓  training only: {RECIPE['augmentation']} (channels and labels together)"
             + ("; VSCP copy-paste adds batch // 2 samples" if RECIPE["vscp"] else ""),
             f"   {a.model} → {a.classes} × {P} × {P} class scores",
             "        ↓  test: softmax averaged over 8 views (4 rotations × 2 flips)",
             "   per-pixel class probabilities → argmax",
             "```", "",
             "All 26 channels enter the first layer together (early fusion): every first-layer filter "
             "sees all 26 channels of its 7×7 (U-Net) or patch-embedding (SegNeXt/MariNeXt) window, so the "
             "network can combine bands, indices and texture from the first layer on.", "",
             "## Input channels", "", "| # | Channel | Group | Scaling | Train mean | Train std |",
             "|---|---|---|---|---|---|"]
    for k, c in enumerate(F26.CHANNELS):
        if k < 11:
            sc = "raw reflectance" if a.bands == "raw" else "standardised (paper stats)"
            mu, sd = (st["mean"][k], st["std"][k]) if a.bands == "raw" else (F26.BAND_MEAN[k], F26.BAND_STD[k])
        else:
            sc, mu, sd = "clip 1–99 %, standardised", st["mean"][k], st["std"][k]
        lines.append(f"| {k + 1} | {c} | {F26.GROUP_OF[c]} | {sc} | {mu:.4g} | {sd:.4g} |")
    lines += ["", "## Training", "", "| setting | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in RECIPE.items()]
    lines += [f"| class weights | {', '.join(f'{n} {v}' for n, v in cfg['ce_weight'].items())} |", "",
              "## Layer summary (torchinfo, depth 4)", "", "See `model_summary.txt`.", ""]
    open(os.path.join(out, "architecture.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:12]))
    print(f"-> {out}/architecture.md, model_summary.txt")


def loaders(a, cfg):
    tr = RunDataset("train", True, a.bands, a.classes)
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(tr, batch_size=RECIPE["batch"], shuffle=True, generator=g, num_workers=0,
                      drop_last=RECIPE["drop_last"]), g


def cmd_check(a):
    out, cfg, w = setup(a)
    seed_all()
    device = torch.device("cuda")
    model = build_model(a.model, a.init, F26.N_CH, a.classes).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"])
    loader, _ = loaders(a, cfg)
    torch.cuda.reset_peak_memory_stats()
    it, t0 = iter(loader), time.time()
    for _ in range(a.batches):
        x, t = with_vscp(*next(it))
        opt.zero_grad()
        loss = crit(model(x.to(device)), t.to(device))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / a.batches
    val = RunDataset("val", False, a.bands, a.classes)
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(10):
            tta_probs(model, val[i][0][None].to(device))
    tv = (time.time() - t1) / 10
    ep = dt * len(loader) / 60 + tv * len(val) / 60
    msg = (f"check {a.run}: {dt:.2f} s/batch, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
           f"loss {loss.item():.3f}; {len(loader)} batches/epoch -> {dt * len(loader) / 60:.1f} min train + "
           f"{tv:.2f} s/patch TTA val x {len(val)} = {tv * len(val) / 60:.1f} min -> ~{ep:.1f} min/epoch, "
           f"~{ep * RECIPE['epochs'] / 60:.1f} h for {RECIPE['epochs']} epochs")
    logging.info(msg)
    json.dump({"s_per_batch": dt, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "tta_s_per_patch": tv, "est_hours": ep * RECIPE["epochs"] / 60},
              open(os.path.join(out, "check.json"), "w"), indent=2)


def cmd_train(a):
    out, cfg, w = setup(a)
    seed_all()
    device = torch.device("cuda")
    names = cfg["class_names"]
    loader, g = loaders(a, cfg)
    val = RunDataset("val", False, a.bands, a.classes)
    model = build_model(a.model, a.init, F26.N_CH, a.classes).to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, RECIPE["lr_steps"], gamma=RECIPE["gamma"])
    ema = None
    if RECIPE["ema"]:                                    # timm ModelEma, as the MADOS code
        from timm.utils import ModelEma
        ema = ModelEma(model, decay=RECIPE["ema_decay"], device=device, resume="")
    evaluated = lambda: ema.ema if ema is not None else model   # noqa: E731  (the model that is validated)
    hist, best, start = [], -np.inf, 1
    state_path = os.path.join(out, "train_state.pth")
    if a.resume and os.path.exists(state_path):
        s = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        if ema is not None:
            ema.ema.load_state_dict(s["ema"])
        hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        g.set_state(s["gen"].cpu()); torch.set_rng_state(s["torch_rng"].cpu())
        torch.cuda.set_rng_state(s["cuda_rng"].cpu()); np.random.set_state(s["np_rng"]); random.setstate(s["py_rng"])
        logging.info(f"resumed after epoch {s['epoch']} (best val macro F1 {best:.4f})")
    elif os.path.exists(state_path):
        sys.exit(f"{state_path} exists: pass --resume to continue it, or delete it to start over")
    for epoch in range(start, RECIPE["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, t in loader:
            x, t = with_vscp(x, t)
            x, t = x.to(device), t.to(device)
            opt.zero_grad()
            loss = crit(model(x), t)
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        # validation scores with soft TTA (the selection criterion)
        yt, pr, _, _ = predict(evaluated(), val, device, tta=True)
        mf1, f1c = quick_scores(yt, pr.argmax(1), a.classes)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot / n,
               "val_macroF1": mf1, **{f"val_F1_{c}": float(v) for c, v in zip(names, f1c)},
               "seconds": time.time() - t0}
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        logging.info(f"epoch {epoch}: train_loss {row['train_loss']:.4f}, val macro F1 {mf1:.4f}, "
                     + ", ".join(f"{c} {v:.3f}" for c, v in zip(names, f1c)) + f" ({row['seconds']:.0f} s)")
        if mf1 > best:
            best = mf1
            torch.save(evaluated().state_dict(), os.path.join(out, "model_best_val_f1.pth"))
            json.dump({"epoch": epoch, "weights": "ema" if ema is not None else "model", "val_macroF1_tta": mf1,
                       **{f"val_F1_{c}": float(v) for c, v in zip(names, f1c)}},
                      open(os.path.join(out, "best_epoch.json"), "w"), indent=2)
        sched.step()
        s = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "ema": ema.ema.state_dict() if ema is not None else None,
             "hist": hist, "best": best, "gen": g.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate()}
        torch.save(s, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    torch.save(evaluated().state_dict(), os.path.join(out, "model_last.pth"))
    os.remove(state_path)
    logging.info("training done")


def cmd_test(a):
    """Soft-TTA predictions of the best checkpoint on val and test, then the full report (report.py)."""
    import report
    out, cfg, _ = setup(a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.model, "scratch", F26.N_CH, a.classes).to(device)        # weights come from the checkpoint
    model.load_state_dict(torch.load(os.path.join(out, "model_best_val_f1.pth"), map_location=device))
    seed_all()
    for split in ("val", "test"):
        ds = RunDataset(split, False, a.bands, a.classes)
        yt, pr, pid, maps = predict(model, ds, device, tta=True, keep_maps=(split == "test"))
        np.savez_compressed(os.path.join(out, f"pred_{split}.npz"), y=yt, prob=pr.astype(np.float32), patch=pid,
                            patch_ids=np.array(ds.ids))
        if maps is not None:
            np.save(os.path.join(out, "prob_maps_test.npy"), maps)
        logging.info(f"{split}: {len(yt):,} annotated pixels predicted")
    report.run_report(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["arch", "check", "train", "test"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", choices=["unet", "segnext", "marinext"], required=True)
    ap.add_argument("--init", choices=["ssl4eo", "scratch"], required=True)
    ap.add_argument("--bands", choices=["raw", "std"], required=True)
    ap.add_argument("--classes", type=int, choices=[4, 11], default=4)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    from runs import name
    a.run = name(a.run)
    sys.stdout.reconfigure(encoding="utf-8")
    {"arch": cmd_arch, "check": cmd_check, "train": cmd_train, "test": cmd_test}[a.command](a)
