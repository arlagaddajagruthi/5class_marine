"""
U-Net with a Sentinel-2 pretrained ResNet-50 encoder, 4 classes (1 water,
2 debris, 3 algae/sargassum, 4 other; unannotated pixels ignored).

Changes from scripts/deeplabv3_resnet50/deeplab.py (everything else is reused):
  - model: segmentation_models_pytorch U-Net (skip connections keep the
    full-resolution detail that thin debris needs), ResNet-50 encoder
  - encoder initialised from SSL4EO-S12 MoCo weights (torchgeo
    ResNet50_Weights.SENTINEL2_ALL_MOCO, pretrained on 13-band Sentinel-2 L1C
    divided by 10000). The conv1 filters of our 11 bands are kept (B9 and B10
    dropped); the 5 extra channels start at zero
  - input: 16 channels = the 11 bands as reflectance (NaN -> band mean, no
    standardisation, matching the pretraining scale) + NDWI and the 4 band
    ratios of the staged ML (red/green, nir/red, nir/green, swir1/nir). Each
    extra channel is clipped to its 1st-99th percentile on the train split and
    standardised (stats in extra_channel_stats.json)
  - checkpoint chosen by validation macro F1 (with test-time augmentation when
    the run uses it), not validation loss
  - test-time augmentation (8 views, per-pixel majority vote) on both datasets
  - VSCP copy-paste switchable per run (--vscp / --no-vscp)

Training recipe otherwise as each dataset's paper code (see deeplab.py):
MARIDA 45 epochs, LR /10 at 40; MADOS 80 epochs, LR /10 at 45 and 65, EMA;
Adam 2e-4, batch 5, weighted CE 1/ln(1.03 + p_c), same augmentation.

Usage (repo root):
  python scripts/unet_ssl4eo/unet.py check --dataset MARIDA --run MARIDA_vscp --vscp
  python scripts/unet_ssl4eo/unet.py train --dataset MARIDA --run MARIDA_vscp --vscp [--resume]
  python scripts/unet_ssl4eo/unet.py test  --dataset MARIDA --run MARIDA_vscp --vscp
Outputs: outputs/unet_ssl4eo/<run>/
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
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deeplabv3_resnet50"))
from deeplab import (CLASS_NAMES, RECIPES, SegDataset, classwise_section, md_table, metrics, seed_all,  # noqa: E402
                     train_class_weights, tta, vscp)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_ROOT = os.path.join(REPO, "outputs", "unet_ssl4eo")
TITLE = "U-Net (ResNet-50, SSL4EO-S12 pretrained)"   # report heading; overridden by scripts/segnext
EXTRA_NAMES = ["NDWI", "red/green", "nir/red", "nir/green", "swir1/nir"]
SSL4EO_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]   # B1-B8, B8A, B11, B12 out of the 13 pretrained bands
EPS = 1e-6


def recipe(dataset, use_vscp):
    r = dict(RECIPES[dataset])
    r.update(tta=True, vscp=use_vscp, select="val macroF1", encoder="resnet50 SSL4EO-S12 MoCo",
             inputs="11 bands (reflectance) + " + ", ".join(EXTRA_NAMES))
    return r


###############################################################
# Data: 11 bands + 5 extra channels
###############################################################
def extra_channels(b):
    """b: (11, H, W) reflectance -> (5, H, W): NDWI and 4 ratios (same formulas as the staged ML)."""
    green, red, nir, swir1 = b[2], b[3], b[7], b[9]
    return torch.stack([(green - nir) / (green + nir + EPS), red / (green + EPS), nir / (red + EPS),
                        nir / (green + EPS), swir1 / (nir + EPS)])


class Seg16(SegDataset):
    """deeplab.SegDataset (same reading, NaN filling and augmentation) with raw-scale bands + extra channels."""

    def __init__(self, dataset, split, train, extra_stats):
        super().__init__(dataset, split, train)
        lo, hi, mu, sd = (torch.tensor(extra_stats[k], dtype=torch.float32).view(-1, 1, 1)
                          for k in ("p1", "p99", "mean", "std"))
        self.lo, self.hi, self.mu, self.sd = lo, hi, mu, sd
        self.m = torch.from_numpy(self.mean).view(-1, 1, 1)
        self.s = torch.from_numpy(self.std).view(-1, 1, 1)

    def __getitem__(self, i):
        img, target = super().__getitem__(i)            # standardised 11 bands, augmented
        bands = img * self.s + self.m                   # back to reflectance
        ex = torch.nan_to_num(extra_channels(bands), nan=0.0, posinf=0.0, neginf=0.0)
        ex = (torch.minimum(torch.maximum(ex, self.lo), self.hi) - self.mu) / self.sd
        return torch.cat([bands, ex]).float(), target


def extra_stats(dataset, out):
    path = os.path.join(out, "extra_channel_stats.json")
    if os.path.exists(path):
        return json.load(open(path))
    ds = SegDataset(dataset, "train", train=False)
    m, s = (torch.from_numpy(a).view(-1, 1, 1) for a in (ds.mean, ds.std))
    rng = np.random.default_rng(0)
    samples = []
    for i in range(len(ds)):
        img, _ = ds[i]
        ex = torch.nan_to_num(extra_channels(img * s + m), nan=0.0, posinf=0.0, neginf=0.0).reshape(5, -1).numpy()
        samples.append(ex[:, rng.choice(ex.shape[1], 2000, replace=False)])
    a = np.concatenate(samples, axis=1)
    p1, p99 = np.percentile(a, 1, axis=1), np.percentile(a, 99, axis=1)
    c = np.clip(a, p1[:, None], p99[:, None])
    st = {"channels": EXTRA_NAMES, "p1": p1.tolist(), "p99": p99.tolist(),
          "mean": c.mean(1).tolist(), "std": np.maximum(c.std(1), 1e-6).tolist(),
          "note": "train split, 2000 random pixels per patch (all pixels, annotated or not)"}
    json.dump(st, open(path, "w"), indent=2)
    return st


###############################################################
# Model
###############################################################
def build_model(pretrained=True):
    model = smp.Unet("resnet50", encoder_weights=None, in_channels=11 + len(EXTRA_NAMES), classes=4)
    if pretrained:
        from torchgeo.models import ResNet50_Weights
        sd = ResNet50_Weights.SENTINEL2_ALL_MOCO.get_state_dict(progress=False)
        w = sd["conv1.weight"]                                       # (64, 13, 7, 7)
        new = torch.zeros(w.shape[0], 11 + len(EXTRA_NAMES), *w.shape[2:])
        new[:, :11] = w[:, SSL4EO_KEEP]
        sd["conv1.weight"] = new
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        # nn.Module's loader: smp's encoder override returns nothing, so mismatches would go unnoticed
        missing, unexpected = nn.Module.load_state_dict(model.encoder, sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"encoder weights do not match: missing {missing}, unexpected {unexpected}")
    return model


###############################################################
# Evaluation
###############################################################
def evaluate(model, loader, crit, device, use_tta):
    """Returns annotated-pixel labels, predictions and (without TTA) the mean val loss."""
    model.eval()
    yt, yp, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if use_tta:                                              # batch 1
                pred = tta(model(tta(image)).argmax(1), reverse_aggregation=True)
            else:
                logits = model(image)
                k = int((target != -1).sum())
                if k:
                    loss_sum += crit(logits, target).item() * k
                    n += k
                pred = logits.argmax(1)
            keep = target != -1
            yt.append(target[keep].cpu().numpy())
            yp.append(pred[keep].cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp), (loss_sum / n if n else float("nan"))


###############################################################
# Commands
###############################################################
def setup(args):
    out = os.path.join(OUT_ROOT, args.run)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out, "log.txt"), filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    r = recipe(args.dataset, args.vscp)
    counts, distr, w = train_class_weights(args.dataset, r["weight_param"])
    st = extra_stats(args.dataset, out)
    info = {"dataset": args.dataset, "recipe": r, "train_annotated_pixels": dict(zip(CLASS_NAMES, counts.tolist())),
            "ce_weight": dict(zip(CLASS_NAMES, w.round(4).tolist())), "extra_channel_stats": st}
    json.dump(info, open(os.path.join(out, "recipe_and_weights.json"), "w"), indent=2)
    return out, r, torch.tensor(w, dtype=torch.float32), st


def train_step(model, opt, crit, image, target, r, device):
    if r["vscp"]:
        ia, ta = vscp(image.numpy(), target.numpy())
        image, target = torch.cat([image, torch.from_numpy(ia)]), torch.cat([target, torch.from_numpy(ta)])
    image, target = image.to(device), target.long().to(device)
    opt.zero_grad()
    loss = crit(model(image), target)
    loss.backward()
    opt.step()
    return loss.item(), target.shape[0]


def cmd_check(args):
    out, r, w, st = setup(args)
    seed_all()
    device = torch.device("cuda")
    model = build_model().to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["decay"])
    loader = DataLoader(Seg16(args.dataset, "train", True, st), batch_size=r["batch"], shuffle=True,
                        drop_last=r["drop_last"])
    torch.cuda.reset_peak_memory_stats()
    it, t0 = iter(loader), time.time()
    for _ in range(args.batches):
        image, target = next(it)
        train_step(model, opt, crit, image, target, r, device)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.batches
    vl = DataLoader(Seg16(args.dataset, "val", False, st), batch_size=1)
    t1, k = time.time(), 0
    with torch.no_grad():
        model.eval()
        for image, _ in vl:
            model(tta(image.to(device))).argmax(1)
            k += 1
            if k == 20:
                break
    tv = (time.time() - t1) / k
    logging.info(f"check {args.run}: {dt:.2f} s/batch, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                 f"{len(loader)} batches/epoch -> ~{dt * len(loader) / 60:.1f} min train; TTA val "
                 f"{tv:.2f} s/patch x {len(vl)} -> ~{tv * len(vl) / 60:.1f} min/epoch")


def cmd_train(args):
    out, r, w, st = setup(args)
    seed_all()
    g = torch.Generator()
    g.manual_seed(0)
    device = torch.device("cuda")
    train_loader = DataLoader(Seg16(args.dataset, "train", True, st), batch_size=r["batch"], shuffle=True,
                              drop_last=r["drop_last"], generator=g)
    val_loader = DataLoader(Seg16(args.dataset, "val", False, st), batch_size=r["batch"], shuffle=False)
    val_tta = DataLoader(Seg16(args.dataset, "val", False, st), batch_size=1, shuffle=False)
    model = build_model().to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, r["lr_steps"], gamma=r["gamma"])
    ema = None
    if r["ema"]:
        from timm.utils import ModelEma
        ema = ModelEma(model, decay=r["ema_decay"], device=device, resume="")
    hist, best, start = [], -np.inf, 1
    state_path = os.path.join(out, "train_state.pth")
    if args.resume and os.path.exists(state_path):
        s = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        if ema is not None:
            ema.ema.load_state_dict(s["ema"])
        hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        g.set_state(s["gen"].cpu())  # map_location moved the RNG states to the GPU
        torch.set_rng_state(s["torch_rng"].cpu())
        torch.cuda.set_rng_state(s["cuda_rng"].cpu())
        np.random.set_state(s["np_rng"])
        random.setstate(s["py_rng"])
        logging.info(f"resumed after epoch {s['epoch']} (best val macroF1 so far {best:.4f})")
    elif os.path.exists(state_path):
        sys.exit(f"{state_path} exists: pass --resume to continue it, or delete it to start over")
    sel = "ema" if ema is not None else "model"
    for epoch in range(start, r["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for image, target in train_loader:
            loss, b = train_step(model, opt, crit, image, target, r, device)
            if ema is not None:
                ema.update(model)
            tot += loss * b
            n += b
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot / n}
        evals = {"model": model} | ({"ema": ema.ema} if ema is not None else {})
        for name, m in evals.items():
            seed_all()
            yt, yp, vloss = evaluate(m, val_loader, crit, device, use_tta=False)
            acc, _ = metrics(yt, yp)
            row.update({f"{name}_val_loss": vloss, f"{name}_val_macroF1": acc["macroF1"],
                        f"{name}_val_F1_debris": acc["F1_Marine Debris"]})
        yt, yp, _ = evaluate(evals[sel], val_tta, crit, device, use_tta=True)
        acc, _ = metrics(yt, yp)
        row.update({"sel_val_tta_macroF1": acc["macroF1"], "sel_val_tta_IoU": acc["IoU"],
                    "sel_val_tta_F1_debris": acc["F1_Marine Debris"], "seconds": time.time() - t0})
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        logging.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if row["sel_val_tta_macroF1"] > best:
            best = row["sel_val_tta_macroF1"]
            torch.save(evals[sel].state_dict(), os.path.join(out, "model_best_val_f1.pth"))
            json.dump({"epoch": epoch, "weights": sel, "val_macroF1_tta": best,
                       "val_F1_debris_tta": row["sel_val_tta_F1_debris"]},
                      open(os.path.join(out, "best_epoch.json"), "w"))
        sched.step()
        s = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "ema": ema.ema.state_dict() if ema is not None else None, "hist": hist, "best": best,
             "gen": g.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "np_rng": np.random.get_state(), "py_rng": random.getstate()}
        torch.save(s, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    torch.save(evals[sel].state_dict(), os.path.join(out, "model_last.pth"))
    os.remove(state_path)
    logging.info("training done")


def cmd_test(args):
    out, r, w, st = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    best = json.load(open(os.path.join(out, "best_epoch.json")))
    same = best["epoch"] == r["epochs"]
    dist = {}
    for split in ("train", "val", "test"):
        ds = SegDataset(args.dataset, split, train=False)
        c = np.zeros(4, np.int64)
        for i in range(len(ds)):
            m = ds.raw(i)[1]
            c += np.bincount(m[m > 0].ravel() - 1, minlength=4)
        dist[split] = c
    dist_df = pd.DataFrame({f"{s} pixels": c for s, c in dist.items()} |
                           {f"{s} %": 100 * c / c.sum() for s, c in dist.items()}, index=CLASS_NAMES)
    lines = [f"# {TITLE}, {args.dataset}, run {args.run}, 4 classes "
             "(annotated pixels only)", "", f"Recipe: {json.dumps(r)}", "",
             f"Selected checkpoint: epoch {best['epoch']} of {r['epochs']} ({best['weights']} weights), highest val "
             f"macro F1 with TTA {best['val_macroF1_tta']:.4f}. Test-time augmentation: {r['tta']}. "
             f"VSCP: {r['vscp']}.", "", "## Class distribution (annotated pixels per split)", ""]
    lines += md_table(dist_df, lambda c, v: f"{int(v):,}" if "pixels" in c else f"{v:.2f}") + [""]
    rows, cws = [], {}
    names = ["model_best_val_f1"] if same else ["model_best_val_f1", "model_last"]
    if same:
        lines += ["model_best_val_f1 and model_last hold the same weights (best epoch = last epoch), "
                  "so the test results are reported once.", ""]
    for name in names:
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(out, name + ".pth"), map_location=device))
        for split in (("test", "val") if name == "model_best_val_f1" else ("test",)):
            seed_all()
            loader = DataLoader(Seg16(args.dataset, split, False, st), batch_size=1, shuffle=False)
            yt, yp, _ = evaluate(model, loader, crit, device, use_tta=r["tta"])
            tag = name if split == "test" else f"{name}_val"
            acc, cm, cw, sec = classwise_section(yt, yp, out, tag)
            rows.append({"checkpoint": name, "split": split, "kappa": cohen_kappa_score(yt, yp), **acc})
            cws[(name, split)] = cw
            lines += [f"## {name}, {split} split", ""] + sec
    key = ("model_best_val_f1", "test"), ("model_best_val_f1", "val")
    gap = pd.DataFrame({f"{m} val": cws[key[1]][m] for m in ("precision", "recall", "F1", "IoU")} |
                       {f"{m} test": cws[key[0]][m] for m in ("precision", "recall", "F1", "IoU")} |
                       {"F1 test - val": cws[key[0]]["F1"] - cws[key[1]]["F1"]})
    lines += ["## Validation vs test, per class (model_best_val_f1)", ""]
    lines += md_table(gap, lambda c, v: f"{v:+.4f}" if "-" in c else f"{v:.4f}")
    lines += ["", "Large negative F1 test - val means the class generalises worse to the test scenes than "
              "validation suggested.", ""]
    pd.DataFrame(rows).to_csv(os.path.join(out, "test_metrics.csv"), index=False)
    open(os.path.join(out, "test_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    logging.info("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "train", "test"])
    ap.add_argument("--dataset", choices=["MARIDA", "MADOS"], required=True)
    ap.add_argument("--run", required=True, help="output folder name under outputs/unet_ssl4eo/")
    ap.add_argument("--vscp", action=argparse.BooleanOptionalAction, required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "test": cmd_test}[a.command](a)
