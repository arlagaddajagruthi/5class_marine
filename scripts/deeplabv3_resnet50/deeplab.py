"""
DeepLabv3-ResNet50 pixel-level segmentation, 4 classes (1 water, 2 debris,
3 algae/sargassum, 4 other; class 0 = unannotated/background is ignored).

Neither paper trained DeepLabv3. The model is new; the TRAINING PROCEDURE of
each dataset's own paper is copied from its official code:

MARIDA: the MARIDA U-Net recipe (Kikaki et al. 2022,
  marine-debris.github.io semantic_segmentation/unet/{train.py,dataloader.py})
  - weighted cross-entropy, ignore_index -1 (unannotated pixels), weight
    w_c = 1 / ln(1.03 + p_c), p_c = fraction of class c among annotated TRAIN
    pixels (the published constants equal the train-split distribution exactly)
  - Adam lr 2e-4, weight decay 0, MultiStepLR milestones [40], gamma 0.1,
    45 epochs, batch 5
  - augmentation: rotation by one of {-90, 0, 90, 180} degrees + random
    horizontal flip, applied to image and mask together
  - NaN -> band mean, per-band standardisation with the MARIDA mean/std
  - checkpoint chosen by lowest validation loss (the paper: "early stopping
    based on the loss of the validation set")
MADOS: the MADOS/MariNeXt recipe (Kikaki et al. 2024, gkakogeorgiou/mados
  marinext/{train.py,evaluation.py,test_time_aug.py}, utils/{dataset.py,vscp.py})
  - weighted cross-entropy, ignore_index -1, w_c = 1 / ln(1.03 + p_c), p_c from
    the MADOS train split
  - Adam lr 2e-4, weight decay 0, MultiStepLR milestones [45, 65], gamma 0.1,
    80 epochs, batch 5, drop_last, label smoothing 0, no gradient clipping
  - augmentation: random flip (one of h / v / both / none, as in the code) and,
    with probability 0.8, rotation by k*90 degrees; then VSCP copy-paste (adds
    batch//2 extra samples per batch)
  - EMA of the weights (timm ModelEma, decay 0.999); the EMA model is evaluated
  - test: 8-view test-time augmentation (4 rotations x hflip), per-pixel
    majority vote
  - NaN -> band mean, standardisation with the MADOS mean/std
  - checkpoint chosen by lowest validation loss of the EMA model (the code
    leaves the choice to the user; fixed here in advance)

Model: torchvision deeplabv3_resnet50, no pretrained weights (both papers train
from scratch), first convolution replaced by an 11-band 7x7 conv, 4 output
classes, no auxiliary head, torchvision default initialisation.
Seed 0, cudnn deterministic.

Metrics on annotated pixels only: the papers' Evaluation (macro / micro /
weighted precision, recall, F1, macro IoU) plus per-class F1 / IoU and the
confusion matrix, in the same form as scripts/staged.

Usage (repo root):
  python scripts/deeplabv3_resnet50/deeplab.py cache --dataset MADOS   # stacks MADOS bands once (disk)
  python scripts/deeplabv3_resnet50/deeplab.py check --dataset MARIDA  # GPU memory / speed
  python scripts/deeplabv3_resnet50/deeplab.py train --dataset MARIDA
  python scripts/deeplabv3_resnet50/deeplab.py test  --dataset MARIDA
Outputs: outputs/deeplabv3_resnet50/<dataset>/
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
import torchvision.transforms as transforms
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import hflip

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resnet_paper"))
from resnet_ml import STATS, RandomRotationTransform, read_patch, split_ids  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_ROOT = os.path.join(REPO, "outputs", "deeplabv3_resnet50")
CLASS_NAMES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
RECIPES = {
    "MARIDA": dict(epochs=45, batch=5, lr=2e-4, decay=0.0, lr_steps=[40], gamma=0.1, weight_param=1.03,
                   aug="marida", vscp=False, ema=False, tta=False, drop_last=False),
    "MADOS": dict(epochs=80, batch=5, lr=2e-4, decay=0.0, lr_steps=[45, 65], gamma=0.1, weight_param=1.03,
                  aug="mados", vscp=True, ema=True, ema_decay=0.999, tta=True, drop_last=True),
}


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
def cache_paths(dataset, split):
    d = os.path.join(OUT_ROOT, "cache")
    return os.path.join(d, f"{dataset}_{split}_X.npy"), os.path.join(d, f"{dataset}_{split}_y.npy")


def cmd_cache(args):
    """Stack each MADOS patch's 11 band files into one array on disk (values unchanged, NaNs kept)."""
    os.makedirs(os.path.join(OUT_ROOT, "cache"), exist_ok=True)
    for split in ("train", "val", "test"):
        px, py = cache_paths(args.dataset, split)
        if os.path.exists(px) and os.path.exists(py):
            continue
        ids = split_ids(args.dataset, split)
        img0, m0 = read_patch(args.dataset, ids[0])
        X = np.lib.format.open_memmap(px + ".tmp.npy", mode="w+", dtype=np.float32, shape=(len(ids),) + img0.shape)
        Y = np.empty((len(ids),) + m0.shape, np.int8)
        t0 = time.time()
        for i, p in enumerate(ids):
            X[i], Y[i] = read_patch(args.dataset, p)
        X.flush()
        del X
        os.replace(px + ".tmp.npy", px)
        np.save(py, Y)
        print(f"{args.dataset} {split}: {len(ids)} patches cached ({time.time() - t0:.0f} s)", flush=True)


class SegDataset(Dataset):
    def __init__(self, dataset, split, train):
        self.dataset, self.train = dataset, train
        self.ids = split_ids(dataset, split)
        mean, std = (np.array(v, np.float32) for v in STATS[dataset])
        self.mean, self.std = mean, std
        px, py = cache_paths(dataset, split)
        self.X = np.load(px, mmap_mode="r") if os.path.exists(px) else None
        self.Y = np.load(py) if os.path.exists(py) else None
        self.rot = transforms.Compose([transforms.ToTensor(), RandomRotationTransform([-90, 0, 90, 180]),
                                       transforms.RandomHorizontalFlip()])
        self.norm = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.ids)

    def raw(self, i):
        if self.X is not None:
            return np.array(self.X[i]), self.Y[i].astype(np.int64)
        return read_patch(self.dataset, self.ids[i])

    def __getitem__(self, i):
        img, mask = self.raw(i)
        target = mask - 1                                          # 0 (unannotated) -> -1 ignored; 1..4 -> 0..3
        img = np.moveaxis(img, 0, -1).astype(np.float32)           # C,H,W -> H,W,C
        nan = np.isnan(img)
        img[nan] = np.broadcast_to(self.mean, img.shape)[nan]
        if self.dataset == "MARIDA":                               # MARIDA U-Net dataloader
            stack = np.concatenate([img, target[:, :, None]], axis=-1).astype(np.float32)
            stack = self.rot(stack) if self.train else transforms.ToTensor()(stack)
            img, target = stack[:-1], stack[-1].long()
            return self.norm(img), target
        target = target[:, :, None]                                # MADOS dataset.py
        if self.train:
            f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]
            if f != 2:
                img, target = (np.flip(a, f) if f != -1 else np.fliplr(np.flipud(a)) for a in (img, target))
            if np.random.random() < 0.8:
                k = np.random.randint(0, 4)
                img, target = np.rot90(img, k, (1, 0)), np.rot90(target, k, (1, 0))
        img = (img.astype(np.float32).transpose(2, 0, 1) - self.mean.reshape(-1, 1, 1)) / self.std.reshape(-1, 1, 1)
        return torch.from_numpy(img.copy()), torch.from_numpy(target.squeeze().copy()).long()


def vscp(image, target):
    """gkakogeorgiou/mados utils/vscp.py, unchanged."""
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


def tta(img, reverse_aggregation=False):
    """gkakogeorgiou/mados test_time_aug.py, unchanged."""
    out = []
    if not reverse_aggregation:
        for k in [0, 1, 2, 3]:
            im = torch.rot90(img, k=k, dims=[-2, -1])
            out.append(im)
            out.append(hflip(im))
        return torch.cat(out)
    for k in [3, 2, 1, 0]:
        out.append(torch.rot90(hflip(img[k * 2 + 1]), k=-k, dims=[-2, -1]))
        out.append(torch.rot90(img[k * 2], k=-k, dims=[-2, -1]))
    return torch.mode(torch.stack(out), dim=0, keepdim=True)[0]


def train_class_weights(dataset, c):
    ds = SegDataset(dataset, "train", train=False)
    counts = np.zeros(4, np.int64)
    for i in range(len(ds)):
        m = ds.raw(i)[1]
        counts += np.bincount(m[m > 0].ravel() - 1, minlength=4)
    distr = counts / counts.sum()
    return counts, distr, 1 / np.log(c + distr)


###############################################################
# Model
###############################################################
class DeepLab(nn.Module):
    def __init__(self, in_ch=11, n_classes=4):
        super().__init__()
        self.net = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=n_classes, aux_loss=False)
        self.net.backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.net.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")  # torchvision ResNet default

    def forward(self, x):
        return self.net(x)["out"]          # already upsampled (bilinear) to the input size


###############################################################
# Evaluation
###############################################################
def metrics(y_true, y_pred):
    lab = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=lab)
    f1c = f1_score(y_true, y_pred, labels=lab, average=None, zero_division=0)
    iouc = jaccard_score(y_true, y_pred, labels=lab, average=None, zero_division=0)
    return {
        "macroPrec": precision_score(y_true, y_pred, labels=lab, average="macro", zero_division=0),
        "microPrec": precision_score(y_true, y_pred, labels=lab, average="micro", zero_division=0),
        "weightPrec": precision_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0),
        "macroRec": recall_score(y_true, y_pred, labels=lab, average="macro", zero_division=0),
        "microRec": recall_score(y_true, y_pred, labels=lab, average="micro", zero_division=0),
        "weightRec": recall_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0),
        "macroF1": float(f1c.mean()),
        "microF1": f1_score(y_true, y_pred, labels=lab, average="micro", zero_division=0),
        "weightF1": f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0),
        "IoU": float(iouc.mean()),
        **{f"F1_{n}": float(v) for n, v in zip(CLASS_NAMES, f1c)},
        **{f"IoU_{n}": float(v) for n, v in zip(CLASS_NAMES, iouc)},
    }, cm


def classwise(cm):
    """Per-class table from a confusion matrix (rows true, columns predicted)."""
    tp = np.diag(cm).astype(np.int64)
    support, predicted, total = cm.sum(1), cm.sum(0), cm.sum()
    fn, fp = support - tp, predicted - tp
    tn = total - tp - fn - fp
    div = lambda a, b: np.divide(a, b, out=np.zeros(len(a)), where=b > 0)
    prec, rec = div(tp, predicted), div(tp, support)
    return pd.DataFrame({
        "support": support, "share_%": 100 * support / total, "predicted": predicted,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "class_acc": rec, "OvR_acc": (tp + tn) / total,
        "precision": prec, "recall": rec, "F1": div(2 * prec * rec, prec + rec), "IoU": div(tp, tp + fp + fn),
        "specificity": div(tn, tn + fp), "FP_per_TP": div(fp, tp),
    }, index=CLASS_NAMES)


def md_table(df, fmt):
    cols = list(df.columns)
    out = ["| class | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(fmt(c, row[c]) for c in cols) + " |")
    return out


def classwise_section(yt, yp, out, name):
    """Markdown lines + CSVs with the full class-wise analysis of one checkpoint."""
    acc, cm = metrics(yt, yp)
    cw = classwise(cm)
    cw.to_csv(os.path.join(out, f"classwise_{name}.csv"), index_label="class")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(out, f"confusion_{name}.csv"))
    rown = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    coln = 100 * cm / np.maximum(cm.sum(0, keepdims=True), 1)
    pd.DataFrame(rown, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(out, f"confusion_rowpct_{name}.csv"))
    pd.DataFrame(coln, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(os.path.join(out, f"confusion_colpct_{name}.csv"))
    ints = {"support", "predicted", "TP", "FP", "FN", "TN"}
    fmt = lambda c, v: f"{int(v):,}" if c in ints else (f"{v:.2f}" if c in ("share_%", "FP_per_TP") else f"{v:.4f}")
    lines = ["### Overall", "", "| metric | value |", "|---|---|",
             f"| overall accuracy | {acc['microF1']:.4f} |",
             f"| balanced accuracy (macro recall) | {acc['macroRec']:.4f} |",
             f"| Cohen's kappa | {cohen_kappa_score(yt, yp):.4f} |"]
    lines += [f"| {k} | {acc[k]:.4f} |" for k in ("macroPrec", "macroRec", "macroF1", "IoU", "microPrec", "microRec",
                                                    "microF1", "weightPrec", "weightRec", "weightF1")]
    lines += ["", f"Annotated pixels: {cm.sum():,}. Macro scores weight every class equally; micro/weighted "
              "scores are dominated by Water.", "", "### Per class", ""]
    lines += md_table(cw, fmt)
    lines += ["", "class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, "
              "= recall = producer's accuracy; their mean is the balanced accuracy).",
              "OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the "
              "many pixels of other classes; shown for completeness).",
              "FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).",
              "", "### Confusion matrix, pixel counts (rows true, columns predicted)", ""]
    cmdf = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cmdf["total true"] = cm.sum(1)
    lines += md_table(cmdf, lambda c, v: f"{int(v):,}")
    lines += [f"| total predicted | " + " | ".join(f"{int(v):,}" for v in cm.sum(0)) + f" | {cm.sum():,} |", "",
              "### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)", ""]
    lines += md_table(pd.DataFrame(rown, index=CLASS_NAMES, columns=CLASS_NAMES), lambda c, v: f"{v:.2f}")
    lines += ["", "### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)", ""]
    lines += md_table(pd.DataFrame(coln, index=CLASS_NAMES, columns=CLASS_NAMES), lambda c, v: f"{v:.2f}")
    lines += ["", "### Where the errors go", ""]
    for i, c in enumerate(CLASS_NAMES):
        miss = [(CLASS_NAMES[j], cm[i, j]) for j in np.argsort(-cm[i]) if j != i and cm[i, j] > 0]
        fals = [(CLASS_NAMES[j], cm[j, i]) for j in np.argsort(-cm[:, i]) if j != i and cm[j, i] > 0]
        fn, fp = cm[i].sum() - cm[i, i], cm[:, i].sum() - cm[i, i]
        ms = ", ".join(f"{n} {v:,} ({100 * v / fn:.1f}%)" for n, v in miss) or "none"
        fs = ", ".join(f"{n} {v:,} ({100 * v / fp:.1f}%)" for n, v in fals) or "none"
        lines += [f"- **{c}**: {fn:,} missed pixels went to: {ms}.",
                  f"  {fp:,} false-positive pixels came from: {fs}."]
    return acc, cm, cw, lines + [""]


def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval()
    yt, yp, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if use_tta:                                            # batch 1 only, as in the MADOS code
                pred = tta(model(tta(image)).argmax(1), reverse_aggregation=True)
            else:
                logits = model(image)
                k = int((target != -1).sum())
                if k:
                    loss_sum += criterion(logits, target).item() * k
                    n += k
                pred = logits.argmax(1)
            keep = target != -1
            yt.append(target[keep].cpu().numpy())
            yp.append(pred[keep].cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp), (loss_sum / n if n else float("nan"))


###############################################################
# Commands
###############################################################
def setup(dataset):
    out = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out, "log.txt"), filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return out


def build(dataset, out):
    r = RECIPES[dataset]
    counts, distr, w = train_class_weights(dataset, r["weight_param"])
    info = {"recipe": r, "train_annotated_pixels": dict(zip(CLASS_NAMES, counts.tolist())),
            "train_class_fraction": dict(zip(CLASS_NAMES, distr.round(5).tolist())),
            "ce_weight": dict(zip(CLASS_NAMES, w.round(4).tolist()))}
    json.dump(info, open(os.path.join(out, "recipe_and_weights.json"), "w"), indent=2)
    logging.info(json.dumps(info))
    return r, torch.tensor(w, dtype=torch.float32)


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
    out = setup(args.dataset)
    r, w = build(args.dataset, out)
    seed_all()
    device = torch.device("cuda")
    model = DeepLab().to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["decay"])
    loader = DataLoader(SegDataset(args.dataset, "train", True), batch_size=r["batch"], shuffle=True,
                        drop_last=r["drop_last"])
    torch.cuda.reset_peak_memory_stats()
    it, t0 = iter(loader), time.time()
    for _ in range(args.batches):
        image, target = next(it)
        train_step(model, opt, crit, image, target, r, device)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.batches
    logging.info(f"check {args.dataset}: {dt:.2f} s/batch, peak GPU memory "
                 f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, {len(loader)} batches/epoch -> "
                 f"~{dt * len(loader) / 60:.1f} min/epoch train")


def cmd_train(args):
    out = setup(args.dataset)
    r, w = build(args.dataset, out)
    seed_all()
    g = torch.Generator()
    g.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SegDataset(args.dataset, "train", True), batch_size=r["batch"], shuffle=True,
                              drop_last=r["drop_last"], generator=g)
    val_loader = DataLoader(SegDataset(args.dataset, "val", False), batch_size=r["batch"], shuffle=False)
    model = DeepLab().to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean", weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, r["lr_steps"], gamma=r["gamma"])
    ema = None
    if r["ema"]:
        from timm.utils import ModelEma
        ema = ModelEma(model, decay=r["ema_decay"], device=device, resume="")
    hist, best, start = [], np.inf, 1
    state_path = os.path.join(out, "train_state.pth")
    if args.resume and os.path.exists(state_path):
        s = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        if ema is not None:
            ema.ema.load_state_dict(s["ema"])
        hist, best, start = s["hist"], s["best"], s["epoch"] + 1
        g.set_state(s["gen"])
        torch.set_rng_state(s["torch_rng"])
        torch.cuda.set_rng_state(s["cuda_rng"])
        np.random.set_state(s["np_rng"])
        random.setstate(s["py_rng"])
        logging.info(f"resumed after epoch {s['epoch']} (best val loss so far {best:.4f})")
    elif not args.resume and os.path.exists(state_path):
        sys.exit(f"{state_path} exists: pass --resume to continue it, or delete it to start over")
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
        evals = {"model": model}
        if ema is not None:
            evals["ema"] = ema.ema
        for name, m in evals.items():
            seed_all()
            yt, yp, vloss = evaluate(m, val_loader, crit, device)
            acc, _ = metrics(yt, yp)
            row.update({f"{name}_val_loss": vloss, f"{name}_val_macroF1": acc["macroF1"], f"{name}_val_IoU": acc["IoU"],
                        f"{name}_val_F1_debris": acc["F1_Marine Debris"]})
        sel = "ema" if ema is not None else "model"
        row["seconds"] = time.time() - t0
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        logging.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if row[f"{sel}_val_loss"] < best:
            best = row[f"{sel}_val_loss"]
            torch.save(evals[sel].state_dict(), os.path.join(out, "model_best_val_loss.pth"))
            json.dump({"epoch": epoch, "weights": sel, "val_loss": best}, open(os.path.join(out, "best_epoch.json"), "w"))
        sched.step()
        model.train()
        s = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "ema": ema.ema.state_dict() if ema is not None else None, "hist": hist, "best": best,
             "gen": g.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "np_rng": np.random.get_state(), "py_rng": random.getstate()}
        torch.save(s, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)      # a crash mid-save keeps the previous epoch's state
    torch.save((ema.ema if ema is not None else model).state_dict(), os.path.join(out, "model_last.pth"))
    os.remove(state_path)
    logging.info("training done")


def cmd_test(args):
    out = setup(args.dataset)
    r, w = build(args.dataset, out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.CrossEntropyLoss(ignore_index=-1, weight=w.to(device))
    best = json.load(open(os.path.join(out, "best_epoch.json")))
    same = best["epoch"] == r["epochs"]          # best epoch is the last epoch -> identical weights
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
    lines = [f"# DeepLabv3-ResNet50, {args.dataset}, 4 classes (annotated pixels only)", "",
             f"Recipe: {json.dumps(r)}", "",
             f"Selected checkpoint: epoch {best['epoch']} of {r['epochs']} ({best['weights']} weights), lowest val "
             f"loss {best['val_loss']:.4f}. Test-time augmentation: {r['tta']}.", "",
             "## Class distribution (annotated pixels per split)", ""]
    lines += md_table(dist_df, lambda c, v: f"{int(v):,}" if "pixels" in c else f"{v:.2f}") + [""]
    rows, cws = [], {}
    names = ["model_best_val_loss"] if same else ["model_best_val_loss", "model_last"]
    if same:
        lines += ["model_best_val_loss and model_last hold the same weights (best epoch = last epoch), "
                  "so the test results are reported once.", ""]
    for name in names:
        model = DeepLab().to(device)
        model.load_state_dict(torch.load(os.path.join(out, name + ".pth"), map_location=device))
        splits = ("test", "val") if name == "model_best_val_loss" else ("test",)
        for split in splits:
            seed_all()
            loader = DataLoader(SegDataset(args.dataset, split, False), batch_size=1 if r["tta"] else r["batch"],
                                shuffle=False)
            yt, yp, _ = evaluate(model, loader, crit, device, use_tta=r["tta"])
            tag = name if split == "test" else f"{name}_val"
            acc, cm, cw, sec = classwise_section(yt, yp, out, tag)
            rows.append({"checkpoint": name, "split": split, "kappa": cohen_kappa_score(yt, yp), **acc})
            cws[(name, split)] = cw
            lines += [f"## {name}, {split} split", ""] + sec
    key = ("model_best_val_loss", "test"), ("model_best_val_loss", "val")
    gap = pd.DataFrame({f"{m} val": cws[key[1]][m] for m in ("precision", "recall", "F1", "IoU")} |
                       {f"{m} test": cws[key[0]][m] for m in ("precision", "recall", "F1", "IoU")} |
                       {"F1 test - val": cws[key[0]]["F1"] - cws[key[1]]["F1"]})
    lines += ["## Validation vs test, per class (model_best_val_loss)", ""] + md_table(gap, lambda c, v: f"{v:+.4f}" if "-" in c else f"{v:.4f}")
    lines += ["", "Large negative F1 test - val means the class generalises worse to the test scenes than the "
              "validation loss suggested.", ""]
    pd.DataFrame(rows).to_csv(os.path.join(out, "test_metrics.csv"), index=False)
    open(os.path.join(out, "test_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    logging.info("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cache", "check", "train", "test"])
    ap.add_argument("--dataset", choices=["MARIDA", "MADOS"], required=True)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="train: continue from <dataset>/train_state.pth")
    a = ap.parse_args()
    {"cache": cmd_cache, "check": cmd_check, "train": cmd_train, "test": cmd_test}[a.command](a)
