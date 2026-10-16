"""
ResNet baseline exactly as in the MARIDA paper (Kikaki et al. 2022, PLOS ONE),
re-run with the 4-class remapping (1 water, 2 debris, 3 algae/sargassum, 4 other).

The paper's ResNet is NOT a pixel classifier. It is a weakly supervised
MULTI-LABEL PATCH CLASSIFIER: input = one whole 11-band patch, output = for
each class, "is this class present somewhere in the patch?" (yes/no).
Reference code: github.com/marine-debris/marine-debris.github.io,
multi-label/resnet/{resnet.py, dataloader.py, train.py, evaluation.py}.
The MADOS paper/code (MariNeXt) has no ResNet; MADOS is run with the same
MARIDA ResNet recipe, using MADOS's own band mean/std (from the MADOS repo).

Kept exactly as in the reference code
- ResNet-50, no pretraining, conv1 replaced by an 11-band 7x7 conv, global
  average pooling, one linear layer (2048 -> n_classes); Kaiming-normal init
  of every Conv2d and Linear weight.
- NaN pixels imputed with the band mean, then per-band standardisation.
- Augmentation (train only): rotation by one of {-90, 0, 90, 180} degrees,
  random horizontal flip (p = 0.5).
- BCEWithLogitsLoss(pos_weight = w), w_c = 1 / ln(1.6 + pos_c / neg_c), where
  pos_c / neg_c = number of TRAIN patches with / without class c.
- Adam, lr 2e-4, weight decay 1e-6, batch 32, 20 epochs, MultiStepLR
  milestones [5, 10, 15], gamma 0.2; evaluation on val every epoch.
- Threshold 0.5 on the sigmoid; metrics = the reference Evaluation_ML.
- seed 0, cudnn deterministic.

Differences (do not change the maths)
- Patches are read from disk on each access instead of all held in RAM.
- num_workers 0 (Windows; also avoids extra RAM).
- Only the best (min val loss) and last checkpoints are kept, not all 20.
- The reference code leaves the test epoch to the user; here it is fixed in
  advance: the epoch with the lowest val loss (the paper used early stopping
  on val loss for its U-Net). The last epoch is reported as well.

Usage (repo root):
  python scripts/resnet_paper/resnet_ml.py check --dataset MARIDA   # labels, weights, GPU memory/speed
  python scripts/resnet_paper/resnet_ml.py train --dataset MARIDA
  python scripts/resnet_paper/resnet_ml.py test  --dataset MARIDA
Outputs: outputs/resnet_paper/<dataset>/
"""
import argparse
import glob
import json
import logging
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import (accuracy_score, classification_report, coverage_error, f1_score, hamming_loss,
                             label_ranking_loss, multilabel_confusion_matrix, precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLASSES = [1, 2, 3, 4]
CLASS_NAMES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]

# Band statistics from the reference repositories (computed by the authors on the original data).
STATS = {
    "MARIDA": (  # marine-debris.github.io multi-label/resnet/dataloader.py
        [0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
         0.03875053, 0.03436435, 0.0392113, 0.02358126, 0.01588816],
        [0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
         0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]),
    "MADOS": (  # gkakogeorgiou/mados utils/dataset.py
        [0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
         0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944],
        [0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
         0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115]),
}
ROOTS = {"MARIDA": os.path.join(REPO, "data", "MARIDA_5Class"), "MADOS": os.path.join(REPO, "data", "MADOS_5Class")}

# Original MARIDA labels_mapping.txt index (class k at position k-1) -> remapped class,
# the same mapping as scripts/utilities/convert.py (classes 6 clouds and 13 cloud shadows -> 0, dropped).
MARIDA_GROUPS = {1: [7, 8, 10, 11, 12, 14, 15], 2: [1], 3: [2, 3], 4: [4, 5, 9]}

OPTIONS = dict(epochs=20, batch=32, lr=2e-4, decay=1e-6, lr_steps=[5, 10, 15], gamma=0.2,
               weight_param=1.6, threshold=0.5, seed=0, eval_batch=5)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################
# Model (reference resnet.py)
###############################################################
def weights_init_kaiming(m):
    if m.__class__.__name__.find("Conv2d") != -1 or type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class ResNet(nn.Module):
    def __init__(self, input_bands=11, output_classes=4):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool)
        self.fc = nn.Linear(2048, output_classes)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.view(x.size(0), -1))


###############################################################
# Data
###############################################################
_RHORC_RE = re.compile(r"_L2R_rhorc_(\d+)_\d+\.tif$")


def read_patch(dataset, pid, mask_only=False):
    """Raw 11-band image (C, H, W) with NaNs kept, and the remapped 4-class mask."""
    root = ROOTS[dataset]
    if dataset == "MARIDA":
        scene = pid.rsplit("_", 1)[0]
        base = os.path.join(root, "patches", f"S2_{scene}", f"S2_{pid}")
        if mask_only:
            return None, tifffile.imread(base + "_cl.tif").astype(np.int64)
        img = tifffile.imread(base + ".tif").astype(np.float32)
        if img.shape[-1] == 11:
            img = np.transpose(img, (2, 0, 1))
        mask = tifffile.imread(base + "_cl.tif").astype(np.int64)
        return img, mask
    scene, idx = pid.rsplit("_", 1)
    d = os.path.join(root, scene)
    mask = tifffile.imread(os.path.join(d, "10", f"{scene}_L2R_cl_{idx}.tif")).astype(np.int64)
    if mask_only:
        return None, mask
    files = sorted(glob.glob(os.path.join(d, "*", f"{scene}_L2R_rhorc_*_{idx}.tif")),
                   key=lambda f: int(_RHORC_RE.search(f).group(1)))
    if len(files) != 11:
        raise ValueError(f"{pid}: {len(files)} bands")
    h, w = mask.shape
    bands = []
    for f in files:   # 20 m / 60 m -> 10 m, nearest neighbour (as in MADOS)
        b = tifffile.imread(f).astype(np.float32)
        fy, fx = h // b.shape[0], w // b.shape[1]
        bands.append(np.repeat(np.repeat(b, fy, 0), fx, 1) if (fy, fx) != (1, 1) else b)
    return np.stack(bands), mask


def split_ids(dataset, split):
    return list(np.genfromtxt(os.path.join(ROOTS[dataset], "splits", f"{split}_X.txt"), dtype=str))


def patch_labels(dataset, split):
    """Multi-hot (n_patches, 4): class c present in the patch.
    MARIDA: the official labels_mapping.txt, remapped; checked against the remapped masks.
    MADOS (no label file): presence of the class in the remapped mask (identical construction)."""
    ids = split_ids(dataset, split)
    from_mask = np.array([[int((read_patch(dataset, p, mask_only=True)[1] == c).any()) for c in CLASSES] for p in ids], np.float32)
    if dataset == "MARIDA":
        L = json.load(open(os.path.join(ROOTS[dataset], "labels_mapping.txt")))
        y = np.array([[max(L[f"S2_{p}.tif"][k - 1] for k in MARIDA_GROUPS[c]) for c in CLASSES] for p in ids],
                     np.float32)
        if not np.array_equal(y, from_mask):
            raise RuntimeError("labels_mapping.txt and the remapped masks disagree")
        return ids, y
    return ids, from_mask


class RandomRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, random.choice(self.angles))


class PatchDataset(Dataset):
    def __init__(self, dataset, ids, y, train):
        mean, std = (np.array(v, np.float32) for v in STATS[dataset])
        self.dataset, self.ids, self.y, self.mean = dataset, ids, y, mean
        aug = [transforms.ToTensor()]
        if train:
            aug += [RandomRotationTransform([-90, 0, 90, 180]), transforms.RandomHorizontalFlip()]
        self.transform = transforms.Compose(aug)
        self.standardization = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img, _ = read_patch(self.dataset, self.ids[i])
        img = np.moveaxis(img, 0, -1).astype(np.float32)          # C,H,W -> H,W,C
        nan = np.isnan(img)
        img[nan] = np.broadcast_to(self.mean, img.shape)[nan]       # impute band mean
        img = self.standardization(self.transform(img))
        return img, torch.tensor(self.y[i])


def label_weights(y_train, c):
    pos = y_train.sum(0)
    pos_weight = (len(y_train) - pos) / pos            # neg / pos per class, train patches
    return pos, pos_weight, 1 / np.log(c + 1 / pos_weight)


###############################################################
# Metrics (reference utils/metrics.py Evaluation_ML)
###############################################################
def evaluation_ml(y_pred, y_true):
    return {
        "macroPrec": precision_score(y_true, y_pred, average="macro"),
        "microPrec": precision_score(y_true, y_pred, average="micro"),
        "samplePrec": precision_score(y_true, y_pred, average="samples"),
        "macroRec": recall_score(y_true, y_pred, average="macro"),
        "microRec": recall_score(y_true, y_pred, average="micro"),
        "sampleRec": recall_score(y_true, y_pred, average="samples"),
        "macroF1": f1_score(y_true, y_pred, average="macro"),
        "microF1": f1_score(y_true, y_pred, average="micro"),
        "sampleF1": f1_score(y_true, y_pred, average="samples"),
        "HammingLoss": hamming_loss(y_true, y_pred),
        "subsetAcc": accuracy_score(y_true, y_pred),
        "coverageError": coverage_error(y_true, y_pred),   # reference passes the 0/1 predictions here
        "rankLoss": label_ranking_loss(y_true, y_pred),
    }


def predict(model, loader, criterion, device):
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for img, y in loader:
            img, y = img.to(device), y.to(device)
            logits = model(img)
            loss_sum += criterion(logits, y).item() * y.shape[0]
            n += y.shape[0]
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), loss_sum / n


###############################################################
# Commands
###############################################################
def setup(dataset):
    out = os.path.join(REPO, "outputs", "resnet_paper", dataset)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out, "log.txt"), filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return out


def labels_and_weights(dataset, out):
    data = {s: patch_labels(dataset, s) for s in ("train", "val", "test")}
    pos, pw, w = label_weights(data["train"][1], OPTIONS["weight_param"])
    info = {"train_patches_with_class": dict(zip(CLASS_NAMES, pos.tolist())),
            "neg_over_pos": dict(zip(CLASS_NAMES, pw.round(4).tolist())),
            "bce_pos_weight": dict(zip(CLASS_NAMES, w.round(4).tolist())),
            "patches": {s: len(v[0]) for s, v in data.items()},
            "patches_with_no_label": {s: int((v[1].sum(1) == 0).sum()) for s, v in data.items()},
            "label_counts": {s: dict(zip(CLASS_NAMES, v[1].sum(0).astype(int).tolist())) for s, v in data.items()}}
    json.dump(info, open(os.path.join(out, "labels_and_weights.json"), "w"), indent=2)
    return data, torch.tensor(w, dtype=torch.float32), info


def cmd_check(args):
    out = setup(args.dataset)
    data, w, info = labels_and_weights(args.dataset, out)
    logging.info(json.dumps(info, indent=2))
    device = torch.device("cuda")
    seed_all(OPTIONS["seed"])
    model = ResNet(11, 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=OPTIONS["lr"], weight_decay=OPTIONS["decay"])
    crit = nn.BCEWithLogitsLoss(pos_weight=w.to(device))
    ds = PatchDataset(args.dataset, *data["train"], train=True)
    loader = DataLoader(ds, batch_size=OPTIONS["batch"], shuffle=True)
    torch.cuda.reset_peak_memory_stats()
    t0, tl, n = time.time(), 0.0, 0
    it = iter(loader)
    for b in range(args.batches):
        t1 = time.time()
        img, y = next(it)
        tl += time.time() - t1
        img, y = img.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(img), y)
        loss.backward()
        opt.step()
        n += 1
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    nb = len(loader)
    logging.info(f"check {args.dataset}: {n} batches, {dt:.2f} s/batch (data loading {tl / n:.2f} s), "
                 f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; "
                 f"{nb} batches/epoch -> ~{dt * nb / 60:.1f} min/epoch train")


def cmd_train(args):
    out = setup(args.dataset)
    logging.info(f"options: {json.dumps(OPTIONS)}")
    data, w, info = labels_and_weights(args.dataset, out)
    logging.info(json.dumps(info))
    seed_all(OPTIONS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PatchDataset(args.dataset, *data["train"], train=True),
                              batch_size=OPTIONS["batch"], shuffle=True)
    val_loader = DataLoader(PatchDataset(args.dataset, *data["val"], train=False),
                            batch_size=OPTIONS["batch"], shuffle=False)
    model = ResNet(11, 4).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=OPTIONS["lr"], weight_decay=OPTIONS["decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, OPTIONS["lr_steps"], gamma=OPTIONS["gamma"])
    hist, best = [], np.inf
    for epoch in range(1, OPTIONS["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for img, y in train_loader:
            img, y = img.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(img), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.shape[0]
            n += y.shape[0]
        probs, yv, vloss = predict(model, val_loader, crit, device)
        acc = evaluation_ml((probs >= OPTIONS["threshold"]).astype(np.float32), yv)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot / n, "val_loss": vloss,
               **{f"val_{k}": v for k, v in acc.items()}, "seconds": time.time() - t0}
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        logging.info(f"epoch {epoch}: train loss {row['train_loss']:.4f} val loss {vloss:.4f} "
                     f"val macroF1 {acc['macroF1']:.4f} ({row['seconds']:.0f} s)")
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(out, "model_best_val_loss.pth"))
            json.dump({"epoch": epoch, "val_loss": vloss}, open(os.path.join(out, "best_epoch.json"), "w"))
        sched.step()
    torch.save(model.state_dict(), os.path.join(out, "model_last.pth"))
    logging.info("training done")


def classwise_ml(yt, yp):
    """Per-class table for multi-label predictions, from the per-class 2x2 confusion matrices."""
    rows = {}
    for cm, c in zip(multilabel_confusion_matrix(yt, yp), CLASS_NAMES):
        (tn, fp), (fn, tp) = cm
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        rows[c] = {"support": tp + fn, "predicted": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                   "accuracy": (tp + tn) / cm.sum(), "precision": prec, "recall": rec, "specificity": spec,
                   "balanced_acc": (rec + spec) / 2, "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return pd.DataFrame(rows).T


def cmd_test(args):
    out = setup(args.dataset)
    data, w, _ = labels_and_weights(args.dataset, out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = nn.BCEWithLogitsLoss(pos_weight=w.to(device))
    best = json.load(open(os.path.join(out, "best_epoch.json")))
    lines = [f"# ResNet-50 multi-label patch classification, {args.dataset} (4 classes), test split", "",
             f"Selected epoch (lowest val loss): {best['epoch']} (val loss {best['val_loss']:.4f}). "
             f"Threshold {OPTIONS['threshold']}.", ""]
    for name in ("model_best_val_loss", "model_last"):
        model = ResNet(11, 4).to(device)
        model.load_state_dict(torch.load(os.path.join(out, name + ".pth"), map_location=device))
        loader = DataLoader(PatchDataset(args.dataset, *data["test"], train=False),
                            batch_size=OPTIONS["eval_batch"], shuffle=False)
        probs, yt, tloss = predict(model, loader, crit, device)
        yp = (probs >= OPTIONS["threshold"]).astype(np.float32)
        acc = evaluation_ml(yp, yt)
        json.dump({f"S2_{p}.tif" if args.dataset == "MARIDA" else p: v.tolist()
                   for p, v in zip(data["test"][0], yp)}, open(os.path.join(out, f"test_predictions_{name}.json"), "w"))
        pd.DataFrame([acc]).to_csv(os.path.join(out, f"test_metrics_{name}.csv"), index=False)
        lines += [f"## {name}", "", f"Test loss {tloss:.4f}", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in acc.items()]
        cw = classwise_ml(yt, yp)
        cw.to_csv(os.path.join(out, f"classwise_{name}.csv"), index_label="class")
        lines += ["", "### Per class (each class is a yes/no 'present in the patch' decision)", "",
                  "| class | patches with class | predicted present | TP | FP | FN | TN | accuracy | precision | "
                  "recall | specificity | balanced acc | F1 |", "|---" * 14 + "|"]
        for c, r in cw.iterrows():
            lines.append(f"| {c} | " + " | ".join(f"{int(r[k]):,}" for k in ("support", "predicted", "TP", "FP", "FN", "TN"))
                         + " | " + " | ".join(f"{r[k]:.4f}" for k in ("accuracy", "precision", "recall", "specificity",
                                                                      "balanced_acc", "F1")) + " |")
        lines += ["", "accuracy = (TP + TN) / test patches; recall = share of patches containing the class that were "
                  "flagged; specificity = share of patches without it that were left unflagged; balanced acc = "
                  "(recall + specificity) / 2, which is not inflated when the class is rare.", "",
                  "```", classification_report(yt, yp, target_names=CLASS_NAMES, digits=4, zero_division=0), "```", ""]
    open(os.path.join(out, "test_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    logging.info("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "train", "test"])
    ap.add_argument("--dataset", choices=["MARIDA", "MADOS"], required=True)
    ap.add_argument("--batches", type=int, default=6, help="check: number of training batches to time")
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "test": cmd_test}[a.command](a)
