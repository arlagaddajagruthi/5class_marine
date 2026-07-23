import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'mados-master', 'utils'))
from dataset import MADOS, MARIDADataset

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv2 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        
        logits = self.outc(x)
        return logits

def train_and_eval(dataset_name, dataset_path, batch_size=2, epochs=1):
    print(f"\n{'='*50}\nStarting New U-Net Pipeline for {dataset_name}\n{'='*50}")
    splits_path = os.path.join(dataset_path, 'splits')
    
    if "MARIDA" in dataset_name:
        train_ds = MARIDADataset(dataset_path, splits_path, 'train')
        val_ds = MARIDADataset(dataset_path, splits_path, 'val')
    else:
        train_ds = MADOS(dataset_path, splits_path, 'train')
        val_ds = MADOS(dataset_path, splits_path, 'val')
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cpu')
    model = SimpleUNet(11, 5).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for images, targets in tqdm(train_loader, desc="Training"):
            images, targets = images.to(device), targets.long().to(device)
            optimizer.zero_grad()
            logits = model(images)
            
            # Ensure logits match target size
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear')
                
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear')
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = targets.cpu().numpy()
            
            mask = targets != -1
            all_preds.extend(preds[mask])
            all_targets.extend(targets[mask])
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\nMetrics:")
    print("Precision:", precision_score(all_targets, all_preds, average='macro', zero_division=0))
    print("Recall:", recall_score(all_targets, all_preds, average='macro', zero_division=0))
    print("F1 Score:", f1_score(all_targets, all_preds, average='macro', zero_division=0))
    print("IoU (Jaccard):", jaccard_score(all_targets, all_preds, average='macro', zero_division=0))
    
    print("\nDetailed Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    with open(f"{dataset_name}_unet_results.txt", "w") as f:
        f.write(f"Results for {dataset_name}\n")
        f.write(f"Precision: {precision_score(all_targets, all_preds, average='macro', zero_division=0)}\n")
        f.write(f"Recall: {recall_score(all_targets, all_preds, average='macro', zero_division=0)}\n")
        f.write(f"F1 Score: {f1_score(all_targets, all_preds, average='macro', zero_division=0)}\n")
        f.write(f"IoU (Jaccard): {jaccard_score(all_targets, all_preds, average='macro', zero_division=0)}\n")
        f.write("\nDetailed Confusion Matrix:\n")
        f.write(str(confusion_matrix(all_targets, all_preds)) + "\n")
        
    print(f"Results saved to {dataset_name}_unet_results.txt\n")

if __name__ == "__main__":
    datasets = [
        ("MARIDA_5Class", r"c:\Users\CB.SC.U4CSE23709\Desktop\Marine Datasets\MARIDA_5Class")
    ]
    
    for name, path in datasets:
        if os.path.exists(path):
            train_and_eval(name, path)
        else:
            print(f"Path not found for {name}: {path}")
