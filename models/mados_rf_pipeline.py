import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import rasterio
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import joblib
from joblib import Parallel, delayed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import MADOS

def extract_glcm_features(gray_img, window_size=13):
    # Quantize to 16 bins
    gray_quant = np.digitize(gray_img, bins=np.linspace(0, np.max(gray_img), 16)) - 1
    gray_quant = np.clip(gray_quant, 0, 15).astype(np.uint8)
    
    # Pad image
    pad = window_size // 2
    padded = np.pad(gray_quant, pad_width=pad, mode='reflect')
    
    h, w = gray_img.shape
    features = np.zeros((h, w, 6))
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size]
            glcm = graycomatrix(window, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            features[i, j, 0] = graycoprops(glcm, 'contrast')[0, 0]
            features[i, j, 1] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features[i, j, 2] = graycoprops(glcm, 'homogeneity')[0, 0]
            features[i, j, 3] = graycoprops(glcm, 'energy')[0, 0]
            features[i, j, 4] = graycoprops(glcm, 'correlation')[0, 0]
            features[i, j, 5] = graycoprops(glcm, 'ASM')[0, 0]
            
    return features

def process_single_patch_data(im, im_cl):
    # im is loaded by MADOS dataset as (11, 256, 256)
    im = np.moveaxis(im, 0, -1) # Now (256, 256, 11)
    
    mask = im_cl > -1 # MADOS dataset subtracts 1, so background (0) is now -1
    if not np.any(mask):
        return None
        
    # Calculate Spectral Indices for MADOS
    # Sorted order of MADOS bands: 413 (0), 492 (1), 559 (2), 665 (3), 704 (4), 739 (5), 782 (6), 833 (7), 864 (8), 1614 (9), 2202 (10)
    
    # NDVI = (NIR - Red) / (NIR + Red)
    # NIR is 833 (index 7), Red is 665 (index 3)
    ndvi = (im[:, :, 7] - im[:, :, 3]) / (im[:, :, 7] + im[:, :, 3] + 1e-8)
    ndwi = (im[:, :, 2] - im[:, :, 7]) / (im[:, :, 2] + im[:, :, 7] + 1e-8)
    fai = im[:, :, 7] - (im[:, :, 3] + (im[:, :, 9] - im[:, :, 3]) * ((833 - 665) / (1614 - 665)))
    fdi = im[:, :, 7] - (im[:, :, 4] + (im[:, :, 9] - im[:, :, 4]) * ((833 - 704) / (1614 - 704)))
    
    indices = np.stack([ndvi, ndwi, fai, fdi], axis=-1)
    
    # Grayscale image (approx) from RGB (indices 3, 2, 1) -> 665, 559, 492
    rgb = im[:, :, [3, 2, 1]]
    gray = np.mean(rgb, axis=-1)
    
    glcm_features = extract_glcm_features(gray, window_size=13)
    
    all_features = np.concatenate([im, indices, glcm_features], axis=-1)
    
    X = all_features[mask]
    y = im_cl[mask].flatten()
    
    return X, y

def main(args):
    print("Loading MADOS Train dataset...")
    splits_dir = os.path.join(args.path, 'splits')
    train_dataset = MADOS(args.path, splits_dir, mode='train')
    
    print("Loading MADOS Test dataset...")
    test_dataset = MADOS(args.path, splits_dir, mode='test')
    
    print("Extracting features in parallel for train set (this will take a while)...")
    if args.debug_limit:
        train_X = train_dataset.X[:args.debug_limit]
        train_y = train_dataset.y[:args.debug_limit]
        test_X_data = test_dataset.X[:args.debug_limit]
        test_y = test_dataset.y[:args.debug_limit]
    else:
        train_X = train_dataset.X
        train_y = train_dataset.y
        test_X_data = test_dataset.X
        test_y = test_dataset.y
    
    results_train = Parallel(n_jobs=-1, verbose=10)(delayed(process_single_patch_data)(train_X[i], train_y[i]) for i in range(len(train_X)))
    
    X_train, y_train = [], []
    for res in results_train:
        if res is not None:
            X, y = res
            X_train.append(X)
            y_train.append(y)
            
    print("Extracting features in parallel for test set...")
    results_test = Parallel(n_jobs=-1, verbose=10)(delayed(process_single_patch_data)(test_X_data[i], test_y[i]) for i in range(len(test_X_data)))
    
    X_test, y_test = [], []
    for res in results_test:
        if res is not None:
            X, y = res
            X_test.append(X)
            y_test.append(y)
            
    if not X_train:
        print("No training data found in this subset.")
        return
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    if not X_test:
        print("No test data found in this subset, evaluating on training data for debug purposes...")
        X_test = X_train
        y_test = y_train
    else:
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        
    print(f"Training RandomForest on {X_train.shape[0]} pixels with {X_train.shape[1]} features...")
    clf = RandomForestClassifier(n_estimators=125, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = clf.predict(X_test)
    
    print("Metrics:")
    print("Precision:", precision_score(y_test, preds, average='macro'))
    print("Recall:", recall_score(y_test, preds, average='macro'))
    print("F1 Score:", f1_score(y_test, preds, average='macro'))
    print("IoU (Jaccard):", jaccard_score(y_test, preds, average='macro'))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nSaving model to mados_rf_model.joblib...")
    joblib.dump(clf, os.path.join(args.path, 'mados_rf_model.joblib'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Path to MADOS dataset (e.g. MADOS_5Class)')
    parser.add_argument('--debug_limit', type=int, default=0, help='Limit number of patches for debugging')
    args = parser.parse_args()
    main(args)
