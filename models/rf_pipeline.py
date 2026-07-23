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
from assets import s2_mapping

def extract_glcm_features(gray_img, window_size=13):
    # Quantize to 16 bins
    gray_quant = np.digitize(gray_img, bins=np.linspace(0, np.max(gray_img), 16)) - 1
    gray_quant = np.clip(gray_quant, 0, 15).astype(np.uint8)
    
    # Pad image
    pad = window_size // 2
    padded = np.pad(gray_quant, pad_width=pad, mode='reflect')
    
    h, w = gray_img.shape
    features = np.zeros((h, w, 6))
    
    # This loop is slow but serves as the explicit baseline requested
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

def process_patch(img_path):
    with rasterio.open(img_path) as ds:
        im = ds.read()
    im = np.moveaxis(im, 0, -1)
    
    # Load class and confidence
    if '_rhorc_' in img_path:
        cl_path = img_path.replace('_rhorc_', '_cl_')
        conf_path = img_path.replace('_rhorc_', '_conf_')
    else:
        cl_path = img_path.replace('.tif', '_cl.tif')
        conf_path = img_path.replace('.tif', '_conf.tif')
    
    with rasterio.open(cl_path) as ds_cl:
        im_cl = ds_cl.read(1)[..., np.newaxis]
    
    with rasterio.open(conf_path) as ds_conf:
        im_conf = ds_conf.read(1)[..., np.newaxis]
    
    # 0 is unlabelled or background, we skip it
    mask = im_cl[:, :, 0] > 0
    
    if not np.any(mask):
        return None, None
    
    # Calculate Spectral Indices
    # Assuming S2 bands: nm490 is index 1, nm560 is index 2, nm665 is index 3, nm842 is index 7
    # NDVI = (NIR - Red) / (NIR + Red) -> (nm842 - nm665) / (nm842 + nm665)
    ndvi = (im[:, :, 7] - im[:, :, 3]) / (im[:, :, 7] + im[:, :, 3] + 1e-8)
    ndwi = (im[:, :, 2] - im[:, :, 7]) / (im[:, :, 2] + im[:, :, 7] + 1e-8)
    fai = im[:, :, 7] - (im[:, :, 3] + (im[:, :, 9] - im[:, :, 3]) * ((842 - 665) / (1600 - 665)))
    fdi = im[:, :, 7] - (im[:, :, 4] + (im[:, :, 9] - im[:, :, 4]) * ((842 - 705) / (1600 - 705)))
    
    indices = np.stack([ndvi, ndwi, fai, fdi], axis=-1)
    
    # Grayscale image (approx) from RGB (indices 3, 2, 1)
    rgb = im[:, :, [3, 2, 1]]
    gray = np.mean(rgb, axis=-1)
    
    # Calculate GLCM features
    glcm_features = extract_glcm_features(gray, window_size=13)
    
    # Concatenate features
    all_features = np.concatenate([im, indices, glcm_features], axis=-1)
    
    # Flatten and mask
    X = all_features[mask]
    y = im_cl[mask].flatten()
    conf = im_conf[mask].flatten()
    
    return X, y

def process_single_patch(patch, X_train_list, X_test_list):
    basename = os.path.basename(patch).split('.tif')[0]
    if 'rhorc' in basename:
        splited_name = basename.split('_')
        img_name = '_'.join(splited_name[:-3]) + '_' + splited_name[-1]
    else:
        img_name = basename.replace('S2_', '')
    
    X, y = process_patch(patch)
    if X is not None:
        if img_name in X_train_list:
            return ('train', X, y)
        elif img_name in X_test_list:
            return ('test', X, y)
    return None

def main(args):
    print("Gathering patches...")
    patches = glob(os.path.join(args.path, '**', '*.tif'), recursive=True)
    patches = [p for p in patches if '_cl.' not in p and '_conf.' not in p and '_rep.' not in p and '_cl_' not in p and '_conf_' not in p and '_rep_' not in p]
    
    root_path = os.path.dirname(args.path)
    if not root_path:
        root_path = args.path
    
    X_train_list = np.genfromtxt(os.path.join(root_path, 'splits', 'train_X.txt'), dtype='str')
    X_test_list = np.genfromtxt(os.path.join(root_path, 'splits', 'test_X.txt'), dtype='str')
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("Extracting features in parallel (this will be much faster!)...")
    patch_list = patches[:args.debug_limit] if args.debug_limit else patches
    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_single_patch)(patch, X_train_list, X_test_list) for patch in patch_list)
    
    for res in results:
        if res is not None:
            split, X, y = res
            if split == 'train':
                X_train.append(X)
                y_train.append(y)
            elif split == 'test':
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
    
    print("\nSaving model to rf_model.joblib...")
    joblib.dump(clf, os.path.join(root_path, 'rf_model.joblib'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Path to dataset patches (e.g. MARIDA_5Class/patches)')
    parser.add_argument('--debug_limit', type=int, default=0, help='Limit number of patches for debugging')
    args = parser.parse_args()
    main(args)
