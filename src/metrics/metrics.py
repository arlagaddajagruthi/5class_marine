import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    jaccard_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, num_classes=5):
    # Overall metrics (macro average)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    iou_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    recall_per = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    iou_per = jaccard_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    return {
        "overall": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
            "iou": iou_macro,
            "accuracy": accuracy
        },
        "per_class": {
            "precision": precision_per,
            "recall": recall_per,
            "f1": f1_per,
            "iou": iou_per
        },
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import pandas as pd

def save_metrics_table(results, class_names, save_path):
    iou_per = results['per_class']['iou']
    recall_per = results['per_class']['recall']
    precision_per = results['per_class']['precision']
    f1_per = results['per_class']['f1']
    
    iou_avg = results['overall']['iou']
    recall_avg = results['overall']['recall']
    precision_avg = results['overall']['precision']
    f1_avg = results['overall']['f1']
    acc_avg = results['overall']['accuracy']
    
    # Calculate per-class accuracy from confusion matrix: (TP + TN) / Total
    cm = results['confusion_matrix']
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (fp + fn + tp)
    acc_per = (tp + tn) / total
    
    data = {
        "Class": class_names + ["Average"],
        "IoU": list(iou_per) + [iou_avg],
        "F1": list(f1_per) + [f1_avg],
        "Precision": list(precision_per) + [precision_avg],
        "Recall (PA)": list(recall_per) + [recall_avg],
        "Accuracy": list(acc_per) + [acc_avg]
    }
    
    df = pd.DataFrame(data)
    
    for col in ["IoU", "F1", "Precision", "Recall (PA)", "Accuracy"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    df.to_csv(save_path + ".csv", index=False)

    with open(save_path + ".md", "w") as f:
        f.write(df.to_markdown(index=False))