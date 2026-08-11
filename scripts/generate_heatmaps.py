import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

CLASS_NAMES = ['Marine Debris', 'Sargassum/\nVeg', 'Natural\nPhenom/Foam', 'Ships/\nInfrastr', 'Water/\nOther']

cms = {
    'MADOS_5Class_XGBoost': np.array([
        [99958, 17, 0, 0, 25],
        [0, 78716, 293, 22, 20969],
        [0, 73, 638, 6, 256],
        [0, 800, 6, 2075, 208],
        [0, 6905, 256, 19, 55885]
    ]),
    'MARIDA_5Class_XGBoost': np.array([
        [92228, 7105, 103, 125, 439],
        [36187, 63416, 206, 10, 181],
        [51, 6, 303, 1, 20],
        [56, 0, 2, 1538, 45],
        [209, 55, 74, 1, 1271]
    ]),
    'MADOS_5Class_ResNet18': np.array([
        [41571653, 0, 0, 0, 0],
        [294020, 0, 0, 0, 0],
        [973, 0, 0, 0, 0],
        [3089, 0, 0, 0, 0],
        [63065, 0, 0, 0, 0]
    ]),
    'MARIDA_5Class_ResNet18': np.array([
        [23369053, 0, 0, 0, 0],
        [154739, 0, 0, 0, 0],
        [381, 0, 0, 0, 0],
        [1641, 0, 0, 0, 0],
        [1610, 0, 0, 0, 0]
    ]),
}

out_dir = r'c:\Users\cb.sc.u4cse23709\.gemini\antigravity\brain\9062dba9-f340-4526-a41d-7da210f2cf66'
os.makedirs(out_dir, exist_ok=True)

for name, cm in cms.items():
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    title = name.replace('_', ' ')
    fig.suptitle(title + ' - Confusion Matrix', fontsize=15, fontweight='bold', y=1.01)

    sns.heatmap(cm, ax=axes[0], annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar=True)
    axes[0].set_title('Raw Pixel Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontsize=10)
    axes[0].set_ylabel('Actual Class', fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)

    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar=True, vmin=0, vmax=100)
    axes[1].set_title('Recall-Normalized (%)', fontweight='bold')
    axes[1].set_xlabel('Predicted Class', fontsize=10)
    axes[1].set_ylabel('Actual Class', fontsize=10)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    out_path = os.path.join(out_dir, name + '_confusion_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: ' + out_path)

print('All heatmaps generated!')
