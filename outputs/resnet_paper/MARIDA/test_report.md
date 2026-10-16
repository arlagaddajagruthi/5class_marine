# ResNet-50 multi-label patch classification, MARIDA (4 classes), test split

Selected epoch (lowest val loss): 18 (val loss 0.5038). Threshold 0.5.

## model_best_val_loss

Test loss 0.5284

| metric | value |
|---|---|
| macroPrec | 0.4374 |
| microPrec | 0.7431 |
| samplePrec | 0.7591 |
| macroRec | 0.3940 |
| microRec | 0.6876 |
| sampleRec | 0.6945 |
| macroF1 | 0.4042 |
| microF1 | 0.7143 |
| sampleF1 | 0.6915 |
| HammingLoss | 0.1950 |
| subsetAcc | 0.4067 |
| coverageError | 2.4290 |
| rankLoss | 0.3152 |

```
                 precision    recall  f1-score   support

          Water     0.8517    0.9702    0.9071       302
  Marine Debris     0.4767    0.4059    0.4385       101
Algae/Sargassum     0.0000    0.0000    0.0000        26
          Other     0.4211    0.2000    0.2712        80

      micro avg     0.7431    0.6876    0.7143       509
      macro avg     0.4374    0.3940    0.4042       509
   weighted avg     0.6661    0.6876    0.6678       509
    samples avg     0.7591    0.6945    0.6915       509

```

Per-class confusion (rows: annotated no/yes, columns: predicted no/yes):

- Water: [[6, 51], [9, 293]]
- Marine Debris: [[213, 45], [60, 41]]
- Algae/Sargassum: [[330, 3], [26, 0]]
- Other: [[257, 22], [64, 16]]

## model_last

Test loss 0.5261

| metric | value |
|---|---|
| macroPrec | 0.4312 |
| microPrec | 0.7262 |
| samplePrec | 0.7400 |
| macroRec | 0.4214 |
| microRec | 0.7033 |
| sampleRec | 0.7001 |
| macroF1 | 0.4207 |
| microF1 | 0.7146 |
| sampleF1 | 0.6861 |
| HammingLoss | 0.1992 |
| subsetAcc | 0.4039 |
| coverageError | 2.4290 |
| rankLoss | 0.3187 |

```
                 precision    recall  f1-score   support

          Water     0.8504    0.9603    0.9020       302
  Marine Debris     0.4660    0.4752    0.4706       101
Algae/Sargassum     0.0000    0.0000    0.0000        26
          Other     0.4082    0.2500    0.3101        80

      micro avg     0.7262    0.7033    0.7146       509
      macro avg     0.4312    0.4214    0.4207       509
   weighted avg     0.6612    0.7033    0.6773       509
    samples avg     0.7400    0.7001    0.6861       509

```

Per-class confusion (rows: annotated no/yes, columns: predicted no/yes):

- Water: [[6, 51], [12, 290]]
- Marine Debris: [[203, 55], [53, 48]]
- Algae/Sargassum: [[333, 0], [26, 0]]
- Other: [[250, 29], [60, 20]]
