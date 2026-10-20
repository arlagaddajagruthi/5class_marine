# ResNet-50 multi-label patch classification, MADOS (4 classes), test split

Selected epoch (lowest val loss): 10 (val loss 0.4887). Threshold 0.5.

## model_best_val_loss

Test loss 0.5205

| metric | value |
|---|---|
| macroPrec | 0.5766 |
| microPrec | 0.6932 |
| samplePrec | 0.7017 |
| macroRec | 0.4965 |
| microRec | 0.6822 |
| sampleRec | 0.7001 |
| macroF1 | 0.5110 |
| microF1 | 0.6877 |
| sampleF1 | 0.6740 |
| HammingLoss | 0.2009 |
| subsetAcc | 0.4492 |
| coverageError | 2.4176 |
| rankLoss | 0.3594 |

### Per class (each class is a yes/no 'present in the patch' decision)

| class | patches with class | predicted present | TP | FP | FN | TN | accuracy | precision | recall | specificity | balanced acc | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 469 | 566 | 412 | 154 | 57 | 105 | 0.7102 | 0.7279 | 0.8785 | 0.4054 | 0.6419 | 0.7961 |
| Marine Debris | 124 | 117 | 57 | 60 | 67 | 544 | 0.8255 | 0.4872 | 0.4597 | 0.9007 | 0.6802 | 0.4730 |
| Algae/Sargassum | 45 | 11 | 4 | 7 | 41 | 676 | 0.9341 | 0.3636 | 0.0889 | 0.9898 | 0.5393 | 0.1429 |
| Other | 306 | 235 | 171 | 64 | 135 | 358 | 0.7266 | 0.7277 | 0.5588 | 0.8483 | 0.7036 | 0.6322 |

accuracy = (TP + TN) / test patches; recall = share of patches containing the class that were flagged; specificity = share of patches without it that were left unflagged; balanced acc = (recall + specificity) / 2, which is not inflated when the class is rare.

```
                 precision    recall  f1-score   support

          Water     0.7279    0.8785    0.7961       469
  Marine Debris     0.4872    0.4597    0.4730       124
Algae/Sargassum     0.3636    0.0889    0.1429        45
          Other     0.7277    0.5588    0.6322       306

      micro avg     0.6932    0.6822    0.6877       944
      macro avg     0.5766    0.4965    0.5110       944
   weighted avg     0.6788    0.6822    0.6694       944
    samples avg     0.7017    0.7001    0.6740       944

```

## model_last

Test loss 0.5195

| metric | value |
|---|---|
| macroPrec | 0.5396 |
| microPrec | 0.6916 |
| samplePrec | 0.7081 |
| macroRec | 0.5103 |
| microRec | 0.6864 |
| sampleRec | 0.7056 |
| macroF1 | 0.5153 |
| microF1 | 0.6890 |
| sampleF1 | 0.6812 |
| HammingLoss | 0.2009 |
| subsetAcc | 0.4615 |
| coverageError | 2.3915 |
| rankLoss | 0.3530 |

### Per class (each class is a yes/no 'present in the patch' decision)

| class | patches with class | predicted present | TP | FP | FN | TN | accuracy | precision | recall | specificity | balanced acc | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 469 | 519 | 396 | 123 | 73 | 136 | 0.7308 | 0.7630 | 0.8443 | 0.5251 | 0.6847 | 0.8016 |
| Marine Debris | 124 | 133 | 62 | 71 | 62 | 533 | 0.8173 | 0.4662 | 0.5000 | 0.8825 | 0.6912 | 0.4825 |
| Algae/Sargassum | 45 | 17 | 4 | 13 | 41 | 670 | 0.9258 | 0.2353 | 0.0889 | 0.9810 | 0.5349 | 0.1290 |
| Other | 306 | 268 | 186 | 82 | 120 | 340 | 0.7225 | 0.6940 | 0.6078 | 0.8057 | 0.7068 | 0.6481 |

accuracy = (TP + TN) / test patches; recall = share of patches containing the class that were flagged; specificity = share of patches without it that were left unflagged; balanced acc = (recall + specificity) / 2, which is not inflated when the class is rare.

```
                 precision    recall  f1-score   support

          Water     0.7630    0.8443    0.8016       469
  Marine Debris     0.4662    0.5000    0.4825       124
Algae/Sargassum     0.2353    0.0889    0.1290        45
          Other     0.6940    0.6078    0.6481       306

      micro avg     0.6916    0.6864    0.6890       944
      macro avg     0.5396    0.5103    0.5153       944
   weighted avg     0.6765    0.6864    0.6779       944
    samples avg     0.7081    0.7056    0.6812       944

```
