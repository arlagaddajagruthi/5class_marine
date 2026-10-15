# Verification diagnostics (MARIDA)

## C. Rainfall before scenes with and without labelled debris (ERA5, 7 days before the image)

| Region | Group | scenes | median 7-day rain (mm) | median 72 h rain (mm) |
|---|---|---|---|---|
| Honduras/Guatemala tiles (16P*, 16Q*) | with debris | 29 | 29.3 | 11.4 |
| Honduras/Guatemala tiles (16P*, 16Q*) | without debris | 4 | 9.8 | 2.0 |
| all MARIDA | with debris | 55 | 24.4 | 7.5 |
| all MARIDA | without debris | 8 | 6.9 | 3.3 |

## B. LightGBM, stage 5 features: unweighted vs log weights

| Weighting | train F1 | test F1 | train logloss at iter 1 / 10 / 50 / 100 (min, at iter) | predicted test pixels per class (water, debris, algae, other) |
|---|---|---|---|---|
| none | 0.9283 | 0.7449 | 0.0460 / 0.0233 / 0.0053 / 0.0694 (min 0.0052 at 49) | [153639, 1094, 1699, 1939] |
| log | 0.9856 | 0.8747 | 0.6983 / 0.2096 / 0.0261 / 0.0078 (min 0.0078 at 100) | [154089, 585, 1647, 2050] |

True test pixels per class: [154739, 381, 1641, 1610]

## A1. Can the scene be identified from the environmental features?

- Patches: 1381 in 63 scenes; distinct environmental vectors: 1381.
- Random forest predicting the scene from the 7 env features (5-fold, patches of scenes with ≥5 patches, 1354 patches / 54 scenes): accuracy **0.935** (chance ≈ 0.019).

## A2. Adding env features under two cross-validation schemes (MARIDA train split, log weights)

| CV scheme | Model | stage 5 F1 | stage 6 F1 | Δ env | Δ env (debris F1) |
|---|---|---|---|---|---|
| random pixel folds (scenes seen) | rf | 0.9602 | 0.9815 | +0.0212 | +0.0275 |
| random pixel folds (scenes seen) | lgbm | 0.9618 | 0.9839 | +0.0221 | +0.0295 |
| scene-grouped folds (scenes unseen) | rf | 0.7446 | 0.6834 | -0.0613 | +0.1077 |
| scene-grouped folds (scenes unseen) | lgbm | 0.6609 | 0.5389 | -0.1220 | -0.1026 |

