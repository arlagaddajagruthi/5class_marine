# Does GLCM add anything? (stage 7 vs stage 1, mean of 5 models)

Rule: run stage 8 if any delta > 0.01 (any weighting, any dataset, macro or debris F1).

| weighting   | dataset   |   stage1_macroF1 |   stage7_macroF1 |   delta_macroF1 |   stage1_debrisF1 |   stage7_debrisF1 |   delta_debrisF1 |
|:------------|:----------|-----------------:|-----------------:|----------------:|------------------:|------------------:|-----------------:|
| balanced    | MARIDA    |           0.7284 |           0.7384 |          0.0099 |            0.4076 |            0.4011 |          -0.0066 |
| balanced    | MADOS     |           0.5919 |           0.6384 |          0.0464 |            0.3228 |            0.3216 |          -0.0011 |
| log         | MARIDA    |           0.7942 |           0.7897 |         -0.0045 |            0.528  |            0.5008 |          -0.0273 |
| log         | MADOS     |           0.6396 |           0.6778 |          0.0382 |            0.4203 |            0.3933 |          -0.0271 |
| none        | MARIDA    |           0.7637 |           0.746  |         -0.0178 |            0.4286 |            0.338  |          -0.0906 |
| none        | MADOS     |           0.6315 |           0.6713 |          0.0398 |            0.3944 |            0.3794 |          -0.015  |

Decision: **run stage 8**
