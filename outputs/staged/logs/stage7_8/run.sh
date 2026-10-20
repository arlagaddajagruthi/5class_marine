# Stage 7 (bands + GLCM) for all three class weightings; stage 8 (bands + NDWI + ratios + GLCM)
# only if glcm_decision.py says GLCM adds something. Re-runnable: finished stage folders are redone
# (each takes ~20 min), GLCM caches are reused.
cd /c/PROJECT/5class_marine_pollution
L=outputs/staged/logs/stage7_8
st () { echo "$1 $(date)" >> $L/status.txt; }
st "start"
python -u scripts/staged_ml/glcm_features.py > $L/glcm_features.log 2>&1 || { st "glcm_features FAILED"; exit 1; }
st "glcm caches done"
for w in balanced log none; do
  case $w in balanced) R=outputs/staged; F="--weighting balanced";;
             log) R=outputs/staged/log_weights; F="--weighting log";;
             none) R=outputs/staged/no_class_weights; F="--no-class-weights";; esac
  python -u scripts/staged_ml/stage7_glcm.py $F --out-root $R > $L/stage7_$w.log 2>&1 || { st "stage7 $w FAILED"; exit 1; }
  st "stage7 $w done"
done
python -u scripts/staged_ml/glcm_decision.py > $L/decision.log 2>&1; d=$?   # 0 = run stage 8, 2 = skip
if [ $d -eq 0 ]; then
  st "decision: run stage 8"
  for w in balanced log none; do
    case $w in balanced) R=outputs/staged; F="--weighting balanced";;
               log) R=outputs/staged/log_weights; F="--weighting log";;
               none) R=outputs/staged/no_class_weights; F="--no-class-weights";; esac
    python -u scripts/staged_ml/stage8_ndwi_ratios_glcm.py $F --out-root $R > $L/stage8_$w.log 2>&1 || { st "stage8 $w FAILED"; exit 1; }
    st "stage8 $w done"
  done
elif [ $d -eq 2 ]; then st "decision: GLCM adds nothing, stage 8 skipped"
else st "decision script FAILED"; exit 1; fi
for w in balanced log none; do
  case $w in balanced) R=outputs/staged;; log) R=outputs/staged/log_weights;; none) R=outputs/staged/no_class_weights;; esac
  python -u scripts/staged_ml/compare_stages.py --out-root $R > $L/compare_$w.log 2>&1
done
st "ALL DONE"
