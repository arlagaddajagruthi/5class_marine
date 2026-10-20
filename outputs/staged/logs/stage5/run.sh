cd /c/PROJECT/5class_marine_pollution
L=outputs/staged/logs/stage5
for spec in "balanced outputs/staged" "log outputs/staged/log_weights" "none outputs/staged/no_class_weights"; do
  set -- $spec
  python -u scripts/staged_ml/stage5_ndwi_ratios.py --weighting $1 --out-root $2 > $L/stage5_$1.log 2>&1
  echo "stage5 $1 exit $?" >> $L/status.txt
  python -u scripts/staged_ml/compare_stages.py --out-root $2 > $L/compare_$1.log 2>&1
  echo "compare $1 exit $?" >> $L/status.txt
done
