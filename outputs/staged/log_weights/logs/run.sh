cd /c/PROJECT/5class_marine_pollution
R=outputs/staged/log_weights
for s in stage1_bands stage2_single_index stage3_all_indices stage4_indices_ratios; do
  python -u scripts/staged_ml/$s.py --weighting log --out-root $R > $R/logs/$s.log 2>&1
  echo "$s exit $?" >> $R/logs/status.txt
done
python -u scripts/staged_ml/compare_stages.py --out-root $R > $R/logs/compare.log 2>&1
echo "compare exit $?" >> $R/logs/status.txt
