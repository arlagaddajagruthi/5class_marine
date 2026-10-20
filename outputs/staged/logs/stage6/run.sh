cd /c/PROJECT/5class_marine_pollution
L=outputs/staged/logs/stage6
F=data/env_sst_chl_wind/logs/fetch_cmems.log
until grep -q "^Done;" $F; do sleep 30; done
if ! grep -q "^Done; 0 failed" $F; then   # retry only the missing files once
  python -u scripts/env_sst_chl_wind/fetch_cmems.py > data/env_sst_chl_wind/logs/fetch_cmems_retry.log 2>&1
  echo "fetch retry exit $?" >> $L/status.txt
fi
python -u scripts/env_sst_chl_wind/build_env_table.py > $L/build_env_table.log 2>&1
echo "build_env_table exit $?" >> $L/status.txt
for spec in "balanced outputs/staged" "log outputs/staged/log_weights" "none outputs/staged/no_class_weights"; do
  set -- $spec
  python -u scripts/staged_ml/stage6_env.py --weighting $1 --out-root $2 > $L/stage6_$1.log 2>&1
  echo "stage6 $1 exit $?" >> $L/status.txt
  python -u scripts/staged_ml/compare_stages.py --out-root $2 > $L/compare_$1.log 2>&1
  echo "compare $1 exit $?" >> $L/status.txt
done
