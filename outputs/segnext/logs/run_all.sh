# SegNeXt-T (from scratch): MARIDA without VSCP, MARIDA with VSCP, MADOS with VSCP; train + test each.
# Safe to re-run after a crash: finished runs (test_report.md present) are skipped, unfinished ones resume.
cd /c/PROJECT/5class_marine_pollution
L=outputs/segnext/logs
U="python -u scripts/segnext/segnext.py"
run () {  # dataset run vscp-flag
  [ -f outputs/segnext/$2/test_report.md ] && { echo "$2 already done $(date)" >> $L/status.txt; return 0; }
  echo "$2 start $(date)" >> $L/status.txt
  if [ -f outputs/segnext/$2/model_last.pth ]; then rc=0; else
    $U train --dataset $1 --run $2 $3 --resume >> $L/train_$2.log 2>&1; rc=$?; echo "$2 train exit $rc $(date)" >> $L/status.txt
  fi
  [ $rc -eq 0 ] || return $rc
  $U test --dataset $1 --run $2 $3 > $L/test_$2.log 2>&1; rc=$?; echo "$2 test exit $rc $(date)" >> $L/status.txt
  return $rc
}
run MARIDA MARIDA_novscp --no-vscp && run MARIDA MARIDA_vscp --vscp && run MADOS MADOS_vscp --vscp \
  && echo "ALL DONE $(date)" >> $L/status.txt || echo "STOPPED WITH AN ERROR $(date)" >> $L/status.txt
