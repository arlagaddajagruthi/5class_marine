cd /c/PROJECT/5class_marine_pollution
L=outputs/staged_valselect/logs
V="python -u scripts/staged_valselect/valselect.py"
$V worker > $L/worker_main.log 2>&1 &
P1=$!
$V worker --heavy-only > $L/worker_heavy.log 2>&1 &
P2=$!
wait $P1; echo "worker_main exit $?" >> $L/status.txt
wait $P2; echo "worker_heavy exit $?" >> $L/status.txt
$V decide > $L/decide.log 2>&1; echo "decide exit $?" >> $L/status.txt
$V worker > $L/worker_phase2.log 2>&1; echo "worker_phase2 exit $?" >> $L/status.txt
$V report > $L/report.log 2>&1; echo "report exit $?" >> $L/status.txt
