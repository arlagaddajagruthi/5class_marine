cd /c/PROJECT/5class_marine_pollution
L=outputs/resnet_paper/logs
V="python -u scripts/resnet_paper/resnet_ml.py"
for D in MARIDA MADOS; do
  $V train --dataset $D > $L/train_$D.log 2>&1; echo "$D train exit $?" >> $L/status.txt
  $V test --dataset $D > $L/test_$D.log 2>&1; echo "$D test exit $?" >> $L/status.txt
done
