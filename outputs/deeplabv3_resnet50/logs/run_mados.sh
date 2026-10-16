# DeepLabv3 MADOS train + test. Safe to re-run after a crash: --resume continues from MADOS/train_state.pth.
cd /c/PROJECT/5class_marine_pollution
L=outputs/deeplabv3_resnet50/logs
V="python -u scripts/deeplabv3_resnet50/deeplab.py"
echo "MADOS run start $(date)" >> $L/status.txt
$V train --dataset MADOS --resume >> $L/train_MADOS.log 2>&1; rc=$?; echo "MADOS train exit $rc $(date)" >> $L/status.txt
[ $rc -eq 0 ] || exit $rc
$V test --dataset MADOS > $L/test_MADOS.log 2>&1; echo "MADOS test exit $? $(date)" >> $L/status.txt
