# Waits for the MADOS training process (PID 21172) to exit, then runs the MADOS test and the two ResNet tests.
cd /c/PROJECT/5class_marine_pollution
L=outputs/deeplabv3_resnet50/logs
RL=outputs/resnet_paper/logs
echo "after_mados waiting for PID 21172 $(date)" >> $L/status.txt
while tasklist //FI "PID eq 21172" 2>/dev/null | grep -q 21172; do sleep 60; done
if [ ! -f outputs/deeplabv3_resnet50/MADOS/model_last.pth ]; then
  echo "MADOS training process ended WITHOUT finishing (no model_last.pth) $(date); tests not run" >> $L/status.txt
  exit 1
fi
echo "MADOS train finished $(date)" >> $L/status.txt
python -u scripts/deeplabv3_resnet50/deeplab.py test --dataset MADOS > $L/test_MADOS.log 2>&1
echo "MADOS test exit $? $(date)" >> $L/status.txt
for D in MARIDA MADOS; do
  python -u scripts/resnet_paper/resnet_ml.py test --dataset $D > $RL/test_${D}_classwise.log 2>&1
  echo "resnet $D test exit $? $(date)" >> $L/status.txt
done
echo "after_mados all done $(date)" >> $L/status.txt
