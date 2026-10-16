# Waits for the ResNet run (outputs/resnet_paper) to finish, then runs DeepLabv3-ResNet50.
cd /c/PROJECT/5class_marine_pollution
L=outputs/deeplabv3_resnet50/logs
V="python -u scripts/deeplabv3_resnet50/deeplab.py"
echo "waiting for resnet_paper $(date)" >> $L/status.txt
until grep -q "MADOS test exit" outputs/resnet_paper/logs/status.txt 2>/dev/null; do sleep 120; done
echo "start $(date)" >> $L/status.txt
step() { $V $1 --dataset $2 > $L/$1_$2.log 2>&1; rc=$?; echo "$2 $1 exit $rc $(date)" >> $L/status.txt; return $rc; }
step cache MADOS || exit 1
for D in MARIDA MADOS; do
  step check $D || continue
  step train $D && step test $D
done
echo "all done $(date)" >> $L/status.txt
