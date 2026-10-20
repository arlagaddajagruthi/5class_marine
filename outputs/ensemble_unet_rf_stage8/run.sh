cd /c/PROJECT/5class_marine_pollution
export PYTHONIOENCODING=utf-8
python scripts/ensemble/unet_rf.py --datasets MARIDA MADOS --rf-stage 8 > outputs/ensemble_unet_rf_stage8/run.log 2>&1 && echo "ALL DONE" >> outputs/ensemble_unet_rf_stage8/run.log || echo "FAILED" >> outputs/ensemble_unet_rf_stage8/run.log
