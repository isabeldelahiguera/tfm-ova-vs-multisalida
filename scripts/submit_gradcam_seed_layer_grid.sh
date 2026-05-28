#!/bin/bash
set -euo pipefail

# Submit a compact Grad-CAM robustness grid:
# - several seeds;
# - several VGG convolutional layers;
# - CSV-only by default to avoid generating thousands of PNG panels.

DATASET="${DATASET:-brisc}"
SEEDS="${SEEDS:-1 2 3}"
TARGET_LAYERS="${TARGET_LAYERS:-last features.7 features.2}"
CAM_METHOD="${CAM_METHOD:-gradcam++}"
CAM_TARGET="${CAM_TARGET:-predicted}"
SELECTION="${SELECTION:-all}"
NUM_IMAGES="${NUM_IMAGES:-860}"
REQUIRE_MASK="${REQUIRE_MASK:-1}"
INCLUDE_ALL_CLASS_METRICS="${INCLUDE_ALL_CLASS_METRICS:-1}"
INCLUDE_ALL_OVA_CAMS="${INCLUDE_ALL_OVA_CAMS:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
NODE="${NODE:-zeus}"

echo "Submitting Grad-CAM grid"
echo "DATASET=${DATASET}"
echo "SEEDS=${SEEDS}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "CAM_METHOD=${CAM_METHOD}"
echo "CAM_TARGET=${CAM_TARGET}"
echo "SAVE_IMAGES=${SAVE_IMAGES}"

for seed in ${SEEDS}; do
  for target_layer in ${TARGET_LAYERS}; do
    echo "Submitting seed=${seed}, target_layer=${target_layer}"
    DATASET="${DATASET}" \
    SEED="${seed}" \
    TARGET_LAYER="${target_layer}" \
    CAM_METHOD="${CAM_METHOD}" \
    CAM_TARGET="${CAM_TARGET}" \
    SELECTION="${SELECTION}" \
    NUM_IMAGES="${NUM_IMAGES}" \
    REQUIRE_MASK="${REQUIRE_MASK}" \
    INCLUDE_ALL_CLASS_METRICS="${INCLUDE_ALL_CLASS_METRICS}" \
    INCLUDE_ALL_OVA_CAMS="${INCLUDE_ALL_OVA_CAMS}" \
    SAVE_IMAGES="${SAVE_IMAGES}" \
      sbatch --nodelist="${NODE}" scripts/run_explicabilidad_gradcam_slurm.sh
  done
done
