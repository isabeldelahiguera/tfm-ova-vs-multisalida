#!/bin/bash
#SBATCH --job-name=tfm_tb_vgg
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export PATH="/opt/anaconda/bin:/opt/anaconda/anaconda3/bin:$PATH"
export PS1="${PS1-}"
eval "$(conda shell.bash hook)"

cd /mnt/homeGPU/imhiguera
mkdir -p slurm_logs resultados_slurm

conda activate /mnt/homeGPU/imhiguera/conda_env

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

MODEL_ARCH="${MODEL_ARCH:-vgg}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COUPLING_MODES="${COUPLING_MODES:-ova}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
TB_ROOT="${TB_ROOT:-/mnt/homeGPU/imhiguera/data/tb_chest_xray}"
JOB_ID="${SLURM_JOB_ID:-local}"

python run_experiments.py \
  --task classification \
  --dataset tb_chest_xray \
  --tb-root "${TB_ROOT}" \
  --image-size "${IMAGE_SIZE}" \
  --model-arch "${MODEL_ARCH}" \
  --seeds ${SEEDS} \
  --coupling-modes ${COUPLING_MODES} \
  --epochs "${EPOCHS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --output-csv "resultados_slurm/exp_tb_${MODEL_ARCH}_${IMAGE_SIZE}_${JOB_ID}.csv" \
  --summary-csv "resultados_slurm/exp_tb_${MODEL_ARCH}_${IMAGE_SIZE}_${JOB_ID}_summary.csv"
