#!/bin/bash
#SBATCH --job-name=tfm_classic
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
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

MODEL_ARCH="${MODEL_ARCH:-mlp}"
DATASETS="${DATASETS:-iris wine breast_cancer digits}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
COUPLING_MODES="${COUPLING_MODES:-ova}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-32 16}"
JOB_ID="${SLURM_JOB_ID:-local}"

for DATASET in ${DATASETS}; do
  echo "Launching dataset=${DATASET}, model_arch=${MODEL_ARCH}, seeds=${SEEDS}"
  python run_experiments.py \
    --task classification \
    --dataset "${DATASET}" \
    --model-arch "${MODEL_ARCH}" \
    --hidden-layers ${HIDDEN_LAYERS} \
    --seeds ${SEEDS} \
    --coupling-modes ${COUPLING_MODES} \
    --epochs "${EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --output-csv "resultados_slurm/exp_${DATASET}_${MODEL_ARCH}_${JOB_ID}.csv" \
    --summary-csv "resultados_slurm/exp_${DATASET}_${MODEL_ARCH}_${JOB_ID}_summary.csv"
done
