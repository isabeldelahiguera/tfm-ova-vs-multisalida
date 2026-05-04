#!/bin/bash
#SBATCH --job-name=tfm_exp
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export PATH="/opt/anaconda/bin:/opt/anaconda/anaconda3/bin:$PATH"
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

python run_experiments.py \
  --task classification \
  --dataset cifar10 \
  --seeds 1 \
  --coupling-modes ova \
  --output-csv resultados_slurm/exp_cifar10_${SLURM_JOB_ID}.csv \
  --summary-csv resultados_slurm/exp_cifar10_${SLURM_JOB_ID}_summary.csv

