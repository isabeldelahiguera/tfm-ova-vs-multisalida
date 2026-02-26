#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name unlearn
#SBATCH --output=slrums/output_%A_%a.out
#SBATCH --exclude=titan,atenea
#SBATCH --partition=dgx2,dgx,dios

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"

conda activate ../env/unlearning/
export PYTHONPATH='.'

echo "Running unlearning script"

python ./src/trainer.py --dataset iris --seed 4050
python ./src/trainer.py --dataset wine --seed 4050 
python ./src/trainer.py --dataset digits  --seed 4050 
python ./src/trainer.py --dataset breast_cancer --seed 4050 

python ./src/trainer.py --dataset iris --hidden-layers 64 --seed 4050 
python ./src/trainer.py --dataset wine --hidden-layers 64 --seed 4050 
python ./src/trainer.py --dataset digits --hidden-layers 64 --seed 4050 
python ./src/trainer.py --dataset breast_cancer --hidden-layers 64 --seed 4050 

python ./src/trainer.py --dataset iris --hidden-layers 64 32 16 --seed 4050 
python ./src/trainer.py --dataset wine --hidden-layers 64 32 16 --seed 4050 
python ./src/trainer.py --dataset digits --hidden-layers 64 32 16 --seed 4050 
python ./src/trainer.py --dataset breast_cancer --hidden-layers 64 32 16 --seed 4050