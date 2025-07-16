#!/bin/bash
#SBATCH -J maest_64
#SBATCH -p impa
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/scardenas/smc-master-thesis/scripts/models
#SBATCH -o job_logs/%N.%J.out # STDOUT
#SBATCH -e job_logs/%N.%j.err # STDERR

#Load Miniconda module 
module load Anaconda3/2024.02-1

#Enable the bash shell
eval "$(conda shell.bash hook)"

#Enable the conda environment
conda activate env

source ../../../.env
cd ../../laion_clap

python -m training.main \
    --save-frequency 5 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --batch-size=64 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=45 \
    --workers=1 \
    --use-bn-sync \
    --amodel MAEST-10s \
    --tmodel roberta \
    --warmup 3200 \
    --report-to "wandb" \
    --wandb-notes "MAEST-64" \
    --datasetnames "mtg_jamendo/split-0" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs logs \
    --seed 42 \
    --datasetpath ../data \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --freeze-text
