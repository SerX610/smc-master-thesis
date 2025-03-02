#!/bin/bash
#SBATCH -J webdataset
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 2 
#SBATCH --chdir=/home/scardenas/smc-master-thesis/scripts/data
#SBATCH --time=2:00:00
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

#Load Miniconda module 
module load Miniconda3/4.9.2

#Enable the bash shell
eval "$(conda shell.bash hook)"

#Enable the conda environment
conda activate env

#Run the Python script
python create_mtg_jamendo_webdataset.py
