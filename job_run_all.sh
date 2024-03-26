#!/bin/bash
#SBATCH --job-name=dvc_repro
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=25gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lukas.erhard@sowi.uni-stuttgart.de

module load lib/cudnn/8.5.0-cuda-11.6

bash run_all.sh
