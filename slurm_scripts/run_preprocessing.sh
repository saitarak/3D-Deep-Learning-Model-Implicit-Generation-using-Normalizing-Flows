#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=50G

conda activate NDF
python ../generate.py --config ../configs/shapenet_cars.txt