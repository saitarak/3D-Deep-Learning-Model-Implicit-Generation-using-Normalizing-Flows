# 3D Model Implicit Generation Using Normalizing Flows

## Setup

For the project, GPU is required to run the experiments. 
NDF.yml file contains all the python dependencies used in the project.

To create an environment, go to ndf directory and use the following lines of code:
'''
conda env create -f NDF.yml
conda activate NDF
'''

To perform the experiments ShapeNetCore v2 dataset cars foler named as 02958343 is downloaded. 
1. Dataprocessing is performed using:
'''
python ../dataprocessing/preprocess.py --config ../configs/shapenet_cars.txt
'''

2. The split of training, test or validation split data using:
'''
python ../dataprocessing/create_split.py --config ../configs/shapenet_cars.txt
'''

3. Train the Real NVP model with the following command:
''' 
python ../train_rnvp.py --config ../configs/shapenet_cars.txt
'''

4. Visualization of graphs:
'''
tensorboard --logdir ../models/experiments/shapenet_cars/summary
'''

5. Generate 3D car models:
''' 
python ../generate.py --config ../configs/shapenet_cars.txt
'''

Checkpoints can be found in ../models/experiments/shapenet_cars for NDF model to freeze the checkpoint.

For running the python files the ../slurm_scripts folder contains shell scripts.

Note: Checkpoints have to be changed to the format checkpoint:h:m_s_.tar from checkpoint_h_m_s_.tar before running the experiments.
