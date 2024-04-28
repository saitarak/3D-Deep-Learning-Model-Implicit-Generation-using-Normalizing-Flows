# 3D Model Implicit Generation Using Normalizing Flows

## Abstract

In recent times, the researchers from deep learning community started focusing on 3D modeling for broader adoption. The algorithms are being
implemented at a fast pace to work with 3D data. The main goal is to reconstruct and generate 3D models that can help save time in modeling 3D
data in various industry domains like robotics, computer vision, and virtual
reality, etc., However, the quality of 3D generation is not attained to the
required level.
In this thesis, we present an implicit generative framework that merges implicit representation and normalizing flow techniques to generate novel shapes.
Implicit representation excels at intricate 3D shape learning while normalizing
flow-based generation facilitates novel 3D shape generation offering tractable
log-likelihoods and efficient sampling. This work further analyzes the latent
spaces in neural implicit representation models, aiming to identify the optimal
shape learning models. This enhances the ability to generate novel shapes by
learning the optimal shape encoding. By capturing 3D shapes as dense point
clouds, our approach advances generative 3D modeling.
We conduct our experimentation on the ’cars’ object category from the
ShapeNet dataset for its intricate internal details and complexity. To quantitatively assess the efficacy of our 3D reconstruction models, we employ
the chamfer distance metric. Additionally, we present qualitative outcomes
of our generation approach to provide a comprehensive view of our model
performance.

## Setup

For the project, GPU is required to run the experiments. 
NDF.yml file contains all the python dependencies used in the project.

To create an environment, go to ndf directory and use the following lines of code:
   ```
   conda env create -f NDF.yml
   conda activate NDF
   ```

To perform the experiments ShapeNetCore v2 dataset cars foler named as 02958343 is downloaded. 
1. Dataprocessing is performed with:
   ```
   python ../dataprocessing/preprocess.py --config ../configs/shapenet_cars.txt
   ```

2. The split of training, test or validation split data using:
   ```
   python ../dataprocessing/create_split.py --config ../configs/shapenet_cars.txt
   ```

3. Train the Real NVP model with the following command:
   ``` 
   python ../train_rnvp.py --config ../configs/shapenet_cars.txt
   ```

4. Visualization of graphs:
   ```
   tensorboard --logdir ../models/experiments/shapenet_cars/summary
   ```

5. Generate 3D car models:
   ``` 
   python ../generate.py --config ../configs/shapenet_cars.txt
   ```

Checkpoints can be found in ../models/experiments/shapenet_cars for NDF model to freeze the checkpoint.

For running the python files the ../slurm_scripts folder contains shell scripts.

Note: Checkpoints have to be changed to the format checkpoint:h:m_s_.tar from checkpoint_h_m_s_.tar before running the experiments.
