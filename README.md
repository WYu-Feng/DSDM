# DSDM: Dual-Space Diffusion Model for Universal Image Restoration

This repository contains the implementation code for the paper **"DSDM: Dual-Space Diffusion Model for Universal Image Restoration"**. The code provides the necessary scripts to train and test the FRM and MU-Net for universal image restoration tasks.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.10
- PyTorch 2.1.1+cu118
- torchvision 0.16.1+cu118

And download the following files to the project folder:


### Train
 To train the FRM and MU-Net models, run the following script: `python train.py`. This script will start the training process using the default configuration. 

### Test
To test the trained models, run the following script: `python test.py`. Before running the test script, make sure to update the checkpoint path in `test.py` to load your pre-trained model: `path = 'your checkpoint path'`. Replace `'your checkpoint path'` with the actual path to your trained model checkpoint. 
