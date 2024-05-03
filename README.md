# vision transformer

## Introduction
Using ViT to classify CIFAR10, employing transfer learning, after training for five epochs, the validation accuracy reached 98.7%.

## Installation
Clone this repository:
```
git clone https://github.com/zlfffan/ViT.git
```
## Install dependencies
```
conda update conda
conda create -n env_name python=x.x
pip install matplotlib
```
## Usage
1. Run `train.py` to train the FCN model. (First, download the pre-trained weights as described in train.py.)
2. Run `predict.py` to view the model's prediction results.
3. Run the following command to view the training process:
```
tensorboard --logdir model_pre/VIT
```
