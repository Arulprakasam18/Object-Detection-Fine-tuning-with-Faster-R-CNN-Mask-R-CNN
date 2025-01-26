# Object Detection Fine-tuning with Faster R-CNN & Mask R-CNN

This repository contains the implementation of fine-tuning object detection models (Faster R-CNN and Mask R-CNN) for instance segmentation tasks using the PennFudan dataset. The goal is to apply various custom transformations, optimizers, and learning rate schedulers to train a model that can detect and segment pedestrians.

## Overview

The project involves the following key steps:

1. **Dataset Preparation**: The PennFudan dataset, which contains images of pedestrians with corresponding masks, is used for training.
2. **Model Selection**: Pre-trained Faster R-CNN and Mask R-CNN models from TorchVision are used as the base models.
3. **Transformations and Data Augmentation**: Custom transformations, such as random horizontal flips and tensor scaling, are applied to the images and masks.
4. **Training and Evaluation**: The model is trained for a defined number of epochs, with an optimizer (SGD) and learning rate scheduler. The performance is evaluated using standard metrics.

---

## Project Structure

Object-Detection-FineTuning/
│
├── data/                         # PennFudan dataset
│   ├── PennFudanPed/             # Dataset directory
│   │   ├── PNGImages/           # Images folder
│   │   ├── PedMasks/            # Masks folder
│
├── engine.py                     # Contains training and evaluation functions
├── utils.py                      # Helper functions
├── main.py                       # Main file to train the model
├── requirements.txt              # Dependencies
└── README.md                     # Project Documentation



---

## Requirements

To install the necessary dependencies, run:


Dependencies include:
- PyTorch
- TorchVision
- torchvision.transforms
- NumPy
- Matplotlib
- OpenCV

---

## Dataset

The PennFudan dataset contains two classes: background and person. The dataset has a set of images with corresponding pedestrian masks. You can download the dataset from [here](https://www.cis.upenn.edu/~jshi/ped_html/).

---

## Training the Model

1. **Prepare the Dataset**: Download the PennFudan dataset and place it inside the `data/PennFudanPed` directory.

2. **Define Transformations**: 
    - Random Horizontal Flip for augmentation during training.
    - Convert the images to PyTorch tensors.

3. **Split the Dataset**: The dataset is split into training and test subsets, where 50 samples are used for testing.

4. **Set up DataLoader**: Data loaders are used to load the data in batches, with custom collation functions to handle varying sizes of images and masks.

5. **Model**: A pre-trained Faster R-CNN model with a ResNet-50 backbone and FPN (Feature Pyramid Networks) is used for training. Mask R-CNN is used for the instance segmentation task.

6. **Optimizer and Scheduler**: The optimizer is defined as SGD with a learning rate scheduler that decreases the learning rate after every 3 epochs.

7. **Training Loop**: The training loop runs for 2 epochs, printing losses and evaluating the model on the test set.

---

## Code Explanation

**Training Loop in `main.py`:**

