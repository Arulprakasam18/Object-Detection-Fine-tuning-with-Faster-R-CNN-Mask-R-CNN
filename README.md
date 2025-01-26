# Object Detection Fine-tuning with Faster R-CNN & Mask R-CNN

It ontains the implementation of fine-tuning object detection models (Faster R-CNN and Mask R-CNN) for instance segmentation tasks using the PennFudan dataset. The goal is to apply various custom transformations, optimizers, and learning rate schedulers to train a model that can detect and segment pedestrians.

## Overview

The project involves the following key steps:

1. **Dataset Preparation**: The PennFudan dataset, which contains images of pedestrians with corresponding masks, is used for training.
2. **Model Selection**: Pre-trained Faster R-CNN and Mask R-CNN models from TorchVision are used as the base models.
3. **Transformations and Data Augmentation**: Custom transformations, such as random horizontal flips and tensor scaling, are applied to the images and masks.
4. **Model Fine-tuning**: The pre-trained models are modified to handle the number of classes in our dataset, and the classifier heads are replaced to allow training on our dataset.
5. **Training and Evaluation**: The model is trained for a defined number of epochs, with an optimizer (SGD) and learning rate scheduler. The performance is evaluated using standard metrics.

---

## Project Structure

The directory structure for this project is as follows:

Object-Detection-FineTuning/  
├── data/                   # Directory for datasets  
│   ├── PennFudanPed/       # PennFudan dataset directory  
│       ├── PNGImages/      # Folder containing images  
│       ├── PedMasks/       # Folder containing masks  
├── engine.py               # File containing training and evaluation functions  
├── utils.py                # File for helper functions  
├── main.py                 # Main script to train the model  
├── requirements.txt        # File listing project dependencies  
└── README.md               # Documentation for the project  







