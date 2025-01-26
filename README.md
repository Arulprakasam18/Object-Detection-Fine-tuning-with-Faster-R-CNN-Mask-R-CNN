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

### 1. **Prepare the Dataset**
   - Download the PennFudan dataset and place it inside the `data/PennFudanPed` directory. The directory should look like this:
   
     ```
     data/
     └── PennFudanPed/
         ├── PNGImages/  # Contains all image files
         └── PedMasks/   # Contains all mask files
     ```

### 2. **Define Transformations**
   - **Random Horizontal Flip**: This transformation is applied during training to augment the dataset by randomly flipping images horizontally.
   - **Tensor Conversion**: Images and masks are converted to PyTorch tensors, and the pixel values are scaled to a range of 0 to 1 for model compatibility.

### 3. **Split the Dataset**
   - The dataset is split into a training set and a test set. For simplicity, 50 samples are used for testing, and the remaining samples are used for training.

### 4. **Set up DataLoader**
   - Data loaders are used to load the data in batches. A custom collation function is used to ensure that images and corresponding masks are padded and resized appropriately.

### 5. **Model Selection and Fine-tuning**
   - **Pre-trained Model**: A pre-trained Faster R-CNN or Mask R-CNN model with a ResNet-50 backbone is used as the base. We modify the model's head to adapt it to our dataset.
     - **Faster R-CNN**: The classifier head (`FastRCNNPredictor`) is replaced to accommodate the number of classes in our dataset (background and person).
     - **Mask R-CNN**: The mask predictor head is also replaced to handle binary segmentation masks.

### 6. **Optimizer and Scheduler**
   - The optimizer used is SGD (Stochastic Gradient Descent) with a learning rate scheduler that reduces the learning rate after every 3 epochs. This helps in fine-tuning the model effectively.

### 7. **Training Loop**
   - The model is trained for a set number of epochs, and during each epoch, the model is trained on the training dataset and evaluated on the test dataset. The performance is logged, and the learning rate is adjusted.

---

## Code Explanation

### `main.py`

The core file that handles model training and evaluation.

1. **Loading the Pre-trained Model**:

    ```python
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    ```

   - This loads a pre-trained Faster R-CNN model with a ResNet-50 backbone and Feature Pyramid Networks (FPN).

2. **Modify the Classifier and Mask Heads**:

    - **For Faster R-CNN**:
      ```python
      num_classes = 2  # 1 class (person) + background
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      ```

    - **For Mask R-CNN**:
      ```python
      in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
      model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes=num_classes)
      ```

   These lines replace the pre-trained classification and mask predictor heads to match the number of classes in our dataset (background and pedestrian).

3. **Optimizer and Scheduler**:

    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    ```

   The optimizer is SGD, and the learning rate is reduced after every 3 epochs.

4. **Training Loop**:

    ```python
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
    ```

   The model is trained for 2 epochs, and the learning rate is updated after each epoch.

---

## Inference

Once the model is trained, you can run inference on new images as follows:

```python
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]  # Example random images
predictions = model(x)  # Get predictions
print(predictions[0])  # Display predictions for the first image





