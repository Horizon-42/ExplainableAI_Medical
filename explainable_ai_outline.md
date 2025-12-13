# Data analysis
## Overview
- Orignal Dataset
    | Dataset | Total | Normal | Pneumonia | Normal % | Pneumonia % | Ratio |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Train** | 5216 | 1341 | 3875 | 25.7% | 74.3% | 1:2.89 |
    | **Validation** | 16 | 8 | 8 | 50.0% | 50.0% | 1:1.00 |
    | **Test** | 624 | 234 | 390 | 37.5% | 62.5% | 1:1.67 |
The validation set of original dataset is too small, to address this problem, we combine the train/val dataset and divide them with ratio **0.8:0.2**.
- Redistributed Dataset
    | Dataset | Total | Normal | Pneumonia | Normal % | Pneumonia % | Ratio |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Train** | 4185 | 1079 | 3106 | 25.8% | 74.2% | 1:2.88 |
    | **Validation** | 1047 | 270 | 777 | 25.8% | 74.2% | 1:2.88 |
    | **Test** | 624 | 234 | 390 | 37.5% | 62.5% | 1:1.67 |

## Leakage Check
- Train-Val overlap: 0
- Train-Test overlap: 0
- Val-Test overlap: 0
No data leakage detected!

---

## Class Distribution
![alt text](class_distribution.png)
Two class is imbalance, so we need **oversample** during model finetuning.

---
## Pixel Intensity Analysis
![alt text](pixel_intensity.png)
The intensity distributions differ across the training, validation, and test datasets. Therefore, **intensity-** and **contrast-** based data augmentation is necessary to improve the model’s ability to generalize to images from different datasets.

---

## Sample Visulization
![alt text](train_samples.png)
![alt text](test_samples.png)
Pneumonia images has bigger opacity: **Ground-Glass Opacity**.

---
![alt text](<figures/Train Set_average_images.png>)
![alt text](<figures/Test Set_average_images.png>)
The average images indicate that the most prominent differences are concentrated in the lung regions of the chest, where pneumonia images exhibit higher opacity.

---

# Model 
## Model Setup
- Base Model: Resnet 18.
- Loss Function: 
- Data augmentation: 
- Oversample: 

## Cross Validation 
Run 5-Fold cross validation to pick best augmentation hyperparameters.

### Comparision 


# Explain
## Grad Cam

## LIME

## 