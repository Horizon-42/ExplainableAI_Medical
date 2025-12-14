---
marp: true
size: 16:9
style: |
  img {
    display: block;
    margin: 0 auto;
  }
---

# Data Analysis

---

## Original Dataset

| Dataset | Total | Normal | Pneumonia | Normal % | Pneumonia % | Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 5216 | 1341 | 3875 | 25.7% | 74.3% | 1:2.89 |
| **Validation** | 16 | 8 | 8 | 50.0% | 50.0% | 1:1.00 |
| **Test** | 624 | 234 | 390 | 37.5% | 62.5% | 1:1.67 |

Validation set is too small (only 16 samples)

---

## Redistributed Dataset

Combine train/val and split with ratio **0.8:0.2**

| Dataset | Total | Normal | Pneumonia | Normal % | Pneumonia % | Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4185 | 1079 | 3106 | 25.8% | 74.2% | 1:2.88 |
| **Validation** | 1047 | 270 | 777 | 25.8% | 74.2% | 1:2.88 |
| **Test** | 624 | 234 | 390 | 37.5% | 62.5% | 1:1.67 |

---

## Leakage Check

| Check | Result |
| :--- | :---: |
| Train-Val overlap | 0 ✅ |
| Train-Test overlap | 0 ✅ |
| Val-Test overlap | 0 ✅ |

**No data leakage detected!**

---

## Class Distribution

![h:450 center](class_distribution.png)

Classes are imbalanced, Use **oversampling** during training

---

## Pixel Intensity Analysis

![w:1000 center](pixel_intensity.png)

Use **intensity/contrast augmentation** to improve generalization

---

## Sample Visualization (Train)

![w:1000 center](train_samples.png)

---

## Sample Visualization (Test)

![w:1000 center](test_samples.png)

Pneumonia images show higher opacity: **Ground-Glass Opacity**

---

## Average Image Analysis (Train)

![w:1000 center](<figures/Train Set_average_images.png>)

---

## Average Image Analysis (Test)

![w:1000 center](<figures/Test Set_average_images.png>)

Most differences are in **lung regions** where pneumonia shows higher opacity

---

# Model 
---
## Model Setup
- Base Model: Resnet 18.
- Loss Function: **BCEWithLogitsLoss**
- Optimizer: Adam
- Learning Rate Schedule: **ReduceLROnPlateau**
- Early Stop: quit training after 7 epoch no improvement.
- Oversample: Use **WeightedRandomSampler** for training set.

---

## Cross Validation 
Run 5-Fold cross validation to pick best augmentation hyperparameters.
![alt text](cv_results.png)

---

## Comparison 
| Configuration | Accuracy | Recall | Specificity | F1 Score | AUC |
|---------------|----------|--------|-------------|----------|-----|
| Baseline (jitter=0.1, No crop) | 0.8606 | 0.9949 | 0.6368 | 0.8992 | 0.9662 |
| jitter=0.3, crop=0.08-1.0 | **0.9503** | 0.9949 | **0.8761** | **0.9616** | **0.9923** |
| jitter=0.2, crop=0.5-1.0 | 0.9391 | 0.9949 | 0.8462 | 0.9533 | 0.9886 |

**Key Observations:**
- All models achieve the same high Recall(0.9949), detecting nearly all Pneumonia cases
- The main improvement is in **Specificity** (Normal class detection): 0.6368 to 0.8761



---
# Explain
---
## Grad Cam

## LIME

## 