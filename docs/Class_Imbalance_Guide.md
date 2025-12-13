# Handling Class Imbalance in Medical Image Classification

Class imbalance is a common challenge in medical imaging, where one class (e.g., "diseased") is significantly underrepresented compared to another (e.g., "healthy"). This guide covers practical techniques to address this issue.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Data-Level Techniques](#2-data-level-techniques)
3. [Algorithm-Level Techniques](#3-algorithm-level-techniques)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Practical Recommendations](#5-practical-recommendations)
6. [References](#6-references)

---

## 1. The Problem

### Why Imbalance Hurts Model Performance

Consider your pneumonia dataset:

| Dataset | Normal | Pneumonia | Ratio |
| :--- | :---: | :---: | :---: |
| **Train** | 1341 (25.7%) | 3875 (74.3%) | 1:2.89 |

When training on imbalanced data:

- **Bias toward majority class:** The model learns to predict "Pneumonia" more often because it minimizes overall loss.
- **Poor minority class recall:** Normal cases get misclassified frequently.
- **Misleading accuracy:** 90% accuracy means nothing if 90% of samples are one class.

---

## 2. Data-Level Techniques

### A. Oversampling (Increase Minority Class)

**Concept:** Duplicate or synthesize samples from the minority class.

#### i. Random Oversampling

Simply duplicate minority class samples randomly.

```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights (inverse of class frequency)
class_counts = [1341, 3875]  # [Normal, Pneumonia]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in train_labels]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True  # Allow duplicates
)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

**Pros:** Simple, effective.  
**Cons:** Risk of overfitting to duplicated samples.

#### ii. SMOTE (Synthetic Minority Over-sampling Technique)

Creates synthetic samples by interpolating between existing minority samples.

```python
from imblearn.over_sampling import SMOTE

# For tabular/flattened image features
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Note:** SMOTE works best for tabular data. For images, data augmentation is preferred.

#### iii. Data Augmentation (Best for Images)

Generate new training samples through transformations.

```python
from torchvision import transforms

# Aggressive augmentation for minority class
minority_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Pros:** Creates truly new samples, reduces overfitting.  
**Cons:** Requires domain knowledge to choose valid augmentations.

---

### B. Undersampling (Decrease Majority Class)

**Concept:** Remove samples from the majority class to balance the dataset.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Pros:** Faster training, less memory.  
**Cons:** Loses potentially valuable information from majority class.

---

### C. Hybrid: Combine Over + Undersampling

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

---

## 3. Algorithm-Level Techniques

### A. Class Weights in Loss Function

**Concept:** Penalize misclassification of minority class more heavily.

```python
import torch.nn as nn

# Inverse frequency weighting
class_counts = [1341, 3875]  # [Normal, Pneumonia]
total = sum(class_counts)
class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float)
class_weights = class_weights / class_weights.sum()  # Normalize

# Apply to loss function
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**For Binary Classification:**

```python
# BCEWithLogitsLoss with pos_weight
# pos_weight = num_negative / num_positive
pos_weight = torch.tensor([1341 / 3875])  # 0.346
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

---

### B. Focal Loss (Best for Extreme Imbalance)

**Concept:** Down-weight easy examples, focus on hard misclassified samples.

**Formula:**
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $\alpha_t$: Class balancing weight
- $\gamma$: Focusing parameter (typically 2.0)
- $p_t$: Model's predicted probability for the true class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Usage
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**When to use:**
- Extreme imbalance (1:10 or worse)
- Many "easy" examples dominating the loss
- Object detection tasks

---

### C. Threshold Tuning

**Concept:** Adjust the decision threshold based on class distribution.

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities on validation set
val_probs = model_predict(val_loader)
val_labels = get_labels(val_loader)

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")

# Use optimal threshold for predictions
predictions = (val_probs >= optimal_threshold).astype(int)
```

---

## 4. Evaluation Metrics

**Never use accuracy alone for imbalanced data.** Use these instead:

| Metric | Formula | Best For |
| :--- | :--- | :--- |
| **Balanced Accuracy** | (Recall + Specificity) / 2 | Overall performance |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance of precision/recall |
| **AUC-ROC** | Area under ROC curve | Threshold-independent evaluation |
| **Precision-Recall AUC** | Area under PR curve | When positive class is rare |

```python
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

balanced_acc = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
auc = roc_auc_score(y_true, y_probs)

print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
```

---

## 5. Practical Recommendations

### For Your Pneumonia Detection Project

Given your dataset (Normal: 25%, Pneumonia: 75%), here's the recommended approach:

#### Step 1: Use WeightedRandomSampler (Already in your code ✓)

```python
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in train_df['label']]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
```

#### Step 2: Add Class Weights to Loss

```python
# Calculate weights
class_weights = torch.tensor([total/1341, total/3875]).to(device)  # Inverse frequency
class_weights = class_weights / class_weights.sum()  # Normalize

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### Step 3: Use Balanced Accuracy for Model Selection

```python
balanced_acc = (recall + specificity) / 2
if balanced_acc > best_balanced_acc:
    torch.save(model.state_dict(), 'best_model.pth')
```

#### Step 4: Tune Threshold on Validation Set

```python
# Find threshold that maximizes F1 or balanced accuracy
thresholds = np.arange(0.3, 0.7, 0.05)
best_threshold = 0.5
best_f1 = 0
for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
```

---

## 6. References

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - *Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.*
   - [Link to Paper](https://arxiv.org/abs/1106.1813)

2. **Focal Loss**
   - *Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.*
   - Introduced in the RetinaNet paper for object detection.
   - [Link to Paper](https://arxiv.org/abs/1708.02002)

3. **Class Imbalance in Medical Imaging**
   - *Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance.*
   - [Link to Paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0192-5)

4. **Imbalanced-Learn Library**
   - Python library for handling imbalanced datasets.
   - [Documentation](https://imbalanced-learn.org/stable/)

5. **PyTorch WeightedRandomSampler**
   - Official PyTorch documentation.
   - [Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)

---

## Summary Table: When to Use What

| Technique | Imbalance Level | Pros | Cons |
| :--- | :---: | :--- | :--- |
| **WeightedRandomSampler** | Mild (1:3) | Simple, no data loss | May overfit |
| **Class Weights** | Mild-Moderate | Easy to implement | Requires tuning |
| **Data Augmentation** | Any | Creates new samples | Domain knowledge needed |
| **Focal Loss** | Severe (1:10+) | Handles hard examples | Hyperparameter sensitive |
| **SMOTE** | Moderate | Synthesizes samples | Not ideal for images |
| **Threshold Tuning** | Any | Post-hoc adjustment | Requires validation set |

---

*Last updated: December 2025*
