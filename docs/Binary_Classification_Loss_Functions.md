# Loss Functions for Binary Classification

A comprehensive guide to loss functions used in binary classification tasks, with practical implementations and use cases for medical imaging.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Binary Cross-Entropy Loss](#2-binary-cross-entropy-loss-bce)
3. [BCE with Logits Loss](#3-bce-with-logits-loss)
4. [Weighted BCE Loss](#4-weighted-bce-loss)
5. [Focal Loss](#5-focal-loss)
6. [Dice Loss](#6-dice-loss)
7. [Hinge Loss](#7-hinge-loss)
8. [Comparison Table](#8-comparison-table)
9. [Choosing the Right Loss](#9-choosing-the-right-loss)
10. [References](#10-references)

---

## 1. Overview

In binary classification, the loss function measures the difference between the predicted probability and the true label (0 or 1). The goal is to minimize this loss during training.

**Key Concepts:**
- **Logits:** Raw model outputs before applying activation (e.g., sigmoid)
- **Probabilities:** Outputs after sigmoid, in range [0, 1]
- **Labels:** Ground truth, either 0 (negative) or 1 (positive)

---

## 2. Binary Cross-Entropy Loss (BCE)

### Formula

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

Where:
- $y_i$ = true label (0 or 1)
- $\hat{y}_i$ = predicted probability (after sigmoid)
- $N$ = number of samples

### Intuition

- When $y = 1$: Loss = $-\log(\hat{y})$ → Penalizes low predictions
- When $y = 0$: Loss = $-\log(1 - \hat{y})$ → Penalizes high predictions

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Requires sigmoid-activated outputs
criterion = nn.BCELoss()

# Example
probs = torch.sigmoid(model(inputs))  # Apply sigmoid first
labels = torch.tensor([1.0, 0.0, 1.0])
loss = criterion(probs, labels)
```

### When to Use

- Standard binary classification
- Balanced datasets
- When model outputs probabilities

---

## 3. BCE with Logits Loss

### Formula

Same as BCE, but **combines sigmoid and BCE** in one numerically stable operation.

$$\text{BCEWithLogits} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i))\right]$$

Where:
- $z_i$ = logit (raw model output)
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ = sigmoid function

### PyTorch Implementation

```python
import torch.nn as nn

# Takes raw logits as input (no sigmoid needed)
criterion = nn.BCEWithLogitsLoss()

# Example
logits = model(inputs)  # Raw output, no sigmoid
labels = torch.tensor([1.0, 0.0, 1.0]).unsqueeze(1)
loss = criterion(logits, labels)
```

### Advantages Over BCELoss

1. **Numerically stable:** Avoids log(0) issues
2. **More efficient:** Sigmoid + loss in one operation
3. **Recommended by PyTorch** for binary classification

### When to Use

- **Always prefer this over BCELoss** for binary classification
- Standard choice for most tasks

---

## 4. Weighted BCE Loss

### Formula

$$\text{WeightedBCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[w_1 \cdot y_i \log(\hat{y}_i) + w_0 \cdot (1 - y_i) \log(1 - \hat{y}_i)\right]$$

Where:
- $w_1$ = weight for positive class
- $w_0$ = weight for negative class (usually 1.0)

### PyTorch Implementation

```python
import torch.nn as nn

# pos_weight: weight for positive class relative to negative
# If dataset has 1000 negative and 300 positive:
pos_weight = torch.tensor([1000 / 300])  # ≈ 3.33

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Understanding `pos_weight`

`pos_weight` is a **multiplier for the positive class loss**. It answers: "How much more should I penalize missing a positive sample vs. a negative sample?"

**Formula with pos_weight:**

$$\text{WeightedBCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[w \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$

Where $w$ = `pos_weight`

**How It Works:**

| pos_weight | Effect |
| :---: | :--- |
| 1.0 | No change, standard BCE |
| 2.0 | Missing a positive costs 2× more than missing a negative |
| 0.5 | Missing a negative costs 2× more than missing a positive |

**Example:**
```python
# Dataset: 740 Pneumonia (positive), 260 Normal (negative)
# Without pos_weight: Model may predict all Pneumonia (74% accuracy!)

# Option 1: Inverse frequency (recommended)
pos_weight = 260 / 740  # ≈ 0.35
# Now missing a Normal case is penalized ~3× more

# Option 2: Equal effective samples
pos_weight = 1.0  # Treat both classes equally

# Option 3: Boost minority recall
pos_weight = 0.25  # Strongly penalize missing Normal cases
```

**For Your Pneumonia Dataset:**
```python
# Normal: 26%, Pneumonia: 74%
# You want to catch all Pneumonia (recall), but also identify Normal

# If Pneumonia is positive (label=1):
pos_weight = torch.tensor([0.35])  # 260/740

# If Normal is positive (label=1):
pos_weight = torch.tensor([2.85])  # 740/260
```

**Important:** `pos_weight` affects **which class the model prioritizes**:
- `pos_weight > 1`: Boost positive class recall (fewer false negatives)
- `pos_weight < 1`: Boost negative class recall (fewer false positives)

### Weight Calculation

```python
# Method 1: Inverse frequency
num_neg = (labels == 0).sum()
num_pos = (labels == 1).sum()
pos_weight = num_neg / num_pos

# Method 2: Effective number (for severe imbalance)
beta = 0.9999
pos_weight = (1 - beta) / (1 - beta ** num_pos)
```

### When to Use

- **Imbalanced datasets** (your pneumonia case: 74% positive, 26% negative)
- When you want to penalize misclassification of minority class more

---

## 5. Focal Loss

### The Problem It Solves

In imbalanced datasets, the model sees many "easy" examples from the majority class. Standard BCE averages over all examples, so hard examples get drowned out.

### What Are "Easy" vs "Hard" Examples?

**Easy Examples:**
- Samples the model classifies correctly with **high confidence**
- Example: Pneumonia image predicted with 0.95 probability (correct)
- The model has already "learned" these patterns
- They contribute very little to improving the model

**Hard Examples:**
- Samples the model classifies incorrectly or with **low confidence**
- Example: Pneumonia image predicted with 0.4 probability (wrong)
- These are the cases where the model needs to improve
- More valuable for learning

**The Problem:**
```
Dataset: 1000 Pneumonia (easy), 100 Normal (hard)

Standard BCE Loss:
- 1000 easy examples contribute ~0.05 loss each = 50 total
- 100 hard examples contribute ~0.8 loss each = 80 total
- Total = 130, but easy examples dominate gradient updates!

Focal Loss (γ=2):
- 1000 easy examples contribute ~0.001 loss each = 1 total
- 100 hard examples contribute ~0.64 loss each = 64 total  
- Total = 65, hard examples now dominate learning!
```

### Formula

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t = \hat{y}$ if $y = 1$, else $1 - \hat{y}$
- $\alpha_t$ = class balancing weight
- $\gamma$ = focusing parameter (typically 2.0)

### Key Insight

$(1 - p_t)^\gamma$ is the **modulating factor**:
- Easy examples ($p_t \approx 1$): Factor → 0, loss → small
- Hard examples ($p_t \approx 0.5$): Factor → large, loss → large

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weight for positive class (0.25 is common)
            gamma: Focusing parameter (2.0 is standard)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: logits, targets: 0 or 1
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)  # p_t = probability of correct class
        
        # Apply focal modulation
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weight
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Usage
criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, labels)
```

### Hyperparameter Tuning

| Parameter | Default | Effect |
| :--- | :---: | :--- |
| $\gamma = 0$ | - | Equivalent to standard BCE |
| $\gamma = 1$ | - | Mild focusing |
| $\gamma = 2$ | ✓ | Standard, works well |
| $\gamma = 5$ | - | Very aggressive focusing |
| $\alpha = 0.25$ | ✓ | Common for object detection |
| $\alpha = 0.5$ | - | Equal class weights |

### When to Use

- **Severe class imbalance** (1:10 or worse)
- When model is overconfident on easy examples
- Object detection, medical imaging with rare findings

---

## 6. Dice Loss

### The Problem It Solves

BCE treats each pixel/sample independently. Dice Loss considers the **overlap** between prediction and ground truth, which is better for segmentation and when class ratios matter.

### Formula

$$\text{Dice Loss} = 1 - \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}$$

Where:
- $p_i$ = predicted probability
- $g_i$ = ground truth (0 or 1)
- $\epsilon$ = small constant for numerical stability

### Dice Coefficient (F1 Score Connection)

$$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \text{F1 Score}$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# Usage
criterion = DiceLoss()
loss = criterion(logits, labels)
```

### Combined BCE + Dice Loss

```python
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
```

### When to Use

- **Semantic segmentation**
- When optimizing for F1 score directly
- Imbalanced binary classification

---

## 7. Hinge Loss

### Formula

$$\text{Hinge} = \max(0, 1 - y \cdot \hat{y})$$

Where:
- $y \in \{-1, +1\}$ (note: different label encoding!)
- $\hat{y}$ = raw model output (no sigmoid)

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# For SVM-style classification
criterion = nn.HingeEmbeddingLoss()

# Or manually:
def hinge_loss(outputs, targets):
    # targets should be -1 or +1
    return torch.mean(torch.clamp(1 - targets * outputs, min=0))
```

### When to Use

- **Support Vector Machines (SVM)**
- When you want hard margins
- Less common in deep learning (BCE preferred)

---

## 8. Comparison Table

| Loss Function | Handles Imbalance | Numerical Stability | Focuses on Hard Examples | Best For |
| :--- | :---: | :---: | :---: | :--- |
| **BCELoss** | ❌ | ⚠️ | ❌ | Balanced datasets |
| **BCEWithLogitsLoss** | ❌ | ✅ | ❌ | General binary classification |
| **Weighted BCE** | ✅ | ✅ | ❌ | Moderate imbalance |
| **Focal Loss** | ✅ | ✅ | ✅ | Severe imbalance, hard examples |
| **Dice Loss** | ✅ | ✅ | ❌ | Segmentation, F1 optimization |
| **Hinge Loss** | ❌ | ✅ | ❌ | SVM-style classifiers |

---

## 9. Choosing the Right Loss

### Decision Flowchart

```
Is your dataset balanced?
├── Yes → BCEWithLogitsLoss
└── No → How severe is the imbalance?
    ├── Mild (1:2 to 1:5) → Weighted BCE
    └── Severe (1:10+) → Are there many easy examples?
        ├── Yes → Focal Loss
        └── No → Weighted BCE or Dice Loss
```

### For Your Pneumonia Project

Your dataset: Normal 26% : Pneumonia 74% (ratio ~1:2.9)

**Recommendations:**

1. **Start with:** `BCEWithLogitsLoss` (baseline)
2. **If recall is low:** Add `pos_weight` for weighted BCE
3. **If model is overconfident:** Try Focal Loss with $\gamma = 2$

```python
# Your current setup (good choice!)
criterion = nn.BCEWithLogitsLoss()

# If you need to boost Normal class recall:
pos_weight = torch.tensor([0.35])  # 26% / 74% ≈ 0.35
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# If many easy examples dominate:
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

---

## 10. References

1. **Binary Cross-Entropy**
   - *Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.*
   - Chapter 6: Deep Feedforward Networks
   - [Online Book](https://www.deeplearningbook.org/)

2. **Focal Loss**
   - *Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection.*
   - Introduced in RetinaNet paper
   - [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

3. **Dice Loss**
   - *Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*
   - [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)

4. **Class Imbalance Survey**
   - *Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance.*
   - [Journal of Big Data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0192-5)

5. **PyTorch Loss Functions**
   - Official PyTorch documentation
   - [torch.nn Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

6. **Effective Number of Samples**
   - *Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples.*
   - [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)

---

## Quick Reference Code

```python
import torch
import torch.nn as nn

# ============ STANDARD ============
criterion = nn.BCEWithLogitsLoss()

# ============ WEIGHTED ============
pos_weight = torch.tensor([num_neg / num_pos])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ============ FOCAL ============
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# ============ DICE ============
criterion = DiceLoss(smooth=1e-6)

# ============ COMBINED ============
criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
```

---

*Last updated: December 2025*
