# Model Evaluation Function Documentation

## Overview

The `eval_epoch()` function evaluates a trained neural network model on validation or test data for binary classification of pneumonia detection in chest X-ray images. It computes multiple medical-relevant metrics to assess model performance.

---

## Function Signature

```python
def eval_epoch(model, dataloader, criterion):
    """Evaluate model performance on validation/test dataset."""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | The trained neural network model (e.g., ResNet50) |
| `dataloader` | `torch.utils.data.DataLoader` | DataLoader containing validation/test images and labels |
| `criterion` | `torch.nn.Module` | Loss function (e.g., CrossEntropyLoss) - not actively used but kept for API consistency |

### Returns

| Return Value | Type | Range | Description |
|--------------|------|-------|-------------|
| `val_recall` | `float` | 0.0 - 1.0 | Sensitivity/Recall for pneumonia class (positive) |
| `val_specificity` | `float` | 0.0 - 1.0 | Specificity/Recall for normal class (negative) |
| `val_f1` | `float` | 0.0 - 1.0 | Weighted F1-score across both classes |
| `val_auc` | `float` | 0.0 - 1.0 | Area Under the ROC Curve |

---

## Implementation Details

### Process Flow

```
1. Set model to evaluation mode
   ↓
2. Initialize empty lists for predictions, labels, probabilities
   ↓
3. For each batch in dataloader:
   a. Move data to device (GPU/CPU)
   b. Forward pass (no gradient computation)
   c. Apply softmax to get probabilities
   d. Extract pneumonia probability (column 1)
   e. Apply threshold (0.5) for binary prediction
   f. Collect results
   ↓
4. Compute all metrics using sklearn
   ↓
5. Return metrics
```

### Code Breakdown

```python
model.eval()  # Disable dropout, batch norm in training mode
```
**Why?** Ensures consistent predictions by freezing random components.

```python
all_preds, all_trues, all_probs = [], [], []
```
**Purpose:** Store predictions, ground truth, and probabilities for metric calculation.

```python
with torch.no_grad():
    # ... forward pass ...
```
**Why?** Disables gradient computation to:
- Save memory
- Speed up inference
- Prevent accidental model updates

```python
outputs = model(inputs)  # Shape: (batch_size, 2)
```
**Output format:**
- Column 0: Logit for class 0 (Normal)
- Column 1: Logit for class 1 (Pneumonia)

```python
probs = torch.softmax(outputs, dim=1)[:, 1]
```
**Breakdown:**
- `torch.softmax(outputs, dim=1)`: Converts logits to probabilities
  - `dim=1`: Apply softmax across classes (columns)
  - Output: [[P(Normal), P(Pneumonia)], ...]
- `[:, 1]`: Extract only pneumonia probabilities
  - `:` = all rows (all samples in batch)
  - `1` = second column (pneumonia class)

**Example:**
```python
# outputs (logits):    [[2.1, 3.5], [1.2, 0.8]]
# after softmax:       [[0.20, 0.80], [0.60, 0.40]]
# after [:, 1]:        [0.80, 0.40]
```

```python
preds = (probs >= 0.5).float()
```
**Purpose:** Convert probabilities to binary predictions
- `>= 0.5`: True if pneumonia probability ≥ 50%
- `.float()`: Convert boolean to 0.0 or 1.0

```python
all_probs.extend(probs.cpu().numpy())
```
**Operations:**
- `.cpu()`: Move from GPU to CPU memory
- `.numpy()`: Convert PyTorch tensor to NumPy array
- `.extend()`: Add to list (for sklearn compatibility)

---

## Metrics Explained

### 1. Recall (Sensitivity) - Pneumonia Detection

```python
val_recall = recall_score(all_trues, all_preds, pos_label=1)
```

**Formula:**
```
Recall = TP / (TP + FN)
```

**Components:**
- **TP (True Positives):** Pneumonia cases correctly identified
- **FN (False Negatives):** Pneumonia cases missed by model

**Interpretation:**
- **High recall (e.g., 0.95):** Model catches 95% of pneumonia cases
- **Low recall (e.g., 0.60):** Model misses 40% of pneumonia cases

**Clinical Importance:**
- **Critical metric for medical diagnosis**
- Missing pneumonia (FN) can be life-threatening
- Target: ≥ 0.90 for clinical deployment

**Example:**
```
100 pneumonia cases:
- Model correctly identifies 92 → TP = 92
- Model misses 8             → FN = 8
Recall = 92 / (92 + 8) = 0.92 (92%)
```

---

### 2. Specificity - Normal Case Detection

```python
val_specificity = recall_score(all_trues, all_preds, pos_label=0)
```

**Formula:**
```
Specificity = TN / (TN + FP)
```

**Components:**
- **TN (True Negatives):** Normal cases correctly identified
- **FP (False Positives):** Normal cases incorrectly flagged as pneumonia

**Interpretation:**
- **High specificity (e.g., 0.90):** Model correctly identifies 90% of healthy patients
- **Low specificity (e.g., 0.60):** Model has many false alarms

**Clinical Importance:**
- Reduces unnecessary treatments and patient anxiety
- Prevents resource waste (unnecessary tests, antibiotics)
- Maintains trust in AI system

**Example:**
```
200 normal cases:
- Model correctly identifies 180 → TN = 180
- Model falsely flags 20      → FP = 20
Specificity = 180 / (180 + 20) = 0.90 (90%)
```

---

### 3. F1-Score (Weighted) - Balanced Performance

```python
val_f1 = f1_score(all_trues, all_preds, average='weighted')
```

**Formula (per class):**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Where:**
```
Precision = TP / (TP + FP)  # How many predicted positives are correct?
Recall    = TP / (TP + FN)  # How many actual positives are found?
```

**Weighted Average:**
```
F1_weighted = (n_class0 × F1_class0 + n_class1 × F1_class1) / total_samples
```

**Why `average='weighted'`?**
- Accounts for class imbalance
- If dataset has 800 pneumonia and 200 normal cases:
  - Pneumonia F1 gets 4× more weight
  - Prevents model from ignoring minority class

**Interpretation:**
- **High F1 (e.g., 0.88):** Good balance of precision and recall
- **Low F1 (e.g., 0.50):** Poor performance or severe imbalance issues

**Clinical Importance:**
- Single metric for overall model quality
- Balances false positives and false negatives
- Useful for comparing different models

**Example:**
```
Class 0 (Normal):    Precision=0.85, Recall=0.90, F1=0.87, n=200
Class 1 (Pneumonia): Precision=0.92, Recall=0.88, F1=0.90, n=800

F1_weighted = (200×0.87 + 800×0.90) / 1000 = 0.894
```

---

### 4. AUC-ROC - Threshold-Independent Performance

```python
val_auc = roc_auc_score(all_trues, all_probs)
```

**What is ROC?**
- **ROC (Receiver Operating Characteristic):** Curve plotting TPR vs FPR
  - **TPR (True Positive Rate):** Same as Recall/Sensitivity
  - **FPR (False Positive Rate):** 1 - Specificity

**What is AUC?**
- **AUC (Area Under Curve):** Area under the ROC curve
- Range: 0.0 to 1.0

**Interpretation:**

| AUC Score | Performance | Clinical Meaning |
|-----------|-------------|------------------|
| 0.90 - 1.00 | Excellent | Model distinguishes classes very well |
| 0.80 - 0.90 | Good | Acceptable for clinical use with oversight |
| 0.70 - 0.80 | Fair | Needs improvement before deployment |
| 0.50 - 0.70 | Poor | Better than random but not clinically useful |
| 0.50 | Random | No better than coin flip |
| < 0.50 | Worse than random | Model predictions are inverted |

**Why Use Probabilities?**
```python
val_auc = roc_auc_score(all_trues, all_probs)  # Uses probabilities, not binary preds
```
- AUC considers model confidence across all thresholds
- Better than binary predictions (which use fixed 0.5 threshold)

**Clinical Importance:**
- **Threshold-independent:** Works regardless of decision threshold
- **Flexibility:** Allows adjusting threshold based on clinical needs
  - High recall needed? Lower threshold (e.g., 0.3)
  - High specificity needed? Raise threshold (e.g., 0.7)
- **Comparison:** Easy to compare different models

**Example:**
```
Sample predictions:
Patient  True_Label  Probability  Pred(0.5)
   1         1          0.92         1      ✓ TP
   2         0          0.15         0      ✓ TN
   3         1          0.78         1      ✓ TP
   4         0          0.45         0      ✓ TN
   5         1          0.35         0      ✗ FN (but high prob, close call)

AUC considers the full probability distribution, not just binary outcome.
```

---

## Visual Summary: Confusion Matrix

```
                    Predicted
                 Normal    Pneumonia
Actual  Normal  |  TN    |    FP    |  → Specificity = TN/(TN+FP)
        Pneumo  |  FN    |    TP    |  → Recall = TP/(TP+FN)
                           ↓
                   Precision = TP/(TP+FP)
```

---

## Common Pitfalls & Solutions

### Problem 1: Class Imbalance
**Symptom:** High accuracy but poor recall or specificity

**Example:**
```python
# 900 pneumonia, 100 normal
# Model always predicts pneumonia
Accuracy = 900/1000 = 0.90  # Looks good!
Recall = 1.0                 # Perfect!
Specificity = 0.0            # Terrible!
```

**Solution:** Use weighted F1 and monitor both recall/specificity

### Problem 2: Wrong Threshold
**Symptom:** Good AUC but poor recall

**Example:**
```python
# Using threshold=0.5 but model outputs are biased low
AUC = 0.88           # Good discrimination
Recall = 0.65        # Missing too many cases
```

**Solution:** Adjust threshold based on clinical requirements

### Problem 3: Overfitting to Majority Class
**Symptom:** High recall, low specificity

**Solution:**
- Use class weights in loss function
- Apply data augmentation for minority class
- Use stratified sampling

---

## Usage Example

```python
# Initialize model and data
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate
recall, specificity, f1, auc = eval_epoch(model, val_loader, criterion)

# Print results
print(f"Validation Metrics:")
print(f"  Recall (Sensitivity): {recall:.3f}")
print(f"  Specificity:          {specificity:.3f}")
print(f"  F1-Score (Weighted):  {f1:.3f}")
print(f"  AUC-ROC:              {auc:.3f}")

# Clinical interpretation
if recall >= 0.90 and specificity >= 0.85 and auc >= 0.90:
    print("✓ Model meets clinical deployment criteria")
else:
    print("✗ Model needs improvement")
```

**Expected Output:**
```
Validation Metrics:
  Recall (Sensitivity): 0.923
  Specificity:          0.876
  F1-Score (Weighted):  0.901
  AUC-ROC:              0.945
✓ Model meets clinical deployment criteria
```

---

## Clinical Deployment Recommendations

### Minimum Acceptable Thresholds

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| Recall | 0.85 | 0.95 | Critical: Don't miss pneumonia cases |
| Specificity | 0.70 | 0.85 | Important: Minimize false alarms |
| F1-Score | 0.75 | 0.90 | Overall balance |
| AUC | 0.80 | 0.95 | Discrimination ability |

### Recommended Monitoring

1. **Track all four metrics together** - Don't optimize one at expense of others
2. **Test on diverse populations** - Different age groups, X-ray machines
3. **Regular revaluation** - Model performance can drift over time
4. **Human-in-the-loop** - AI assists, doesn't replace radiologists

---

## References

- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Medical AI Guidelines: FDA recommendations for AI/ML in medical devices
- ROC/AUC Tutorial: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

---

## Related Functions

- `train_epoch()`: Training loop counterpart
- `torch.nn.CrossEntropyLoss()`: Loss function used
- `sklearn.metrics`: Source of evaluation metrics