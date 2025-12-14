Image input size is 224, and crop means the image size before it be center crop to 224

# jitter 0.1, crop 224, Baseline model
=== FINAL TEST METRICS ===
Test Accuracy: 0.8606
Recall     : 0.9949
Specificity: 0.6368
F1 Score   : 0.8992
AUC        : 0.9662

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.64      0.77       234
   Pneumonia       0.82      0.99      0.90       390

    accuracy                           0.86       624
   macro avg       0.90      0.82      0.84       624
weighted avg       0.88      0.86      0.85       624

# jitter 0.3, random resize crop 0.08 to 1
=== FINAL TEST METRICS ===
Test Accuracy: 0.9503
Recall     : 0.9949
Specificity: 0.8761
F1 Score   : 0.9616
AUC        : 0.9923

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.88      0.93       234
   Pneumonia       0.93      0.99      0.96       390

    accuracy                           0.95       624
   macro avg       0.96      0.94      0.95       624
weighted avg       0.95      0.95      0.95       624


# jitter 0.2, random resize crop 0.5 to 1
=== FINAL TEST METRICS ===
Test Accuracy: 0.9391
Recall     : 0.9949
Specificity: 0.8462
F1 Score   : 0.9533
AUC        : 0.9886

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.85      0.91       234
   Pneumonia       0.92      0.99      0.95       390

    accuracy                           0.94       624
   macro avg       0.95      0.92      0.93       624
weighted avg       0.94      0.94      0.94       624

---

# Comparison Table

| Configuration | Accuracy | Recall | Specificity | F1 Score | AUC |
|---------------|----------|--------|-------------|----------|-----|
| Baseline (jitter=0.1, No crop) | 0.8606 | 0.9949 | 0.6368 | 0.8992 | 0.9662 |
| jitter=0.3, crop=0.08-1.0 | **0.9503** | 0.9949 | **0.8761** | **0.9616** | **0.9923** |
| jitter=0.2, crop=0.5-1.0 | 0.9391 | 0.9949 | 0.8462 | 0.9533 | 0.9886 |

**Key Observations:**
- All models achieve the same high Recall (0.9949), detecting nearly all Pneumonia cases
- The main improvement is in **Specificity** (Normal class detection): 0.6368 → 0.8761
- Best configuration: **jitter=0.3, RandomResizedCrop scale=(0.08, 1.0)** with 95% accuracy
