Image input size is 224, and crop means the image size before it be center crop to 224

# jitter 0, crop 240

=== FINAL TEST METRICS ===
Test Accuracy: 0.9103
Recall     : 0.9897
Specificity: 0.7778
F1 Score   : 0.9324
AUC        : 0.9763

Classification Report:
              precision    recall  f1-score   support

      Normal       0.98      0.78      0.87       234
   Pneumonia       0.88      0.99      0.93       390

    accuracy                           0.91       624
   macro avg       0.93      0.88      0.90       624
weighted avg       0.92      0.91      0.91       624

# jitter 0.4, crop 280

=== FINAL TEST METRICS ===
Test Accuracy: 0.9279
Recall     : 0.9692
Specificity: 0.8590
F1 Score   : 0.9438
AUC        : 0.9759

Classification Report:
              precision    recall  f1-score   support

      Normal       0.94      0.86      0.90       234
   Pneumonia       0.92      0.97      0.94       390

    accuracy                           0.93       624
   macro avg       0.93      0.91      0.92       624
weighted avg       0.93      0.93      0.93       624


# jitter 0.3, crop 280
=== FINAL TEST METRICS ===
Test Accuracy: 0.9407
Recall     : 0.9744
Specificity: 0.8846
F1 Score   : 0.9536
AUC        : 0.9844

Classification Report:
              precision    recall  f1-score   support

      Normal       0.95      0.88      0.92       234
   Pneumonia       0.93      0.97      0.95       390

    accuracy                           0.94       624
   macro avg       0.94      0.93      0.94       624
weighted avg       0.94      0.94      0.94       624

# jitter 0.3, crop 260
=== FINAL TEST METRICS ===
Test Accuracy: 0.9183
Recall     : 0.9667
Specificity: 0.8376
F1 Score   : 0.9366
AUC        : 0.9741

Classification Report:
              precision    recall  f1-score   support

      Normal       0.94      0.84      0.88       234
   Pneumonia       0.91      0.97      0.94       390

    accuracy                           0.92       624
   macro avg       0.92      0.90      0.91       624
weighted avg       0.92      0.92      0.92       624

# jitter 0.4, crop 280
=== FINAL TEST METRICS ===
Test Accuracy: 0.9327
Recall     : 0.9615
Specificity: 0.8846
F1 Score   : 0.9470
AUC        : 0.9806

Classification Report:
              precision    recall  f1-score   support

      Normal       0.93      0.88      0.91       234
   Pneumonia       0.93      0.96      0.95       390

    accuracy                           0.93       624
   macro avg       0.93      0.92      0.93       624
weighted avg       0.93      0.93      0.93       624

# jitter 0.3, crop 300
=== FINAL TEST METRICS ===
Test Accuracy: 0.9087
Recall     : 0.9590
Specificity: 0.8248
F1 Score   : 0.9292
AUC        : 0.9700

Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.82      0.87       234
   Pneumonia       0.90      0.96      0.93       390

    accuracy                           0.91       624
   macro avg       0.91      0.89      0.90       624
weighted avg       0.91      0.91      0.91       624

# jitter 0.1, crop 224
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

# jitter 0.3, random crop 300
=== FINAL TEST METRICS ===
Test Accuracy: 0.9199
Recall     : 0.9718
Specificity: 0.8333
F1 Score   : 0.9381
AUC        : 0.9677

Classification Report:
              precision    recall  f1-score   support

      Normal       0.95      0.83      0.89       234
   Pneumonia       0.91      0.97      0.94       390

    accuracy                           0.92       624
   macro avg       0.93      0.90      0.91       624
weighted avg       0.92      0.92      0.92       624

# jitter 0.3, random resize crop 0.8 to 1
=== FINAL TEST METRICS ===
Test Accuracy: 0.9038
Recall     : 0.9949
Specificity: 0.7521
F1 Score   : 0.9282
AUC        : 0.9794

Classification Report:
              precision    recall  f1-score   support

      Normal       0.99      0.75      0.85       234
   Pneumonia       0.87      0.99      0.93       390

    accuracy                           0.90       624
   macro avg       0.93      0.87      0.89       624
weighted avg       0.91      0.90      0.90       624

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