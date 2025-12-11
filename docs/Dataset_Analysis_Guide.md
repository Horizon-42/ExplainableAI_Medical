# 📊 Dataset Analysis Guide for Medical Image Classification

## A Comprehensive Guide to Understanding and Analyzing Medical Image Datasets

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Why Dataset Analysis Matters](#2-why-dataset-analysis-matters)
3. [Key Concepts](#3-key-concepts)
4. [Class Distribution Analysis](#4-class-distribution-analysis)
5. [Image Statistics](#5-image-statistics)
6. [Pixel Intensity Analysis](#6-pixel-intensity-analysis)
7. [Dataset Comparison](#7-dataset-comparison)
8. [Data Quality Checks](#8-data-quality-checks)
9. [Visualization Techniques](#9-visualization-techniques)
10. [Best Practices](#10-best-practices)
11. [Code Examples](#11-code-examples)

---

## 1. Introduction

Dataset analysis is the first and most crucial step in any machine learning project. Before training a model, you must understand your data thoroughly. This guide covers essential techniques for analyzing medical image datasets.

### What You'll Learn

```
✓ How to analyze class distributions
✓ How to extract and visualize image statistics
✓ How to detect data quality issues
✓ How to compare train/val/test splits
✓ How to identify potential biases
```

---

## 2. Why Dataset Analysis Matters

### 2.1 Common Problems Discovered Through Analysis

| Problem | Impact | How Analysis Helps |
|---------|--------|-------------------|
| **Class Imbalance** | Model biased toward majority class | Reveals need for resampling/weighting |
| **Data Leakage** | Overly optimistic performance | Detects overlap between splits |
| **Distribution Shift** | Poor generalization | Compares train vs test distributions |
| **Image Quality Issues** | Training instability | Identifies corrupted/unusual images |
| **Inconsistent Preprocessing** | Model confusion | Reveals dimension/intensity variations |

### 2.2 The Dataset Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET ANALYSIS PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Data          ──→  Read paths, labels, metadata        │
│         ↓                                                       │
│  2. Overview Stats     ──→  Count samples, classes, splits      │
│         ↓                                                       │
│  3. Class Analysis     ──→  Distribution, imbalance ratio       │
│         ↓                                                       │
│  4. Image Statistics   ──→  Dimensions, file sizes, formats     │
│         ↓                                                       │
│  5. Pixel Analysis     ──→  Intensity distributions by class    │
│         ↓                                                       │
│  6. Quality Checks     ──→  Duplicates, corrupted, leakage      │
│         ↓                                                       │
│  7. Visualization      ──→  Sample images, average images       │
│         ↓                                                       │
│  8. Report             ──→  Summary findings, recommendations   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Concepts

### 3.1 Train/Validation/Test Splits

```
Total Dataset
     │
     ├── Training Set (60-80%)
     │   └── Used to train the model
     │
     ├── Validation Set (10-20%)
     │   └── Used to tune hyperparameters
     │
     └── Test Set (10-20%)
         └── Used for final evaluation only
```

**Key Rules:**
- **Never** look at test set during model development
- Validation and test should have similar distributions
- Use stratified splitting to maintain class proportions

### 3.2 Class Imbalance

**Definition**: When one class has significantly more samples than another.

```
Balanced:     Class A: 50%  |  Class B: 50%
Imbalanced:   Class A: 20%  |  Class B: 80%
Severe:       Class A: 5%   |  Class B: 95%
```

**Imbalance Ratio**: The ratio of majority to minority class samples.

$$\text{Imbalance Ratio} = \frac{N_{majority}}{N_{minority}}$$

**Example:**
- 1000 Pneumonia, 400 Normal
- Ratio = 1000/400 = 2.5
- Written as 1:2.5 (Normal:Pneumonia)

### 3.3 Class Weights

To handle imbalance, assign higher weights to minority class:

$$w_c = \frac{N_{total}}{N_{classes} \times N_c}$$

Where:
- $w_c$ = weight for class c
- $N_{total}$ = total number of samples
- $N_{classes}$ = number of classes
- $N_c$ = number of samples in class c

**Example:**
```python
# 1000 Pneumonia, 400 Normal (1400 total, 2 classes)
weight_normal = 1400 / (2 × 400) = 1.75
weight_pneumonia = 1400 / (2 × 1000) = 0.70
```

---

## 4. Class Distribution Analysis

### 4.1 What to Analyze

1. **Sample counts** per class per split
2. **Percentages** of each class
3. **Imbalance ratio**
4. **Consistency** across splits

### 4.2 Visualization Types

#### Bar Chart
Best for comparing absolute counts.

```python
import matplotlib.pyplot as plt

classes = ['Normal', 'Pneumonia']
train_counts = [400, 1000]
test_counts = [100, 250]

x = range(len(classes))
width = 0.35

plt.bar([i - width/2 for i in x], train_counts, width, label='Train')
plt.bar([i + width/2 for i in x], test_counts, width, label='Test')
plt.xticks(x, classes)
plt.ylabel('Count')
plt.legend()
plt.title('Class Distribution')
plt.show()
```

#### Pie Chart
Best for showing proportions.

```python
plt.pie([400, 1000], labels=['Normal', 'Pneumonia'], 
        autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Class Proportions')
plt.show()
```

#### Stacked Bar Chart
Best for comparing proportions across datasets.

```python
# Proportions
train_props = [400/1400*100, 1000/1400*100]
test_props = [100/350*100, 250/350*100]

datasets = ['Train', 'Test']
plt.bar(datasets, [train_props[0], test_props[0]], label='Normal')
plt.bar(datasets, [train_props[1], test_props[1]], 
        bottom=[train_props[0], test_props[0]], label='Pneumonia')
plt.ylabel('Percentage')
plt.legend()
plt.show()
```

### 4.3 Statistical Test: Chi-Square

Use chi-square test to check if class distributions differ significantly between splits.

```python
from scipy.stats import chi2_contingency

# Contingency table
#              Train  Val   Test
# Normal       320    80    100
# Pneumonia    800    200   250

contingency = [[320, 80, 100],   # Normal
               [800, 200, 250]]  # Pneumonia

chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Distributions are SIGNIFICANTLY DIFFERENT")
else:
    print("Distributions are SIMILAR (good!)")
```

---

## 5. Image Statistics

### 5.1 Dimensions Analysis

Important to check:
- **Width and Height**: Range, mean, std
- **Aspect Ratio**: Width/Height
- **File Size**: Indicator of quality/compression

```python
from PIL import Image
import numpy as np

def get_image_stats(paths):
    stats = []
    for path in paths:
        img = Image.open(path)
        w, h = img.size
        file_size = os.path.getsize(path) / 1024  # KB
        stats.append({
            'width': w,
            'height': h,
            'aspect_ratio': w / h,
            'file_size_kb': file_size,
            'channels': len(img.getbands())
        })
    return pd.DataFrame(stats)
```

### 5.2 What to Look For

| Statistic | Healthy Range | Red Flag |
|-----------|--------------|----------|
| Width variation | < 20% std | Images of vastly different sizes |
| Aspect ratio | 0.8 - 1.2 | Many non-square images |
| File size | Consistent | Bimodal distribution |
| Channels | All same | Mix of grayscale and RGB |

### 5.3 Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Width distribution
axes[0,0].hist(df['width'], bins=30)
axes[0,0].set_xlabel('Width (pixels)')
axes[0,0].set_title('Width Distribution')

# Height distribution
axes[0,1].hist(df['height'], bins=30)
axes[0,1].set_xlabel('Height (pixels)')
axes[0,1].set_title('Height Distribution')

# Width vs Height scatter
axes[1,0].scatter(df['width'], df['height'], alpha=0.5)
axes[1,0].set_xlabel('Width')
axes[1,0].set_ylabel('Height')
axes[1,0].set_title('Width vs Height')

# File size distribution
axes[1,1].hist(df['file_size_kb'], bins=30)
axes[1,1].set_xlabel('File Size (KB)')
axes[1,1].set_title('File Size Distribution')

plt.tight_layout()
plt.show()
```

---

## 6. Pixel Intensity Analysis

### 6.1 Why It Matters

Pixel intensity distributions can reveal:
- **Acquisition differences** between classes
- **Preprocessing inconsistencies**
- **Scanner/equipment variations**
- **Potential shortcuts** the model might learn

### 6.2 Key Statistics

For each image, compute:
- **Mean intensity**: Average brightness
- **Standard deviation**: Contrast measure
- **Min/Max**: Dynamic range
- **Histogram**: Full distribution

```python
def analyze_intensity(image_path):
    img = np.array(Image.open(image_path).convert('L'))  # Grayscale
    return {
        'mean': img.mean(),
        'std': img.std(),
        'min': img.min(),
        'max': img.max(),
        'median': np.median(img)
    }
```

### 6.3 Comparing Classes

```python
# Box plot by class
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, x='class', y='mean_intensity', ax=axes[0])
axes[0].set_title('Mean Intensity by Class')

sns.boxplot(data=df, x='class', y='std_intensity', ax=axes[1])
axes[1].set_title('Intensity Std by Class')

plt.tight_layout()
plt.show()
```

### 6.4 Average Histogram

Compare average pixel histograms between classes:

```python
def compute_avg_histogram(image_paths, bins=256):
    histograms = []
    for path in image_paths:
        img = np.array(Image.open(path).convert('L'))
        hist, _ = np.histogram(img, bins=bins, range=(0, 256))
        histograms.append(hist)
    return np.mean(histograms, axis=0)

# Compare classes
normal_hist = compute_avg_histogram(normal_paths)
pneumonia_hist = compute_avg_histogram(pneumonia_paths)

plt.plot(range(256), normal_hist, label='Normal', alpha=0.7)
plt.plot(range(256), pneumonia_hist, label='Pneumonia', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Average Frequency')
plt.title('Average Pixel Histogram by Class')
plt.legend()
plt.show()
```

---

## 7. Dataset Comparison

### 7.1 Train vs Test Distribution

Ensure train and test have similar distributions. Differences indicate **distribution shift**.

```python
# Compare class proportions
train_prop = len(train_df[train_df['label']==1]) / len(train_df)
test_prop = len(test_df[test_df['label']==1]) / len(test_df)

print(f"Train Pneumonia %: {train_prop*100:.1f}%")
print(f"Test Pneumonia %: {test_prop*100:.1f}%")
print(f"Difference: {abs(train_prop - test_prop)*100:.1f}%")
```

### 7.2 Feature Distribution Comparison

Use statistical tests to compare distributions:

```python
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test

# Compare mean intensity distributions
stat, p_value = ks_2samp(train_intensity['mean'], test_intensity['mean'])

if p_value < 0.05:
    print("⚠️ Intensity distributions DIFFER between train and test")
else:
    print("✓ Intensity distributions are similar")
```

### 7.3 Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Overlapping histograms
axes[0].hist(train_df['intensity_mean'], alpha=0.5, label='Train', bins=30)
axes[0].hist(test_df['intensity_mean'], alpha=0.5, label='Test', bins=30)
axes[0].legend()
axes[0].set_title('Mean Intensity Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(train_df['intensity_mean'], dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Train)')

# Box plot comparison
combined = pd.concat([
    train_df.assign(split='Train'),
    test_df.assign(split='Test')
])
sns.boxplot(data=combined, x='split', y='intensity_mean', ax=axes[2])
axes[2].set_title('Mean Intensity by Split')

plt.tight_layout()
plt.show()
```

---

## 8. Data Quality Checks

### 8.1 Checklist

| Check | Method | Action if Failed |
|-------|--------|-----------------|
| Duplicate files | Hash comparison | Remove duplicates |
| Missing files | os.path.exists() | Fix paths or remove |
| Corrupted images | Try to open | Remove or re-download |
| Wrong format | Check extension/header | Convert or remove |
| Very small images | Check dimensions | Remove or flag |
| Data leakage | Compare filenames | Re-split data |

### 8.2 Code Implementation

```python
import hashlib

def check_data_quality(df, name):
    """Comprehensive data quality check."""
    issues = []
    
    # 1. Duplicate paths
    dup_paths = df['path'].duplicated().sum()
    if dup_paths > 0:
        issues.append(f"Found {dup_paths} duplicate paths")
    
    # 2. Missing files
    missing = sum(1 for p in df['path'] if not os.path.exists(p))
    if missing > 0:
        issues.append(f"Found {missing} missing files")
    
    # 3. Corrupted images
    corrupted = []
    for path in df['path']:
        try:
            img = Image.open(path)
            img.verify()  # Verify it's a valid image
        except:
            corrupted.append(path)
    if corrupted:
        issues.append(f"Found {len(corrupted)} corrupted images")
    
    # 4. Very small images
    small = []
    for path in df['path']:
        try:
            img = Image.open(path)
            if img.size[0] < 50 or img.size[1] < 50:
                small.append(path)
        except:
            pass
    if small:
        issues.append(f"Found {len(small)} very small images (<50px)")
    
    # 5. Check for duplicate content (by hash)
    hashes = {}
    duplicates = []
    for path in df['path']:
        try:
            with open(path, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
                if h in hashes:
                    duplicates.append((path, hashes[h]))
                else:
                    hashes[h] = path
        except:
            pass
    if duplicates:
        issues.append(f"Found {len(duplicates)} content duplicates")
    
    return issues

# Run checks
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    issues = check_data_quality(df, name)
    print(f"\n{name} Set:")
    if issues:
        for issue in issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  ✓ All checks passed")
```

### 8.3 Data Leakage Detection

```python
def check_leakage(train_df, val_df, test_df):
    """Check for data leakage between splits."""
    
    # Method 1: Check filenames
    train_files = set(os.path.basename(p) for p in train_df['path'])
    val_files = set(os.path.basename(p) for p in val_df['path'])
    test_files = set(os.path.basename(p) for p in test_df['path'])
    
    train_val = train_files & val_files
    train_test = train_files & test_files
    val_test = val_files & test_files
    
    print(f"Train ∩ Val: {len(train_val)} files")
    print(f"Train ∩ Test: {len(train_test)} files")
    print(f"Val ∩ Test: {len(val_test)} files")
    
    # Method 2: Check by content hash
    # (More thorough but slower)
    
    if len(train_test) > 0:
        print("\n⚠️ DATA LEAKAGE DETECTED!")
        print(f"Sample: {list(train_test)[:3]}")
    else:
        print("\n✓ No data leakage detected")

check_leakage(train_df, val_df, test_df)
```

---

## 9. Visualization Techniques

### 9.1 Sample Images Grid

```python
def plot_sample_grid(df, title, rows=2, cols=5):
    """Display a grid of sample images."""
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    samples = df.sample(min(rows*cols, len(df)))
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        ax = axes[idx // cols, idx % cols]
        img = Image.open(row['path'])
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{row['class_name']}\n{img.size}")
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_sample_grid(train_df, "Training Set Samples")
```

### 9.2 Average Image by Class

```python
def compute_average_image(df, label, size=(224, 224), n=100):
    """Compute average image for a class."""
    class_df = df[df['label'] == label].sample(min(n, len(df)))
    
    images = []
    for _, row in class_df.iterrows():
        img = Image.open(row['path']).convert('L').resize(size)
        images.append(np.array(img))
    
    return np.mean(images, axis=0)

# Compute and visualize
avg_normal = compute_average_image(train_df, 0)
avg_pneumonia = compute_average_image(train_df, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(avg_normal, cmap='gray')
axes[0].set_title('Average Normal')
axes[0].axis('off')

axes[1].imshow(avg_pneumonia, cmap='gray')
axes[1].set_title('Average Pneumonia')
axes[1].axis('off')

# Difference image
diff = avg_pneumonia - avg_normal
axes[2].imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
axes[2].set_title('Difference (P - N)')
axes[2].axis('off')

plt.suptitle('Average Images Analysis', fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.3 t-SNE Visualization

Visualize high-dimensional features in 2D:

```python
from sklearn.manifold import TSNE
import torch
from torchvision import transforms, models

# Extract features using pretrained model
def extract_features(paths, model, transform):
    features = []
    for path in tqdm(paths):
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor).squeeze().numpy()
        features.append(feat)
    return np.array(features)

# Load pretrained model (remove classification head)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Extract and visualize
features = extract_features(df['path'][:500], model, transform)
tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
for label, name in [(0, 'Normal'), (1, 'Pneumonia')]:
    mask = df['label'][:500] == label
    plt.scatter(embedded[mask, 0], embedded[mask, 1], label=name, alpha=0.5)
plt.legend()
plt.title('t-SNE Visualization of Image Features')
plt.show()
```

---

## 10. Best Practices

### 10.1 Analysis Checklist

```
□ Load all splits (train, val, test)
□ Create overview table with counts
□ Visualize class distribution
□ Calculate imbalance ratio
□ Analyze image dimensions
□ Analyze pixel intensities
□ Compare distributions across splits
□ Check for data leakage
□ Check for duplicates/corrupted files
□ Visualize sample images
□ Compute average images per class
□ Generate summary report
```

### 10.2 Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Ignoring imbalance | Biased model | Use weights/sampling |
| Not checking leakage | Overfitting | Verify no overlap |
| Skipping visualization | Missing obvious issues | Always look at samples |
| Analyzing only training | Distribution shift | Analyze all splits |
| Single statistic | Missing patterns | Use multiple metrics |

### 10.3 Reporting Template

```markdown
# Dataset Analysis Report

## Overview
- Total samples: X
- Training: X (Y%)
- Validation: X (Y%)
- Test: X (Y%)

## Class Distribution
- Class A: X (Y%)
- Class B: X (Y%)
- Imbalance ratio: 1:Z

## Image Statistics
- Dimensions: min-max, mean
- File sizes: mean ± std

## Key Findings
1. Finding 1
2. Finding 2

## Recommendations
1. Recommendation 1
2. Recommendation 2

## Figures
- Figure 1: Class distribution
- Figure 2: Dimension analysis
- Figure 3: Sample images
```

---

## 11. Code Examples

### 11.1 Complete Analysis Function

```python
def complete_dataset_analysis(train_df, val_df, test_df, output_dir='figures'):
    """
    Perform complete dataset analysis and save all figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overview
    print("="*60)
    print("DATASET ANALYSIS REPORT")
    print("="*60)
    
    datasets = {'Train': train_df, 'Val': val_df, 'Test': test_df}
    
    for name, df in datasets.items():
        n_normal = len(df[df['label'] == 0])
        n_pneumonia = len(df[df['label'] == 1])
        print(f"\n{name}: {len(df)} images")
        print(f"  Normal: {n_normal} ({n_normal/len(df)*100:.1f}%)")
        print(f"  Pneumonia: {n_pneumonia} ({n_pneumonia/len(df)*100:.1f}%)")
    
    # 2. Visualizations
    # ... (add all visualization code)
    
    # 3. Quality checks
    print("\n" + "="*60)
    print("QUALITY CHECKS")
    print("="*60)
    
    for name, df in datasets.items():
        issues = check_data_quality(df, name)
        print(f"\n{name}: ", end="")
        print("✓ Passed" if not issues else f"⚠️ {len(issues)} issues")
    
    # 4. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    return {
        'overview': create_overview_table(train_df, val_df, test_df),
        'train_dims': analyze_dimensions(train_df, 'Train'),
        # ... more results
    }

# Usage
results = complete_dataset_analysis(train_df, val_df, test_df)
```

### 11.2 Quick Analysis Script

```python
#!/usr/bin/env python3
"""Quick dataset analysis script."""

import sys
from pathlib import Path

def quick_analysis(data_dir):
    """Run quick analysis on a dataset directory."""
    
    # Count files
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.*')))
            print(f"{class_dir.name}: {count} files")
    
    # Check for common issues
    all_files = list(Path(data_dir).rglob('*.*'))
    print(f"\nTotal files: {len(all_files)}")
    
    # File types
    extensions = {}
    for f in all_files:
        ext = f.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    print(f"File types: {extensions}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        quick_analysis(sys.argv[1])
    else:
        print("Usage: python quick_analysis.py <data_dir>")
```

---

## Further Reading

### Papers
1. "A Survey on Data Collection for Machine Learning" - Roh et al., 2019
2. "Data Quality for Machine Learning Tasks" - Budach et al., 2022

### Tools
- **pandas-profiling**: Automatic EDA reports
- **sweetviz**: Visual comparison of datasets
- **Great Expectations**: Data validation library

### Tutorials
- Kaggle: "Data Visualization" course
- Fast.ai: "Practical Deep Learning" - Data chapter

---

*Document created for the Explainable AI Medical Image Classification project*
*Last updated: December 2024*

