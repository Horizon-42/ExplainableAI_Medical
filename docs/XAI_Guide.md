# Explainable AI (XAI) for Medical Image Classification
## A Comprehensive Guide to Understanding and Implementing XAI Methods

---

## Table of Contents

1. [Introduction to Explainable AI](#1-introduction-to-explainable-ai)
2. [Why XAI Matters in Medical Imaging](#2-why-xai-matters-in-medical-imaging)
3. [CAM - Class Activation Mapping](#3-cam---class-activation-mapping)
4. [LIME - Local Interpretable Model-agnostic Explanations](#4-lime---local-interpretable-model-agnostic-explanations)
5. [SHAP - SHapley Additive exPlanations](#5-shap---shapley-additive-explanations)
6. [LRP - Layer-wise Relevance Propagation](#6-lrp---layer-wise-relevance-propagation)
7. [Comparison of Methods](#7-comparison-of-methods)
8. [Best Practices](#8-best-practices)
9. [Further Reading](#9-further-reading)

---

## 1. Introduction to Explainable AI

### What is Explainable AI?

Explainable AI (XAI) refers to methods and techniques that make the decisions of AI systems understandable to humans. In the context of deep learning, XAI helps answer the question: **"Why did the model make this prediction?"**

### Types of Explanations

| Type | Description | Example Methods |
|------|-------------|-----------------|
| **Local** | Explains individual predictions | LIME, SHAP (local), LRP |
| **Global** | Explains overall model behavior | Feature importance, SHAP (global) |
| **Post-hoc** | Applied after model training | LIME, SHAP, CAM |
| **Intrinsic** | Built into the model | Attention mechanisms, Decision trees |

### Key Concepts

```
Attribution: Assigning importance scores to input features
Saliency: Highlighting regions that influence the prediction
Perturbation: Modifying inputs to observe changes in output
Backpropagation: Tracing gradients from output to input
```

---

## 2. Why XAI Matters in Medical Imaging

### Critical Importance in Healthcare

1. **Trust**: Doctors need to understand AI recommendations before acting on them
2. **Validation**: Ensure the model focuses on clinically relevant features
3. **Debugging**: Identify if the model learns spurious correlations
4. **Regulatory**: Many jurisdictions require explainability for medical AI
5. **Education**: Help medical students understand disease patterns

### Example: Pneumonia Detection

```
Good explanation: Model highlights lung opacities (actual pneumonia signs)
Bad explanation: Model focuses on text annotations or equipment artifacts
```

---

## 3. CAM - Class Activation Mapping

### 3.1 Overview

**CAM (Class Activation Mapping)** generates visual explanations by leveraging the spatial information preserved in convolutional layers.

**Paper**: "Learning Deep Features for Discriminative Localization" (Zhou et al., 2016)

### 3.2 How CAM Works

#### Architecture Requirement
CAM requires a **Global Average Pooling (GAP)** layer before the final classification layer.

```
Input Image → Conv Layers → Feature Maps → GAP → FC Layer → Output
                              ↓
                        [C × H × W]
                              ↓
                    Global Average Pooling
                              ↓
                          [C × 1]
                              ↓
                    Weighted Sum → Class Score
```

#### Mathematical Formulation

For class `c`, the CAM is computed as:

$$M_c(x, y) = \sum_k w_k^c \cdot f_k(x, y)$$

Where:
- $M_c(x, y)$ = CAM value at spatial location (x, y) for class c
- $w_k^c$ = Weight connecting feature map k to class c
- $f_k(x, y)$ = Activation of feature map k at location (x, y)

### 3.3 Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class CAMGenerator:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        
        # Register hook to capture feature maps
        target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.feature_maps = output.detach()
    
    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Get weights from final FC layer
        weights = self.model.fc.weight[class_idx]  # Shape: [num_features]
        
        # Generate CAM
        # feature_maps shape: [1, C, H, W]
        # weights shape: [C]
        cam = torch.zeros(self.feature_maps.shape[2:])
        
        for i, w in enumerate(weights):
            cam += w * self.feature_maps[0, i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam

# Usage
cam_generator = CAMGenerator(model, model.layer4)
cam = cam_generator.generate(input_image)
```

### 3.4 Variants of CAM

| Variant | Key Idea | Advantage |
|---------|----------|-----------|
| **Grad-CAM** | Uses gradients instead of weights | Works with any CNN architecture |
| **Grad-CAM++** | Weighted combination of gradients | Better for multiple objects |
| **Score-CAM** | Uses activation scores as weights | Gradient-free, more stable |
| **Eigen-CAM** | Uses principal components | Class-agnostic |

### 3.5 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Grad-CAM** generalizes CAM to work with any CNN architecture, removing the requirement for a Global Average Pooling layer. It uses the gradients of the target concept flowing into the final convolutional layer to produce a coarse localization map.

#### Mathematical Formulation

To obtain the class-discriminative localization map $L_{Grad-CAM}^c$ for a class $c$:

1.  **Compute Gradients**: Calculate the gradient of the score for class $c$, $y^c$, with respect to the feature map activations $A^k$ of a convolutional layer:
    $$ \frac{\partial y^c}{\partial A^k} $$

2.  **Global Average Pooling (Neuron Importance)**: These gradients are global-average-pooled to obtain the neuron importance weights $\alpha_k^c$:
    $$ \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} $$
    This weight $\alpha_k^c$ captures the "importance" of feature map $k$ for a target class $c$.

3.  **Weighted Combination**: Perform a weighted combination of forward activation maps, followed by a ReLU:
    $$ L_{Grad-CAM}^c = ReLU\left(\sum_k \alpha_k^c A^k\right) $$
    
    *   **ReLU** is applied because we are only interested in features that have a *positive* influence on the class of interest. Negative pixels are likely to belong to other categories.

#### Interpreting the Heatmap

The resulting heatmap $L_{Grad-CAM}^c$ provides a visual explanation of the model's decision:

*   **Intensity**: The value at each pixel represents the strength of the correlation between that region and the target class.
    *   **Hot Areas (Red/Yellow)**: These regions are highly important. The model detected features here that strongly increased the score for the target class (e.g., "lung opacity" for Pneumonia).
    *   **Cold Areas (Blue/Black)**: These regions had little to no positive contribution to the prediction.
*   **Class Specificity**: Grad-CAM is class-specific. If an image contains both a Cat and a Dog:
    *   Visualizing "Cat" will highlight the cat's ears/whiskers.
    *   Visualizing "Dog" will highlight the dog's snout/eyes.
*   **Resolution**: The heatmap is coarse (low resolution) because it comes from the deeper layers of the CNN (e.g., $7 \times 7$ or $14 \times 14$), and is then upsampled to the original image size.

### 3.6 Grad-CAM Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
```

### 3.7 Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Fast computation | ❌ Requires specific architecture (GAP) |
| ✅ No need for additional training | ❌ Coarse resolution (feature map size) |
| ✅ Intuitive visualization | ❌ Only highlights positive contributions |
| ✅ Works well for localization | ❌ May miss fine-grained details |

---

## 4. LIME - Local Interpretable Model-agnostic Explanations

### 4.1 Overview

**LIME** explains individual predictions by approximating the model locally with an interpretable model (e.g., linear regression).

**Paper**: "Why Should I Trust You?" (Ribeiro et al., 2016)

### 4.2 Core Idea

LIME answers: *"What would happen if we slightly changed the input?"*

```
Original Prediction: Pneumonia (95% confidence)

LIME Process:
1. Create perturbations of the input (e.g., hide parts of the image)
2. Get model predictions for each perturbation
3. Fit a simple model to explain the relationship
4. Identify which parts matter most
```

### 4.3 How LIME Works for Images

#### Step-by-Step Process

```
Step 1: Segmentation
        Original Image → Superpixels (using SLIC, Quickshift, etc.)
        
Step 2: Perturbation
        Generate N samples by randomly hiding superpixels
        
Step 3: Prediction
        Get model predictions for all N perturbed images
        
Step 4: Weighting
        Weight samples by similarity to original (using kernel function)
        
Step 5: Fitting
        Fit interpretable model (e.g., linear regression):
        y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
        
Step 6: Explanation
        Superpixels with highest weights are most important
```

### 4.4 Mathematical Formulation

LIME solves the following optimization:

$$ \xi(x) = \operatorname*{argmin}_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

Where:
- $f$ = Original model (black box)
- $g$ = Interpretable model (e.g., linear)
- $\pi_x$ = Proximity measure (kernel) around sample x
- $\mathcal{L}$ = Loss function measuring fidelity
- $\Omega(g)$ = Complexity penalty

#### Understanding the Weights

It is crucial to distinguish between the two types of weights in LIME:

1. **Similarity Weight ($\pi_x$)**:
   - **What**: A weight assigned to each *perturbed sample*.
   - **How**: Calculated using a kernel function (e.g., Exponential Kernel) based on distance from the original image.
   - **Role**: Tells the linear model to prioritize samples that look like the original image.
   - **Formula**: $\pi_x(z) = \exp(-D(x,z)^2/\sigma^2)$

2. **Feature Weight ($w$)**:
   - **What**: A coefficient assigned to each *superpixel*.
   - **How**: Learned by the linear regression model.
   - **Role**: This **IS the explanation**. It indicates how much a superpixel contributes to the prediction.
   - **Interpretation**: Positive = Supports prediction; Negative = Opposes prediction.

The linear model is trained by minimizing the **Weighted Squared Loss**:

$$ \mathcal{L}(f, g, \pi_x) = \sum_{z} \underbrace{\pi_x(z)}_{\text{Similarity Weight}} \cdot \left( f(z) - \underbrace{w \cdot z}_{\text{Linear Pred}} \right)^2 $$

### 4.5 Implementation

```python
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np

class LIMEExplainer:
    def __init__(self, model, preprocess_fn):
        """
        Args:
            model: PyTorch model
            preprocess_fn: Function to preprocess images for the model
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.explainer = lime_image.LimeImageExplainer(random_state=42)
    
    def predict_batch(self, images):
        """
        Prediction function required by LIME.
        Input: numpy array of images [N, H, W, C], values in [0, 255]
        Output: numpy array of probabilities [N, num_classes]
        """
        self.model.eval()
        batch = torch.stack([self.preprocess_fn(img) for img in images])
        
        with torch.no_grad():
            outputs = self.model(batch.to(device))
            # For binary classification
            probs = torch.sigmoid(outputs)
            # Return [P(Normal), P(Pneumonia)]
            return torch.cat([1 - probs, probs], dim=1).cpu().numpy()
    
    def explain(self, image, num_samples=1000, num_features=10):
        """
        Generate LIME explanation for an image.
        
        Args:
            image: PIL Image or numpy array [H, W, C]
            num_samples: Number of perturbations to generate
            num_features: Number of superpixels to show
        
        Returns:
            explanation: LIME explanation object
        """
        # Convert to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        elif hasattr(image, '__array__'):
            image = np.array(image)
        
        explanation = self.explainer.explain_instance(
            image,
            self.predict_batch,
            top_labels=2,
            hide_color=0,  # Black out hidden superpixels
            num_samples=num_samples,
            num_features=num_features,
            segmentation_fn=SegmentationAlgorithm(
                'quickshift',
                kernel_size=4,
                max_dist=200,
                ratio=0.2
            )
        )
        
        return explanation
    
    def visualize(self, explanation, label=1, positive_only=True):
        """
        Visualize the LIME explanation.
        
        Args:
            explanation: LIME explanation object
            label: Class label to explain
            positive_only: Show only positive contributions
        """
        from skimage.segmentation import mark_boundaries
        import matplotlib.pyplot as plt
        
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=10,
            hide_rest=False
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original
        axes[0].imshow(explanation.image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Segments
        axes[1].imshow(mark_boundaries(explanation.image, explanation.segments))
        axes[1].set_title('Superpixels')
        axes[1].axis('off')
        
        # Explanation
        axes[2].imshow(mark_boundaries(temp / 255.0, mask))
        axes[2].set_title(f'LIME Explanation (class={label})')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig

# Usage
explainer = LIMEExplainer(model, preprocess_fn)
explanation = explainer.explain(image, num_samples=1000)
fig = explainer.visualize(explanation, label=1)
```

### 4.6 Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_samples` | Number of perturbations | 500-5000 |
| `num_features` | Superpixels to highlight | 5-20 |
| `kernel_size` | Superpixel size | 1-10 |
| `hide_color` | Color for hidden regions | 0 (black), mean, blur |

### 4.7 Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Model-agnostic | ❌ Slow (requires many forward passes) |
| ✅ Human-interpretable output | ❌ Randomness in explanations |
| ✅ Works with any input type | ❌ Superpixel boundaries may not align with features |
| ✅ No access to model internals needed | ❌ Local approximation may be poor |

---

## 5. SHAP - SHapley Additive exPlanations

### 5.1 Overview

**SHAP** (SHapley Additive exPlanations) uses game theory (Shapley values) to fairly distribute the "credit" for a prediction among input features.

**Paper**: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)

### 5.2 The Shapley Value Concept

#### Game Theory Analogy

Imagine a team game where players contribute to winning:
- **Players** = Input features (pixels, superpixels)
- **Game outcome** = Model prediction
- **Shapley value** = Fair contribution of each player

#### The Key Question

*"How much does each feature contribute to the difference between the actual prediction and the average prediction?"*

### 5.3 Mathematical Formulation

The Shapley value for feature $i$ is:

$$\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

Where:
- $N$ = Set of all features
- $S$ = Subset of features (coalition)
- $f(S)$ = Model output when only features in S are "present"
- The sum is over all possible coalitions

### 5.4 SHAP Properties (Desirable Axioms)

| Property | Description |
|----------|-------------|
| **Local Accuracy** | Sum of SHAP values = prediction - expected value |
| **Missingness** | Missing features have SHAP value = 0 |
| **Consistency** | If a feature's contribution increases, its SHAP value doesn't decrease |

```
Prediction = Base Value + Σ(SHAP values)

Example:
Base value (average prediction): 0.3
SHAP(feature 1): +0.2
SHAP(feature 2): -0.1
SHAP(feature 3): +0.1
-----------------------------
Final prediction: 0.3 + 0.2 - 0.1 + 0.1 = 0.5
```

### 5.5 SHAP for Images

For images, computing exact Shapley values is intractable (2^n coalitions for n pixels). SHAP uses approximations:

#### Kernel SHAP
- Samples coalitions weighted by Shapley kernel
- Model-agnostic but slow

#### Deep SHAP
- Uses DeepLIFT-style backpropagation
- Fast but requires model access

#### Partition SHAP (for images)
- Groups pixels into superpixels
- Uses hierarchical clustering for efficiency

### 5.6 Implementation

```python
import shap
import numpy as np
import torch

class SHAPExplainer:
    def __init__(self, model, device, masker_type="blur"):
        """
        Args:
            model: PyTorch model
            device: torch device
            masker_type: "blur", "inpaint", or a background dataset
        """
        self.model = model
        self.device = device
        self.masker_type = masker_type
    
    def predict_fn(self, images):
        """
        Prediction function for SHAP.
        Input: uint8 images [N, C, H, W] or [N, H, W, C]
        Output: probabilities [N, 1] or [N, num_classes]
        """
        self.model.eval()
        
        # Handle different input formats
        if images.shape[-1] == 3:  # [N, H, W, C]
            images = np.transpose(images, (0, 3, 1, 2))  # -> [N, C, H, W]
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images_norm = (images.astype(np.float32) / 255.0 - mean) / std
        
        # Predict
        with torch.no_grad():
            tensor = torch.from_numpy(images_norm).float().to(self.device)
            outputs = self.model(tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        return probs
    
    def explain(self, images, max_evals=500):
        """
        Compute SHAP values for images.
        
        Args:
            images: numpy array [N, C, H, W], uint8
            max_evals: Maximum number of model evaluations
        
        Returns:
            shap_values: SHAP Explanation object
        """
        # Create masker
        if self.masker_type == "blur":
            masker = shap.maskers.Image("blur(128,128)", images[0].shape)
        elif self.masker_type == "inpaint":
            masker = shap.maskers.Image("inpaint_telea", images[0].shape)
        else:
            masker = shap.maskers.Image(self.masker_type, images[0].shape)
        
        # Create explainer
        explainer = shap.Explainer(self.predict_fn, masker)
        
        # Compute SHAP values
        shap_values = explainer(
            images,
            max_evals=max_evals,
            batch_size=50
        )
        
        return shap_values
    
    def visualize(self, shap_values, images):
        """
        Visualize SHAP explanations.
        """
        import matplotlib.pyplot as plt
        
        # Transpose for visualization: [N, C, H, W] -> [N, H, W, C]
        if images.shape[1] == 3:
            images_vis = np.transpose(images, (0, 2, 3, 1))
        else:
            images_vis = images
        
        # Get SHAP values
        sv = shap_values.values
        if sv.ndim == 5:  # [N, outputs, C, H, W]
            sv = sv[:, 0, :, :, :]  # Take first output
        
        # Transpose SHAP values
        if sv.shape[1] == 3:
            sv = np.transpose(sv, (0, 2, 3, 1))
        
        # Sum across channels
        sv_sum = sv.sum(axis=-1)
        
        # Plot
        n = len(images)
        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            # Original
            axes[i, 0].imshow(images_vis[i])
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # SHAP heatmap
            max_abs = np.abs(sv_sum[i]).max()
            axes[i, 1].imshow(sv_sum[i], cmap='bwr', vmin=-max_abs, vmax=max_abs)
            axes[i, 1].set_title('SHAP Values')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(images_vis[i])
            axes[i, 2].imshow(sv_sum[i], cmap='bwr', alpha=0.5, 
                            vmin=-max_abs, vmax=max_abs)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        return fig

# Usage
explainer = SHAPExplainer(model, device, masker_type="blur")
shap_values = explainer.explain(images, max_evals=500)
fig = explainer.visualize(shap_values, images)
```

### 5.7 Interpreting SHAP Values

```
SHAP Value Interpretation:
┌─────────────────────────────────────────────────────────┐
│  Positive SHAP (Red)  → Increases prediction            │
│  Negative SHAP (Blue) → Decreases prediction            │
│  Zero SHAP (White)    → No effect on prediction         │
│                                                         │
│  Sum of all SHAP values = Prediction - Base Value       │
└─────────────────────────────────────────────────────────┘
```

### 5.8 Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Strong theoretical foundation | ❌ Computationally expensive |
| ✅ Consistent and fair attribution | ❌ Requires many model evaluations |
| ✅ Additive (values sum to prediction) | ❌ Approximations may introduce error |
| ✅ Model-agnostic versions available | ❌ Choice of baseline affects results |

---

## 6. LRP - Layer-wise Relevance Propagation

### 6.1 Overview

**LRP** (Layer-wise Relevance Propagation) explains predictions by backpropagating "relevance" scores from the output layer to the input layer.

**Paper**: "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation" (Bach et al., 2015)

### 6.2 Core Principle: Conservation

LRP is based on the **conservation principle**:

$$\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)} = ... = f(x)$$

The total relevance is conserved across layers, starting from the output prediction.

### 6.3 LRP Rules

#### Basic LRP-0 Rule

$$R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_k a_k w_{kj}} R_j^{(l+1)}$$

Where:
- $R_i^{(l)}$ = Relevance of neuron i in layer l
- $a_i$ = Activation of neuron i
- $w_{ij}$ = Weight from neuron i to neuron j

#### LRP-ε Rule (Numerical Stability)

$$R_i^{(l)} = \sum_j \frac{a_i w_{ij}}{\sum_k a_k w_{kj} + \epsilon \cdot \text{sign}(\sum_k a_k w_{kj})} R_j^{(l+1)}$$

The small $\epsilon$ prevents division by zero and absorbs some relevance.

#### LRP-γ Rule (Positive Emphasis)

$$R_i^{(l)} = \sum_j \frac{a_i (w_{ij} + \gamma w_{ij}^+)}{\sum_k a_k (w_{kj} + \gamma w_{kj}^+)} R_j^{(l+1)}$$

Where $w^+$ denotes positive weights. Higher $\gamma$ emphasizes positive contributions.

#### LRP-αβ Rule (Separate Positive/Negative)

$$R_i^{(l)} = \sum_j \left( \alpha \frac{(a_i w_{ij})^+}{\sum_k (a_k w_{kj})^+} - \beta \frac{(a_i w_{ij})^-}{\sum_k (a_k w_{kj})^-} \right) R_j^{(l+1)}$$

Constraint: $\alpha - \beta = 1$ (common choice: α=2, β=1 or α=1, β=0)

### 6.4 Layer-Specific Rules

Different rules work better for different layer types:

| Layer Type | Recommended Rule | Reason |
|------------|------------------|--------|
| Dense (upper) | LRP-ε | Stability |
| Dense (lower) | LRP-γ | Sparsity |
| Conv (upper) | LRP-ε | Stability |
| Conv (lower) | LRP-γ or LRP-αβ | Better visualization |
| Input layer | LRP-zB | Bounded input |

### 6.5 Implementation with Captum

```python
from captum.attr import LRP
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LRPExplainer:
    def __init__(self, model, device):
        """
        Initialize LRP explainer using Captum library.
        
        Args:
            model: PyTorch model
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        self.lrp = LRP(model)
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def explain(self, input_tensor, target=None):
        """
        Compute LRP attribution.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            target: Target class index (None for predicted class)
        
        Returns:
            attribution: numpy array [H, W]
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Get prediction if target not specified
        if target is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if output.shape[-1] == 1:  # Binary
                    target = 0  # Single output neuron
                else:
                    target = output.argmax(dim=1).item()
        
        # Compute LRP
        attribution = self.lrp.attribute(input_tensor, target=target)
        
        # Process attribution
        attr_np = attribution.squeeze().cpu().detach().numpy()
        
        # Sum across channels if needed
        if attr_np.ndim == 3:
            attr_np = attr_np.sum(axis=0)
        
        return attr_np
    
    def visualize(self, input_tensor, attribution, title="LRP Explanation"):
        """
        Visualize LRP attribution.
        """
        import cv2
        
        # Denormalize input for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = input_tensor.squeeze().cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # LRP heatmap (diverging colormap)
        max_abs = np.abs(attribution).max() + 1e-10
        axes[1].imshow(attribution, cmap='seismic', vmin=-max_abs, vmax=max_abs)
        axes[1].set_title('LRP Attribution\n(Red=Positive, Blue=Negative)')
        axes[1].axis('off')
        
        # Positive relevance only
        pos_attr = np.maximum(attribution, 0)
        axes[2].imshow(pos_attr, cmap='Reds')
        axes[2].set_title('Positive Relevance')
        axes[2].axis('off')
        
        # Overlay
        attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        heatmap = plt.cm.jet(attr_norm)[:, :, :3]
        overlay = 0.6 * img + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

# Usage
explainer = LRPExplainer(model, device)
attribution = explainer.explain(input_tensor)
fig = explainer.visualize(input_tensor, attribution)
```

### 6.6 Manual LRP Implementation (Educational)

```python
def lrp_linear(layer, a, R, eps=1e-6):
    """
    LRP for linear/dense layer.
    
    Args:
        layer: nn.Linear layer
        a: Input activation [batch, in_features]
        R: Relevance from next layer [batch, out_features]
        eps: Small constant for numerical stability
    
    Returns:
        R_prev: Relevance for previous layer [batch, in_features]
    """
    W = layer.weight  # [out, in]
    
    # Compute z = a @ W.T + b
    z = torch.mm(a, W.T) + layer.bias
    
    # Stabilize
    z = z + eps * torch.sign(z)
    z[z == 0] = eps
    
    # Compute s = R / z
    s = R / z
    
    # Backpropagate: R_prev = a * (s @ W)
    R_prev = a * torch.mm(s, W)
    
    return R_prev

def lrp_conv2d(layer, a, R, eps=1e-6):
    """
    LRP for convolutional layer.
    
    Args:
        layer: nn.Conv2d layer
        a: Input activation [batch, C_in, H, W]
        R: Relevance from next layer [batch, C_out, H', W']
        eps: Small constant
    
    Returns:
        R_prev: Relevance [batch, C_in, H, W]
    """
    # Forward pass to get z
    z = F.conv2d(a, layer.weight, layer.bias, 
                 layer.stride, layer.padding)
    z = z + eps * torch.sign(z)
    z[z == 0] = eps
    
    # Element-wise division
    s = R / z
    
    # Backward pass using transposed convolution
    c = F.conv_transpose2d(s, layer.weight, stride=layer.stride, 
                           padding=layer.padding)
    
    # Multiply by input
    R_prev = a * c
    
    return R_prev

def lrp_relu(a, R):
    """
    LRP passes through ReLU unchanged.
    """
    return R

def lrp_pool(pool_layer, a, R, indices=None):
    """
    LRP for pooling layer.
    Redistributes relevance to the positions that were selected.
    """
    if isinstance(pool_layer, nn.MaxPool2d):
        # Use indices from forward pass
        R_prev = F.max_unpool2d(R, indices, pool_layer.kernel_size,
                                pool_layer.stride, pool_layer.padding)
    elif isinstance(pool_layer, nn.AvgPool2d):
        # Distribute equally
        R_prev = F.interpolate(R, scale_factor=pool_layer.kernel_size)
        R_prev = R_prev / (pool_layer.kernel_size ** 2)
    
    return R_prev
```

### 6.7 Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Pixel-level attribution | ❌ Architecture-dependent rules |
| ✅ Theoretically principled | ❌ Skip connections need special handling |
| ✅ Fast (single backward pass) | ❌ BatchNorm layers can be problematic |
| ✅ Conservation property | ❌ Multiple rule variants to choose from |

---

## 7. Comparison of Methods

### 7.1 Summary Table

| Aspect | CAM | LIME | SHAP | LRP |
|--------|-----|------|------|-----|
| **Type** | Gradient | Perturbation | Perturbation | Backpropagation |
| **Model Access** | Weights needed | Black-box | Black-box* | Weights needed |
| **Speed** | Fast | Slow | Slow | Fast |
| **Resolution** | Coarse | Superpixel | Superpixel | Pixel |
| **Theory** | Heuristic | Local surrogate | Game theory | Conservation |
| **Consistency** | No | No | Yes | Yes |

*Deep SHAP requires model access

### 7.2 When to Use Each Method

```
┌─────────────────────────────────────────────────────────────────┐
│ Use CAM when:                                                   │
│ • You need fast explanations                                    │
│ • Coarse localization is sufficient                             │
│ • Using CNN with GAP layer                                      │
├─────────────────────────────────────────────────────────────────┤
│ Use LIME when:                                                  │
│ • Model is a black box (no internal access)                     │
│ • You need human-interpretable explanations                     │
│ • Superpixel-level granularity is acceptable                    │
├─────────────────────────────────────────────────────────────────┤
│ Use SHAP when:                                                  │
│ • You need theoretically grounded explanations                  │
│ • Consistency and fairness are important                        │
│ • You have computational resources                              │
├─────────────────────────────────────────────────────────────────┤
│ Use LRP when:                                                   │
│ • You need pixel-level detail                                   │
│ • You have access to model architecture                         │
│ • Speed is important                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Visual Comparison

```
Input Image:    [Chest X-ray with pneumonia]
                        ↓
┌──────────┬──────────┬──────────┬──────────┐
│   CAM    │   LIME   │   SHAP   │   LRP    │
├──────────┼──────────┼──────────┼──────────┤
│ Coarse   │ Blocky   │ Smooth   │ Detailed │
│ heatmap  │ regions  │ heatmap  │ heatmap  │
│ focused  │ segments │ w/ both  │ fine-    │
│ on main  │ of       │ pos/neg  │ grained  │
│ features │ interest │ contrib. │ pixels   │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 8. Best Practices

### 8.1 General Guidelines

1. **Use multiple methods**: Different methods reveal different aspects
2. **Sanity checks**: Verify explanations make clinical sense
3. **Compare across samples**: Check consistency of explanations
4. **Document limitations**: Be transparent about what explanations can/cannot show

### 8.2 Medical Imaging Specific

```python
# Checklist for Medical XAI
checklist = {
    "clinical_validity": "Does the explanation highlight anatomically relevant regions?",
    "consistency": "Are similar cases explained similarly?",
    "robustness": "Do small input changes cause large explanation changes?",
    "user_study": "Do clinicians find the explanations helpful?",
    "failure_cases": "What do explanations look like for misclassified images?"
}
```

### 8.3 Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Confirmation bias | Accepting explanations that match expectations | Use quantitative evaluation |
| Over-interpretation | Reading too much into heatmaps | Acknowledge uncertainty |
| Single method reliance | Using only one XAI technique | Ensemble explanations |
| Ignoring failures | Not examining wrong predictions | Always analyze errors |

---

## 9. Further Reading

### Papers

1. **CAM**: Zhou et al., "Learning Deep Features for Discriminative Localization" (CVPR 2016)
2. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
3. **LIME**: Ribeiro et al., "Why Should I Trust You?" (KDD 2016)
4. **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
5. **LRP**: Bach et al., "On Pixel-Wise Explanations for Non-Linear Classifier Decisions" (PLoS ONE 2015)

### Libraries

| Library | Methods | URL |
|---------|---------|-----|
| **Captum** | LRP, IG, DeepLIFT, etc. | https://captum.ai |
| **SHAP** | SHAP variants | https://github.com/slundberg/shap |
| **LIME** | LIME | https://github.com/marcotcr/lime |
| **pytorch-grad-cam** | CAM variants | https://github.com/jacobgil/pytorch-grad-cam |
| **Alibi** | Multiple methods | https://github.com/SeldonIO/alibi |

### Books

1. Molnar, C. "Interpretable Machine Learning" (Online book, free)
2. Samek et al., "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning"

### Online Courses

1. Coursera: "Explainable AI" by Duke University
2. Fast.ai: Practical Deep Learning (includes interpretability)

---

## Appendix: Quick Reference Code

```python
# ============================================================
# QUICK REFERENCE: All XAI Methods in One Place
# ============================================================

# --- CAM ---
from pytorch_grad_cam import GradCAM
cam = GradCAM(model, target_layers=[model.layer4[-1]])
grayscale_cam = cam(input_tensor=img, targets=None)

# --- LIME ---
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image, predict_fn, top_labels=2)

# --- SHAP ---
import shap
masker = shap.maskers.Image("blur(128,128)", image.shape)
explainer = shap.Explainer(predict_fn, masker)
shap_values = explainer(images, max_evals=500)

# --- LRP ---
from captum.attr import LRP
lrp = LRP(model)
attribution = lrp.attribute(input_tensor, target=class_idx)
```

---

*Document created for the Explainable AI Medical Image Classification project*
*Last updated: December 2024*

