# XAI Methods in Chest X-ray Pneumonia Classification

This section discusses the strengths and limitations of the XAI methods used in this project (Grad-CAM, Occlusion Sensitivity, LIME) in the medical (chest X-ray) context, based on the observed results (TP/TN/FP examples and comparison slides).

## 1) Grad-CAM

### Strengths (in this medical context)
- **Fast, intuitive localization**: Grad-CAM quickly produces a coarse heatmap that often highlights *where* the CNN is “looking”. In the TP examples, the highlighted regions overlap with lung fields where pneumonia-related opacity is expected.
- **Model-consistent and class-specific**: It is tied directly to the network’s gradients for a target class score, so it reflects how internal features contribute to the predicted class.
- **Good for sanity checks**: It can reveal shortcuts/spurious focus. In your FP example, the heatmap suggests attention on non-pathology structures (e.g., bone), indicating potential confounding cues.

### Limitations
- **Coarse resolution**: Heatmaps come from deep feature maps and are upsampled, so boundaries are blurry and may not precisely delineate lesions.
- **Positive-evidence only (with ReLU)**: Standard Grad-CAM highlights regions that *increase* the target score; it does not show strong “evidence against” the class unless you explicitly compute the opposite-class map.
- **Susceptible to confounders**: As seen in the FP case, Grad-CAM can highlight non-lung or non-disease regions if the model learned spurious correlations.
- **Not a causal guarantee**: A “hot” region is correlated with the class score under the model’s current representation, but it does not prove the region is medically causal.

## 2) Occlusion Sensitivity

### Strengths (in this medical context)
- **Perturbation-based (more directly behavioral)**: Occlusion tests how predictions change when parts of the image are masked, providing a more “what-if” explanation than purely gradient-based maps.
- **Bias detection and quality control**: Your occlusion results flagged cases where masking **borders/non-lung** areas changed predictions strongly (Normal FP with border focus), which is a useful warning sign for clinical deployment.
- **Interpretable directionality**: Regions that cause large probability drops when occluded are easy to interpret as “important” under the model.

### Limitations
- **Patch/masking artifacts**: The occluding patch can create out-of-distribution inputs; the model’s response may reflect artifact sensitivity rather than true reliance on anatomy.
- **Computationally expensive**: Sliding-window occlusion scales with image size and stride; it is much slower than Grad-CAM.
- **Choice-dependent**: Results depend on patch size, stride, and mask value; different settings can change the map.
- **Still class-specific**: In binary single-logit setups, an occlusion map commonly measures importance for the *positive* class (pneumonia) unless explicitly computed for the negative class.

## 3) LIME (Image Superpixel Explanations)

### Strengths (in this medical context)
- **Local, human-readable regions**: Superpixels often align better with anatomical regions than coarse CNN heatmaps. In your TN/FP examples, LIME’s important regions were mainly in the **lung area**, which matches clinical expectation more closely than border-focused explanations.
- **Shows supportive vs opposing evidence**: With your signed heatmap visualization (negative vs positive weights), LIME can communicate which regions push toward pneumonia vs away from it.
- **Model-agnostic**: Works without changing the model; useful for comparing explanations across different classifiers.

### Limitations
- **Depends strongly on segmentation**: Superpixel size/shape can distort explanation (over-segmentation or under-segmentation), especially on grayscale X-rays with subtle intensity transitions.
- **Approximation error**: LIME fits a simple surrogate locally; if the model is highly non-linear in that neighborhood, the explanation may be unstable or misleading.
- **Perturbation realism**: “Hiding” superpixels produces images that may not be physically plausible, which can change model behavior in ways unrelated to true pathology.
- **Sampling variability**: Different random seeds / numbers of samples can change the learned weights.

## 4) Summary: When each method is most useful
- **Grad-CAM**: Fast localization + quick sanity checks (but coarse and can follow confounders).
- **Occlusion**: Behavioral evidence + strong for bias detection (but slow and can be artifact-sensitive).
- **LIME**: Region-level, signed explanations that can align with lungs (but sensitive to segmentation and sampling).

---

# Evaluation Criteria for Explanation Quality (End-User Focus)

Below are evaluation criteria designed for *clinical end users* (e.g., radiologists) who need explanations that are trustworthy, anatomically meaningful, and robust.

## Criterion 1 — Anatomical Plausibility / Lung-Focus

**Definition**: The explanation should concentrate on clinically relevant anatomy (primarily lung fields for pneumonia) rather than borders, markers, ribs, or background.

**Why it matters**: Your occlusion FP case indicated **border/non-lung** reliance (potential bias). An end user needs a clear signal that the model is using plausible medical evidence.

**How to assess (practical options)**
- **Quantitative (recommended)**: Compute an “inside-lung attribution ratio”
  $$ R = \frac{\sum_{(i,j)\in \text{lung}} |E_{ij}|}{\sum_{(i,j)\in \text{image}} |E_{ij}|} $$
  where $E$ is the explanation map (Grad-CAM intensity, occlusion drop map, or absolute LIME weight map). Higher $R$ is better.
- **Qualitative (end-user)**: A radiologist rates whether highlighted regions correspond to lung opacities/expected patterns vs artifacts (e.g., borders).

## Criterion 2 — Faithfulness (Prediction Change When Evidence Removed)

**Definition**: If an explanation claims a region is important, then removing/occluding that region should meaningfully change the model’s output in the predicted direction.

**Why it matters**: A heatmap that “looks right” but does not affect the prediction is not helpful for trust. Occlusion results already provide a natural framework for this.

**How to assess (practical options)**
- **Deletion test**: Remove the top-$k\%$ highlighted pixels/superpixels (according to the explanation) and measure probability drop.
- **Insertion test**: Start from a blurred/blank baseline and add back highlighted regions first; measure how quickly the model’s confidence recovers.

## Criterion 3 — Stability / Robustness (optional but recommended)

**Definition**: Explanations should be consistent under small, clinically irrelevant changes (tiny intensity shifts, minor crops/translation) and across similar images.

**Why it matters**: In medical imaging, small acquisition differences should not flip the explanation. Unstable explanations reduce end-user confidence.

**How to assess**
- Re-run explanations under small perturbations and compute similarity (e.g., Spearman rank correlation of pixel importance, or IoU of top-$k\%$ regions).

## Criterion 4 — Human Usability / Actionability (optional)

**Definition**: End users should be able to understand the explanation quickly and use it to support (or challenge) the model output.

**Why it matters**: In practice, clinicians need explanations that are interpretable under time constraints.

**How to assess**
- Short clinician questionnaire: clarity, confidence impact, and whether it helped detect failure modes (e.g., border bias).

---

## Recommended minimum set (to satisfy end-user needs)
- **Anatomical Plausibility / Lung-Focus** (Criterion 1)
- **Faithfulness via Deletion/Insertion or Occlusion Drops** (Criterion 2)

These two criteria jointly address: (a) *medical relevance* and (b) *whether the explanation reflects actual model behavior*.
