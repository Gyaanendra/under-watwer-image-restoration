# 🌊 LSUI Dataset Exploration & Physical Degradation Analysis
**Project Documentation — Semester 5 Project**

---

## 📌 1. Executive Summary & Context

The **Large-Scale Underwater Image (LSUI)** dataset is one of the premier benchmarks for supervised underwater image enhancement (UIE). 

Before designing or training deep learning networks (e.g., U-Net, Restormer, NAFNet, Water-Net, HeroMamba, or GAN-based architectures), performing thorough **Exploratory Data Analysis (EDA)** is essential. EDA helps us:
1. **Understand optical physics**: Unpack the physical degradation processes (absorption, scattering, turbidity) present in the data.
2. **Verify integrity**: Ensure zero corrupted files, missing pairs, or dimension anomalies before expensive training runs.
3. **Formulate objective loss functions**: Design loss functions that directly penalize the specific degradation mechanisms observed (e.g., color shift in LAB space, high-frequency edge loss).
4. **Establish baseline difficulty**: Quantify the initial distortion (PSNR, SSIM, Laplacian blur) to measure future model improvements objectively.

---

## 🔬 2. Theoretical Background: The Physics of Underwater Imaging

Underwater images do not degrade like standard atmospheric images (which primarily experience fog or haze). Instead, light propagation in aquatic mediums is governed by two distinct optical processes:

```
                      Sunlight / Artificial Light
                                  │
                                  ▼
                        [Water Interface]
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
1. Selective Absorption                            2. Scattering
(Wavelength-dependent:                              (Suspended particles:
 Red drops first, Blue/Green penetrates)             marine snow, plankton, sand)
        │                                                   │
        ▼                                           ┌───────┴───────┐
   Color Cast                                       ▼               ▼
 (Green / Blue hues)                           Forward Scatter  Backscatter
                                                (Edge Blurring) (Veiling Haze /
                                                                 Low Contrast)
```

### 2.1 The Beer-Lambert Law & Selective Absorption
Light intensity at wavelength $\lambda$ at distance $z$ follows exponential decay:
$$E(\lambda, z) = E_0(\lambda) \, e^{-c(\lambda) z}$$
where $c(\lambda) = a(\lambda) + b(\lambda)$ is the total attenuation coefficient ($a$ = absorption, $b$ = scattering).
* **Red light** ($\lambda \approx 620 - 750\text{ nm}$): Highest absorption coefficient. It is largely extinguished within the first $3 - 5\text{ meters}$.
* **Green light** ($\lambda \approx 495 - 570\text{ nm}$): Moderately absorbed; penetrates well in coastal/inland waters containing phytoplankton.
* **Blue light** ($\lambda \approx 450 - 495\text{ nm}$): Lowest absorption in clear oceanic water; penetrates deepest, producing severe blue casts.

### 2.2 Forward & Backward Scattering
* **Forward Scattering**: Light reflected from the object is deflected at small angles by suspended particles (*marine snow*, plankton, minerals) before reaching the lens. This causes **edge blurring** and **loss of high-frequency texture**.
* **Backward Scattering**: Ambient illumination is scattered directly into the camera optical path without ever striking the subject. This creates a **luminous veiling haze**, drastically reducing scene contrast.

---

## 📂 3. Dataset Anatomy & Verification

### 3.1 Folder Structure
The dataset in our workspace is organized into:
```
dataset/LSUI/
├── input/      # 4,279 raw degraded underwater images (0.jpg -> 4278.jpg)
├── GT/         # 4,279 clear ground-truth reference images (0.jpg -> 4278.jpg)
└── lsui_splits.csv # Generated reproducible Train/Val/Test metadata
```

### 3.2 Integrity Findings
* **Total Pairs**: Exactly **4,279 paired `.jpg` files**.
* **Pair Matching**: **100% 1-to-1 match**. No missing ground truth images and no orphaned inputs.
* **Color Channels**: All 4,279 images are standard 3-channel 8-bit RGB (`mode: RGB`).
* **Corrupt Headers**: **0 corrupt files**. All images successfully parse via both OpenCV and PIL.

---

## 📊 4. Dimensionality & Metadata Profile

Unlike toy datasets (e.g. CIFAR-10) where all images are pre-cropped to $32 \times 32$, LSUI contains real-world captures from various underwater survey cameras, action cameras (GoPro), and submersibles.

### 4.1 Resolution Statistics (Full Dataset Scan)

| Metric | Width (px) | Height (px) | Aspect Ratio | Input Size (KB) | Ground Truth Size (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean** | 447.04 | 309.17 | 1.42 | 49.47 | 65.14 |
| **Standard Deviation** | 211.20 | 128.86 | 0.25 | 82.77 | 89.60 |
| **Minimum** | 256.00 | 225.00 | 1.00 | 5.96 | 10.35 |
| **25th Percentile** | 320.00 | 240.00 | 1.33 | 22.22 | 34.65 |
| **Median (50th %)** | 320.00 | 256.00 | 1.33 | 27.34 | 43.67 |
| **75th Percentile** | 640.00 | 360.00 | 1.78 | 50.60 | 66.03 |
| **Maximum** | 1280.00 | 1024.00 | 2.00 | 690.81 | 721.08 |

### 4.2 Top 5 Resolution Clusters

| Resolution ($W \times H$) | Aspect Ratio | Count | Percentage | Primary Source Type |
| :---: | :---: | :---: | :---: | :--- |
| **$320 \times 240$** | $4:3$ ($1.33$) | 1,700 | 39.7% | Standard sub-surface survey video frames |
| **$640 \times 360$** | $16:9$ ($1.78$) | 943 | 22.0% | Widescreen HD downsampled captures |
| **$400 \times 300$** | $4:3$ ($1.33$) | 773 | 18.1% | Diver handheld camera frames |
| **$256 \times 256$** | $1:1$ ($1.00$) | 498 | 11.6% | Pre-cropped square benchmark patches |
| **$720 \times 405$** | $16:9$ ($1.78$) | 137 | 3.2% | High-definition wide survey captures |

### 4.3 Why Ground Truth Files Are 31% Larger
The average ground truth file size ($65.1\text{ KB}$) is **31.7% larger** than the raw input ($49.5\text{ KB}$).
* JPEG compression exploits spatial frequency redundancy: uniform, foggy, and low-contrast regions compress into fewer bytes.
* Ground truth images recover sharp edges, rich color gradients, and micro-textures, increasing high-frequency entropy and producing larger JPEG files at equivalent quality.

---

## 🔍 5. What Was Done in the Notebook: Section by Section

The notebook [`notebooks/01_lsui_dataset_exploration.ipynb`](file:///e:/sem5_project/notebooks/01_lsui_dataset_exploration.ipynb) executes an end-to-end scientific study:

### Section 1 & 2: Robust Setup & Path Discovery
* Configured reproducible styling (`seaborn-whitegrid`, custom color palettes).
* Created path resolution logic that runs seamlessly regardless of whether the working directory is the repository root or the `notebooks/` subfolder.
* Added fallback handling for `IPython.display` so code runs in both interactive notebook interfaces and headless script executions.

### Section 3 & 4: Integrity Verification & Metadata Profiling
* Programmatically iterated over all 4,279 images using non-decoding PIL headers to build a high-performance DataFrame in seconds.
* Computed summary statistics and generated 4-panel distribution plots:
  - Resolution scatter plot ($W$ vs $H$).
  - Megapixel density distribution (mean $\approx 0.16\text{ MP}$).
  - Aspect ratio frequency bar chart ($4:3$ and $16:9$ dominate).
  - KDE file size density curves (Input vs GT).

### Section 5: Paired Comparison & Absolute Difference Heatmaps
* Built a paired visualization tool that renders:
  1. **Raw Input image**
  2. **Ground Truth image**
  3. **Amplified Absolute Difference Heatmap** ($|I_{\text{GT}} - I_{\text{raw}}|$ via `cv2.absdiff` and `magma` colormap).
* **Observation**: The difference heatmap highlights that enhancement is non-uniform. The strongest modifications occur in the red-spectrum biological structures (corals, fish) and shadow regions, rather than uniformly across the water background.

### Section 6: Degradation Archetypes Categorization
* Implemented optical color-ratio metrics:
  $$\text{Green Bias} = \frac{\bar{G}}{\bar{R} + \bar{B} + \epsilon}, \quad \text{Blue Bias} = \frac{\bar{B}}{\bar{R} + \bar{G} + \epsilon}$$
* Automatically extracted extreme archetypes representing the 4 real-world underwater environments:
  1. **Severe Green Cast**: Coastal/estuarine waters with dense organic matter.
  2. **Dominant Blue Cast**: Open pelagic ocean water.
  3. **Low Illumination**: Deep underwater crevices or cloudy/low-sun conditions.
  4. **Turbid Hazy Scenes**: High-backscatter shallow waters with suspended sediment.

### Section 7: Proving Red Channel Attenuation
* Plotted per-channel RGB histograms for degraded vs enhanced pairs.
* Evaluated dataset-wide mean intensities across samples:
  - **Raw Input Red Mean**: $\approx 60 - 80$ (heavily suppressed).
  - **Ground Truth Red Mean**: $\approx 105 - 125$ (restored dynamic range).
  - Proves mathematically that underwater restoration is primarily a **non-linear spectral rebalancing** task rather than simple brightness amplification.

### Section 8: Decoupled Color Space Decomposition
* Decomposed images into:
  - **CIELAB ($L^*a^*b^*$)**: $L^*$ isolates pure lightness; $a^*$ shows the dramatic shift from negative (green cast) toward positive (red recovery); $b^*$ tracks blue-yellow balance.
  - **HSV**: Isolates Saturation ($S$) and Value ($V$) to show how vibrancy and dynamic range are restored independently of hue.

### Section 9: Sharpness & Frequency Analysis
* Calculated **Laplacian Variance** ($\text{Var}(\nabla^2 I)$) and **RMS Contrast**:
  - Raw images suffer from attenuated high frequencies due to forward scattering.
  - Ground truth images exhibit a **$2.5\times - 4\times$ median boost in Laplacian variance**, demonstrating significant edge and texture recovery.

### Section 10: Reference Baseline Benchmarks (PSNR & SSIM)
* Implemented clean, pure-OpenCV/NumPy versions of PSNR and SSIM:
  - **Baseline PSNR (Raw vs GT)**: Averages $\approx 14 - 17\text{ dB}$. (Values $< 20\text{ dB}$ indicate heavy visual distortion).
  - **Baseline SSIM (Raw vs GT)**: Averages $\approx 0.65 - 0.75$.
* These numbers serve as the **lower bound**; any deep learning model developed in this project must comfortably surpass these scores on the test split.

### Section 11 & 12: Split Generation & Export
* Created a reproducible split generator using random seed `42`:
  - **Train set**: 3,423 pairs ($80\%$)
  - **Validation set**: 428 pairs ($10\%$)
  - **Test set**: 428 pairs ($10\%$)
* Exported the split definitions to [`dataset/LSUI/lsui_splits.csv`](file:///e:/sem5_project/dataset/LSUI/lsui_splits.csv).

---

## 🎯 6. Actionable Takeaways for Model Development

From our exploration, here are the core engineering recommendations for your Semester 5 project:

### 1. Preprocessing & Patching
* **Do NOT simply resize all images to $256 \times 256$**: Sticking a $1280 \times 720$ image directly into a $256 \times 256$ resize distorts aspect ratio and destroys fine coral textures.
* **Recommended Training Strategy**: Use **random cropping of $256 \times 256$ patches** during training. This retains native pixel resolution and sensor noise characteristics while enabling uniform mini-batching.
* **Testing Strategy**: For evaluation, either test on native sizes (if using fully convolutional/attention architectures) or pad images to multiples of 16/32.

### 2. Loss Function Design
A simple Mean Squared Error (MSE / L2) loss will produce blurry images because MSE averages out high-frequency details. Instead, use a multi-term composite loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{color}} \mathcal{L}_{\text{color}}$$
* **$\mathcal{L}_{1}$ (Charbonnier / L1 Loss)**: Penalizes absolute pixel differences; preserves sharp edges much better than L2.
* **$\mathcal{L}_{\text{SSIM}}$**: Enforces local structural and luminance preservation.
* **$\mathcal{L}_{\text{color}}$ (CIELAB or Angle Loss)**: Measures $\Delta E$ in LAB space or cosine angle in RGB space, specifically penalizing the stubborn green/blue color cast and driving red-channel recovery.

### 3. Evaluation Protocol
When testing trained models, evaluate using:
* **Full-Reference Metrics** (comparing against GT):
  - **PSNR** (target: $> 24\text{ dB}$)
  - **SSIM** (target: $> 0.88$)
* **Non-Reference Metrics** (underwater domain specific):
  - **UIQM** (Underwater Image Quality Measure: sharpness, colorfulness, contrast)
  - **UCIQE** (Underwater Color Image Quality Evaluation: chroma variance, contrast, saturation)

---

## 🔗 Related Project Files
* **Exploration Notebook**: [`notebooks/01_lsui_dataset_exploration.ipynb`](file:///e:/sem5_project/notebooks/01_lsui_dataset_exploration.ipynb)
* **Dataset Splits File**: [`dataset/LSUI/lsui_splits.csv`](file:///e:/sem5_project/dataset/LSUI/lsui_splits.csv)
* **Dataset Raw Images**: [`dataset/LSUI/input/`](file:///e:/sem5_project/dataset/LSUI/input/)
* **Dataset Ground Truth**: [`dataset/LSUI/GT/`](file:///e:/sem5_project/dataset/LSUI/GT/)
