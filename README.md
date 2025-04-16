
# 🧠 Brain Tumor Segmentation in MRI with 3D Deep Learning

This project focuses on automated brain tumor segmentation using 3D CNNs on multi-modal MRI images, evaluated on the BraTS dataset. It implements and compares two deep learning architectures — **AttentionUNet** and **UNet3D** — using a **patch-based pipeline** and a **custom ROI extraction strategy** to improve learning efficiency and granularity.

![Model](https://img.shields.io/badge/model-AttentionUNet%2FUNet3D-blue)
![Data](https://img.shields.io/badge/dataset-BraTS18-ff69b4)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Objective

To perform robust and interpretable segmentation of brain tumor subregions using lightweight and patch-efficient 3D models. The primary aim is to:
- Reduce memory usage by limiting training to tumor-centered patches
- Leverage domain-specific ROI masks to focus learning on informative regions
- Compare attention-based and standard UNet architectures

---

## 🧪 Dataset and Input

We use the **BraTS 2018** dataset, which provides multi-modal MRI scans with the following sequences:
- T1-weighted (T1)
- Contrast-enhanced T1 (T1ce)
- T2-weighted (T2)
- Fluid Attenuated Inversion Recovery (FLAIR)

**Label classes:**
- 0 = Background
- 1 = Necrotic core
- 2 = Edema
- 4 = Enhancing tumor
(Label 3 = Non-enhancing tumor was not used)

These are merged into **3 binary classes**:
- **Whole Tumor (WT)**: labels 1, 2, 4
- **Tumor Core (TC)**: labels 1, 4
- **Enhancing Core (EC)**: label 4

---

## 🧼 Preprocessing & Patch Extraction

### 🔍 ROI-Based Masking

Our ROI extraction method is inspired by [CascadeNet (Scientific Reports, 2021)](https://doi.org/10.1038/s41598-021-90428-8):

- We threshold binarized versions of T1ce, T2, and FLAIR (T1ce > 0.9, T2/FLAIR > 0.7)
- Morphological constraints (solidity, area, major axis) are applied on T1ce
- Regions overlapping with FLAIR ∩ T2 are selected

### 🧊 Patch Sampling

- Extract 3D patches: **128×128×16**
- Use only those within ROI and with <95% background
- If no valid patch, fallback to tumor centroid
- Each subject: max 5 random + 1 fallback patch

### 📏 Normalization

- Per-slice Z-score normalization
- Applied independently to all 4 modalities
- Normalized and raw versions are **stacked as 8 channels**

---

## 🧠 Models

### 1. AttentionUNet (3D)

An attention-gated 3D UNet with:
- 8-channel input (4 modalities + 4 normalized)
- Attention gates on decoder skip connections
- `sigmoid` activation (multi-label output)
- Dice + BCE loss with class weights

### 2. UNet3D (Baseline)

Standard 3D UNet with:
- 4-channel input (raw modalities only)
- BatchNorm and Dropout layers
- No attention mechanisms
- Same training/inference pipeline

---

## 📊 Results

We report mean Dice and Sensitivity over 38 validation samples (stored in `results/`):

## 📊 Model Comparison (Median Scores)

| Model        | Class | Dice   | Sensitivity |
|--------------|-------|--------|-------------|
| AttentionUNet | WT    | 0.88   | 0.91        |
| AttentionUNet | TC    | 0.81   | 0.84        |
| AttentionUNet | EC    | 0.70   | 0.76        |
| UNet3D        | WT    | 0.5861 | 0.5753      |
| UNet3D        | TC    | 0.8345 | 0.8511      |
| UNet3D        | EC    | 0.8391 | 0.8788      |

---

## 📁 Repository Structure

```
brain-tumor-segmentation-mri/
├── notebooks/
│   ├── train_attentionunet.ipynb
│   ├── train_unet3d.ipynb
│   └── evaluate_models.ipynb
├── results/               # Dice/Sensitivity metrics in JSON format
├── src/
│   └── preprocess_patches.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧭 Planned Improvements

> CascadeNet implementation in progress — not yet competitive but under refinement.

✅ **Planned next steps:**
- Refactor **CascadeNet** with DWA module and per-pixel prediction
- Improve segmentation granularity by extracting smaller patches (e.g., 96×96×8)
- Add **per-class Precision, Specificity, Hausdorff95** in `evaluate_models.ipynb`
- Introduce **online data augmentation** and weighted loss schemes

---

## 📖 References

- Ranjbarzadeh et al. (2021). *Brain tumor segmentation based on deep learning and an attention mechanism...* [Scientific Reports](https://doi.org/10.1038/s41598-021-90428-8)
- Oktay et al. (2018). *Attention U-Net.* [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)
- Bakas et al. (2017). *The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS).*

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Developed as part of a personal portfolio project by [Kamil Szlachcic](https://github.com/kamilszlachcic). Contributions and feedback are welcome!



