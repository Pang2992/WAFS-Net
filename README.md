# WAFS-Net

Official implementation of **WAFS-Net**, a deep learning framework for ovarian tumor classification using adaptive frequency-spatial feature fusion.

## Overview

WAFS-Net is designed for benign and malignant ovarian tumor classification from medical images. The framework integrates complementary spatial and frequency-domain representations to improve diagnostic feature learning.

The main components include:

- Dual spatial feature encoders based on DenseNet121 and EfficientNet-B0
- Haar discrete wavelet transform (DWT) for frequency-domain feature extraction
- Adaptive frequency-spatial feature fusion
- Classification head for benign/malignant prediction
- Class-weighted optimization for handling class imbalance
- Independent external validation

This repository provides the source code, configuration files, and evaluation scripts used in the study.

## Framework

The proposed WAFS-Net consists of three major stages:

1. **Spatial feature extraction**  
   DenseNet121 and EfficientNet-B0 are employed as complementary spatial encoders.

2. **Frequency-domain feature extraction**  
   Haar discrete wavelet transform is used to extract frequency-domain information from ovarian medical images.

3. **Adaptive frequency-spatial fusion and classification**  
   Spatial and frequency representations are adaptively integrated for final benign/malignant classification.

## Datasets

### MMOTU

The public MMOTU ovarian ultrasound dataset is used for model development and internal evaluation.

The dataset is **not redistributed in this repository**. Users should obtain the dataset from its original source and comply with the corresponding terms of use.

### TCGA-OV

CT data used in the study are derived from the publicly available **The Cancer Genome Atlas Ovarian Cancer (TCGA-OV)** collection.

The original medical images are **not redistributed in this repository**. Users should obtain the data through the authorized data provider and comply with the applicable data-access requirements.

### Independent Clinical Validation Cohort

An independent clinical cohort from **Shengjing Hospital of China Medical University** was used for external validation.

Because these data contain potentially sensitive clinical information and are subject to institutional ethical and privacy restrictions, the clinical images are **not publicly distributed through this GitHub repository**.

De-identified data may be made available from the corresponding author upon reasonable request, subject to institutional approval and applicable data-sharing regulations.

## Repository Structure

```text
WAFS-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── src/
│   ├── models/
│   ├── datasets/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── external_validation.py
└── data/
    └── README.md
```

## Environment

The experiments were implemented using Python and PyTorch.

Recommended environment:

```text
Python >= 3.9
PyTorch
torchvision
numpy
pandas
scikit-learn
scipy
PyWavelets
Pillow
opencv-python
matplotlib
tqdm
```

Detailed package versions will be provided in `requirements.txt`.

## Installation

Clone this repository:

```bash
git clone https://github.com/Pang2992/WAFS-Net.git
cd WAFS-Net
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The original medical imaging datasets are not included in this repository.

After obtaining the corresponding datasets from their official sources, organize the data according to the instructions provided in:

```text
data/README.md
```

## Training

Model training can be performed using:

```bash
python scripts/train.py
```

Experiment settings can be modified through the configuration files in:

```text
configs/
```

## Evaluation

Evaluate a trained model using:

```bash
python scripts/evaluate.py
```

External validation can be performed using:

```bash
python scripts/external_validation.py
```

Clinical data required for external validation are not included in this repository.

## Reproducibility

This repository provides the implementation necessary to reproduce the main computational experiments reported in the associated study.

The repository includes or will include:

- Model architecture
- Data preprocessing procedures
- Training scripts
- Evaluation scripts
- Configuration files
- Random seed settings
- Dependency information

Raw clinical data are excluded because of ethical and privacy restrictions.

## Code Availability

The source code used for model implementation, training, and evaluation is publicly available in this repository.

## Data Availability

The public datasets used in this study are available from their respective original data providers. The MMOTU dataset was used for ovarian ultrasound analysis, while TCGA-OV was used for CT-based analysis.

The independent clinical validation data are not publicly available because they contain potentially sensitive clinical information and are subject to institutional ethical and privacy restrictions. De-identified data may be made available from the corresponding author upon reasonable request, subject to institutional approval and applicable data-sharing regulations.

## Citation

If you use this code in your research, please cite the associated article.

Citation information will be updated after publication.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

For questions regarding the code or data-access procedures, please contact the corresponding author.
