# Data Preparation

The raw medical imaging datasets used in this study are not redistributed in this repository.

## 1. MMOTU Ovarian Ultrasound Dataset

The public MMOTU OTU-2D dataset was used for model development and internal evaluation.

The dataset contains ovarian ultrasound images representing multiple ovarian tumor categories. For the binary classification task in this study, the original categories were grouped into benign and malignant classes according to the experimental protocol described in the manuscript.

The dataset used in this study contained 1,202 ultrasound images:

- Training set: 820 images
- Validation set: 382 images

The benign category included:

- Chocolate cyst
- Serous cystadenoma
- Teratoma
- Theca cell tumor
- Simple cyst
- Mucinous cystadenoma

The malignant category consisted of:

- High-grade serous carcinoma

The original MMOTU data are not included in this repository. Users should obtain the dataset from its original provider and comply with the corresponding license and terms of use.

## 2. TCGA-OV CT Dataset

CT data were obtained from The Cancer Genome Atlas Ovarian Cancer (TCGA-OV) collection.

After quality control, 77 patients were retained for the CT analysis.

The CT data are not redistributed through this repository. Users should obtain TCGA-OV data from the authorized data provider and comply with the applicable data-access and usage requirements.

## 3. Independent Clinical Validation Cohort

An independent clinical cohort from Shengjing Hospital of China Medical University was used for external validation.

The external validation cohort consisted of:

- Benign: 100 cases
- Malignant: 194 cases
- Total: 294 cases

Pathological diagnosis served as the reference standard.

The clinical images are not publicly distributed in this repository because they contain potentially sensitive clinical information and are subject to institutional ethical, privacy, and data-sharing restrictions.

De-identified data may be made available from the corresponding author upon reasonable request, subject to institutional approval and applicable data-sharing regulations.

## 4. Recommended Directory Structure

After obtaining the public datasets, users should organize the data locally according to the preprocessing and training scripts provided in this repository.

Example:

```text
data/
├── raw/
│   ├── US_MMOTU/
│   └── TCGA_OV/
│
├── processed/
│   ├── mmotu_2d_cls/
│   │   ├── train/
│   │   └── val/
│   │
│   └── tcga_ov_ct_volumes/
│
└── clinical_external/
```

## Important

Do not upload patient-level clinical images, identifiable clinical information, or other protected health information to this repository.

The `data/` directory in the public repository contains documentation only and does not contain the original medical imaging datasets.
