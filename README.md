# Histopathology Metastasis Detection with DenseNet201

Transfer learning pipeline for binary image classification on the **PatchCamelyon (PCam)** dataset — detecting metastatic tissue in histopathologic lymph node scans. Built with DenseNet201 pretrained on ImageNet. Includes full training and evaluation scripts with metrics, ROC curve, and confusion matrix outputs.

---

## Overview

This project implements an end-to-end ML pipeline for medical image classification:

- **Training** (`train.py`) — fine-tunes DenseNet201 on PCam histopathology data stored in HDF5 format
- **Evaluation** (`evaluate.py`) — runs inference and generates a full evaluation report with visualizations

---

## Dataset

This project uses the **[PatchCamelyon (PCam)](https://github.com/basveeling/pcam)** benchmark dataset.

- **327,680** color images (96×96px) extracted from histopathologic scans of lymph node sections
- **Task**: binary classification — presence or absence of metastatic tissue in the center 32×32px region of each patch
- **Split**: 262,144 training / 32,768 validation / 32,768 test examples (50/50 class balance)
- **Source**: derived from the [Camelyon16 Challenge](https://camelyon16.grand-challenge.org/) dataset
- **License**: [CC0](https://choosealicense.com/licenses/cc0-1.0/)

Data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB) or [Zenodo](https://zenodo.org/record/2546921).

If you use this dataset, please cite the original paper:

> Veeling et al. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962 (2018)

---

## Model Architecture

- **Backbone**: DenseNet201 (pretrained on ImageNet, top removed)
- **Head**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
- **Loss**: Binary crossentropy
- **Optimizer**: Adam (lr=1e-4)
- **Input shape**: 96×96×3 (configurable)

Supports partial fine-tuning via `--fine-tune-at` layer index.

---

## Data Format

Input data must be stored in **HDF5 files**:

| File | Key | Shape | Description |
|------|-----|-------|-------------|
| `train_x.h5` | `x` | `(N, H, W, 3)` | Images, uint8 [0–255] |
| `train_y.h5` | `y` | `(N,)` or `(N, 1)` | Binary labels (0 or 1) |

Images are normalized to [0, 1] automatically during loading.

---

## Requirements

```bash
pip install tensorflow h5py numpy scikit-learn matplotlib seaborn
```

Tested with Python 3.8+, TensorFlow 2.x.

---

## Usage

### Training

```bash
python train.py \
  --train-x data/train_x.h5 \
  --train-y data/train_y.h5 \
  --epochs 10 \
  --batch-size 32 \
  --output models/densenet201_model.h5
```

**Optional performance flags:**

```bash
python train.py \
  --train-x data/train_x.h5 \
  --train-y data/train_y.h5 \
  --omp-threads 4 \
  --intra-threads 4 \
  --inter-threads 2 \
  --gpu-memory 4096 \
  --low-priority
```

**Quick smoke test** (load only first N samples):

```bash
python train.py --train-x data/train_x.h5 --train-y data/train_y.h5 --test-samples 64
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-x` | required | Path to HDF5 file with training images |
| `--train-y` | required | Path to HDF5 file with training labels |
| `--batch-size` | `16` | Batch size |
| `--epochs` | `3` | Number of training epochs |
| `--output` | `densenet201_model.h5` | Output model path |
| `--test-samples` | `0` | Use only N samples (0 = all) |
| `--omp-threads` | `2` | OMP thread count |
| `--intra-threads` | `2` | TF intra-op threads |
| `--inter-threads` | `1` | TF inter-op threads |
| `--gpu-memory` | `0` | GPU memory cap in MB (0 = no limit) |
| `--low-priority` | `False` | Run process at lower OS priority |

---

### Evaluation

```bash
python evaluate.py \
  --model models/densenet201_model.h5 \
  --test-x data/test_x.h5 \
  --test-y data/test_y.h5 \
  --output-dir results/
```

#### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to saved Keras model |
| `--test-x` | required | Path to HDF5 file with test images |
| `--test-y` | required | Path to HDF5 file with test labels |
| `--batch-size` | `16` | Inference batch size |
| `--test-samples` | `0` | Evaluate only first N samples (0 = all) |
| `--output-dir` | `.` | Directory to save results |

---

## Outputs

After evaluation, the following files are saved to `--output-dir`:

| File | Description |
|------|-------------|
| `evaluation_report.txt` | Accuracy, Precision, Recall, F1, AUC-ROC + full classification report |
| `roc_curve.png` | ROC curve with AUC score |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `metrics_chart.png` | Bar chart of all metrics |

---

## Data Augmentation

Training applies on-the-fly augmentation via `ImageDataGenerator`:

- Horizontal & vertical flip
- Width & height shift (±10%)

---

## Project Structure

```
.
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── data/
│   ├── train_x.h5
│   ├── train_y.h5
│   ├── test_x.h5
│   └── test_y.h5
├── models/
│   └── densenet201_model.h5
└── results/
    ├── evaluation_report.txt
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── metrics_chart.png
```

---

## License

MIT
