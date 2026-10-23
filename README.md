# Bird Species Classifier

COS30082 Applied Machine Learning individual assignment

Fa Zhen NG, student ID 104400141

A Streamlit app that recognises 200 bird species. Choose **Version 1** or
**Version 2**, then select either **EfficientNetV2-S (best)** or **MobileNetV1 (best)**.
Each option loads the checkpoint with the highest validation Top-1 accuracy for
that architecture and version. The app defaults to Version 2 EfficientNetV2-S.

https://cos30082individualassignment-medgwuczqvy53a3gw73dmu.streamlit.app/

## Final training and Test results

Training accuracy below uses deterministic evaluation of the training images at
the selected checkpoint, without random augmentation or dropout. Validation
selects checkpoints within each version before its Test evaluation. Test Top-1
is overall accuracy. Mean per-class accuracy gives each of the 200 species equal weight.

| Version | Model | Train Top-1 | Validation Top-1 | Test Top-1 | Test mean per-class |
|---|---|---:|---:|---:|---:|
| V1 | EfficientNetV2-S | 99.64% | 76.09% | 76.41% | 75.90% |
| V1 | MobileNetV1 | 99.92% | 61.49% | 59.05% | 58.43% |
| V2 | EfficientNetV2-S | 99.92% | 79.92% | 79.15% | 78.51% |
| V2 | MobileNetV1 | 99.43% | 58.28% | 57.64% | 56.63% |

### Selected checkpoints

| Version | Model | Validation-selected recipe | Best epoch / epochs run |
|---|---|---|---:|
| V1 | EfficientNetV2-S | `03_horizontal_flip` | 14 / 15 |
| V1 | MobileNetV1 | `01_fine_tune` | 15 / 15 |
| V2 | EfficientNetV2-S | `01_random_augmentation` | 25 / 33 |
| V2 | MobileNetV1 | `01_random_augmentation` | 32 / 35 |

Both versions use 3,863 training images, 966 validation images and the official
1,204-image Test set, with the same seed-42 split and class order. Image hashes
prevent identical content crossing training and validation. Four official Test
images overlap Train. Excluding them, EfficientNetV2-S reaches **76.67%** Top-1 /
**76.17%** mean per-class in V1, and **79.42%** / **78.79%** in V2.

## Training choices

**Version 1** runs 13 cumulative recipes for each architecture, covering a frozen
baseline, backbone fine-tuning, crops and flips, followed by stronger image
transformations and regularisers. Every recipe restarts from ImageNet weights.
It uses 224-pixel inputs, up to 15 epochs, two classifier warm-up epochs for
fine-tuning, initial backbone/head learning rates 3e-5 / 3e-4, and batch 8 with
four accumulation steps. Recorded total: **12.71 hours for 26 runs**.

**Version 2** tests a frozen baseline and one mild augmentation/fine-tuning
recipe per architecture. Crops retain 85–100% of the image, with horizontal flips
and mild colour jitter. It removes vertical flips, rotation, inversion, tints,
erasing and grayscale, and sets added dropout, weight decay and label smoothing
to zero. Inputs increase to 384 pixels, the fine-tuning budget increases to 35
epochs with three warm-up epochs, backbone LR drops to 1e-5, and batch 2 uses
16 accumulation steps. Recorded total: **7.78 hours for four runs**.

Both versions use ImageNet transfer learning, a new 200-output classifier,
cross-entropy loss, AdamW, cosine learning-rate decay, mixed precision, fixed
BatchNorm running statistics and validation-based early stopping.

## Repository contents

- `app.py`: version/model selection, photo upload and Top-5 predictions.
- `predictor.py`: standalone CPU model reconstruction and preprocessing.
- `requirements.txt`: deployment dependencies.
- `.streamlit/config.toml`: Streamlit configuration.
- `models/manifest.json`: selected recipes, final metrics and checkpoint hashes.
- `models/version_1/` and `models/version_2/`: two selected checkpoints per version.

The main branch contains the deployment app and its selected weights. Training
notebooks, datasets, reports, caches and intermediate checkpoints are kept outside
this deployment repository.

