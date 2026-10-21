# COS30082 Bird Classification

Start with `Bird_Classification_Local_GPU.ipynb`. Two architectures only:
**MobileNetV1 width 1.0** and **EfficientNetV2-S**. Original assignment PDF and
`Data_loading1.ipynb` are preserved. The teacher approved Streamlit.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1
```

Stop old training, restart the notebook kernel and select
`.venv\Scripts\python.exe` (global Python is 2.7). Run the notebook in order.
New results go to **artifacts_retrain_v3**; old `artifacts` and global deployment
files remain untouched and are excluded from the revised notebook/app/package.
Do not run two training processes together on the 4GB GPU.

Each stage trains both architectures, then displays their accuracy comparison
table and per-epoch **Train Accuracy, Train Loss, Val Accuracy, Val Loss** tables.
Model comparison graphs and Grad-CAM panels are removed. Augmentation examples
remain so you can inspect the actual input changes.

| Stage | Added setting (all earlier settings retained) |
|---|---|
| 0 | Frozen ImageNet backbone; train new 200-class linear head |
| 1 | Fine-tune backbone after two head-only warm-up epochs |
| 2 | Random crop area 40–100%, aspect ratio 0.75–1.33 |
| 3 | Horizontal (left/right) flip p=0.5 |
| 4 | Vertical (up/down) flip p=0.2 |
| 5 | Random rotation uniformly from −45° to +45° |
| 6 | ColorJitter brightness/contrast 0.4, saturation 0.5, hue 0.1 |
| 7 | Inversion / red / green / blue / yellow effects, total p=0.25 |
| 8 | Classifier dropout 0.3 |
| 9 | AdamW weight decay 0.001 |
| 10 | Label smoothing 0.1 |
| 11 | Random erasing p=0.25, area 2–15% |
| 12 | Grayscale p=0.2 |

26 training runs, 13 recipes × 2 architectures. Every recipe initializes from
its architecture's same ImageNet weights and seed, never the previous bird
checkpoint. Thus comparisons do not give later recipes extra training epochs.
The baseline's 20–40% is an expectation, not an enforced score. All gains and
declines are reported. Previous-stage deltas are conditional on earlier methods,
not isolated effects. Single-seed experiments do not establish significance.

Pretrained backbones can be fine-tuned on the first bird training; no new bird
classes are required. Fine-tuning changes weights, not pixels. Its configuration
is shown without repeating five identical pictures. Crop previews span five
size bands/positions; rotation previews span five angle bands. Runtime sampling
is independent on every access, not limited to five transformations. Flip and
grayscale previews show their two possible states. Preview RNG states are restored.

Color-effect training selects one effect with conditional weights 10/25/25/25/15%
for inversion/red/green/blue/yellow; inversion therefore affects about 2.5% of
accesses. Tint opacity is uniform 20–45%. Effects are forced and labeled in the
isolated preview. Stronger augmentation can lower accuracy for fine-grained bird
classification. Tinted/inverted RGB is not thermal or infrared data. Deploy the
validation winner rather than assuming the last recipe is best.

Defaults: 224px, batch 8, accumulation 4, AMP, workers 0, max 15 epochs, patience 5,
head LR 3e-4 / backbone LR 3e-5. On OOM use batch 4 / accumulation 8 with a fresh
output directory. BatchNorm running statistics are fixed for all recipes.
MobileNetV1 uses timm `mobilenetv1_100.ra4_e3600_r224_in1k`, mean/std 0.5;
EfficientNetV2-S uses torchvision ImageNet normalization. Metadata preserves these.
Their different pretraining histories limit architecture-only conclusions.

Data must be present as `Train/`, `Test/`, `train.txt`, `test.txt` (not in Git).
4,829 train rows, 200 classes; split 3,863 fit / 966 validation. Six conflicting
duplicate rows remain in fit; identical content never crosses fit/validation.
Final Test has 1,204 images and four Train overlaps, reported alongside clean
metrics. Test is not used for recipe/checkpoint selection. Finalization is gated
on all runs and locks the selection before evaluating Test.

`comparison_display.csv` has readable percentages and changes in percentage
points. `paired_comparison.csv` uses fractions for accuracy. Train Accuracy uses
actual training batches; Train Eval Accuracy uses deterministic inputs, and its
difference from validation is the clean gap. Displayed losses are unsmoothed CE;
the smoothed optimization objective is stored separately.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe tools/package_source.py
```

Streamlit reads only the new study: photo upload/Top-5, checkpoint selection,
comparison/epoch tables, augmentation previews. CPU inference leaves GPU available.
Finalization exports `artifacts_retrain_v3/deployment/bird_model.pth`. Until a new
checkpoint exists the app shows a pending message. It never substitutes old runs.
Only load trusted checkpoints. Probabilities for one photo are not dataset accuracy.

The ZIP `submission/COS30082_source.zip` includes source and available new-study
CSV/JSON/PNG results, excluding weights, dataset images, caches and environments.
Rebuild after training. See [ASSIGNMENT_GUIDE.md](ASSIGNMENT_GUIDE.md) for the
report/video submission and [PROGRESS.md](PROGRESS.md) for verification status.

References: [MobileNetV1](https://arxiv.org/abs/1704.04861),
[EfficientNetV2](https://arxiv.org/abs/2104.00298),
[PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
