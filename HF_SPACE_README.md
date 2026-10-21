---
title: CUB-200 Bird Classifier
emoji: 🐦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Rename this file to README.md **inside the Hugging Face Docker Space**.

Upload `app.py`, `birdlab.py`, `requirements.txt`, `Dockerfile`,
`deployment/bird_model.pth` and `deployment/model_info.json` after final evaluation.
The model is chosen on validation Top-1 accuracy. CPU inference needs no GPU.

Optional: upload `artifacts/comparison.csv`, `artifacts/test_summary.csv` and the
experiment `history.csv` files to enable comparison charts. For that option,
remove `artifacts` from `.dockerignore`. Never upload the training image folders.
