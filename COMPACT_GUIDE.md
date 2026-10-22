# 三轮精简训练版

原 `Bird_Classification_Local_GPU.ipynb`、代码和训练结果完整保留，不改写。
新文件是 `Bird_Classification_Compact_GPU.ipynb`，只有 **3 轮 × 两个架构**。
所有新结果写 `artifacts_compact_v1`，不读取或覆盖旧鸟类 checkpoint。

| 轮次 | 配方 | 最多 epoch |
|---|---|---|
| 1 Baseline | 冻结 ImageNet backbone，只训练新分类头 | 15 |
| 2 Random augmentation | 解冻微调，保留 85–100% 面积裁剪、左右翻转、轻微颜色变化 | 35 |
| 3 Final optimized | 保留适度增强，加 dropout 0.2、weight decay 1e-4、label smoothing 0.05，降低分类头 LR | 60 |

两模型统一 384px、batch 2、累积 16、AMP、workers 0。微调前 warm-up 3 epoch，backbone LR 1e-5，head LR 前两轮 3e-4、最终轮 1e-4。早停 patience 分别 5/8/12，BatchNorm 运行统计固定。最多 220 model-epochs，可能比旧版单个阶段慢；少的是步骤与单元格数量，不是保证更短的总训练时间。显存不足时减 batch 至 1、累积 32，并使用新的输出目录。

384px 增加细节但不保证提分；EfficientNetV2-S 权重官方推理尺寸为 384，MobileNetV1 原预训练尺寸为 224。此处统一正方形缩放保持训练评估部署一致，并非复刻 EfficientNet 官方的保持比例/中心裁剪流程。

每轮从相同 ImageNet 初始权重和 seed 开始，不继承前一轮鸟 checkpoint。不同轮同时改变冻结策略、预算和正则化，属于完整配方比较，不能把提升全部归功于 random crop。三轮都展示实际 accuracy、上轮/基准差值和截至目前最佳；后者单调不降只是择优保留，不是声称每个配方都提高。

Train Accuracy 超过 90% 不等于 Val/Test 超过 90%。固定 split、防止内容泄漏，所有阶段完成才打开 Test。保留官方和去掉四张 Train/Test 重复图片的指标。不要为了达标挑删困难 Test 样本或反复根据 Test 调参。

不使用上下翻转、大角度旋转、反色/色层、灰度或擦除，以减少破坏羽毛颜色与姿态。轻微 ColorJitter 参数为 brightness/contrast/saturation 0.1、hue 0.015；crop ratio 0.9–1.1，flip p=0.5。这些选择受旧 validation 结果启发，新实验不是完全独立的验证发现，应在报告说明。

开始前停止其他 GPU 训练并 Restart Kernel，选择 `.venv`，顺序运行新 notebook。现有旧训练不被自动终止。正式六次训练未自动执行，实际分数等运行后记录。

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_compact.py
.\.venv\Scripts\python.exe tools/package_compact_source.py
```

新 Streamlit 只读取 compact study，默认优先展示导出的 validation 最佳模型；旧应用 `app.py` 保持原样。上传清晰、主体明确、属于 200 已知物种的照片更符合训练条件，但无法保证每张正确。未识别物种也会被强制归到这 200 类之一，高 confidence 不代表一定正确。用独立、有真值的上传照片检查实际表现，不能用单张预测概率代替 accuracy。

最终导出 `artifacts_compact_v1/deployment/bird_model.pth`。源码包是单独的 `submission/COS30082_compact_source.zip`，不替换旧 ZIP；报告/视频仍按原 ASSIGNMENT_GUIDE.md 要求提交。包不含权重和数据图片。

## 已完成的验证

`tools/verify_compact.py` 已通过：两架构三轮共六次小样本 GPU 训练、表格与历史最佳计算、六个 checkpoint 的 CPU Top-5 预测、200 类合成数据的最终评估/重复图排除/导出、Streamlit 模型切换，以及原版 13 阶段流程未被修改。Notebook 格式与所有代码单元通过验证。测试仅使用 17 fit / 9 validation、96px、每轮 2 epoch，放在 `smoke_artifacts/compact_v1`；**这些分数不是正式实验结果**。正式 384px 全量训练与真实上传图片评估尚未执行。
