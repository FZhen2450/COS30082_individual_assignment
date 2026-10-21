# 作业要求与实验安排

依据 `Assignment1_v2.pdf` 第 1–3 页。原始文件保留不变。

## 要交什么

这是个人作业，占 20%，任务是 200 类鸟类图片分类。

1. **PDF 报告，最多 4 页**，包含姓名与学号。两个主体部分：Methodology、Result and Discussion。
2. **Python 源码 ZIP**，代码要有充分注释，体现你理解实现。
3. **最长 10 分钟视频**：解释概念与模型设计、介绍代码、展示训练及结果、演示在 Hugging Face 上运行的 AI 网页应用。建议配 slides。
4. 报告必须包含 **YouTube 视频链接**和 **Hugging Face 应用链接**。

报告必须写模型架构、loss、hyperparameters、避免 overfitting 的方法、不同模型表现的比较，以及最佳模型的选择理由。

PDF 本身没有规定必须先训练出很差的模型。你的老师补充的 baseline 思路应理解为：先建立真实基准，再用实验判断改进是否有效，不能故意训练失败或编造提升。

## 两个必报指标

- **Top-1 accuracy**：全部测试图片中预测正确的比例。
- **Average accuracy per class**：先算每个真实类别中预测正确的比例，再对 200 个类别平均。也就是 macro recall，不是每类 one-vs-rest 二元 accuracy。

例如 accuracy 从 70% 到 75%，提升是 **5 个百分点（pp）**。程序自动计算相对 baseline 的正负变化。

## 本次采用的两个架构

- **MobileNetV3-Large**：作为较轻量的第一个模型，适合这台电脑的 4GB 显存。
- **EfficientNetV2-S**：作为第二个模型；不用原 notebook 中的 V2-M 和 ConvNeXt。

“最多两个 model”在这里按**最多两个架构**理解。同一架构的多个实验需要分别重新训练，才能比较单一方法的效果；目前是 8 个实验 × 2 个架构，共 16 次训练。如果老师指的是最多两次训练总数，就需要先确认，因为这样无法完成独立的消融实验。

## 怎样比较各种方法

| 实验 | 相对 baseline 的变化 | 要分析的问题 |
|---|---|---|
| 00 baseline | 无随机增强、无 dropout / weight decay / label smoothing | 初始性能如何，是否存在过拟合？ |
| 01 geometry | 随机裁剪、水平翻转、小角度旋转 | 对位置、角度、缩放的鲁棒性是否改善？ |
| 02 color | 温和的亮度、对比度、饱和度和色调变化 | 是否减少对光照的依赖？会否破坏鸟类颜色线索？ |
| 03 grayscale | 20% 概率灰度化并保持三通道 | 减少颜色依赖是否有效？ |
| 04 dropout | dropout=0.3 | 泛化是否改善，训练与验证差距是否缩小？ |
| 05 weight decay | AdamW weight_decay=0.001 | 权重正则化是否有效？ |
| 06 label smoothing | smoothing=0.1 | 降低过度自信是否改善验证准确率？ |
| 07 combined | geometry + color + dropout + weight decay + smoothing + erasing | 组合方案是否有效？它可能不如单项实验。 |

01–06 每项都独立与 00 比较，不是接着上一个实验继续训练。07 同时改变多个因素，因此不能声称其提升单独来自 random erasing。每次重置为相同 ImageNet 预训练权重和分类头种子。

所有实验共同使用 classifier warm-up、cosine scheduler、early stopping 和固定 BatchNorm 统计；这些是公共训练设置，不作为已经隔离验证的单项提升。

**random shape** 这里实现为合理的几何变化，不随意扭曲鸟的身体形状。**灰度不等于红外线**：现有 RGB 图像没有真实红外或热成像信息，所以不会把滤镜或灰度伪称为红外数据。

## 数据检查发现的问题

- Train：4,829 张；Test：1,204 张；标签 0–199，共 200 类，标注对应文件均存在。
- Train 中有 3 对 byte-identical 图片，且每对存在类别标签冲突：Crested / Parakeet Auklet、Philadelphia / Warbling Vireo、Red eyed Vireo / Tennessee Warbler。保留原始标注，在报告中披露这一限制。
- 按内容 hash 把 6 张重复图片保留在 fit，再对其余独立图片分层抽样：fit 3,863 张，validation 966 张。200 类均保留，相同图片不会横跨两个分区。
- Test 中 `0197.jpg`、`0273.jpg`、`0918.jpg`、`0976.jpg` 与 Train 图片完全相同。原始 Test 不删除；最后同时报告原始完整测试集指标和去掉这 4 张后的补充指标。
- 训练阶段完全不使用 Test 来选 checkpoint、方法或学习率。模型按 validation Top-1 选择，最终才打开 Test。

## 从哪里开始

1. PowerShell 中运行 `powershell -ExecutionPolicy Bypass -File .\setup_gpu.ps1`。
2. 打开 `Bird_Classification_Local_GPU.ipynb`，在 VS Code 右上角选择 `.venv\Scripts\python.exe` kernel。
3. 顺序运行：GPU 检查 → 参数 → 数据划分 → 增强预览 → MobileNet baseline。
4. 查看第一条真实 learning curve，再运行其余实验和 EfficientNetV2-S。完整实验可能需要数小时。
5. 实验完成后看 `comparison.csv`、准确率与增益图，写出真实结论。训练曲线中 fit-eval 明显高于 validation 才是需要讨论的过拟合证据之一，不能只看 training accuracy。
6. 最后运行 Test evaluation，导出模型；终端运行 `.venv\Scripts\python.exe -m streamlit run app.py`。
7. 上传到 Hugging Face Docker Space，录视频，然后写报告。训练前没有最终准确率或可提交的训练成果。

若 CUDA out of memory，重启 kernel，改 batch_size=4、accumulation_steps=8，用新的 output_dir 重跑。更改模型配置时不能覆盖原实验。相同配置重跑会续训或跳过已完成实验。

## 建议 4 页报告结构

1. **Methodology**：数据与 split、重复数据问题、两个架构、ImageNet transfer learning、输入预处理、loss 和训练参数。
2. **Methodology / Experiments**：baseline 与单项方法表、控制变量、组合方案、checkpoint 选择原则。
3. **Result and Discussion**：真实 validation 对比表与曲线、过拟合分析、两项最终 Test 指标，并附去重复后的补充指标。
4. **Result and Discussion**：错误样本、颜色或形状相近类别的混淆、模型选择理由、4GB/224px/单随机种子的限制；YouTube 和 Hugging Face 链接。

姓名、学号和链接必须自己补上。没有训练完成前，不应编写虚构的 accuracy、讨论或最终报告。
