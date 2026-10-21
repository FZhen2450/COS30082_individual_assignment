# 作业要求与实验安排

依据 `Assignment1_v2.pdf` 第 1–3 页，并按老师最新同意的方案，将应用展示改为 Streamlit。原始 PDF 保留不变；其中 Hugging Face 的要求由这项老师确认的调整替代。

## 要交什么

这是个人作业，占 20%，任务是 200 类鸟类图片分类。

1. **PDF 报告，最多 4 页**，包含姓名与学号。两个主体部分：Methodology、Result and Discussion。
2. **Python 源码 ZIP**，代码要有充分注释，体现你理解实现。
3. **最长 10 分钟视频**：解释概念与模型设计、介绍代码、展示训练及结果、演示 Streamlit AI 网页应用。建议配 slides。
4. 报告包含 **YouTube 视频链接**，并说明 Streamlit 应用如何运行；若有线上部署，可附应用链接。不再要求 Hugging Face 链接。

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
7. 在 Streamlit 中演示上传鸟类图片、Top-5 预测和实验对比，录视频，然后写报告。本机运行即可按上述命令准备演示；训练前没有最终准确率或可提交的训练成果。

若 CUDA out of memory，重启 kernel，改 batch_size=4、accumulation_steps=8，用新的 output_dir 重跑。更改模型配置时不能覆盖原实验。相同配置重跑会续训或跳过已完成实验。

## 建议 4 页报告结构

1. **Methodology**：数据与 split、重复数据问题、两个架构、ImageNet transfer learning、输入预处理、loss 和训练参数。
2. **Methodology / Experiments**：baseline 与单项方法表、控制变量、组合方案、checkpoint 选择原则。
3. **Result and Discussion**：真实 validation 对比表与曲线、过拟合分析、两项最终 Test 指标，并附去重复后的补充指标。
4. **Result and Discussion**：错误样本、颜色或形状相近类别的混淆、模型选择理由、4GB/224px/单随机种子的限制；YouTube 链接与 Streamlit 演示/运行说明（若已线上部署，可附链接）。

姓名、学号和链接必须自己补上。没有训练完成前，不应编写虚构的 accuracy、讨论或最终报告。

## 按最新 rubric 对齐评分

以下依据你提供的 rubric，总分 **20 分**。其中应用评分仍写 Gradio；本项目按老师已同意的调整使用 **Streamlit**，保留对应用设计与展示质量的要求。

| 评分项 | 分值 | 高分需要展示什么 | 当前状态与下一步 |
|---|---:|---|---|
| Report | 7 | 连贯解释 CNN 架构、训练方案和方法差异；用额外定量与定性结果支撑发现 | 实验输出流程已准备；完整训练后才能写真实比较和讨论 |
| Video presentation | 3 | 清晰解释所用 CNN 和代码，展示对方法的理解，而不只是播放运行画面 | 录制尚未完成；需讲清为什么选这两个模型，以及实际改进是否有效 |
| Model deployment and web application | 3 | 完整、易用、设计清楚的模型应用，展示质量超出基础功能 | Streamlit 已有上传、Top-5、模型选择和实验图表；训练后需检查真实使用流程并录制演示 |
| AI modeling framework | 7 | 数据输入、预处理、loader、架构、优化器、指标和训练方案完整；有泛化设计并取得良好准确率 | 训练流程已做小样本验证；正式泛化表现与准确率仍待完整实验验证 |

**代码能运行，不等于已达到高分档。** 目前完成的是实现与流程验证；rubric 中要求的准确率、有效改进和研究发现，必须由实际训练结果支持。rubric 没有给出明确的 accuracy 分数线，不能承诺某个准确率就一定满分。

### 报告中最值得保留的证据

在 4 页限制内，建议优先放以下内容，而不是把所有输出截图塞进报告：

1. **一个精简实验表**：两个架构的 baseline 与各方法的 validation Top-1、Average accuracy per class、相对 baseline 的变化（pp）。明确哪些方法没有提升，单种子结果不当作统计显著结论。
2. **两组关键 learning curves**：各架构 baseline 与 validation 最佳方案的 fit-eval / validation 曲线；结合准确率差距和 loss 讨论过拟合，而不是只写“使用了 dropout”。
3. **一个最终测试表**：两个架构的 baseline 与 validation 最佳方案，报告两项必需指标；模型选择不能依据这个测试表重新调整。
4. **少量正确与错误样本**：说明哪些鸟类容易混淆，以及颜色、姿态、背景或裁剪可能造成的影响。观察到的现象与未经验证的原因要分开描述。
5. **数据质量发现**：披露重复图与冲突标签、说明如何避免 fit/validation 泄漏，并比较完整 Test 与去重复子集指标。这个发现有助于支撑分析深度，但不能保证额外得分。

具体可分析的问题包括：颜色增强有没有破坏细粒度的羽毛颜色线索？灰度增强是否降低表现？组合方案是否反而不如某个单项方案？较大的模型是否值得额外训练时间？这些都要等真实结果出来再回答。

完整的 200×200 confusion matrix 放在结果文件里；报告优先选择少数典型混淆类别或错误图片，保证图中文字可读。两个模型的结构说明应涉及实际采用的模块、200 类分类头、冻结/解冻策略和 loss，不能只列模型名称。

### 视频建议安排（控制在 10 分钟内）

| 时间 | 内容 |
|---|---|
| 0:00–1:00 | 200 类分类任务、数据集与两项评估指标 |
| 1:00–3:00 | 两个 CNN 的主要结构、transfer learning、分类头与训练方案 |
| 3:00–5:00 | 关键代码：split、增强、loader、优化器、checkpoint 与指标计算 |
| 5:00–7:00 | 展示训练记录、baseline/改进对比、过拟合与错误分析 |
| 7:00–9:00 | Streamlit 实际上传预测、Top-5、模型选择与实验对比 |
| 9:00–9:30 | 有证据支持的最佳方案、局限与结论，保留少量时间余量 |

不用在视频里等待完整训练；展示已保存的实际训练记录，并讲清如何生成结果。避免逐行朗读全部代码，重点解释设计选择和证据。

### Streamlit 应用的演示检查

- 图片预览、预测鸟种、Top-5 概率是否清楚，模型与实验名称是否易于理解。
- 同一张图片可以切换已训练模型查看差异；概率高不代表一定正确，也不能当作总体 accuracy。
- 实验对比图是否标清 validation、百分比和提升百分点；最终 Test 结果与 validation 结果是否可区分。
- 上传无效图片或尚未有 checkpoint 时，是否给出清楚提示；演示前需用真实训练权重完成一次完整操作。
- 当前页面已有上述主要功能，但还不能仅凭功能清单断言符合 rubric 的 outstanding design；真实使用体验与演示清晰度仍需检查。

### 接下来优先做什么

先跑完 MobileNet baseline 并看曲线，再完成两个架构的对照实验；在查看 Test 前完成验证集分析与方案选择。随后运行最终评估、检查 Streamlit 真实预测流程，最后写报告与录视频。优先投入时间在 rubric 共占 **14/20 分**的报告和建模证据上，同时完成应用与视频的展示要求。
