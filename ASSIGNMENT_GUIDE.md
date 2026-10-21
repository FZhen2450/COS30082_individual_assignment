# 作业要求与新版实验安排

依据 `Assignment1_v2.pdf`，任务是个人完成 200 类鸟类分类，占 20%。老师已同意用 **Streamlit**，保留这项确认作为依据；原始 PDF 不变。

## 提交内容

1. **最多 4 页 PDF 报告**，姓名与学号，主体为 Methodology、Result and Discussion。
2. **Python 源码 ZIP**，有注释并能说明实现。
3. **最长 10 分钟视频**：解释 CNN、训练方案、关键代码、实验结果，并演示 Streamlit。
4. 报告附 YouTube 视频链接与 Streamlit 运行说明；有线上部署再附应用链接。

`submission` 是本地整理提交材料的文件夹，不代表必须把整个文件夹交上去。打包脚本生成其中的 `COS30082_source.zip`；报告 PDF 和源码 ZIP 按课程提交入口要求一起提交，视频按要求提供链接。权重和数据图片不在源码 ZIP 中，若老师另行要求权重，应另附文件或链接。

## 新一轮怎样开始

停止旧训练，Restart Kernel，选择 `.venv\Scripts\python.exe`，从 notebook 顶部顺序运行。新版会把旧输出清空，代码重新生成；已有模型文件不删除。

**正式新结果只写 `artifacts_retrain_v3`。** 旧 `artifacts` 不参与新表格、Streamlit 或提交打包。每个阶段两个模型都完成再进入下一阶段；中断可从该轮最后完整 epoch 恢复。改训练参数时另设新文件夹，避免混用实验。

两个架构是 MobileNetV1 width 1.0 和 EfficientNetV2-S。13 阶段 × 2 架构，共 26 次训练，不是增加到 26 个架构。每次从同一套初始 ImageNet 权重与 seed 开始，避免后期方法因额外训练时间而占优势。

| Stage | 新增方法，保留前面所有设置 |
|---|---|
| 0 | 冻结 ImageNet backbone，只训练新 200 类分类头 |
| 1 | 前 2 epoch 训练分类头，再解冻 backbone、用较小 LR 微调 |
| 2 | RandomResizedCrop：面积 40–100%，长宽比 0.75–1.33 |
| 3 | 左右 horizontal flip，p=0.5 |
| 4 | 上下 vertical flip，p=0.2 |
| 5 | 均匀随机旋转 −45° 到 +45° |
| 6 | 更强 ColorJitter：亮度/对比度 0.4、饱和度 0.5、色调 0.1 |
| 7 | 反色、红/绿/蓝/黄色层，总概率 0.25，色层强度随机 20–45% |
| 8 | 分类头 dropout 0.3 |
| 9 | AdamW weight decay 0.001 |
| 10 | Label smoothing 0.1 |
| 11 | Random erasing，p=0.25，面积 2–15% |
| 12 | Grayscale，p=0.2 |

Stage 7 触发后，五种效果的条件概率依次为 10/25/25/25/15%，所以训练图片访问中约 2.5% 会反色。它是额外实验，并非必然有益的默认结论。五个单项示例强制展示各效果，完整配方预览和训练使用实际概率。

第一次训练你的鸟数据也能 fine-tune：backbone 已经在 ImageNet 训练过。冻结分类头之外的层是迁移学习策略，随后解冻适应鸟类；并非只有新增类别才需要。它修改权重，不修改图片，所以只展示设置表。Dropout、weight decay、label smoothing 同样不伪造输入图片。

五张 crop 示例覆盖不同面积区间与位置；五张 rotation 示例覆盖不同角度区间，让变化容易看清。训练每次加载图片都会独立随机抽样，远不止五种。左右翻转和上下翻转是两个不同阶段，每种只有原图/翻转两种状态。预览保留同一张 validation 原图与 hash，并恢复 RNG，不改变训练数据与随机序列。

反色、强色层、上下翻转可能破坏羽毛颜色或姿态线索。**不保证每一步提高 accuracy**；负提升也是实验发现。RGB 色层不是红外线或温度检测。模型比较图片与 Grad-CAM 已移除，保留增强示例及结果表格。

## 指标和对比表格

每个 epoch 显示 **Train Accuracy、Train Loss、Val Accuracy、Val Loss**。另列 Val Average Accuracy per Class、Best Val Accuracy、Train Eval Accuracy、Gap。Best 是验证集选择的最好 epoch；Gap 是关闭随机增强/dropout 的 Train Eval Accuracy 减 Val Accuracy。在线 Train Accuracy 用实际训练 batch，两者不能混淆。

展示的训练/验证 loss 都是未平滑交叉熵；平滑后的优化目标另存 `train_objective`。每个阶段结束后显示两模型的完整 epoch 表与逐阶段比较表。新版不自动画比较图或训练曲线。

最终必须报告 Test 的 **Top-1 accuracy** 和 **Average accuracy per class（200 类 macro recall）**，不是只报告 Train/Val。Test 只在所有配方完成并冻结选择后打开；比较两个 baseline 与各自 validation 最佳方案。最后一步未必最好。

`comparison_display.csv` 中 accuracy 为百分数，变化为百分点 pp。例如 30% 到 40% 是 +10 pp。`paired_comparison.csv` 是机器可读版本，accuracy 为 0–1。Previous delta 是在前面方法已存在的条件下新增方法的变化，不能解释为单独使用该方法的收益。单 seed 不声称统计显著性，不能人为把 baseline 压到 20–40%。

## 数据与报告证据

Train 4,829，Test 1,204，200 类。Train 有 3 对内容完全相同且标签冲突的图片，六行保留 fit；其余分层划分得到 fit 3,863 / validation 966，双方包含全部类别，内容不泄漏。保持原标签并披露局限。

Test 的 `0197.jpg`、`0273.jpg`、`0918.jpg`、`0976.jpg` 与 Train 内容重复。保留官方完整 Test，同时提供移除四张后的 clean 指标。Test 不能用来调整增强、LR 或选择模型。

| Rubric | 分数 | 报告和演示要体现的证据 |
|---|---|---|
| Report | 7 | 架构、训练方案、定量比较、定性分析及有根据的发现 |
| AI modeling framework | 7 | 数据/loader、预处理、优化、指标、训练、泛化设计与实际准确率 |
| Video | 3 | 清楚解释方法、代码与真实结果，展示理解 |
| Deployment/web | 3 | 依老师同意用 Streamlit，展示可用预测及清晰结果表 |

报告优先放精简逐阶段表、baseline/最佳方案的关键 epoch 指标表、最终 Test 表、少量真实错误案例及数据质量发现。解释两个 CNN 的具体结构、分类头、冻结解冻、loss、超参数。讨论强颜色增强是否破坏物种线索、垂直翻转是否有害、模型表现与训练成本。把观察与推测区分开。代码运行成功不等于达到高分档，结论必须有真实结果支持。

视频建议：任务/指标 1 分钟，CNN 与迁移学习 2 分钟，关键代码 2 分钟，结果分析 2 分钟，Streamlit 上传/Top-5/模型切换/对比表 2 分钟，局限与结论半分钟。不要在视频中等待完整训练。

## 应用与打包

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe tools/package_source.py
```

应用只读取新 study，可选择 checkpoint、上传图片、查看 Top-5、逐阶段/逐 epoch 表和增强示例。单张图片概率不是整个模型 accuracy。最终部署文件为 `artifacts_retrain_v3/deployment/bird_model.pth`，包含预处理及类别映射。没有新 checkpoint 时显示待训练状态。

打包只收集源码和新 study 的 CSV/JSON/PNG，不带旧实验、模型权重、环境、cache 或数据图片。完整训练后重新打包再交。姓名学号、报告、视频录制仍需你完成。
