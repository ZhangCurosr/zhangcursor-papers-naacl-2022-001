---
title: "FACTPEGASUS-Factuality-Aware-Pre-training-and-Fine-tuning-fo"
source: https://aclanthology.org/2022.naacl-main.74.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:02"
field: "自然语言生成-事实性摘要"
keywords: ["abstractive summarization", "factuality", "hallucination", "contrastive learning", "pre-training objective", "PEGASUS", "BART"]
innovations: ["事实感知预训练目标factGSG，融合ROUGE与FactCC进行句子选择", "Corrector-Contrastor-Connector三模块微调框架，端到端解决幻觉问题", "文档-摘要对比学习范式，提升事实性同时保持抽象能力"]
benchmarks: ["XSum", "WikiHow", "Gigaword"]
---

# 论文速读：FACTPEGASUS: Factuality-Aware Pre-training and Fine-tuning for Abstractive Summarization

## 一句话总结
本文提出FACTPEGASUS，通过在预训练阶段引入事实感知目标（factGSG）和在微调阶段设计三个互补模块（Corrector、Contrastor、Connector），系统性地解决摘要是生成中的幻觉问题，在XSum、WikiHow和Gigaword三个数据集上显著提升了事实性指标。

## 研究问题与动机
1. **幻觉问题的严重性**：当前抽象式摘要模型生成的内容常包含原文中不存在的实体或事实，严重影响模型的可靠性和实际应用安全性。
2. **预训练目标忽视事实性**：现有预训练方法（如PEGASUS的GSG）主要优化ROUGE质量指标，未考虑生成句子与原文的事实一致性，可能导致选择高ROUGE但含幻觉的句子。
3. **微调阶段的模仿错误**：下游数据集中存在大量幻觉摘要（如XSum中约70%的摘要含幻觉），模型直接微调会产生"模仿性虚假"，甚至遗忘预训练阶段学到的事实性行为。
4. **现有解决方案的局限**：后处理方法依赖额外资源训练校正模型；基于对比学习的方法仅在摘要之间进行对比，未充分利用文档上下文；过滤非事实训练数据会大幅减少可用样本量。

## 核心贡献（创新点）
1. **事实感知预训练目标factGSG**：在PEGASUS的GSG基础上，将FactCC事实性指标融入句子选择策略，选择既重要又符合事实的句子作为伪摘要，与传统仅基于ROUGE的选择方法本质不同。
2. **Corrector模块**：通过三种策略（Replace/Remove/Combined）移除训练数据参考摘要中的幻觉实体，使模型能在完整训练集上学习而不受非事实行为污染，区别于已有工作仅过滤整个训练样本。
3. **Contrastor模块**：采用文档-摘要之间的对比学习（而非摘要-摘要对比），以NT-Xent损失鼓励模型区分事实性摘要与非事实性摘要，更贴近"忠实性"的定义。
4. **Connector技术**：利用GSG风格目标，在下游任务输入文档中插入mask token，模拟预训练任务设置以桥接预训练与微调差距，是一种零样本/少样本知识迁移的有效机制。

## 方法详解

### 3.1 事实感知预训练（factGSG）
- **原始GSG问题**：PEGASUS使用ROUGE-1作为句子重要性衡量标准，但高ROUGE分句子可能包含幻觉（如图1a中句子C因提及Denver火灾而获得高分，但原文其余部分讲述Seattle火灾）。
- **factGSG改进**：句子得分公式为 $s_i = rouge(x_i, D \setminus \{x_i\}) + FactCC(x_i, D \setminus \{x_i\})$，同时考虑重要性和事实性。
- **gap sentence ratio调整**：从原始30%改为仅选择1个句子，减少幻觉可能性。
- **FactCC使用**：二值事实一致性指标（0或1），计算速度快且与人类判断高度相关。

### 3.2 事实感知微调

**Connector（桥接预训练与微调）**：
- 在下游任务输入文档的句子间插入mask token，模拟预训练时的GSG任务设置。
- 最佳位置通过对验证集ROUGE评估确定，实验发现放在第一段前效果最佳（与XSum数据集生成方式一致）。
- 本质上是一种null-prompting策略。

**Corrector（校正幻觉）**：
- **Replace策略**：用文档中相同NER标签的相似实体替换幻觉实体，保证语法正确性。
- **Remove策略**：激进地移除幻觉实体及相关依存弧词语，避免产生错误信息。
- **Combined策略**：先Replace再对剩余幻觉执行Remove，平衡事实性与语法性。
- 幻觉检测基于spaCy NER：摘要中无法在文档中找到相同字符串的实体视为幻觉。

**Contrastor（对比学习）**：
- 以文档$D_i$为anchor，参考摘要$S_i$为正样本，生成非事实摘要$N_i$为负样本。
- 负样本生成：替换实体为同类型随机实体，分为intrinsic（用文档内实体）和extrinsic（用训练集中非文档实体）两类。
- 使用NT-Xent损失：$l_{D_i, S_i} = -\log \frac{\exp(\sin(z_{D_i}, z_{S_i})/\tau)}{\sum_{S_j \in N_i \cup \{S_i\}} \exp(\sin(z_{D_i}, z_{S_j})/\tau)}$
- 总损失：$L = L_{CE} + \lambda L_{CL}$，其中$\lambda=5, \tau=0.05$。
- 与Cao & Wang (2021)的区别：在文档-摘要间对比（而非摘要-摘要间对比），更符合忠实性定义。

## 实验与结果

### 数据集与基线
- **预训练**：C4数据集（realnewslike子集）
- **微调任务**：XSum（204K）、WikiHow（157K）、Gigaword（3.8M）
- **基线**：BART-base、PEGASUS*（自复现版本）、DAE（Goyal & Durrett, 2021）、CLIFF（Cao & Wang, 2021）

### 主要结果（Table 1）
**XSum数据集**：
- FACTPEGASUS vs BART-base：token error下降51%（12.38→6.07）、sentence error下降36%（60.70→38.66）、FactCC提升43%（23.99→34.32）
- 相比DAE，FactCC提升34%（25.43→34.32）
- ROUGE-L下降约2点（33.78→31.17），但事实性提升远大于质量损失

**WikiHow数据集**：
- sentence error下降约3点（45.77→42.40）
- FactCC提升0.32点（99.09→99.41）

**Gigaword数据集**：
- FactCC提升4.36点（55.66→60.02）

### 零样本与少样本实验（Table 4, Figure 4）
- **零样本**：factGSG+mask vs GSG+mask，sentence error降低5点，FactCC提升约10点，证明预训练目标的有效性。
- **少样本**：仅10个样本时，PEGASUS*事实性开始退化，而FACTPEGASUS保持事实性并随样本增加持续改善。

### 事实性动态分析（Figure 5）
- BART-base微调过程中token error和sentence error分别增加2点和8点
- FACTPEGASUS仅增加1点和4.8点，退化程度减半

### 事实性与抽象度权衡（Figure 6）
- FACTPEGASUS位于faithfulness-abstractiveness曲线上方，证明事实性提升并非单纯依赖增加抽取程度。

### 人工评估（Table 2）
- FACTPEGASUS事实性得分39.66，显著高于BART-base（24.67）、PEGASUS*（27.33）和CLIFF（29.33）
- 信息量无显著差异（p>0.15）

## 相关工作脉络

1. **PEGASUS（Zhang et al., 2020）**：GSG预训练目标的基础工作，本文在其基础上引入事实性指标扩展预训练目标。
2. **DAE（Goyal & Durrett, 2021）**：使用DEP-Entail过滤非事实训练样本，但仅能使用部分训练数据；本文通过Corrector校正而非删除样本。
3. **CLIFF（Cao & Wang, 2021）**：摘要间的对比学习，负样本为自动生成幻觉摘要；本文扩展为文档-摘要对比学习，更贴近忠实性定义。
4. **SimCLS（Liu & Liu, 2021）**：摘要质量对比学习框架；本文将其应用于事实性，并以文档而非摘要作为anchor。
5. **Post-processing方法（Chen et al., 2021; Dong et al., 2020）**：依赖外部模型进行后处理；本文端到端集成事实性优化。
6. **Nan et al. (2021)**：实体级事实一致性过滤；本文保留完整训练集并通过校正而非过滤处理幻觉。

## 局限性与未来方向

1. **Corrector的语法限制**：Remove策略在特定句法结构（如主语或及物动词宾语位置的幻觉实体）可能产生不grammatical句子，需更完善的校正算法。
2. **预训练计算成本**：FactCC计算虽只针对ROUGE top-5句子，但仍增加额外开销。
3. **依赖FactCC的局限性**：作为二值分类器，FactCC可能无法捕捉细粒度的事实性差异。
4. **未来方向**：探索更系统的自动校正算法、将方法扩展到多模态摘要、研究更大规模预训练下事实性的保持。

## 研究启发与可借鉴点

1. **预训练目标的系统性扩展**：将下游评估指标（如事实性）融入预训练目标，而非仅在微调阶段优化，是值得借鉴的整体训练框架设计思路。
2. **文档-摘要对比学习范式**：以文档为anchor、摘要为正样本的对比学习设置，比摘要间对比更符合忠实性定义，可迁移至其他生成任务。
3. **Mask token作为prompt**：利用预训练任务自然的mask token进行零样本/少样本知识迁移，为pretrain-finetune gap bridging提供了简洁有效的技术方案。
4. **训练数据校正而非过滤**：Corrector的Replace+Remove策略证明可以对含噪声的训练数据进行校正而非简单删除，从而保留更多训练信号。
5. **多指标联合优化**：同时考虑ROUGE（质量）和FactCC（事实性）的sentence selection策略，为多目标预训练提供可行方案。

## 关键术语表

**Factuality/Faithfulness**：摘要与源文档的事实一致性，要求摘要中的所有陈述都能在文档中找到依据。

**Hallucination**：摘要中出现的在源文档中不存在的事实或实体，是抽象摘要模型的核心问题。

**Gap Sentence Generation (GSG)**：PEGASUS的预训练目标，随机选择重要句子作为伪摘要并掩码原文中对应部分进行生成。

**FactCC**：基于NLI的事实一致性评估指标，输出二值判断（1=一致，0=不一致），计算快速且与人类判断高度相关。

**Imitative Falsehood**：模型在微调阶段从训练数据中学习到并复现的幻觉行为，导致遗忘预训练学到的事实性。

**NT-Xent Loss**：Normalized Temperature-scaled Cross Entropy loss，对比学习的标准损失函数，用于拉近正样本对、推远负样本对。

**Intrinsic/Extrinsic Hallucination**：Intrinsic指文档中存在但被错误表述的实体，Extrinsic指文档中完全不存在却被引入的实体。

**Null-Prompting**：使用预设的mask token而非人工设计的文本prompt来引导模型行为的少样本/零样本学习方法。

## 可复现要素

- **预训练数据**：C4数据集（realnewslike子集），论文未提供预处理脚本
- **微调数据集**：XSum、WikiHow、Gigaword，均通过HuggingFace datasets库获取
- **代码开源**：论文未提及代码开源链接
- **模型权重**：PEGASUS*为作者自复现版本，无公开checkpoint
- **关键超参**：
  - 预训练：lr=1e-4, weight_decay=0.01, batch_size=256, max_input=512, max_output=256, 750K steps（full model）
  - 微调：lr=3e-5, batch_size=256, beam_size=6, label_smoothing=0.1
  - Contrastor：λ=5, τ=0.05, 5个负样本
