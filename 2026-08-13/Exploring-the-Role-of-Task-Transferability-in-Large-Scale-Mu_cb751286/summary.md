---
title: "Exploring-the-Role-of-Task-Transferability-in-Large-Scale-Mu"
source: https://aclanthology.org/2022.naacl-main.183.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:36:32"
field: "多任务学习与迁移学习"
keywords: ["multi-task learning", "task transferability", "pre-finetuning", "representation learning", "negative transfer"]
innovations: ["系统性解耦多任务规模与任务相关性对下游性能的影响，首次在任务组层面验证 transferability", "证明当目标任务已知时，小规模相关任务子集的预微调可匹敌大规模多任务训练并大幅降低计算成本"]
benchmarks: ["GLUE", "SuperGLUE", "MRQA", "CONLL-2003", "SQuAD"]
---

# 论文速读：Exploring the Role of Task Transferability in Large-Scale Multi-Task Learning

## 一句话总结
本文通过系统实验揭示了大规模多任务学习（scale）与任务可迁移性（transferability）之间的权衡关系：当目标任务未知时，扩大多任务规模能获得更优的下游表现；而当目标任务已知时，针对相关任务子集进行小规模预微调可达到与大模型相当的下游性能，且计算成本更低。

## 研究问题与动机
1. **矛盾现象**：Aghajanyan et al. (Muppet, 2021) 发现使用大量多样化任务（约15个）的多任务预微调能均匀提升下游性能；而 task transferability 文献（Vu et al., 2020; Pruksachatkun et al., 2020）则表明中间任务的选择对下游性能影响巨大，存在高度方差。
2. **核心疑问**：如果目标任务已知，那么针对一个与目标任务相关的较小任务组进行预微调，能否达到与大规模多任务训练相当的性能？
3. **现有方法不足**：大规模多任务训练虽然鲁棒但计算昂贵；仅依赖任务格式启发式分组进行任务选择尚不明确其实际效果边界，尤其是负迁移（negative transfer）在多任务规模下的表现。
4. **研究空白**：此前关于任务相似性的工作主要聚焦于单个中间任务与单个目标任务之间的迁移预测，尚未扩展到任务组（task groups）层面。

## 核心贡献（创新点）
1. **系统性解耦 scale 与 transferability 效应**：通过遍历所有任务组组合（7种），在统一实验设置下对比分析"任务数量"和"任务相关性"对下游性能的独立与交互影响，这是首次将单任务层面的 transferability 结论推广到任务组层面。
2. **揭示小群相关任务的性价比优势**：发现 Only-SL 和 Only-QA 在对应未见过目标任务上与全量三组（SL+QA+C）性能相当，但计算成本显著降低——本质区别在于本文证明在目标明确时，精心选择相关子集可以替代盲目扩大任务规模。
3. **发现多任务训练降低随机重启方差**：大规模多任务预微调（SL+QA+C）在所有目标任务上均呈现最低的跨5次随机种子性能波动，相比单任务组设置显著降低结果方差，这对工程部署具有直接价值。
4. **证明简单启发式分组的合理性**：尽管基于输出格式（task format）的任务分组是一种"不完美的启发式"，但在隔离 transferability 效应方面仍然是合理的选择，为后续研究提供了基线参照。

## 方法详解
- **实验范式**：遵循 Aghajanyan et al. (2021) 的两步流水线——第一步用 XLM-RoBERTa-base 在任务集上进行 **pre-finetuning**（共享编码器 + 各任务专属 head），第二步在12个未见过目标任务上分别 **finetune**（新随机初始化 head）。
- **任务分组**：将29个预微调任务和12个目标任务按输出空间划分为三类：**Sequence Labelling (SL)**、**Extractive Question Answering (QA)**、**Classification (C)**。
- **实验组合**：对所有7种非空组合进行 pre-finetuning——Only-SL、Only-C、Only-QA、SL+C、SL+QA、QA+C、SL+QA+C，baseline 为直接用预训练模型 finetune 目标任务。
- **训练细节**：cross-entropy loss，batch size=128，从验证集按均值 loss early stopping（连续3轮不提升则停止），learning rate 从 1e-3 到 1e-5 网格搜索。
- **关键设计**：使用 heterogeneous batches（每个 batch 中各任务样本按数据集比例采样），不使用 loss scaling（前期实验发现不加 loss scaling 效果更好）。
- **评估**：每个设置在5个不同随机种子上的平均验证集性能 ± 标准差。

## 实验与结果
- **数据集**：29个预微调任务（15个SL来自CONLL/Ontonotes等、9个C来自GLUE、6个QA来自MRQA等）+ 12个未见目标任务（4个SL、4个C来自SuperGLUE、4个QA来自SQuAD/QED/SubjQA/XQuad）。
- **最强结果**：全量任务组合 **SL+QA+C**（29个任务）在所有目标任务上取得最高平均分数 **76.483**，并在5个单独任务上领先。
- **相关子集竞争力**：
  - Only-SL 在未见 SL 任务上达 **80.844**，与 SL+QA+C 的 **80.750** 相当；
  - Only-QA 在未见 QA 任务上达 **75.252**，与 SL+QA+C 的 **75.678** 接近。
- **负迁移严重**：Only-C 预微调在 SL 目标任务上比 baseline 下降 **9.6%**；Only-SL 预微调在 QA 目标任务上比 baseline 下降 **20.3%**。
- **配对任务不如单群**：Only-QA > SL+C 在 QA 目标上，Only-SL > QA+C 在 SL 目标上，Only-C > SL+QA 在 C 目标上——说明小规模下任务选择尤为关键。
- **方差降低**：SL+QA+C 的全局平均标准差为 **0.852**，远低于 baseline 的 **2.662**。
- **计算成本对比**：SL+QA+C 每 epoch 4884s vs Only-SL 每 epoch 1131s（约 4.3x 差距）。

## 相关工作脉络
1. **Muppet (Aghajanyan et al., 2021)**：提出大规模多任务 pre-finetuning 作为任务无关的预训练第二階段；本文在其基础上系统拆解了"规模 vs 相关性"的作用，而非仅报告规模效益。
2. **Task Transferability (Pruksachatkun et al., 2020; Vu et al., 2020)**：研究单个中间任务对单个目标任务的迁移效果；本文将其扩展至任务组层面（groups of tasks）。
3. **EXT5 (Aribandi et al., 2021)**：并行工作，使用 T5-style 模型研究任务族间迁移；本文使用 shared encoder（XLM-RoBERTa）方案，证明 transferability 现象与模型架构无关，且本文额外报告了性能方差分析。
4. **Task Embeddings (Achille et al., 2019; Wang et al., 2019)**：通过梯度或文本表示学习任务向量以预测迁移性；本文采用更简单的基于输出格式的分组启发式，验证其有效性。
5. **Phang et al. (2018)**：早期发现多任务训练降低随机重启方差；本文在更大规模和更细粒度设置下复现并扩展了这一观察。

## 局限性与未来方向
1. **任务分组方式的近似性**：基于任务输出格式（format）的分组是粗糙启发式，未利用基于梯度/表征的更精确任务相似性度量（如 Task2vec）；更优分组是一个组合优化问题，计算代价高。
2. **各组任务数量和示例量不均衡**：SL（225K）、C（944K）、QA（436K）训练样本量差异大，控制示例数量会忽略任务难度差异，如何公平归一化是开放问题。
3. **仅覆盖三类常见 NLP 格式**：实验限于 SL/QA/C 三种标准任务类型，对非标准或新兴任务类型的适用性未知。
4. **负迁移缓解方法缺失**：论文暗示需要研究能缓解大规模多任务训练中负迁移的建模技术，但未给出具体方案。

## 研究启发与可借鉴点
1. **双阶段流水线设计**："pre-finetune（共享编码器）→ finetune（新 head）"的两阶段范式可作为其他研究中控制变量、隔离 scale/transferability 效应的标准实验框架，值得在本团队研究中复用。
2. **方差作为评估指标**：除均值外，跨随机种子的性能标准差同样重要——多任务训练不仅提升均值还降低方差，可作为模型鲁棒性的补充指标纳入实验设计。
3. **小场景 vs 大场景的分情况决策**：当目标任务已知时优先选择相关任务子集以节省算力；目标任务不确定时再使用大规模多任务方案——这一原则可作为团队在实际部署中的决策指南。
4. **损失缩放（loss scaling）的反例**：本文在 preliminary 实验中不使用 loss scaling 反而取得更好结果，提示超参敏感性，团队在其他实验中应同时对比有/无 loss scaling 的情形。
5. **任务分组作为初步分析手段**：即使基于格式的分组较粗糙，仍能有效隔离 transferability 信号；在研究初期可用此方法快速定位关键因素，再逐步引入更精细的相似性度量。

## 关键术语表
- **Pre-finetuning**：在预训练模型之后、目标任务微调之前的中间监督多任务训练阶段，用于学习共享表示。
- **Task Transferability（任务可迁移性）**：指某个源/中间任务的知识对特定目标任务性能影响的程度。
- **Heterogeneous Batches**：在每个 mini-batch 中包含来自多个任务的样本，按数据集规模比例采样。
- **Negative Transfer（负迁移）**：中间任务对目标任务产生有害影响，导致下游性能低于单独在目标任务上微调的效果。
- **Task Format（任务格式）**：按任务输出空间划分的方式（如分类、序列标注、抽取式问答），本文分组的主要依据。
- **Representational Collapse（表征坍缩）**：多任务训练中不同任务的表征趋同退化的现象（Aghajanyan et al., 2020）。
- **GLUE / SuperGLUE**：NLP 通用语言理解基准评测套件；本文 C 类任务的预微调/目标数据均来源于此。
- **MRQA**：大规模多领域阅读理解数据集集合；本文 QA 类预微调任务主要来源于此。

## 可复现要素
- **数据集**：SL 任务来自 CONLL、Ontonotes、GMB、ANEM、Few-NERD 等公开数据集；C 类任务来自 GLUE/SuperGLUE；QA 类来自 MRQA 及 Huggingface Datasets 库。所有数据均公开可用。
- **代码/权重**：论文未提供代码和预训练权重的开源链接。
- **关键超参**：batch size=128；learning rate sweep: {1e-3, 1e-4, 1e-5}；early stopping 阈值=连续3个 epoch 验证 loss 不下降；随机种子数=5；Adam 默认参数（来自 Huggingface Trainer）。
- **硬件**：Amazon p3.16xlarge EC2，8× Tesla V100 GPU。
