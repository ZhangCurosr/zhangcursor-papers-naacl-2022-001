---
title: "Disentangled-Action-Recognition-with-Knowledge-Bases"
source: https://aclanthology.org/2022.naacl-main.41.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:55:38"
field: "组合动作识别与零样本学习"
keywords: ["zero-shot compositional action recognition", "disentangled representation", "knowledge graph", "affordance prior", "macro open world evaluation", "Epic-Kitchen benchmark"]
innovations: ["因子化解耦动词/名词特征并将 KG 节点复杂度从二次降至线性", "利用外部语料学习物性先验约束动词-名词组合合理性", "提出宏开放世界评估协议与基于 Epic-Kitchen 的大规模基准"]
benchmarks: ["Epic-Kitchen v2", "Charades"]
---

# 论文速读：Disentangled-Action-Recognition-with-Knowledge-Bases

## 一句话总结
本文提出 **DARK** 方法，通过解耦动词与名词特征、利用知识图谱预测未见概念的分类权重，并引入物性先验（affordance prior）约束动词-名词组合，解决组合动作识别中**零样本泛化**问题，在 Charades 数据集上取得 SOTA 性能，并构建了基于 Epic-Kitchen 的大规模新基准。

## 研究问题与动机
- **标签空间组合爆炸**：视频动作常表示为“动词-名词”组合（如 move chair, peel apple），可能组合数随动词数 $N_v$ 与名词数 $N_n$ 呈 $O(N_v \times N_n)$ 二次增长，无法为所有组合收集训练数据。
- **现有 KG 方法扩展性差**：先前工作（如 GCNCL）在知识图谱中显式构建动词、名词及**组合动作节点**，训练时节点数二次增长，内存消耗巨大，难以扩展到大词汇表。
- **动词与名词非完全独立**：尽管可分别建模，但动词与名词之间存在类型约束（如 banana 可被 cut 但不会被 peel car），忽略该约束会降低对未见组合的泛化能力。
- **现有评估协议不能反映真实难度**：多数工作仅在数据集已出现的“有效组合类”上进行测试，而真实部署需在全量 $N_v \times N_n$ 标签空间（包含无效组合）中进行预测与评估。

## 核心贡献（创新点）
- **因子化解耦模型**：将动词与名词特征分别提取并解耦，分类器复杂度从 $O(N_v \times N_n)$ 降至 $O(N_v + N_n)$，显著提升可扩展性。
- **知识图谱驱动的新概念分类器生成**：利用 GCN 在外部分开的动词/名词知识图谱上传播语义信息，直接预测未见概念的**分类权重**，而非传播视觉特征。
- **外部语料驱动的物性先验**：从 HowTo100M 字幕语料中自动抽取动词-名词对，训练一个评分函数作为组合约束，使模型在复合动作时能排除不合逻辑的配对。
- **宏开放世界（macro open world）评估协议**：提出在完整 $N_v \times N_n$ 空间进行预测与评估的新设置，并采用 AUC 指标综合衡量 seen/unseen 类别的性能权衡。
- **大规模 Epic-Kitchen 基准**：构建了一个规模约为 Charades 十倍的动作识别基准（3629 个组合类、76605 个视频），并公布详细统计。

## 方法详解
- **因子化解耦动词-名词分类器**：视频 $X$ 经两个独立特征提取器 $\mathcal{F}_v$（动词）和 $\mathcal{F}_n$（名词）得到解耦表征；分别训练线性分类权重 $\mathcal{W}_v^{seen}$、$\mathcal{W}_n^{seen}$，使用交叉熵损失 $\mathcal{L}_{cls}^v$、$\mathcal{L}_{cls}^n$ 进行监督。为解耦，Epic-Kitchen 上通过类无关目标检测器裁剪交互物体（并单独加入手部掩码）作为动词输入，避免视觉特征掺杂对象信息。
- **GCN 学习新概念分类权重**：对动词/名词分别构建知识图谱，以词嵌入（GloVe/BERT）作为节点初始特征，以前期学到的 $\mathcal{W}^{seen}$ 作为监督信号，通过多层 GCN 传播预测未见节点的分类权重 $\hat{\mathcal{W}}^{unseen}$，训练损失为 MSE：$\mathcal{L}_{GCN} = \mathcal{L}_{mse}(\hat{\mathcal{W}}^{seen}, \mathcal{W}^{seen})$。
- **物性先验（Affordance prior）学习**：从 HowTo100M 字幕中提取动词-名词对，训练一个评分函数 $\mathcal{A}(s_v, s_n)$：将动词词嵌入投影到名词嵌入空间后计算余弦距离，经 sigmoid 输出合理性分数。同时训练映射模块 $\mathcal{M}$ 将视觉动词特征映射到语义嵌入空间，损失为 $\mathcal{L}_{mse}(\mathcal{M}(X_v), s_v)$。
- **整体训练与推理**：训练分四步——① 联合训练 $\mathcal{F}_v, \mathcal{F}_n$ 及 seen 分类权重；② 用词嵌入和 $\mathcal{W}^{seen}$ 训练 GCN；③ 用抽取的动词-名词对训练物性评分函数；④ 训练视觉到词嵌入的映射。推理时，动作 $(v,n)$ 的概率由三部分乘积构成：$\mathcal{P}(v,n) = \mathcal{P}(v) * \mathcal{P}(n) * \mathcal{A}(\mathcal{M}(X_v), s_n)$，其中 $\mathcal{P}(v)=\sigma(\mathcal{W}_v * \mathcal{F}_v(X))$，$\mathcal{W}_v$ 对 seen 用已学权重、对 unseen 用 GCN 预测权重。

## 实验与结果
- **数据集**：Epic-Kitchen v2（新基准，3629 个组合类、76605 视频，90 动词、249 名词）与 Charades（9625 视频，34 动词、37 名词，149 组合类）。
- **评估基线**：Triplet、SES、DEM、GCNCL（及其带 GT/Affd/Both 约束的变体）。
- **主要结果（Epic-Kitchen，表1）**：DARK 在所有指标上大幅领先。AUC Macro Open Top1 达 **1.69%**，显著优于 GCNCL+GT（0.044）、GCNCL+Affd（0.064）等；AUC Open Top1 为 **2.04%**；mAP All 为 **2.39%**，mAP Zero-shot class 为 **1.22%**。DARK 将图节点数从 GCNCL 的 22749 降至 **339**。
- **消融实验**：
  - 动词/名词分类器的零样本学习模块：**KG（GloVe）组合**最优（1.81%），优于 SES、ConSE（表3）。
  - 动词知识图谱构建：**VN tree（one-way）** 最佳（1.93%），优于 WN dis（1.86%）和 VN group（0.83%）（表4）。
  - 物性学习：BERT + Proj-Cosine + Visual 映射配置最优，AUC Macro Open Top1 达 **1.69%**，显著优于 Word-only（1.59%）及 Lookup Table（1.35%）（表5）。
- **Charades 结果（表6，GZL 设置）**：DARK 取得 mAP All **11.21%**、mAP Zero-shot **8.38%**，优于 GCNCL-I+A（10.48/7.95）及 Triplet（10.41/7.82）。因数据集较小且为第三人称视角，解耦采用对抗式 discriminator 实现，且未使用物性先验。

## 相关工作脉络
- **零样本学习与知识图谱**：Wang et al. (2018) 利用 GCN 在 KG 上蒸馏语义关系生成视觉分类器；Kampffmeyer et al. (2019) 引入密集连接与分层传播。本文沿用 GCN 预测分类权重的思路，但将其应用于**动词/名词分支**而非整体动作节点，实现线性扩展。
- **组合动作识别**：Zhukov et al. (2019) 利用动词-名词组合分解任务，但属弱监督非零样本设置，且未强制特征解耦；Materzynska et al. (2020) 处理 seen 动词/名词的未见组合，但未覆盖 unseen 概念。本文聚焦**真正的零样本组合泛化**并引入物性约束。
- **GCNCL（Kato et al., 2018）**：最接近的前作，在 KG 中构建动词、名词及组合动作节点，通过特征传播学习新动作表征。其节点数随词汇量二次增长；本文通过**因子化解耦**将复杂度降至线性，并额外引入外部语料约束。
- **物性先验在动作识别中的应用**：Zhuang et al. (2017)、Lu et al. (2016) 利用语言信息作为视觉关系检测的先验。本文将其扩展到**动词-名词组合合理性评分**，并设计可端到端学习的视觉-语义映射模块。
- **零样本评估协议**：Chao et al. (2016) 提出广义零样本学习（GZL）与 AUC 指标；Misra et al. (2017)、Purushwalkam et al. (2019) 等在属性-对象分类中探索 close/open world。本文指出组合动作因存在大量无效配对，现有 open world 评估仍不够严格，故提出 **macro open world** 设置。
- **大尺度动作基准构建**：Epic-Kitchen 原为第一人称厨房交互数据集；本文依据预训练骨干可见类、长尾分布等原则构建组合切分，为社区提供一个规模约十倍于 Charades 的新基准。

## 局限性与未来方向
- **物性先验依赖语言语料**：从字幕自动抽取的动词-名词对可能存在噪声，且基于 open-world 假设（未观察到不等于不可行），未来可探索更精细的视觉-语言对齐或常识知识库。
- **Charades 结果提升有限**：因数据集较小、第三人称视角导致手部掩码不可用，解耦改用对抗式损失，物性先验亦未使用，泛化收益不明显；这提示方法在**小数据、复杂视角**场景下仍有改进空间。
- **宏观开放世界评估的计算开销**：在全量 $N_v \times N_n$ 空间预测会随词汇量平方增长，实际部署需设计高效的剪枝或检索策略。
- **未充分探索其他零样本分类器生成技术**：如生成对抗特征（Geng et al., 2020a）、解释性 GCN（Geng et al., 2020b）等，可进一步集成以提升 unseen 概念的表征质量。

## 研究启发与可借鉴点
- **解耦表征 + 外部约束的组合建模范式**：将动作分解为独立语义实体（动词/名词）并分别学习，再通过显式的物性/类型约束进行组合，这一思路可迁移至**属性-对象识别、视觉关系检测**等组合任务。
- **GCN 直接预测分类权重**：区别于传统在特征空间传播的方式，本文用 GCN 输出分类器的 weight vector，使零样本推理等价于**线性分类**，实现简单且易于集成到现有视觉骨干中。
- **基于字幕语料自动构建先验**：利用 HowTo100M 等大规模指令视频字幕抽取动词-名词共现，无需额外人工标注即可学习合理性评分，为**弱监督常识学习**提供了低成本路径。
- **评估协议的细致设计**：区分 close/open/macro open 世界设置并强调全空间评估，对**零样本/泛化学习任务**的 benchmark 设计具有参考价值，可促使后续工作更贴近真实部署场景。
- **解耦技术的场景自适应**：针对第一人称视频采用几何裁剪，针对第三人称视频采用对抗式 discriminator，提示在不同数据分布下应灵活选择解耦策略。

## 关键术语表
- **零样本组合动作识别**：在训练集未出现的动词或名词组合上，仍能正确识别视频动作的任务。
- **解耦特征（Disentangled representation）**：将动词与名词的视觉表征分别提取并相互独立，避免一方信息污染另一方。
- **知识图谱（KG）**：以节点和边表示实体及其语义关系的结构化知识库，本文用于动词和名词的独立图谱。
- **GCN（图卷积网络）**：在图结构上进行信息聚合的神经网络，此处用于从词嵌入传播生成 unseen 节点的分类权重。
- **物性先验（Affordance prior）**：基于物体形状/功能所暗示的人类合理交互动作的常识约束，本文从字幕语料中学习。
- **宏开放世界（Macro open world）**：提出的一种评估设置，要求模型在整个 $N_v \times N_n$ 标签空间（含无效组合）进行预测与评估。
- **Epic-Kitchen 基准**：本文构建的基于 Epic-Kitchen v2 的组合动作识别数据集，规模约 76605 视频、3629 组合类。
- **AUC（Area Under Curve）**：用于衡量 seen/unseen 类别之间性能权衡的曲线下面积指标，本文采用的核心评估度量。

## 可复现要素
- **数据集**：Charades（公开，http://vuchallenge.org/charades.html）与 Epic-Kitchen v2（Creative Commons Attribution-NonCommercial 4.0，公开）。
- **代码/权重**：论文未明确声明代码开源链接；部分组件使用开源模型（Faster R-CNN/Detectron2、I3D/Kinetics 预训练、GloVe/BERT 词嵌入、WordNet、VerbNet、HowTo100M 字幕）。
- **关键超参**：特征维度（Faster R-CNN ResNet-101 输出 2048 维，I3D 双流各 1024 维合并为 2048 维，Epic-Kitchen 加入手部后为 4096 维）；优化器 Adam；训练硬件约 20 GPU（单次运行需 5 GPU）；Epic-Kitchen 采样整视频均值池化，Charades 每视频 10 clip 最大值池化。具体学习率、GCN 层数、阈值等细节见附录。
