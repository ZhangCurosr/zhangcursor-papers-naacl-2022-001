---
title: "Bridging-the-Gap-between-Language-Models-and-Cross-Lingual-S"
source: https://aclanthology.org/2022.naacl-main.139.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:52:39"
field: "跨语言自然语言理解"
keywords: ["跨语言序列标注", "预训练-微调对齐", "对比学习", "少样本学习", "Span Extraction"]
innovations: ["CLISM：通过跨语言平行语料构建掩码span问答对，桥接预训练与微调目标差距", "CACR：三重视角对比一致性正则化，学习噪声不变的跨语言对齐表征", "仅需1M平行数据（4语言）即超越130M数据（94语言）的SOTA模型"]
benchmarks: ["MLQA", "XQUAD", "CoNLL", "WikiAnn"]
---

# 论文速读：Bridging-the-Gap-between-Language-Models-and-Cross-Lingual-S

## 一句话总结
本文针对跨语言序列标注（xSL）任务中预训练与微调目标不一致的问题，提出了一种新的预训练任务CLISM和对比一致性正则化策略CACR，仅使用100万平行数据（4种语言）即在多个基准上超越了使用1.3亿数据的SOTA模型，并在few-shot设定下取得显著突破。

## 研究问题与动机
- **预训练-微调目标鸿沟**：现有xPLMs在预训练阶段使用MLM等局部理解任务，而xSL微调阶段需要全局上下文推理（如span extraction），两者目标不一致导致性能损失。
- **跨语言对齐不足**：平行语料表示之间存在错位，模型容易学到与噪声（如[QUE]、[MASK]标记）共变的表征，而非真正的跨语言语义对齐。
- **低资源场景瓶颈**：翻译增强方法受限于翻译质量，且现有工作未系统探索预训练目标与xSL任务的桥接。
- **few-shot场景缺乏有效范式**：仅有几百个训练样本时，现有xPLMs性能大幅下降，亟需更高效的预训练策略。

## 核心贡献（创新点）
- **提出CLISM预训练任务**：通过跨语言平行语料构建"掩码源语言-span + 目标语言未掩码-span"的问答对，使预训练目标与xSL的span extraction目标一致。
- **设计CACR对比正则化策略**：利用对比学习强化源语言原始句、掩码句与目标语言句三者表征的一致性，迫使模型学习噪声不变的跨语言对齐表征。
- **小数据高效预训练**：仅用100万平行句子（4种语言）训练的模型，在xMRC和xNER基准上超越使用1.3亿数据、94种语言的Info-XLM和XLM-Align。
- **few-shot场景显著突破**：在MLQA数据集仅512个训练样本时达到50.8%/35.9% EM/F1，比基线提升约24%/20%。
- **验证[QUE] token的微调价值**：消融表明微调时保留[QUE] token可显著提升小样本性能，因该token已编码问题选择信息。

## 方法详解
- **CLISM任务构建**：从平行语料中选取源语言实体span（使用Spacy NER工具），通过GIZA++对齐工具找到目标语言对应span；将源语言span替换为特殊标记[QUE]，目标语言对齐span作为ground-truth答案，拼接为[CLS] + masked源句 + [SEP] + 目标句 + [SEP]的输入序列。
- **CLISM损失函数**：对每个[QUE]标记，计算动态start/end向量 $\mathbf{s}_q = \mathbf{W}_s \mathbf{x}_q$、$\mathbf{e}_q = \mathbf{W}_e \mathbf{x}_q$，通过softmax得到token级起止概率分布，优化交叉熵损失 $\mathcal{L}_{clism} = -\log \mathcal{P}(s=a_i^s|\mathbf{X}) - \log \mathcal{P}(e=a_i^e|\mathbf{X})$。
- **CACR三重视角对比学习**：对同一平行句子分别编码原始源句 $\mathbf{s}^s$、掩码源句 $\hat{\mathbf{s}}^s$、目标句 $\mathbf{s}^t$，经mean-pooling聚合为全局表征 $\mathbf{r}^s, \hat{\mathbf{r}}^s, \mathbf{r}^t$；将三者互为正样本对，优化多对对比损失 $\mathcal{L}_{cacr} = \mathcal{L}(\mathbf{r}^t, \hat{\mathbf{r}}^s) + \mathcal{L}(\mathbf{r}^s, \hat{\mathbf{r}}^s) + \mathcal{L}(\mathbf{r}^t, \mathbf{r}^s)$，温度参数 $\tau=20$。
- **多任务预训练目标**：总损失为 $\mathcal{L} = \mathcal{L}_{clism} + \mathcal{L}_{cacr} + \mathcal{L}_{mlm}$，保留标准MLM以维持语言建模能力。
- **微调策略适配**：在xMRC微调时将[QUE] token追加至问题之后，利用其表征进行span选择，而非标准[CLS]，显著提升小样本性能。

## 实验与结果
- **数据集**：xMRC使用MLQA和XQUAD（英文训练，测试阿拉伯语、德语、西班牙语、印地语、越南语）；xNER使用CoNLL和WikiAnn（西班牙语、荷兰语、英语、德语、印地语、阿拉伯语）。
- **基线模型**：M-BERT、XLM、XLM-R、Info-XLM（130M数据/94语言）、XLM-Align（130M数据/94语言）、CalibreNet、AA-CL。
- **xMRC结果**：基于XLM-R的模型在MLQA上达67.34%/49.11%（vs. Info-XLM的65.85%/48.23%），XQUAD上达76.73%/60.87%（vs. Info-XLM的75.79%/59.50%）。
- **xNER结果**：XLM-R基线在CoNLL上F1达80.63%（vs. Info-XLM的79.52%），WikiAnn上达73.31%（vs. XLM-Align的72.66%）。
- **Few-shot最强结果**：512样本时MLQA达50.8% EM / 35.9% F1（XLM-R基线仅15.5%/26.8%），XQUAD达53.94%/37.1%（基线仅19.1%/33.35%）。
- **预训练数据效率**：1M数据即达峰值，增至4M/8M/12M后性能下降（过拟合）。
- **消融结论**：移除CLISM导致性能骤降（MLQA从67.34/49.11降至64.56/46.80），CACR和MLM各自贡献约1%增益。

## 相关工作脉络
- **Info-XLM / XLM-Align**：使用1.3亿数据+94语言训练，引入信息论或 word alignment 损失，但预训练目标仍与xSL不匹配；本文用1%数据取得更优效果。
- **CalibreNet / AA-CL**：专为xSL设计的微调方法，分别关注边界校准和对比学习增强，但依赖标准xPLM预训练；本文从预训练阶段桥接目标差距。
- **REPT（Jiao et al., 2021）**：通过检索式预训练桥接MLM与MRC目标，但聚焦单语言qa，未探索跨语言对齐。
- **Few-shot QA预训练（Ram et al., 2021）**：同期工作同样关注pretrain-finetune gap，但仅针对单语言few-shot qa，未处理跨语言迁移。
- **XLM-R / XLM**：大规模多语言预训练模型，提供强基线但未针对xSL任务优化；本文在其基础上增量预训练即可超越。
- **跨语言数据增强（Singh et al., 2019; Yuan et al., 2020）**：通过机器翻译或Wikipedia弱标注扩充低资源数据，受限于翻译质量；本文直接利用平行语料无需翻译。

## 局限性与未来方向
- **语言覆盖有限**：仅使用英语、阿拉伯语、越南语、印地语4种语言训练，未验证对其他语言的泛化能力。
- **依赖外部工具**：实体选取依赖Spacy NER，对齐依赖GIZA++，工具误差可能传递至预训练信号。
- **任务范围局限**：目前仅验证xMRC和xNER，未扩展到关系抽取、事件提取等其他xSL任务。
- **数据规模上限**：预训练数据超过1M后性能下降，数据质量与数量的平衡机制尚不明确。
- **未来方向**：作者计划将方法扩展至其他NLP任务，可探索对话问答、核心指代消解等序列标注变体。

## 研究启发与可借鉴点
- **目标对齐预训练范式**：CLISM将预训练任务形式化为span extraction问答，为其他下游任务（如关系抽取、事件触发识别）设计定制化预训练任务提供了可复用框架。
- **对比一致性正则化思路**：CACR通过多视角（原始/掩码/跨语言）对比学习强制噪声不变表征，可迁移至单语言预训练中的鲁棒性增强。
- **特殊token的微调利用**：[QUE] token在预训练中学到问题选择语义，微调时直接复用替代[CLS]，这一策略可推广至其他需要query感知的序列标注任务。
- **小数据高效率验证**：证明精心设计的预训练任务可在极少量平行数据上充分发挥xPLMs潜力，对低资源场景具有实用价值。
- **实验设计参考**：few-shot设置下多seed采样（5次）+ 小步数优化（200 steps）的评估协议，值得在类似研究中复现。

## 关键术语表
- **xSL（Cross-lingual Sequence Labeling）**：跨语言序列标注，将高资源语言的标注知识迁移至低资源语言的序列标注任务（如xMRC、xNER）。
- **xPLM（Cross-lingual Pre-trained Language Model）**：多语言预训练语言模型，如XLM-R、Info-XLM，在大规模多语言语料上预训练。
- **CLISM（Cross-lingual Language Informative Span Masking）**：本文提出的预训练任务，通过跨语言平行语料构建掩码span问答对桥接预训练-微调目标差距。
- **CACR（ContrAstive-Consistency Regularization）**：对比一致性正则化，利用对比学习促使源/目标语言平行句及其掩码版本的表征保持一致。
- **[QUE] token**：特殊标记，用于替换源语言中被掩码的实体span，在预训练和微调中均充当"问题"角色以选择答案span。
- **span extraction**：跨度抽取，从上下文中定位并提取答案文本的起始和结束位置，是MRC和NER的核心子任务。
- **objective gap**：目标鸿沟，指预训练任务（如MLM）与下游微调任务（如span extraction）之间学习目标不一致的现象。
- **few-shot setting**：少样本设置，指微调阶段仅使用极少数训练样本（如64/128/512条）进行评估的场景。

## 可复现要素
- **预训练数据**：MT数据集（Conneau & Lample, 2019）中英/阿/越/印4语言平行句共100万条；数据集公开可下载。
- **代码/权重**：论文未提供开源代码与模型权重（仅说明基于Hugging Face Transformers初始化）。
- **关键超参**：预训练batch size=64，步数=15K，lr=1e-5，warmup=1.5K，序列最大长度=256（每句128），温度τ=20；微调batch size=32，lr=3e-5（xMRC）/5e-5（xNER），epochs=5，few-shot优化步数=200。
- **硬件**：8×V100-32G GPU，预训练耗时4-5小时。
- **基线初始化**：XLM-R base和Info-XLM base（均来自Hugging Face）。
