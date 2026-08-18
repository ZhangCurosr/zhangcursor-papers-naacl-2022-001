---
title: "Improving-Multi-Document-Summarization-through-Referenced-Fl"
source: https://aclanthology.org/2022.naacl-main.120.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:38:51"
---

# 论文速读：Improving-Multi-Document-Summarization-through-Referenced-Fl

## 一句话总结
本文针对多文档摘要（MDS）输入过长问题，提出基于预训练语言模型的抽取-生成分层框架 REFLECT，通过伪Oracle松弛、摘要参考生成与信用感知自批判强化学习，缓解提取器监督信号次优及训练-测试目标不一致等瓶颈，在 Multi-News、Multi-XScience 和 WikiCatSum 上均取得最优结果。

## 研究问题与动机
1. MDS 输入文档簇长度极长，直接截断会导致关键信息丢失，而传统 LSTM 提取器无法充分利用预训练语言模型的表征能力。
2. 现有方法依赖贪婪算法生成的伪 Oracle（pseudo oracle）进行监督学习，其仅基于单一摘要的词汇相似度，且不同评分指标（precision/recall）导致提取结果次优且不稳定。
3. 提取器仅凭特定摘要推导伪 Oracle 缺乏充分证据，难以隐式推断潜在的多义摘要，造成提取精度受限。
4. 提取器的训练目标（最大化与人类摘要的词汇重叠）与测试目标（为抽象器提供利于高质量重写的输入）存在不一致，纯 MLE 训练无法桥接该 gap。

## 核心贡献（创新点）
1. **层次化抽取-生成架构**：使用 RoBERTa-base 构建词级与句级双级编码器，在满足 Transformer 长度约束（M=512）的同时保留预训练知识，区别于仅靠段落排名或局部注意力的长文本基线。
2. **伪 Oracle 松弛（POR）**：提出基于 ROUGE-1 的连续损失加权机制，打破二元监督限制，与全量 MLE 相比显著降低对次优伪标签的过度依赖。
3. **摘要参考（SR）**：利用微调后的抽象器生成摘要作为提取器的辅助上下文信号，替代人工 Gold 摘要，避免训练-测试分布偏移并提升泛化性。
4. **信用感知自批判（CASC）**：将句子选取建模为单轮 CMAB 问题，仅对随机策略与贪心策略产生分歧的动作分配优势值并更新梯度，直接优化测试期 ROUGE-L 目标，优于传统序列决策式 RL。

## 方法详解
- **层次化摘要器（H-SUM）**：输入文档按长度阈值 $M$ 切分为 $K$ 个句子块，经 T-ENC（前 $l$ 层）与 S-ENC（后 $12-l$ 层）编码后，由 SS 输出选中句子索引集 $\hat{e}$，ABS（BART）对拼接后的句子重写生成摘要。
- **伪 Oracle 松弛（POR）**：将句子划分为 Oracle 集 $S$ 与非 Oracle 集 $S^c$，$S$ 的损失权重固定为 1，$S^c$ 的权重设为 $w_i = (1 - \text{ROUGE-1}(x_i, y))^\gamma$ 并平移归一化至最大值为 1，加权交叉熵损失如公式(4)，强调高/低相似度关键句的预测精度。
- **摘要参考（SR）**：训练/推理时先用当前抽象器对选中句子生成参考 $r$，将其与原文块特征相加后送入 S-ENC（公式5），为提取器提供可泛化的语义证据。
- **信用感知自批判（CASC）**：采用自批判策略梯度，主策略 $\pi_\theta$ 按 logits 采样集合 $S$，基线策略 $\tilde{\pi}_\theta$ 为概率阈值 0.5 的贪心集合。奖励 $R(S)=\text{ROUGE-L}(\text{ABS}(S), y)$，优势 $a=R(S)-R(\tilde{S})$，仅对两策略选择存在分歧的句子（公式11指示函数）计算梯度，直接对齐测试目标。

## 实验与结果
- **数据集与指标**：Multi-News（新闻）、Multi-XScience（科学论文）、WikiCatSum（Wikipedia 三域），评估 ROUGE-1/2/L、BERTScore、FactCC 事实一致性。
- **Main Results**：在 Multi-News 上，REFLECT (CASC) 取得 R-1=49.27、R-2=19.96、R-L=24.76，平均 ROUGE 达 31.33，全面超越 PEGASUS、BART-Long-Graph、GraphSum 等强基线；在 Multi-XScience 与 WikiCatSum 各子域同样取得最优，ROUGE-2 与事实一致性提升尤为显著。
- **消融结论**：POR 提升提取召回率；SR 改善抽象性能并验证了生成参考优于 Gold 参考的泛化性；CASC 相比普通 Self-Critic 在最终摘要指标上大幅领先，证实桥接 train-test 目标的关键作用。

## 相关工作脉络
1. **传统统计抽取方法**（Goldstein et al., 2000; Erkan & Radev, 2004）：依赖 MMR/TextRank 等图或启发式规则，缺乏连贯性；本文转向神经抽取-生成范式。
2. **早期抽取-生成框架**（Gehrmann et al., 2018; Chen & Bansal, 2018）：使用 LSTM 提取器+复制机制，受限于长程依赖；本文替换为 RoBERTa 层次化编码器。
3. **长文本 Transformer 基线**（Liu & Lapata, 2019a; Pasunuru et al., 2021; Beltagy et al., 2020）：通过段落排序、局部注意力或图结构压缩输入；本文通过显式两阶段选择+SR参考实现更细粒度的信息筛选。
4. **预训练 Seq2Seq 端到端模型**（PEGASUS, BART）：直接处理截断输入；本文指出截断损失信息，采用分离架构保留完整文档关键句。
5. **强化学习文档选择**（Bražinskas et al., 2021; Chen & Bansal, 2018）：将选取建模为序列决策或使用预计算词汇特征；本文改为单轮 CMAB+信用感知分配，更适配 Transformer 且降低长序列方差。

## 局限性与未来方向
- 摘要参考质量直接影响提取性能，当前仅用单轮抽象器生成，未探索多轮迭代 refine 或引入外部知识辅助参考。
- CASC 需在训练期多次调用抽象器计算奖励，计算开销高于纯 MLE；虽训练用 BART-base、推理用 BART-large，但整体效率仍有优化空间。
- 评估以 ROUGE、BERTScore 和 FactCC 为主，未涵盖人工评测、结构连贯性或下游任务实证。
- 作者指出未来可探索提取器与抽象器的双向交互（如迭代参考、反向梯度传递）及更高效参考生成机制。

## 研究启发与可借鉴点
1. **伪监督的软化处理**：在缺乏精确抽取标注的任务中，可设计与目标输出的相似度连续的损失权重（类 POR），替代硬标签缓解次优伪监督的负面迁移。
2. **自生成参考缓解分布偏移**：用自身或同伴模型的生成结果作为辅助训练信号（类 SR），比直接使用 Gold 标签更能平滑 train-test gap，适用于标注成本高或噪声大的生成任务。
3. **信用感知策略梯度**：在组合选择类 RL 中
