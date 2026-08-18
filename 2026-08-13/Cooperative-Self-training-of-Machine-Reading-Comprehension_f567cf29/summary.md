---
title: "Cooperative-Self-training-of-Machine-Reading-Comprehension"
source: https://aclanthology.org/2022.naacl-main.18.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:54:33"
field: "机器阅读理解与合成数据生成"
keywords: ["cooperative self-training", "machine reading comprehension", "synthetic question generation", "expectation-maximization", "maximum mutual information", "extractive QA", "domain adaptation"]
innovations: ["RGX三模块合作式自训练框架", "基于QAE损失的EM自适应难题筛选", "引入AER的高效MMI推理策略"]
benchmarks: ["SQuAD v1.1", "MRQA (BioASQ, TextbookQA, RACE, RelationExtraction, DuoRC, DROP)", "Natural Questions"]
---

# 论文速读：Cooperative-Self-training-of-Machine-Reading-Comprehension

## 一句话总结
本文提出 **RGX** 合作式自训练框架，通过答案实体识别器（AER）、问题生成器（QG）和答案提取器（QAE）三模块协作，自动从无标注语料生成高质量、非平凡的问题-答案对，从而在零/少标注的域外机器阅读理解任务中显著提升模型性能。

## 研究问题与动机
1. 预训练语言模型（如 BERT/ELECTRA）虽大幅提升了 NLU 性能，但抽取式 QA 仍依赖大量**人工标注的问答对**。
2. 已有自训练方法（如直接生成问题）存在**种子域与目标域之间的领域偏移**，且合成问题的难度分布不均，易陷入“过于简单”或“噪声过大”的两极。
3. 单纯奖励正确回答会使 QG 收敛到**平凡局部最优**，仅依赖低困惑度会引入大量语法/事实错误问题。
4. 如何在**无目标域标注**的前提下，自动筛选出“困难但可回答”的问答对并同步提升 QG 与 QAE，仍是开放问题。

## 核心贡献（创新点）
1. **合作式自训练框架 RGX**：AER、QG、QAE 三个模块在 Masked 实体预测任务上协同训练，无需目标域标注即可生成合成 QA。
   - 区别于此前“教师-学生”单向蒸馏，RGX 让生成器与提取器**互相提供信号**，形成闭环。
2. **基于 EM 的合成 QA 筛选**：用 QAE loss 将问题分为低/中/高难度三桶，仅用低+中难度进行微调，避免噪声与平凡解。
   - 阈值通过 EM 自适应确定，而非固定超参，适应不同域的损失分布。
3. **基于 AER 的最大互信息（MMI）推理**：在测试时用 $\alpha \log P_{QG}(q|p,a) + \beta \log P_{QAE}(a|p,q)$ 重排候选答案。
   - 首次将 MMI 用于 QA 推理，并利用 AER 提供高效候选集，解决了此前 $P(q|p,a)$ 计算开销过大的瓶颈。
4. **广泛的零/半标注域外实验**：在 SQuAD 与 MRQA 六个子集（BioASQ、TextbookQA、RACE、RelationExtraction、DuoRC、DROP）上验证，多数域取得 SOTA。

## 方法详解
- **AER（Answer Entity Recognition）**：在种子语料（NQ/SQuAD）上训练，支持 BIO 序列标注（AER-Tag）或抽取式 span 预测（AER-LM）。先用 sentence-level 输入缓解长文档稀疏性，再从候选 span 中选出 top-K 实体。
- **QG（Question Generation）**：以 BART 为 backbone，输入为“掩码化 passage + 实体”，最大化 $P_{QG}(q|[p^*,e])$。
- **QAE（Question-Answering Extractor）**：以 ELECTRA 为 backbone，输入为 $[q,p]$，预测答案 span 的起始/结束位置 $I_{st}, I_{ed}$，用交叉熵损失分别训练。
- **EM 筛选**：对每个候选 QA 对计算 QAE loss，自适应划分三桶（低/中/高），仅用低+中桶训练，迭代更新 QG 与 QAE。
- **MMI 推理（测试时）**：$a^* = \arg\max_a [\alpha \log P_{QG}(q|p,a) + \beta \log P_{QAE}(a|p,q)]$，其中 $\beta=1$，$\alpha$ 由输入问题与生成问题的相似度自适应计算，避免 QG 对所有问题都赋低困惑度造成的偏差。
- **合作自训练流程**：AER 识别实体 → 掩码生成问题 → QAE 提取答案 → EM 筛选 → 更新 QG/QAE → 循环。

## 实验与结果
- **种子语料**：Natural Questions（106,926 题）+ SQuAD v1.1（107,785 对）。
- **目标域**：MRQA 的 BioASQ、TextbookQA、RACE、RelationExtraction、DuoRC、DROP，每域采样 3000 段生成合成 QA（总数约 70 万）。
- **主要结果**（MRQA 零标注）：
  - 以 NQ 为源时，RGX 平均 F1 = **61.9**，相比 ELECTRA-large（54.2）提升 **+7.7**；相比 QAGen2S（56.0）提升 **+5.9**。
  - 最大单项增益：NQ→TextbookQA，EM +18.0、F1 +19.4。
  - 在 SQuAD 半标注设置下，RGX 达到 EM 83.1 / F1 90.7，逼近全监督 ELECTRA-large（89.7/94.9）。
- **消融**：去掉 EM 或合作自训练（CST）后性能显著下降；MMI 进一步提升 EM/F1；AER 比 CONLL NER 效果好得多。
- **DROP 下降原因**：合成器倾向生成“安全”单步问题，难以覆盖多步推理。

## 相关工作脉络
1. **QAGen2S**（Shakeri et al., 2020）：端到端合成 QA，无 AER/MMI/EM，仅用 target 域无标注文本。
2. **SynQA**（Bartolo et al., 2021）：借助 AdversarialQA 复杂标注与多模型投票，需人工标注成本；RGX 完全零标注。
3. **Sachan & Xing（2018）**：早期自训练 ask+answer，未做难度筛选与互信息推理。
4. **Lewis et al.（2019a）** BART、**Clark et al.（2020）** ELECTRA：本文的 QG/QAE backbone，属预训练层面对比基线。
5. **Li & Jurafsky（2016）**、**Tang et al.（2017）**：MMI 用于翻译/联合生成-回答，但未在 QA 测试时推理中落地。
6. **Puri et al.（2020）**、**Bartolo et al.（2020）**：合成数据增强 QA 鲁棒性，本文在此基础上引入“困难但可回答”的自动筛选。

## 局限性与未来方向
- **多步推理不足**：DROP 上提升有限，合成器偏好安全单步问题。
- **迭代收益递减**：训练超过 1 轮后性能不再提升，自训练存在优化困难。
- **实体重叠问题**：抽取式 AER 会产生重叠 span，需后处理去重，可能牺牲多样性。
- **单轮问答假设**：当前框架未建模多轮对话与上下文依赖。
- **未来方向**：引入多步推理增强、更深层的合作机制（如对抗/博弈）、利用生成困惑度+事实一致性联合评分。

## 研究启发与可借鉴点
1. **EM 难度分桶策略**可迁移到其他自训练/半监督场景，避免噪声累积与平凡解。
2. **AER + QG + QAE 三模块闭环**的设计思想可用于其他生成-判别联合任务（如摘要、解释生成）。
3. **MMI 推理在 QA 中的落地**表明：只要提供高效候选集（如 AER），互信息重排即可实用化。
4. **零标注域外适配**思路对低资源领域（医学、法律）的 QA 落地有直接参考价值。
5. **自适应阈值**替代固定超参，可推广至其他基于 loss 分布的样本筛选。

## 关键术语表
- **RGX**：Cooperative Self-training 框架，包含 Answer entity Recognizer、Question Generator、Answer eXtractor。
- **AER（Answer Entity Recognition）**：答案实体识别，从 passage 中定位潜在答案 span。
- **QG（Question Generation）**：问题生成，基于掩码实体生成可回答的自然语言问题。
- **QAE（Question-Answering Extractor）**：问答提取器，从问题+原文中抽取答案 span。
- **EM（Expectation-Maximization）**：用于自适应划分合成 QA 难度桶的迭代筛选算法。
- **MMI（Maximum Mutual Information）**：测试时结合 QG 与 QAE 概率重排候选答案的推理策略。
- **SQuAD / MRQA**：斯坦福问答数据集与机器阅读理解共享任务（含 6 个子域）。
- **Cooperative Self-training**：生成器与提取器相互提供训练信号的合作式自训练范式。

## 可复现要素
- **数据集**：SQuAD v1.1、Natural Questions、MRQA（BioASQ、TextbookQA、RACE、RelationExtraction、DuoRC、DROP）均公开。
- **代码**：论文声明接收后开源，附录 B/C/D 提供详细实现与示例。
- **模型**：BERT（AER）、BART-large（406M，QG）、ELECTRA-large（335M，QAE）。
- **超参**：学习率从 $\{3e{-5}, 4e{-5}, 5e{-5}\}$ 选优；$eps=1e{-6}$，warmup steps $s_w=2000$，无 weight decay。
- **硬件**：2×Tesla V100，训练约 4 小时。
