---
title: "DISAPERE-A-Dataset-for-Discourse-Structure-in-Peer-Review-Di"
source: https://aclanthology.org/2022.naacl-main.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:54:54"
field: "学术文本分析"
keywords: ["peer review", "discourse structure", "argument mining", "academic publishing", "text annotation", "review-rebuttal interaction"]
innovations: ["首个 sentence-level peer review discourse 数据集", "14类 rebuttal action 标签体系与多对多 context 标注", "agreeability 指标用于识别争议性评审"]
benchmarks: ["AMPERE", "AMSR", "ASAP-Review", "APE"]
---

# 论文速读：DISAPERE-A-Dataset-for-Discourse-Structure-in-Peer-Review-Di

## 一句话总结
论文提出了 DISAPERE，一个包含 506 个 review-rebuttal 对（共 2 万句子）的同行评审对话 discourse 结构标注数据集；首次系统地标注了作者 rebuttal 与其回应的 review 上下文之间的 discourse 关系，为理解审稿交互提供了高质量数据基础。

## 研究问题与动机
1. **同行评审文本分析缺乏系统性 discourse 数据集**：现有工作（如 AMPERE、AMSR）主要关注论证级标注，忽略了 rebuttal 与 review 之间的 discourse 关系，以及非论证性句子的标注。
2. **现有标注粒度不一致**：已有数据集在 review 论证标注上粒度差异大、标签体系不兼容，无法直接用于跨数据集比较或统一建模。
3. **决策者需要基于内容而非仅分数的洞察**：Area chair 目前依赖评分方差作为启发式规则，但缺少从 discourse 层面识别"争议性 review"的方法。
4. **crowdsourcing 不适用于该任务**：正确理解 discourse 结构需要领域知识，无法用众包完成标注。

## 核心贡献（创新点）
1. **首个 sentence-level 的 peer review discourse 数据集**：与 prior work 在 argument span 级别标注不同，DISAPERE 对每一句都进行标注，避免 argument detection pipeline 误差传播。
2. **统一的层级化 review discourse 标签体系**：综合 AMPERE、AMSR、ASAP-Review 等标签集，扩展出细粒度的 request 子类型（解释请求、实验请求、修改请求等），填补了 prior taxonomies 不兼容的空白。
3. **首创 rebuttal 句子的 discourse action 与 context 标注**：14 类 REBUTTAL-ACTION 标签刻画作者意图，并建立多对多的 review-rebuttal 上下文映射关系。
4. **提出 agreeability 指标用于识别争议性评审**：用 CONCUR/ dispute 比例衡量 rebuttal 可接受度，发现低 agreeability 但评分方差小的案例可补充传统启发式规则的盲区。
5. **开源完整标注工具与规范**：提供自定义标注软件和详细 guidelines，支持未来跨领域扩展标注。

## 方法详解
**标注体系设计**：
- Review 句子标注四个属性：REVIEW-ACTION（6 粗粒度类别：Evaluative/Structuring/Request/Fact/Social/Other）、FINE-REVIEW-ACTION（10 子类）、ASPECT（9 维度：Substance/Clarity/Soundness 等）、POLARITY（Positive/Negative）。
- Rebuttal 句子标注两个属性：REBUTTAL-ACTION（14 类别，分为 concur/dispute/non-arg 三 stance）和 CONTEXT（指向 review 句子子集，支持 global context 和 empty set）。

**数据来源与切分**：
- 从 OpenReview 获取 ICLR 2019/2020 的英文 review-rebuttal 对，按 manuscript 划分 train/dev/test（3:1:2），确保同一稿件全文不跨 split。
- 使用 spaCy 进行分句，Stanza 进行 tokenization。

**标注流程**：
- 10 名计算机研究生经过培训和校准完成标注，总计 >850 人时。
- 测试集全部双标注（65 对），计算 Cohen's κ 评估 IAA。
- 提供两阶段标注工具：第一阶段逐句标注 review；第二阶段按顺序处理 rebuttal 句子，选择对应 review 上下文并标注 REBUTTAL-ACTION。

**基线模型**：
- Sentence classification：BERT-base + feedforward on [CLS] token，报告 macro F1。
- Context alignment：BM25 检索基线 + Siamese-BERT (S-BERT) 双塔模型，用 cosine similarity 计算相似度，NO_MATCH 类别处理无上下文情况；评估指标 MRR 和 MAP。

## 实验与结果
**数据集规模**：506 对（train 251 / dev 88 / test 167），20k+ 句子，~39 万 tokens。

**IAA 结果（Table 5）**：
- REVIEW-ACTION: κ=0.605，FINE-REVIEW-ACTION: κ=0.583，POLARITY: κ=0.561
- REBUTTAL-STANCE: κ=0.513，REBUTTAL-ACTION: κ=0.479
- 整体处于 moderate-substantial 区间，与 discourse 标注任务常见水平一致。

**分类任务（Table 7）**：
- REVIEW-ACTION: Macro F1=60.42%（7 labels）
- POLARITY: Macro F1=70.88%（3 labels，最高）
- REBUTTAL-ACTION: Macro F1=31.23%（17 labels，最低）
- ASPECT F1=38.28%，虽 κ 在中等范围但任务本身难度高。

**对齐任务（Table 9）**：
- BM25: MRR=0.5980, MAP=0.5174（surprisingly 优于神经模型）
- S-BERT: MRR=0.5022, MAP=0.4409
- 说明 lexical overlap 是重要信号，但远未饱和；当前模型未利用 rebuttal 上下文信息是主要局限。

**语料分析发现**：
- 84.81% rebuttal 句子有 review 上下文链接
- Spearman ρ 中位数 0.794，说明 order-based alignment 不充分
- 低 agreeability 且非高分差 varience 的稿件占 18%，提示 agreeability 可补充现有 heuristics

## 相关工作脉络
1. **AMPERE (Hua et al., 2019)**：首个 peer review 论证级标注数据集，标注 EVALUATION/REQUEST/FACT 等标签；DISAPERE 扩展至 sentence level 并加入 rebuttal discourse 标注。
2. **AMSR (Fromm et al., 2020)**：论证 stance 导向标注，侧重 acceptance/rejection 立场；DISAPERE 在此基础上细化了 request 子类型和非论证句子。
3. **ASAP-Review (Yuan et al., 2021)**：aspect-based polarity 标注；DISAPERE 吸收其 aspect 标签体系但应用于句子级而非 argument span。
4. **APE (Cheng et al., 2020)**：首个 review-rebuttal discourse relation 标注，但仅覆盖部分句子对；DISAPERE 提供完整的全句对齐标注。
5. **Gao et al. (2019)**：实证研究发现 rebuttal 很少改变评分；DISAPERE 的形式化标签体系将其观察转化为可计算特征。

## 局限性与未来方向
1. **领域单一**：数据仅来自 ICLR（机器学习领域），跨领域泛化能力待验证。
2. **仅限首轮 rebuttal**：未覆盖多轮讨论（multi-turn），Extended discussion 留待未来。
3. **对齐任务性能不足**：当前 BM25/S-BERT 基线 far from satisfactory，Lexical 信号有效但语义理解不足。
4. **标注歧义性**：如 rhetorical question 边界模糊、作者将评价句当作请求处理等，导致 IAA 下降。
5. **未来方向**：引入上下文感知的对齐模型、扩展到 multi-turn、跨会议/领域迁移、结合 decision outcome 预测。

## 研究启发与可借鉴点
1. **Sentence-level 标注避免 pipeline 误差传播**：与 prior work 依赖 argument detection 不同，DISAPERE 直接在句级标注，更稳健且覆盖更全面。
2. **多层级标签体系设计**：粗粒度 REVIEW-ACTION → 细粒度 FINE-REVIEW-ACTION 的分层设计兼顾建模可行性和分析深度。
3. **Agreeability 指标的可迁移性**：将 discourse 统计量转化为决策辅助指标的思路可直接迁移至其他学术评审或辩论场景。
4. **双阶段标注工具设计**：将 review 标注与 rebuttal-context 选择分离，降低 annotator 认知负荷，可复用至其他对话标注任务。
5. **公开 guidelines 与软件**：详细 annotation guidelines + 定制工具的开源自描述提升了可复现性和社区扩展可能性。

## 关键术语表
**DISAPERE**：DIscourse Structure in Academic PEer REview 的缩写，本文提出的同行评审 discourse 结构标注数据集。
**REVIEW-ACTION**：标注 review 句子功能的六大粗粒度类别（Evaluative/Structuring/Request/Fact/Social/Other）。
**REBUTTAL-ACTION**：14 类标注 rebuttal 句子意图的标签，分为 concur/dispute/non-arg 三种 stance。
**CONTEXT 标注**：建立 rebuttal 句子与 review 句子的多对多映射关系，支持 global context 和 empty set。
**Agreeability**：定义为 CONCUR 句子数除以 argumentative 句子总数，用于衡量 rebuttal 对 review 评论的接受程度。
**Macro F1**：数据集评估采用的宏平均 F1 分数，对各 label 类别平等对待。
**Siamese-BERT (S-BERT)**：用于句子对齐的双塔 BERT 模型，独立编码 query/key 后计算 cosine similarity。
**Cohen's κ**：用于评估标注者间一致性（IAA）的 chance-corrected  agreement 系数。

## 可复现要素
- **数据集**：DISAPERE 已随论文发布，包含 train/dev/test 划分
- **代码**：自定义 annotation tool 已开源（论文声明）
- **权重**：BERT-base 和 S-BERT 使用预训练权重（论文未提及额外权重发布）
- **关键超参**：论文未详细列出分类器超参；alignment 任务使用预训练 S-BERT 模型
- **数据源**：ICLR 2019/2020 OpenReview 公开数据
- **标注人数**：10 名 CS 研究生，>850 人时标注量
