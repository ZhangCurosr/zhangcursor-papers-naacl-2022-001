---
title: "Explaining-Why-How-Instructions-and-User-Interfaces-Impact-A"
source: https://aclanthology.org/2022.naacl-main.38.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:36:09"
field: "可解释自然语言处理"
keywords: ["rationale annotation", "explainable NLP", "human-computer interaction", "data labeling", "sentiment analysis", "user study"]
innovations: ["首次系统研究指令和界面交互对rationale选择的影响", "发现界面affordance（拖拽功能）对rationale特征的影响远超指令措辞", "提出并验证双边rationale概念，揭示语言中的对立信号"]
---

# 论文速读：Explaining-Why-How-Instructions-and-User-Interfaces-Impact-A

## 一句话总结
本文通过332名参与者的对照实验，系统研究了不同**指令**和**用户界面设计**如何影响人类标注器在情感分析任务中选择rationales（支持标签的输入token子集），发现界面交互方式对rationale质量的影响远大于指令措辞，且标注一致性显著低于分类标签一致性。

## 研究问题与动机
- **现有工作缺乏对人机标注过程的系统研究**：尽管NLP社区越来越多地收集人类提供的rationales以辅助模型训练，但指令措辞（如"most important words" vs "ALL words"）是否影响结果尚无定论。
- **界面交互设计的缺失**：现有工作通常不报告或控制annotation UI的交互方式（如拖拽选词 vs 逐个点击），这些设计可能显著影响rationale的完整性和一致性。
- **Rationale质量参差不齐**：Carton et al. (2020) 等研究发现现有rationale数据集的质量存在较大方差，学习此类数据反而可能损害模型性能，需从源头理解annotation过程的变异性。
- **效率与收益的权衡**：收集rationale比仅标注标签耗时更长（约2.5倍），需要量化其额外价值以判断是否值得投入。

## 核心贡献（创新点）
1. **首次系统性研究rationale收集过程中的HCI设计因素**：相较于以往工作聚焦于构建数据集或模型评估，本文从人机交互视角探究指令、界面交互对rationale特征的因果影响。
2. **揭示界面affordance（拖拽选词能力）比指令措辞更能影响rationale特征**：证明用户能否一次性拖拽选中多词直接决定rationale是包含上下文短语还是孤立词，而不同指令集对选择比例和类型影响微弱。
3. **提出并验证双边rationale（two-sided rationales）概念**：不仅为所选标签标注支持证据，也为未选标签标注相反证据，发现显著比例的词同时隐含对立情感，挑战了传统one-sided假设。
4. **建立annotation效率基准**：量化labeling alone vs one-sided vs two-sided rationales的时间成本比（1 : 2.5 : 3.8），为后续工作提供收益-成本权衡参考。

## 方法详解
- **实验设计**：三阶段between-subjects在线用户研究，共332名参与者（Phase 1: 119人，Phase 2: 125人，Phase 3: 88人）。
- **任务**：标注10条IMDB电影评论的情感（positive/negative），并为所选标签选择支持性tokens作为rationale。
- **Phase 1（指令变量）**：六组指令条件——Baseline（中性描述）、Generalize（强调泛化性）、Zaidan（引用Zaidan & Eisner, 2008）、Sen（引用Sen et al., 2020）、百分比提示（%, Not Shown / %, Shown）。
- **Phase 2（双边rationale）**：在Phase 1最优三条指令基础上，增加为相反标签选择rationale的任务，对比one-sided vs two-sided。
- **Phase 3（界面与任务变量）**：
  - Zaidan基线条件（支持拖拽，选"words and phrases"）
  - Zaidan, No Dragging（强制逐词点击，仍选"words and phrases"）
  - Zaidan, No Dragging, Words Only（强制逐词点击，仅选"words"）
  - Labels Only（仅标注情感，不提供rationale，作为时间基准）
- **数据集**：从50,000条IMDB评论中精选10条——2条高正、2条高负、2条长评论、2条短评论、2条模糊评论（以覆盖多样性）。
- **评估指标**：
  - RQ1：选中token比例（fraction of words selected）
  - RQ2：标注者间一致性（Krippendorff's α）
  - RQ3：选定词的语义特征（词性分布、短语长度、上下文完整性）
  - RQ4：任务耗时

## 实验与结果
- **RQ1：选中词比例**
  - 中位数参与者选择约**12%**的token作为rationale（有拖拽功能时），不同指令间无显著差异。
  - **无拖拽功能**时降至约**5%**；仅选"words"时进一步降至**4.5%**。
  - 双边任务中，opposite rationales仅占**~4%**，regular rationales占**~11%**。
  
- **RQ2：一致性**
  - 情感标签高度一致：6/10评论Krippendorff's α = **1.0**，其余至少**0.8**。
  - Rationale一致性低：one-sided条件下α约为**0.28–0.35**；two-sided条件下regular rationales α = **0.27–0.29**，opposite rationales α = **0.19–0.26**。
  - "Words Only"条件一致性最高（α = **0.38**），但仍属低水平。

- **RQ3：选定词特征**
  - 少数词被几乎所有参与者选中，长尾词仅在特定条件下被选中。
  - 界面影响远大于指令：相关系数分析显示，Phase 1各条件间Pearson相关系数达**0.56–0.65**（3人样本）至**0.91–0.95**（18人样本）；而含拖拽 vs 不含拖拽条件间仅**0.51–0.53**（3人）。
  - 短语长度：有拖拽时常见完整短语（如"worst movie I have seen"），无拖拽时仅选"worst movie"或"worst"。
  - 词性分布：Zaidan条件中名词占26.9%、动词17.8%、形容词17.1%；No Dragging, Words Only中形容词升至**37.4%**，成为最常用词性。
  - Bifacial words现象：在two-sided条件下，**72–78%**参与者至少一次为同一词标注regular和opposite rationale（如"not very interesting" vs "interesting"）。

- **RQ4：耗时**
  - One-sided rationales耗时约为Labels Only的**2.5倍**。
  - Two-sided rationales耗时约为Labels Only的**3.8倍**。
  - No Dragging条件耗时显著少于有拖拽条件，但选词比例也更低。

## 相关工作脉络
1. **Explanation Datasets**：McDonnell et al. (2016, 2017)、DeYoung et al. (2019)、Sen et al. (2020) 等构建了rationale数据集，但未系统研究annotation过程的人机交互设计。
2. **Attention-based Annotation Assistants**：Choi et al. (2019) 使用注意力模型辅助高亮候选词，但未研究界面设计对selection行为的影响。
3. **Rationale Quality Evaluation**：Carton et al. (2020) 发现现有rationale数据集存在充足性/全面性问题；Plumb et al. (2020)、Ross et al. (2017) 发现学习rationale可能损害模型性能，提示需从源头改善annotation过程。
4. **Instruction Sensitivity in Crowdsourcing**：Chang et al. (2017) 发现模糊指令导致crowdsourcing标注分歧，启发了本文对rationale指令设计的考察。
5. **Negativity Bias in Sentiment**：Aithal & Tan (2021) 发现负面评论中仍存在正面词汇，与本文双边rationale发现相呼应。
6. **Human Explanation Diversity**：Tan (2021) 指出多样指令可收集不同类型的rationale，但本文从UI交互维度补充了这一讨论。

## 局限性与未来方向
- **任务单一**：仅研究情感分析，未验证在其他NLP任务（如NER、文本蕴含）中的泛化性。
- **评论数量少**：仅10条评论可能不足以覆盖文本类型的全部方差。
- **工作流程局限**：仅考察"先标标签后标rationale"的顺序，未探索同步标注或其他workflow。
- **未来方向**：开发能建模rationale分布而非二值ground truth的算法；设计降低two-sided annotation时间的UX策略；开源annotation UI以促进可复现性。

## 研究启发与可借鉴点
1. **界面设计优先级高于指令优化**：若团队需收集高一致性rationale，应优先考虑限制拖拽功能或限定"words only"而非反复调整指令措辞。
2. **双边rationale的价值被低估**：对于情感分析等任务，忽略opposite信号可能导致模型过度拟合spurious correlations，可探索在训练数据中包含双向标注。
3. **Phrase-level rationale优于word-level**：拖拽功能使参与者更倾向选择含上下文的完整短语，这对依赖token bag-of-words的下游方法不利，需推动sequence-aware模型利用rationale。
4. **报告UI设计细节的必要性**：不同研究的rationale差异可能源于不可比界面，建议社区强制要求公开annotation tool代码以增强可比性。
5. **成本-收益框架**：2.5×时间成本意味着同等资源下仅能获得40%的数据量，团队需在rationale质量增益与数据规模损失间做权衡决策。

## 关键术语表
**Rationale**：标注者为支持其选择标签而从输入文本中选出的tokens子集（通常为词或短语），用作解释模型预测的依据。
**Affordance**：用户界面对用户操作能力的暗示或支持，本文中指"能否通过拖拽一次性选中多个词"的交互特性。
**One-sided Rationale**：仅标注支持所选标签的文本证据；传统工作普遍采用此形式。
**Two-sided Rationale**：同时标注支持所选标签和不选标签的文本证据，捕捉文本中的矛盾/复杂信号。
**Krippendorff's α**：衡量多标注者间一致性的统计量，取值范围[-1, 1]，越高表示一致性越强，0.3左右表示低一致性。
**Bifacial Words**：在同一文本中同时被选为regular rationale和opposite rationale的词，反映语言的语境依赖性。
**Bag-of-words Approach**：将文本视为无序词袋的分析方法，无法利用rationale的句法/顺序信息。
**Inter-annotator Consistency**：不同标注者对同一文本选择rationale的重合程度，本文发现远低于标签一致性。

## 可复现要素
- **数据集**：IMDB电影评论（Maas et al., 2011），精选10条评论用于实验；原始数据集公开可用。
- **代码/工具**：React框架开发的rationale-annotation UI已开源（GitHub: https://github.com/UChicagoSUPERgroup/rationales-naacl22）。
- **数据补充**：在线附录含所有10条评论的heatmaps及92%参与者的完整数据（rationale、点击轨迹、计时、问卷）。
- **关键超参**：未明确提及模型超参（本作为用户研究非模型训练）；参与者要求：美国/英国居住、Prolific通过率≥95%、完成次数≥100。
