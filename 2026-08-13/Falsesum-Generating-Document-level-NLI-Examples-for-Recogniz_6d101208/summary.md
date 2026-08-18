---
title: "Falsesum-Generating-Document-level-NLI-Examples-for-Recogniz"
source: https://aclanthology.org/2022.naacl-main.199.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:36:31"
field: "摘要事实一致性检测"
keywords: ["factual inconsistency", "document-level NLI", "controllable text generation", "summarization hallucination", "data augmentation"]
innovations: ["基于可控生成器的文档级NLI数据自动合成管道", "通过OpenIE span掩码与控制代码区分intrinsic/extrinsic不一致错误类型", "在四个摘要一致性基准上超越FactCC取得当时SOTA（整体74.17）"]
benchmarks: ["FactCC", "Ranksum", "QAGS", "SummEval"]
---

# 论文速读：Falsesum: Generating Document-level NLI Examples for Recognizing Factual Inconsistency in Summarization

## 一句话总结
论文提出 Falsesum，一种基于可控文本生成的数据增强管道，能从仅有正样本（文档-摘要对）的训练数据中自动合成多样化、 plausible 的事实不一致摘要，用于扩充文档级 NLI 数据集，从而提升 NLI 模型在摘要事实一致性检测任务上的泛化性能，在四个 benchmark 上均超越 prior SOTA（FactCC）。

## 研究问题与动机
- **核心问题**：如何为"识别摘要中的事实不一致（Factual Inconsistency）"这一下游任务构建高质量、多样化的文档级 NLI 训练数据？
- **现有方法不足**：
  1. 直接用 off-the-shelf 句子级 NLI 模型（如 MNLI 训练的 RoBERTa）在文档级摘要任务上表现很差，存在输入粒度 mismatch（句子 premise vs 多句文档）与领域不一致。
  2. 已有合成 NLI 数据集（如 FactCC）依赖规则变换或简单的语言模型替换，错误类型单一、多样性不足，且难以复现真实摘要模型中发生的复杂 hallucination 模式。
  3. DocNLI、ANLI 等文档级 NLI 数据集虽长度匹配，但其前提-假设对并非来自摘要场景， entailment 分布与目标任务偏差较大。
  4. 直接获取标注的一致/不一致摘要对成本极高，缺乏大规模自动合成方法。

## 核心贡献（创新点）
1. **可控文本生成驱动的事实不一致数据合成框架**：用 T5 学习在给定源文档与金标准摘要的条件下，按控制代码（intrinsic/extrinsic）生成 plausible 的不一致摘要；与规则/词典替换方法相比，能产出风格、长度、词汇分布都与金标准高度匹配的"负样本"。
2. **基于 OpenIE 与 span masking 的输入格式化策略**：将文档与摘要切割为谓词/参数 spans，通过掩码 `<span_i>` 控制错误插入位置，并剥离部分 spans 迫使生成器完成"拼凑式"或"幻觉式"填充，避免模型仅靠复制即可完成任务。
3. **仅用已标注一致摘要进行监督**：整个 pipeline 不需要任何人工标注的不一致摘要；利用 CNN/DailyMail 已有 (D, S⁺) 对即可训练生成器，显著降低数据获取成本。
4. **在四个事实一致性 benchmark 上取得 SOTA**：Falsesum-augmented MNLI 在 FactCC、Ranksum、QAGS、SummEval 上的综合 balanced accuracy 达 74.17，较 prior SOTA FactCC 提升约 5.15 个百分点。

## 方法详解
- **生成目标**：训练 generator $\mathcal{G}$，使其在输入 $(P, A, M, c)$ 条件下，输出 perturbed 摘要 $S^-$；其中 $P$ 为从文档和 gold summary 抽取的 predicate spans 列表，$A$ 为 argument spans 列表，$M$ 为含 `<span_i>` 占位符的掩码摘要，$c \in \{\text{intrinsic}, \text{extrinsic}\}$ 为控制代码。
- **输入预处理**：
  - 使用 PredPatt OpenIE 从源文档与参考摘要中抽取关系三元组 $(ARG_0, PRED, \ldots, ARG_n)$。
  - 随机删除每个 tuple 中一个 argument 制造信息缺失；并对 span 做 lemmatization，强制模型在插入时恢复语法形态。
- **输入格式化四步**：
  1. **Span Removal**：测试时从 $P$ 与 $A$ 中移除来自 gold summary 的 spans，迫使模型不能简单还原原文；extrinsic 训练时也移除以避免过拟合。
  2. **Span Reduction**：以 10% 概率随机删除形容词/副词，使 span 更粗粒度，鼓励模型"幻觉"出未出现的修饰成分。
  3. **Control Code**：以等概率附加 `code: intrinsic` 或 `code: extrinsic`，分别对应"错误整合已有信息"和"捏造新信息"两类错误。
  4. **Summary Masking**：将 gold summary 中部分 predicate/argument spans 替换为 `<span_0>`（pred）、`<span_1…>`（args），标记错误应插入的位置。
- **训练设置**：基于 CNN/DailyMail 训练集（单句摘要切割），生成 394,774 条训练 / 262,692 条测试格式的示例；使用 T5-base，epoch=3，batch=24，lr=3e-5，max_src=256，max_tgt=42，beam=2，rep_penalty=2.5。
- **下游 NLI 训练**：将 Falsesum 生成的 $(D, S^+, Y=1)$ 与 $(D, S^-, Y=0)$ 混合到 MNLI，用 RoBERTa-base 进行二分类 fine-tuning（entailment vs non-entailment）。在文档级输入下，采用 sentence-wise 聚合策略：对每对 $(d_i, s_j)$ 计算 $F(d_i,s_j)$ 后取 max，再对摘要所有句子取平均。

## 实验与结果
- **数据集与基线**：
  - 生成数据源：CNN/DailyMail（Nallapati et al., 2016）。
  - 对比的 augmentation 数据集：ANLI、DocNLI、FactCC；各自采样 100k 示例。
  - 评测基准：FactCC、Ranksum、QAGS、SummEval（均为 binary consistent/inconsistent）。
- **主要结果（Table 2）**：
  - 纯 MNLI-128 整体 57.06；MNLI-512 仅 51.43（说明更长输入未带来收益，反而引入噪声）。
  - [split-doc] MNLI-128 达到 66.63，但需额外计算开销。
  - FactCC-augmented 整体 69.02；**Falsesum-augmented 整体 74.17**，相对 FactCC 提升 +5.15。
  - 各分集指标：FactCC 83.52 / Ranksum 72.90 / QAGS 75.05 / SummEval 65.18，均为当前最高。
- **消融（Table 3）**：去除 contrastive（仅保留正/负之一）损失 1.06；去除 extrinsic 损失 2.22；去除 intrinsic 损失最大 5.03（因 CNN/DailyMail 基准以 intrinsic 错误为主）。
- **细粒度分析（Figure 3）**：在 lexical overlap 高（extractive）子集上，FactCC 严重依赖词法重叠出现假阳性，Falsesum 在该子集提升最显著，说明其数据降低了模型对 surface 特征的依赖。
- **数据质量（Table 4）**：人工校验显示 86% intrinsic / 81% extrinsic 示例符合预期 label；但 extrinsic 的 type 遵循率仅 65%，模型倾向于复制而非真正"幻觉"。
- **自然性（Table 5）**：hypothesis-only 分类器在 Falsesum 上仅 69.38%，显著低于 FactCC（82.15%）与 DocNLI（78.46%），说明 Falsesum 生成的不一致摘要难以通过表面特征区分。

## 相关工作脉络
- **Goodrich et al. (2019)**：基于 OpenIE 关系 tuple 匹配计算摘要事实性，属于 extractive 路线；本文聚焦用 NLI 进行端到端分类，并解决泛化问题。
- **Kryscinski et al. (2020) — FactCC**：规则驱动的 NLI 合成数据；本文指出其错误类型单调、与真实 hallucination 分布偏离，提出可控生成替代规则变换。
- **Falke et al. (2019) — Ranksum**：将一致性建模为候选排序问题；本文沿用其 benchmark 作评估，方法定位不同。
- **Wang et al. (2020) — QAGS**：利用 QA 模型验证一致性；与本文 NLI 视角不同但共享同一评测基准。
- **Yin et al. (2021) — DocNLI**：将 QA 转换为 NLI 并 LM 替换生成文档级对；本文强调其并非源自摘要域，entailment 分布失配。
- **Maynez et al. (2020)**：提出 intrinsic / extrinsic 一致性错误分类学，本文为该分类提供自动化生成途径。

## 局限性与未来方向
- **Extrinsic 错误遵循率偏低（65%）**：生成器倾向复制而非创造新短语，导致部分 extrinsic 示例实质退化为首选项拼接，可能影响多样性的上限。
- **仅基于 CNN/DailyMail**：数据域局限于英文新闻摘要，跨域（如科学、对话摘要）的泛化能力未知。
- **OpenIE 覆盖有限**：仅抽取前 15 句、每句最多 2 个 tuple，会丢失远距离依赖或隐含事实。
- **二分类简化**：将 neutral 与 contradiction 合并为 non-entailment，可能损失细粒度一致性信号。
- **未来方向**：引入更大/更强 generator（如 T5-3B、Pegasus）提升 hallucination 自然度；扩展至多域、多语言；结合 human-in-the-loop 或对抗过滤提升标签质量；探索与 FRANK、SummaC 等细粒度一致性感知的结合。

## 研究启发与可借鉴点
1. **"仅正样本训练不一致生成器"的思路可迁移**：在其它生成质量评测（如机器翻译、代码生成）中，可用同样范式由正确输出合成合理错误，用于训练检测模型。
2. **控制代码驱动的错误类型分化**：将错误分解为 intrinsic/extrinsic 等维度并作为 conditioning，有助于构造"可控误差分布"，值得在其他鲁棒性评估任务中复用。
3. **span masking + pred/arg 拆分的输入构造**：能有效打破模型对表层 lexical overlap 的依赖，对训练不依赖词法捷径的分类器具有通用参考价值。
4. **Sentence-wise 聚合策略适用于文档级 NLI**：$\sigma(D,S)=\frac{1}{m}\sum_j \max_d F(d,s_j)$ 的计算简便、可微近似友好，可作为文档级 entailment 评估的通用 aggregation 模板。
5. **消融设计（contrastive/intrinsic/extrinsic 逐项剥离）**：为证明数据构造各组件的有效性提供了清晰可复用的实验范式。

## 关键术语表
- **Factual Inconsistency**：摘要中与源文档语义不蕴含甚至矛盾的信息现象，是 abstractive 模型常见 hallucination 的一种。
- **Intrinsic / Extrinsic 错误**：Maynez et al. (2020) 分类——前者为对源文档信息的错误整合/曲解，后者为捏造源文中未出现的"新事实"。
- **OpenIE (Open Information Extraction)**：无需预定义本体即可从句子中抽取关系三元组（谓词-参数）的技术，本文用 PredPatt 实现。
- **Span Masking**：将句子中若干 token span 替换为特殊占位符 `<span_i>`，用于控制生成模型在何处插入不一致信息。
- **Control Code**：输入中附加的 `intrinsic`/`extrinsic` 标记，用于条件化控制生成错误的类型。
- **Balanced Accuracy**：正负类 recall 的均值，适用于类别不平衡数据集的评估指标。
- **MNLI-128 / MNLI-512**：分别在最大序列长度 128 / 512 下训练的 RoBERTa-based MNLI 分类器，用于对照输入长度的影响。
- **Hypothesis-only 模型**：仅看到假设句、无前提的判别器，用于检验生成数据是否存在表面 artifacts。

## 可复现要素
- **数据集**：CNN/DailyMail（Nallapati et al., 2016）公开可用；评测基准 FactCC、Ranksum、QAGS、SummEval 均有公开版本或附录给出构造细节。
- **代码/权重**：论文未明确提供开源代码与模型权重链接（仅给出超参与训练细节）。
- **关键超参**：T5-base，epoch=3，batch=24，lr=3e-5，max_src=256，max_tgt=42，seed=11；beam=2，min_len=10，max_len=60，rep_penalty=2.5，length_penalty=1.0；RoBERTa-base fine-tune epoch=3，batch=32，lr=1e-5，max_len ∈ {128,512}，5 次随机种子 {11,12,13,14,15}。
