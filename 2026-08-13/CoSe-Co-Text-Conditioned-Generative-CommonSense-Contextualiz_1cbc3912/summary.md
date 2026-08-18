---
title: "CoSe-Co-Text-Conditioned-Generative-CommonSense-Contextualiz"
source: https://aclanthology.org/2022.naacl-main.83.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:54:17"
field: "常识推理与知识增强语言模型"
keywords: ["common sense reasoning", "knowledge graphs", "generative language models", "question answering", "commonsense knowledge", "sentence-path pairing"]
innovations: ["提出以句子为条件的生成式常识推理框架CoSe-Co，解决train-inference输入类型不匹配问题", "构建首个Sentence-Path配对数据集（290K对），填补无现成数据的空白", "实体掩码训练策略使模型更好地捕获上下文，生成包含23%新颖实体的多样路径"]
benchmarks: ["CommonsenseQA", "ARC", "QASC", "OBQA", "MRPC"]
---

# 论文速读：CoSe-Co: Text Conditioned Generative CommonSense Contextualizer

## 一句话总结
论文提出了 **CoSe-Co**（CommonSense Contextualizer），一种以句子为条件的生成式常识推理框架，能够根据输入的自然语言句子动态生成与整体上下文相关的多跳常识路径。该方法在 CommonsenseQA、ARC、QASC、OBQA 等常识推理任务及 paraphrase 生成任务上均取得了优于现有 SOTA 的结果。

## 研究问题与动机
1. **静态 KG 检索方法覆盖有限**：基于检索的常识增强方法依赖固定结构的 KG（如 ConceptNet），知识来源静态且稀疏，难以覆盖任务所需的全部常识。
2. **生成方法存在 train-inference 输入类型不匹配**：现有生成式方法（如 COMET、PGQA）在 KG 符号实体-关系三元组上训练，无法直接以自然语言句子为输入；应用在下游任务时需额外进行实体提取，限制了通用性。
3. **忽略整体上下文**：PGQA 等方法仅以"问题-选项"实体对为条件生成路径，忽略了句子的整体语义上下文，导致生成的常识可能与问题意图不符。
4. **缺乏句子-常识路径配对数据**：此前没有现成数据集包含自然语言句子与其相关常识路径的配对，阻碍了以句子为条件的生成式常识模型训练。

## 核心贡献（创新点）
1. **提出 CoSe-Co 框架**：首次将生成式常识推理的输入从"实体对"扩展到"自然语言句子"，使模型能捕获整体上下文并动态选择句子中的实体/短语作为推理起点，与 PGQA 等仅依赖实体对的方法形成本质区别。
2. **构建首个 Sentence-Path 配对数据集**：设计了一套从 ConceptNet 采样多跳路径、通过 Solr 检索语义相似句子并过滤的 pipeline，创建了 290K 句子-常识路径配对数据，填补了数据空白。
3. **生成知识包含新颖实体**：由于基于预训练语言模型训练，CoSe-Co 能生成包含 23.28% 新颖实体（不在原 KG 训练路径中）的推理路径，突破了静态 KG 的覆盖限制。
4. **跨任务通用性验证**：不仅在多选择题 QA（CSQA）和开放式常识推理（OpenCSR）上超越 SOTA，还证明其在无选项的文本生成任务（paraphrase generation on MRPC）中同样有效，体现了框架的任务无关性。

## 方法详解
**数据集构建流程（§3.1）**：
- 从 ConceptNet KG 中通过随机游走采样 28M 条多跳路径（路径长度 2–5 跳）。
- 使用 Apache Solr 索引 Wikipedia 语料库（9260万句子），为每条路径生成两种查询模板检索相似句子：
  - **Q1**：提取非连续实体-关系三元组 `(e_i, r_i, e_{i+2})` 和 `(e_i, r_{i+1}, e_{i+2})`，保留关系信息；
  - **Q2**：提取相邻实体对 `(e_i, e_{i+1})`。
- 用 SBERT 计算句子与查询的语义相似度，取 Top-K'=10 条句子与完整路径配对，最终获得 290K 对。
- 检查 n-gram 重叠（CSQA test 与训练集 1-gram 重叠仅 0.15）验证无数据泄露；路径-句子余弦相似度平均 0.783 验证相关性。

**模型架构（§3.2）**：
- 以 T5-base 为骨干，编码器 $E_{\theta_1}$ 处理输入句子 $s$，解码器 $D_{\theta_2}$ 自回归生成路径 token 序列 $p$。
- 损失函数：$\mathcal{L} = -\sum_{t=1}^{N} \log P(x_t^p | x_{<t}^p, O_E)$，其中 $O_E$ 为编码器输出。
- **实体掩码训练策略**：以概率 $p_{mask}=0.33$ 随机屏蔽句子中与路径共现的一个实体，迫使模型从上下文推断被屏蔽实体以生成路径，增强上下文建模能力。

**推理解码（§3.3）**：
- 采用 **diverse-path search**：在第一步采样 top-k 高概率 token，然后对每个 token 分别解码最优序列，返回 k 条 diverse 路径（论文取 k=5）。
- 该策略平衡了路径相关性与多样性（BLEU 0.436，多样性 0.43）。

## 实验与结果
**数据集与基线**：
- 主要数据集：ConceptNet（800万节点，2100万链接）+ Wikipedia（500万文章，9260万句子）。
- 评测基准：CSQA（多选题 QA）、ARC/QASC/OBQA（OpenCSR 开放式推理）、MRPC（paraphrase 生成）。
- 基线包括：MH-GRN、QA-GNN（静态 KG 方法）、PGQA（生成式基线）、COMET、DrFact（OpenCSR SOTA）。

**核心结果**：
- **CSQA**：CoSe-Co 在 100% 训练数据下 IHtest 准确率达 **72.87%**，超越 PGQA（71.19%）1.68pp，超越 QA-GNN（72.29%）0.58pp；在低数据 regime（20%）下增益更大（+2% over QA-GNN），泛化更鲁棒。
- **OpenCSR**（Hits@50 / Recall@50）：
  - ARC：+8% / +8% over DrFact；+2.3% / +1.2% over T5-base 基线。
  - QASC：+6% / +3% over DrFact；+3.9% / +2.5% over T5-base。
  - OBQA：+10% / +6% over DrFact；+7.5% / +3.7% over T5-base。
- **MRPC Paraphrase**：BLEU-4 从 43.10 提升至 **44.50**，SPICE 从 47.10 提升至 48.50。

**消融结果**（Table 3）：
- $p_{mask}=0.33$ 最优（78.15% vs. 无掩码 77.52%）。
- Q1+Q2 联合查询模板最优（优于单独 Q1 或 Q2）。
- 推理时不掩码略优于掩码变体。
- GPT-2 作为 backbone 与 T5-base 性能相当（Appendix B），证实提升来自方法设计而非骨干选择。

## 相关工作脉络
1. **COMET（Bosselut et al., 2019）**：将常识获取建模为 KG 补全任务，以 head entity + relation 生成 tail entity；局限在于输入为符号实体，无法直接处理句子，且仅生成单跳三元组。CoSe-Co 以句子为输入、生成多跳路径，从根本上解决了 train-inference 不匹配问题。
2. **PGQA（Wang et al., 2020b）**：生成连接"问题实体-选项实体"对的多跳路径；局限在于仅以实体对为条件，忽略句子整体上下文，且需预提取实体。CoSe-Co 直接以完整句子为输入，生成的路径更能捕捉语境含义。
3. **QA-GNN（Yasunaga et al., 2021）**：结合 GNN 编码的 KG 嵌入与 LM 表示进行静态 KG 检索推理；受限于 KG 的稀疏性和固定范围。CoSe-Co 通过生成式方法动态扩展知识，不依赖静态 KG 覆盖。
4. **MH-GRN（Feng et al., 2020）**：基于 GNN 的多跳关系推理网络，在静态子图上进行推理。与 CoSe-Co 相比，MH-GRN 无法生成 KG 中不存在的新颖关系路径。
5. **DrFact（Lin et al., 2021a）**：OpenCSR 任务的当前 SOTA，基于 BERT-base 的区分式推理方法。CoSe-Co 通过生成式常识增强 T5 解码器，在 ARC/QASC/OBQA 上全面超越 DrFact。

## 局限性与未来方向
1. **依赖单一 KG（ConceptNet）**：论文Acknowledged 目前仅使用 ConceptNet，未探索其他常识 KG（如 ATOMIC、ConceptCT）的融合，不同 KG 的覆盖范围和关系类型可能带来互补收益。
2. **生成路径的正确性未充分验证**：23.28% 的新颖实体路径中，部分可能引入噪声；论文仅通过 Bilinear AVG 打分（均分 0.414/1.0）间接评估，缺乏人工校验。
3. **推理效率**：每个句子需生成 k=5 条路径，增加了下游任务的计算开销；在低资源场景下可能成为瓶颈。
4. **未来方向**：① 扩展到多种 KG 的融合生成；② 探索更高效的解码策略；③ 将框架应用于更多 NLP 任务（如 NLI、文本蕴含）；④ 研究生成路径的过滤/排序机制以降低噪声影响。

## 研究启发与可借鉴点
1. **句子-知识路径配对数据构建方法**：Q1（含关系三元组）+ Q2（相邻实体对）双层查询模板设计值得借鉴，可用于其他需要句子与结构化知识配对的场景（如知识增强生成、事实核查）。
2. **实体掩码训练策略**：以概率 $p_{mask}$ 随机屏蔽句子中的共现实体，迫使模型从上下文推断缺失信息，这一技巧可迁移到任何"条件生成结构化输出"的任务中。
3. **Diverse-path search 解码策略**：在第一步采样 top-k token 后分别解码的策略，在保持相关性的同时保证输出多样性，适用于任何需要多样化生成结果的序列生成任务。
4. **生成式常识的低数据泛化优势**：CoSe-Co 在 20% 训练数据下比 QA-GNN 提升 2%（相对 PGQA 提升 3%），表明生成式方法比检索式方法在低资源场景下更具鲁棒性，这一发现对数据稀缺场景下的模型选择有指导意义。
5. **任务无关性验证范式**：在 QA、OpenCSR、paraphrase generation 三类不同任务上验证框架有效性，这种跨任务验证策略可作为后续工作的标准评估流程。

## 关键术语表
- **CoSe-Co（CommonSense Contextualizer）**：以句子为条件的生成式常识推理框架，输入自然语言句子，输出与该句子上下文相关的多跳常识路径。
- **Commonsense Knowledge Graph（KG）**：以实体为节点、常识关系为边的结构化知识库，本文使用 ConceptNet（800万节点，34种关系类型）。
- **Sentence-Path Paired Dataset**：由自然语言句子与对应常识推理路径构成的配对数据集，本文构建的 290K 对数据为该类任务的首个公开数据集。
- **OpenCSR（Open-ended Common-Sense Reasoning）**：将多选题 QA 数据集改造为开放式生成任务的评测范式，答案以 blanks 形式标注，模型需生成候选答案集合。
- **Diverse-path Search**：CoSe-Co 的推理解码策略，通过在第一步采样 top-k 高概率 token 并分别解码，生成 k 条既相关又多样的路径。
- **Entity Masking Training**：训练时将句子中概率为 $p_{mask}$ 的共现实体替换为 [MASK] token，迫使模型利用上下文推断被屏蔽实体以指导路径生成。
- **Q1 / Q2 Query Templates**：构建 sentence-path 数据集时的两种 Solr 查询模板，Q1 提取含关系的非连续三元组，Q2 提取相邻实体对。

## 可复现要素
- **数据集**：Sentence-Path 配对数据集已公开发布（论文 footnote 1 和 4 提及），基于公开资源 ConceptNet + Wikipedia 构建。
- **代码**：已开源（论文 footnote 1 提及 GitHub 链接）。
- **模型权重**：已提供训练好的 CoSe-Co 模型。
- **关键超参**：路径长度范围 [2, 5]；$p_{mask}=0.33$；推理路径数 $k=5$；训练 epochs=5；learning rate=$5e{-4}$；weight decay=0.01；batch size=8，gradient accumulation steps=4；优化器 AdamW；ε=$1e{-8}$；硬件：单卡 A-100 GPU。
- **评估指标**：CSQA 用 Accuracy；OpenCSR 用 Hits@K 和 Recall@K；MRPC 用 BLEU-4、METEOR、ROUGE-L、CIDEr、SPICE。
