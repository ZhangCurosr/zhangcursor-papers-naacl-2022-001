---
title: "CERES-Pretraining-of-Graph-Conditioned-Transformer-for-Semi"
source: https://aclanthology.org/2022.naacl-main.16.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:52:54"
field: "会话推荐与预训练"
keywords: ["semi-structured session", "pretraining", "graph-conditioned transformer", "masked language modeling", "e-commerce recommendation", "session encoding"]
innovations: ["提出图条件掩码语言建模（GMLM）联合学习项目内语义与项目间关系", "设计跨注意力Transformer通过latent conditioning tokens传播会话级上下文", "位置感知GNN将时序位置嵌入图节点以编码会话图结构"]
benchmarks: ["Product Search", "Query Search", "Entity Linking"]
---

# 论文速读：CERES-Pretraining-of-Graph-Conditioned-Transformer-for-Semi

## 一句话总结
论文提出 **CERES**（Graph Conditioned Encoder Representations for Session Data），一种面向半结构化电商会话数据的预训练模型，通过图条件掩码语言建模（GMLM）和图条件Transformer架构，联合学习项目内文本语义与项目间交互关系。在 Amazon 4.68 亿会话上预训练后，CERES 在三个下游任务（产品搜索、查询搜索、实体链接）上分别以最高 9% 的增益超越最强基线。

## 研究问题与动机
1. **现有方法无法同时建模文本与图结构**：会话推荐方法（如 SR-GNN、NISER+）将会话视为序列或图，忽略项目内丰富的文本语义；预训练语言模型（如 BERT、RoBERTa）擅长文本但无法建模会话关系图。
2. **已有知识图谱增强方法局限**：K-BERT、KG-BERT 等工作利用知识图谱调整实体语义，但无法编码会话中通用的异构图结构（查询-商品交互关系）。
3. **缺乏面向半结构化会话的自监督预训练**：目前缺少能够统一捕捉项目内语义（intra-item）与项目间交互（inter-item）的预训练框架。
4. **小样本下性能退化严重**：传统大模型在少量标注数据时难以充分训练，亟需高效利用会话级上下文信息的预训练方法。

## 核心贡献（创新点）
1. **提出 GMLM（Graph-Conditioned Masked Language Modeling）预训练任务**：通过掩码 token 预测同时利用项目内上下文与图级项目间上下文，区别于仅依赖单项目文本的 MLM。
2. **设计图条件 Transformer 架构**：结合 Item Transformer Encoder（捕获项目内语义）和 Graph-Conditioned Session Transformer（聚合传播会话级上下文），实现端到端联合建模。
3. **提出位置感知 GNN + 跨注意力 Transformer 的信息传播机制**：通过 latent conditioning tokens 将图级信息注入 token 级别，同时防止会话信息被项内信息稀释。
4. **构建大规模电商会话预训练数据集与评测基准**：使用 4.68 亿 Amazon 会话进行预训练，并在产品搜索、查询搜索、实体链接三个任务上系统评估，证明 CERES 在低资源场景下的高效性。

## 方法详解
**GMLM 预训练任务**：给定会话图 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ 和 token 掩码序列，优化：
$$p_{\text{GMLM}}(v_{\text{masked}}) = \prod_j \mathbb{P}(v_{ij} | \mathcal{G}, \{v_{ik}\}_{k \text{ unmasked}})$$
掩码策略：长序列（商品标题/描述）按 15% 随机掩码，短序列（查询/属性）有 50% 概率整个被掩码，掩码内再随机选 50% token。

**Item Transformer Encoder**：对每个 item（查询或商品）独立运行 Transformer，得到 token 级嵌入 $\mathbf{v}_{ij}$ 和 pooled 嵌入 $\mathbf{v}_i$。查询以特殊 token [SEARCH] 开头，商品属性以 [ATTRTYPE] 开头，商品嵌入取各属性句嵌入的平均池化。

**Position-Aware GNN（PGNN）**：将 pooled item embeddings 作为节点隐状态，加入位置嵌入后送入两层 Graph Attention Network，捕获会话图内项目间关系，输出 session-level item embeddings $\mathbf{v}_i^h$。

**Cross-Attention Transformer**：将 $\mathbf{v}_i^h$ 通过 MLP 扩展为 K 个 latent conditioning tokens，每个 token 取邻域平均后与原 item token embeddings 拼接，送入浅层 Transformer。关键设计：latent tokens 仅做 self-attention，禁止 attend 原始 item tokens，防止会话信息被项内信息稀释。

**Fine-tuning**：下游任务中，会话嵌入取所有 item 嵌入的平均；单 item 嵌入仅用 Item Transformer Encoder。相似性通过线性变换后余弦相似度计算，优化 hinge loss。

## 实验与结果
**数据集**：Amazon 电商会话，468,199,822 个会话用于预训练（2020.8），30,000 会话用于下游任务（2020.9），时间不重叠防泄漏。共涉及 37,580,637 个商品，平均每会话 3.24 个查询、4.36 个商品。

**基线分类**：
- 通用预训练语言模型：BERT、RoBERTa、ELECTRA
- 领域会话预训练模型：Product-BERT、SQSP-BERT
- 会话推荐方法：SR-GNN、NISER+、MERLIN

**主要结果（MAP@1）**：
- 产品搜索：CERES 92.63 vs Product-BERT 91.03（+1.6%）
- 查询搜索：CERES 59.94 vs Product-BERT 52.72（+7.2%）
- 实体链接：CERES 75.48 vs Product-BERT 66.83（+8.7%）

**小样本优势**：当训练样本仅为 300 时，CERES 在 Product Search 达 37.55%、Query Search 达 36.37%，而 BERT /Product-BERT 无法充分训练。

**消融**：移除 Graph-Conditioned Transformer（CERES w/o Cond）导致 Product Search MAP@64 下降 0.1%；移除 GNN（CERES w/o GNN）下降 1.13%；从头训练 Graph-Conditioned Transformer（CERES w/o Pretrain）反而低于 Product-BERT。

**推理开销**：相比同等规模的 12 层 BERT，额外增加约 20% 推理时间。

## 相关工作脉络
1. **预训练语言模型**（BERT、RoBERTa、ELECTRA）：擅长通用文本表示，但不建模图结构，本文将其作为通用基线对比。
2. **知识图谱增强语言模型**（K-BERT、KG-BERT、ERNIE 2.0/3.0）：利用 KG 调整实体语义，但依赖预定义知识图谱，无法编码会话异构图结构。
3. **图神经网络预训练**（GCPN、GCC、GPT-GNN）：面向节点特征已提取的图数据，不能直接处理以文本为节点的会话图。
4. **会话推荐方法**（SR-GNN、NISER+、GC-SAN）：聚焦交互序列/图结构，忽略文本语义，导致 item 表示质量受限。
5. **SQSP-BERT**：仅利用查询-商品二元交互对进行预训练，缺乏完整会话上下文建模，本文在其基础上引入图结构增强。
6. **位置感知图网络**：本文借鉴 BERT 的 positional embedding 思想到 GNN 节点层面，使 GNN 能感知会话中项目的时序位置。

## 局限性与未来方向
1. **仅支持文本模态**：当前 CERES 仅处理会话中的文本描述，未整合商品图像、客户评论等多模态信息。
2. **隐私风险**：会话数据包含个性化用户行为，需严格匿名化；论文建议使用时避免泄露用户画像。
3. **图结构固定**：当前 session graph 基于固定时间窗口（600 秒）内交互构建，未探索动态图或跨会话图结构。
4. **扩展方向**：作者指出可推广至任意异构数据，如含图像的 multimodal session graph。

## 研究启发与可借鉴点
1. **GMLM 任务设计可迁移**：图条件掩码语言建模思路可推广至其他半结构化数据（如社交网络帖子-用户交互、对话日志）的预训练。
2. **跨注意力 + 禁止反向 attend 的设计**：latent conditioning tokens 只 self-attend 不 attend item tokens 的约束，有效防止信息污染，该技巧适用于其他多层信息融合场景。
3. **位置嵌入在 GNN 上的适配**：将 BERT 式 positional embedding 引入 GNN 节点，使 GCN 感知节点时序位置，对任意有序图结构均有参考价值。
4. **小样本高效性验证**：CERES 在 300 样本下仍表现良好，表明图条件预训练可显著降低下游任务数据需求，适合低资源电商场景。
5. **双阶段评估框架**：预训练与微调分离、时间不重叠防泄漏的数据划分策略，值得在电商预训练工作中复用。

## 关键术语表
**CERES**：Graph Conditioned Encoder Representations for Session Data，面向半结构化电商会话的预训练模型。
**GMLM（Graph-Conditioned Masked Language Modeling）**：图条件掩码语言建模，预训练任务，预测掩码 token 时同时利用项目内上下文和图级项目间上下文。
**Item Transformer Encoder**：基于 Transformer 的项目编码器，独立捕获每个会话 item 的文本语义。
**PGNN（Position-Aware Graph Neural Network）**：位置感知图神经网络，在 GNN 节点上加入时序位置嵌入以感知会话顺序。
**Cross-Attention Transformer**：跨注意力 Transformer，通过 latent conditioning tokens 将会话级图信息传播至 token 级别。
**Product Sequence**：商品序列，商品标题与 bullet description 的拼接文本。
**SQSP（Single-Query Single-Product）**：单查询-单商品交互对，用于预训练的数据单元。
**MAP@K / Recall@K**：Mean Average Precision@K 和 Recall@K，检索任务常用评估指标。

## 可复现要素
- **数据集**：Amazon 电商会话，预训练 468,199,822 个会话（2020.8），下游任务 30,000 会话（2020.9），论文未公开代码和权重。
- **预训练超参**：hidden size 768，12 层 Transformer 主干，2 层 GAT + 3 层 Transformer 作为图条件层，总参数 141M，batch size 512，300,000 步，peak LR 3e-5，1% warmup，Adam 优化器，16×A400 GPU。
- **微调超参**：每任务 30,000 训练/3,000 验证/5,000 测试，10 epochs，LR 从 [1e-4, 1e-5, 5e-5, 5e-6] 中择优，其余同预训练。
