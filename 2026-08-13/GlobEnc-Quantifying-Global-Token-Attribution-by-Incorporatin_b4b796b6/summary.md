---
title: "GlobEnc-Quantifying-Global-Token-Attribution-by-Incorporatin"
source: https://aclanthology.org/2022.naacl-main.19.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:43"
field: "Natural Language Processing (NLP) Explainability"
keywords: ["Transformer Interpretability", "Token Attribution", "Explainable AI", "Attention Mechanism", "Layer Normalization", "Rollout Aggregation"]
innovations: ["Proposed GlobEnc to quantify global token attribution by incorporating the entire encoder layer components", "Revealed that coupling both Layer Normalizations (LN1 and LN2) is crucial due to their counteracting outlier weights", "Demonstrated that norm-based methods with dynamic residual connections significantly outperform weight-based attention analysis"]
benchmarks: ["SST-2", "MNLI", "HateXplain"]
---

# 论文速读：GlobEnc: Quantifying Global Token Attribution by Incorporating the Whole Encoder Layer in Transformers

## 一句话总结
论文提出了 **GlobEnc**，一种基于 Transformer 编码器层整体输出的 Token 归因分析方法，通过整合向量范数、残差连接与双层 Layer Normalization，并采用改进的 Attention Rollout 技术跨层聚合，实现了比现有方法更准确、更具解释性的高阶（全局）Token 重要性评估。

## 研究问题与动机
1. **现有注意力分析过于简化**：早期研究多直接利用自注意力权重作为解释依据，但纯权重忽略变换后向量的范数信息，导致归因不可靠。
2. **组件覆盖不全**：现有的基于范数的分析方法（如 Kobayashi et al. 的工作）主要聚焦于注意力模块内部，忽略了 Encoder 层中后续的全连接前馈网络（FFN）、残差连接（RES#2）及输出层归一化（LN#2）的影响。
3. **单层分析与全局行为的脱节**：大多数方法仅停留在单（层）局部归因，缺乏有效的聚合机制来量化整个多层 Encoder 对输入 Token 的全局贡献。
4. **梯度方法的计算瓶颈**：虽然梯度类方法（如 Saliency、HTA）理论上更可靠，但计算成本过高，难以在长序列或大规模实验中常规应用，需要一种高效且高保真度的替代方案。

## 核心贡献（创新点）
1. **提出 GlobEnc 全局归因框架**：首次将 Encoder 层的完整组件（从输入到 LN#2 输出）纳入统一的范数归因分析，并设计了高效的跨层聚合策略。
2. **揭示 Layer Normalization 的协同作用**：发现单独引入 LN#1 反而降低归因精度，而同时将 LN#1 与 LN#2 纳入分析能显著抵消彼此异常权重带来的干扰，从而提升相关性。
3. **实验验证了各组件的关键角色**：系统性地证明向量范数、动态残差比例、以及跨层聚合均为提高归因保真度的关键要素。
4. **在多项基准上实现 SOTA**：在 SST-2、MNLI、HateXplain 等多个任务及不同规模模型（BERT-base/large, ELECTRA）上，GlobEnc 与梯度显著性分数的 Spearman 相关性均显著优于现有主流基线方法。

## 方法详解
1. **编码器层输出归因（Local Attribution）**：
   - 基于 Kobayashi et al. 的范数分析 $\mathcal{N}$，进一步将分析范围从注意力模块扩展到整个 Encoder 层输出。
   - 公式 (11) 定义了 $\mathcal{N}_{\mathrm{ENC}}$，通过将残差输出 $\tilde{z}_{ij}$ 经过 LN#2 分解，近似提取每个输入 Token $j$ 对输出 Token $i$ 的贡献：
     $$ \tilde{x}_{ij} \approx g_{\tilde{z}_i^+}(\tilde{z}_{ij}) = \frac{\tilde{z}_{ij} - m(\tilde{z}_{ij})}{s(\tilde{z}_i^+)} \odot \gamma $$
   - 其中忽略了 FFN 的非线性影响，但保留了 LN#2 对尺度 $s(\tilde{z}_i^+)$ 的调整作用。

2. **跨层聚合（Global Aggregation via Rollout）**：
   - 采用 Abnar & Zuidema 提出的 Attention Rollout 技术进行信息流聚合，但做了关键修改：对于已包含残差信息的范数方法（如 $\mathcal{N}_{\mathrm{ENC}}$），直接使用其归因矩阵相乘（Eq. 12）；对于未显式建模残差的基线，则通过添加恒等矩阵并重新归一化（Eq. 13）来固定残差贡献比例（约为 0.5）。

3. **多基线对比设置**：
   - 覆盖了 Weight-based（$\mathcal{W}, \mathcal{W}_{\mathrm{FIXEDRES}}, \mathcal{W}_{\mathrm{RES}}$）与 Norm-based（$\mathcal{N}, \mathcal{N}_{\mathrm{FIXEDRES}}, \mathcal{N}_{\mathrm{RES}}, \mathcal{N}_{\mathrm{RESLN}}, \mathcal{N}_{\mathrm{ENC}}$）两大系列，以便消融分析各组件（范数、残差、LN）的贡献。

## 实验与结果
- **数据集**：SST-2（情感分类）、MNLI（自然语言推理）、HateXplain（仇恨言论检测）。
- **模型**：BERT-base-uncased, BERT-large, ELECTRA-base。
- **评估指标**：Spearman 等级相关系数，衡量归因向量与梯度 Saliency 得分的一致性。
- **主要结果**：
  - 在 BERT-base/SST-2 上，**GlobEnc ($\mathcal{N}_{\mathrm{ENC}}$)** 取得最高相关系数 **0.77 ± 0.12**，大幅领先次优方法 $\mathcal{N}_{\mathrm{RES}}$ (0.73)。
  - 权重类方法（$\mathcal{W}$）相关系数多为负值或接近零，验证了纯权重的不可靠性。
  - 单独加入 LN#1 ($\mathcal{N}_{\mathrm{RESLN}}$) 导致相关性骤降（如 SST-2 上跌至 -0.21），印证了 LN 异常权重相互抵消的发现。
  - 跨层 Rollout 聚合相比单层原始归因（Table 2）显著提升深层 Layer 的相关性。
  - 在 BERT-large (SST-2: 0.83) 和 ELECTRA-base (SST-2: 0.64) 上，GlobEnc 同样保持最佳性能。

## 相关工作脉络
1. **Kobayashi et al. (2020, 2021)**：提出了引入向量范数和残差连接的归因方法 $\mathcal{N}$ 和 $\mathcal{N}_{\mathrm{RESLN}}$，但未涵盖 Encoder 后半段组件（LN#2, FFN 影响）。GlobEnc 在此基础上扩展至整个 Encoder 层，并发现单独加 LN#1 有害，需双 LN 耦合。
2. **Abnar & Zuidema (2020)**：提出了基于图论的 Attention Rollout 和 Max-flow 聚合方法，假设残差贡献固定为 0.5。GlobEnc 借用了 Rollout 的聚合框架，但将其应用于包含完整 Encoder 组件信息的归因矩阵，并通过消融实验证明了动态残差建模的重要性。
3. **Brunner et al. (2020)**：提出了基于梯度的 Hidden Token Attribution (HTA) 方法，计算成本极高。本文将其改进为 HTA x Inputs 作为高成本的"黄金标准”验证手段，并证明 GlobEnc 能在效率与精度间取得更好平衡。
4. **Clark et al. (2019), Wiegreffe & Pinter (2019)**：早期关于注意力可解释性的争论文献，指出原始注意力权重不能直接解释模型行为。GlobEnc 的研究进一步支持了“必须结合多层组件和范数信息”的观点。

## 局限性与未来方向
1. **FFN 非线性忽略**：由于 FFN 中的激活函数是非线性的，无法像 LN 那样进行线性分解，因此 GlobEnc 仅近似保留了 FFN 通过 LN 尺度参数产生的间接影响，未直接量化 FFN 的贡献。
2. **模型/任务泛化性待扩展**：目前主要在基于 Masked Language Modeling (MLM) 或 Discriminative Pretraining 的 Encoder-only 模型（BERT, ELECTRA）上验证，在 Decoder 或 Encoder-Decoder 架构上的适用性尚待研究。
3. **未来方向**：作者计划将 GlobEnc 应用于更多样化的数据集和模型架构，以深入洞察模型决策机制，并探索其对其他可解释性任务（如对抗攻击分析）的迁移价值。

## 研究启发与可借鉴点
1. **整体论视角**：在分析复杂神经网络组件时，不能孤立看待子模块（如仅看 Attention），而应尽可能将相关的上下游组件（如 LN、Residual）纳入统一框架，以获得更准确的因果/贡献解释。
2. **组件间的相互作用发现**：实验揭示了 LN#1 和 LN#2 的异常权重呈负相关，这提示我们在解释模型时，需警惕单一组件分析的陷阱，关注多层结构中可能存在的补偿或抵消机制。
3. **混合评估策略**：结合高效的注意力归因与高成本的梯度归因（Saliency/HTA）作为验证基线，是一种务实且严谨的实验设计，可在可解释性研究中广泛采用。
4. **聚合方法的灵活性**：针对是否包含残差信息的不同基线，设计了动态（基于 context-mixing ratio）和固定（0.5）两种 Rollout 适配策略，这种针对不同属性矩阵设计适配聚合规则的思路值得借鉴。

## 关键术语表
- **GlobEnc**: 本文提出的全局 Token 归因方法，整合整个 Transformer Encoder 层组件并进行跨层聚合。
- **Token Attribution**: 衡量输入序列中每个 Token 对模型最终预测或某一输出位置的贡献程度。
- **Norm-based Analysis**: 一种归因方法，不仅考虑注意力权重，还结合变换后向量（Value projection output）的范数，比纯权重分析更准确。
- **Layer Normalization (LN)**: Transformer 中的归一化层，用于稳定训练。本文发现 LN#1 和 LN#2 的异常维度会相互抵消。
- **Residual Connection**: 跳过某些计算层的直接连接（如 $x + \text{Sublayer}(x)$），本文证明其对保留原始 Token 信息至关重要。
- **Attention Rollout**: 一种将多层注意力权重沿路径累积相乘以估计全局信息流的聚合技术。
- **Saliency Scores**: 基于梯度的显著性分数（如 Gradient $\times$ Input），常被视为衡量输入重要性的可靠基准。
- **Context-mixing Ratio**: 衡量一个 Token 的输出表示中，来自其他 Token 的上下文信息与自身原始信息的比例。

## 可复现要素
- **数据集**：SST-2, MNLI, HateXplain（均为公开数据集）。
- **代码**：已开源，地址 https://github.com/mohsenfayyaz/GlobEnc。
- **模型权重**：使用 HuggingFace Transformers 库的标准 BERT-base-uncased, BERT-large, ELECTRA-base 预训练模型并进行微调。
- **关键超参**：Fine-tuning epochs 3-5, batch size 32, learning rate 3e-5。
- **评估实现**：Spearman 相关系数计算，HTA 计算仅在一个小的子集（256 samples）上进行以控制成本。
