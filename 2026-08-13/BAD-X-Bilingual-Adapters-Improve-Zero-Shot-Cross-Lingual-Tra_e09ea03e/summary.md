---
title: "BAD-X-Bilingual-Adapters-Improve-Zero-Shot-Cross-Lingual-Tra"
source: https://aclanthology.org/2022.naacl-main.130.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:52:21"
---

# 论文速读：BAD-X: Bilingual Adapters Improve Zero-Shot Cross-Lingual Transfer

## 一句话总结
本文提出 BAD-X 框架，通过为特定源-目标语言对训练双语适配器（Bilingual Adapters, BAs）替代 MAD-X 中的独立单语语言适配器，在零样本跨语言迁移的 POS、句法依存分析和 NLI 任务上显著提升了对低资源语言的性能。

## 研究问题与动机
1. **核心问题**：如何在零样本设定下，针对固定的源语言（$L_s$）到目标语言（$L_t$）迁移方向优化性能，尤其针对低资源语言。
2. **MMTs 的容量瓶颈**：大型多语言 Transformer（如 mBERT、XLM-R）受“多语言诅咒”制约，表征容量向高资源语言倾斜，低资源语言与远距离语言对迁移表现下降。
3. **现有 adapter 方法的方向性缺失**：MAD-X 等为每种语言单独训练单语语言适配器（LA），虽然高度模块化，但无法为特定源-目标对进行定向适配；且低资源 $L_t$ 未标注数据稀缺时，独立 LA 极易过拟合。
4. **模块化与性能的权衡**：追求任意组合 LA 的“零成本迁移”可能牺牲针对具体应用方向的优化空间，需探索语言对级联合适配的潜力。

## 核心贡献（创新点）
1. **提出 BAD-X 双语适配器框架**：用语言对级别的 BA 替换 MAD-X 中的单语 LA，使 MMT 表征更贴合特定源-目标迁移方向。
2. **交替 MLM 预训练机制**：在 BA 训练中按预设比例（1:1、1:N、N:1）交替采样 $L_s$ 与 $L_t$ 的 Wikipedia 数据，既缓解低资源语言过拟合，又促进跨语言词汇/结构信息流动。
3. **系统验证定向适配优势**：在 POS、DP、NLI 三个标准任务及 20 种低资源语言上证明 BAD-X 全面超越 MAD-X 与多语言适配器（MA），且在 diverse language families 上增益一致。
4. **揭示数据配比的任务依赖性**：发现最优源-目标数据采样比例并非固定，句法分析（DP）受益于源语言数据占比更高（5:1），而词性标注（POS）和 NLI 则在目标语言占比更高时（1:2）表现最优。

## 方法详解
1. **Adapter 基础结构**：沿用 MAD-X 的残差 adapter 设计，每层包含下投影 $\mathbf{D}_l \in \mathbb{R}^{d \times h}$、ReLU 激活与上投影 $\mathbf{U}_l \in \mathbb{R}^{h \times d}$，计算式为 $A_l(\mathbf{h}_l, \mathbf{r}_l) = \mathbf{U}_l(\mathrm{ReLU}(\mathbf{D}_l(\mathbf{h}_l))) + \mathbf{r}_l$。
2. **双语适配器（BA）训练**：放弃为 $L_s$ 和 $L_t$ 分别训 LA，改为训练单一 BA。使用 MLM 目标，训练时按固定 batch 比例交替输入两语言 Wikipedia 句子，最终选取 $L_t$ 上 perplexity 最低的 checkpoint。
3. **任务适配器（TA）训练**：在冻结的 BA 上方堆叠 TA，仅使用 $L_s$ 的带标注任务数据微调（POS/DP：15,000 steps，batch=8，lr=5e-5；NLI：5 epochs，batch=32，lr=2e-5）。
4. **推理配置**：保持 BA+TA 参数冻结，直接处理目标语言输入，实现参数高效的定向迁移。
5. **对照设计**：除 Balanced BAD-X（1:1）外测试 1:N 与 N:1 配比；同时构建 Multilingual Adapter（MA），将 $L_s$ 与所有 $N$ 个 $L_t$ 的未标注数据混合训练，用于评估参数共享的效率-性能 trade-off。

## 实验与结果
1. **实验设置**：基础模型为 mBERT；POS 与 DP 在 UD 2.7 的 10 种低资源语言上评估，NLI 在 AmericasNLI 的 10 种语言上评估；源语言固定为英语；全部实验在单张 NVIDIA GeForce RTX 3090 上完成。
2. **整体性能**：Balanced BAD-X 在多数语言上显著优于 MAD-X 与 MA；POS 平均提升 +1.06% accuracy / +0.66% F1，DP 提升 +2.62% UAS / +2.38% LAS，NLI 提升 +2.4% accuracy。
3. **最强结果与亮点**：Wolof（WO）在 DP 任务上提升约 9%（UAS/LAS）；Wixarika（HCH）在 NLI 上提升 6.67%；BAD-X 在 8/10（POS）、9/10（DP）、8/10（NLI）的语言上取得统计显著增益（Student’s t-test, p=0.05）。
4. **数据配比结论**：所有 BAD-X 变体均优于 MAD-X，但最优比例因任务而异：DP 以 5:1 最佳，POS 与 NLI 以 1:2 最佳，表明未标注数据的语言间配比需根据下游任务复杂度调节。
5. **MA 对照结论**：多语言共享适配器虽训练更高效，但受参数容量限制在多语言场景下落后于 MAD-X 和 BAD-X，证实了语言对级专业化的必要性。
6. **鲁棒性验证**：在 8 种语言上重复 3 次不同随机种子实验，BAD-X 平均得分持续高于 MAD-X，结果稳定可靠。

## 相关工作脉络
1. **MAD-X (Pfeiffer et al., 2020b)**：本文最直接的对标基线；MAD-X 追求极致模块化（任意组合单语 LA），BAD-X 则牺牲部分模块化合并 LA 为双语 BA 以换取特定方向性能。
2. **XTREME / Cross-lingual BERT (Hu et al., 2020; K et al., 2019)**：奠定基础零样本跨语言迁移评测范式；本文指出其未针对特定目标语言或源-目标对进行适配的结构性局限。
3. **MAD-G / 上下文参数生成 (Ansell et al., 2021)**：利用生成网络动态产出 adapter 参数以实现高效多语言适配；本文明确将其列为未来融合方向以提升 BA 训练效率。
4. **UDapter / Monolingual Adapters (Üstün et al., 2020; Philip et al., 2020)**：早期针对单一语言或单一任务（依存分析、机器翻译）的 adapter 适配工作；本文将其思路统一推广至多任务、多语言对的通用零样本迁移框架
