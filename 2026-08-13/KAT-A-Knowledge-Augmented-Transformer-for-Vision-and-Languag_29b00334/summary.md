---
title: "KAT-A-Knowledge-Augmented-Transformer-for-Vision-and-Languag"
source: https://aclanthology.org/2022.naacl-main.70.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:38:57"
field: "多模态视觉问答与知识增强"
keywords: ["Vision-and-Language", "Knowledge-Augmented", "OK-VQA", "GPT-3", "CLIP", "Multi-modal Reasoning", "Open-Domain VQA"]
innovations: ["提出双源知识提取机制（CLIP显式检索+GPT-3隐式提示）", "设计端到端编码器-解码器联合推理模块实现异构知识自适应融合", "在OK-VQA上取得54.41%新SOTA，较PICa-Full提升6.41%绝对值"]
benchmarks: ["OK-VQA"]
---

# 论文速读：KAT-A-Knowledge-Augmented-Transformer-for-Vision-and-Languag

## 一句话总结
本文提出知识增强Transformer（KAT），通过对比学习检索Wikidata显式知识、利用冻结GPT-3提取隐式常识与支撑证据，并在编码器-解码器架构中设计跨注意力联合推理模块，首次在开放域多模态任务OK-VQA上取得SOTA（集成准确率54.41%，较PICa-Full提升6.41%绝对值）。

## 研究问题与动机
- OK-VQA等开放域视觉问答任务要求模型不仅识别图像内容，还需调用图像外部的显式事实与隐式常识进行逻辑推理。
- 现有方法多依赖物体标签或关键词检索显式知识，容易引入噪声或低相关性条目；且多数工作仅关注显式知识库，缺乏对模型内部隐式常识的系统利用。
- 已有大模型驱动方法（如PICa）将GPT-3作为黑盒隐式知识库，但未解决异构知识（显式/隐式）在生成过程中的联合推理与噪声抑制问题。
- 简单拼接多源知识会破坏语义结构并引入冗余，需设计端到端的统一架构实现知识源的自适应加权融合。

## 核心贡献（创新点）
- **双源异构知识提取机制**：分别设计基于CLIP的显式检索器与基于GPT-3提示工程的隐式抽取器，显著提升了知识条目的细粒度与相关性。
- **编码器-解码器联合推理模块**：通过Sentinel Token区分知识类型，在Decoder的Cross-Attention层对显式与隐式嵌入进行自适应加权，避免简单拼接带来的噪声干扰。
- **OK-VQA新SOTA**：KAT-ensemble在官方测试集达到54.41%准确率，较当时最优方法PICa-Full提升6.41%绝对值，覆盖全部11类知识类别。
- **可解释性分析框架**：通过消融实验与定性案例揭示显式（细粒度实体）与隐式（通用常识/逻辑链）知识的互补机制，验证了联合推理的必要性。

## 方法详解
- **问题定义**：将OK-VQA建模为序列生成任务，输入图像$ v_i $与问题$ q_i $，自回归生成答案$ a_i $，区别于主流的分类头方法。
- **显式知识检索**：采用滑动窗口裁剪图像区域$\{v_i^1,...,v_i^N\}$，用CLIP-ViT/16的图像编码器$E_{img}$与实体编码器$E_{ent}$计算归一化内积相似度$s = E_{ent}(e)^T E_{img}(v_i^j)$，从过滤后的Wikidata子集（423,520条目，覆盖动物/车辆/工具等8类）中检索Top-$ m $条目作为$ x^{exp} $，索引使用FAISS加速。
- **隐式知识检索**：先用图像描述模型将$ v_i $转为文本$ C $，构造含指令、$ C $、$ q_i $及相似训练样本的Prompt输入冻结GPT-3（175B）获取候选答案；再构造“$(q_i)?$ $(a)$ This is because”提示二次查询，提取支撑证据，合并为$ x^{imp} $。
- **推理模块**：将$ q_i $与各知识条目配对，添加Sentinel Token（`question:`, `entity:`, `description:`, `candidate:`, `evidence:`）后经Encoder生成显式矩阵$ X^{exp} \in \mathbb{R}^{m \times d} $与隐式矩阵$ X^{imp} \in \mathbb{R}^{p \times d} $；拼接为全局表示$\bar{X}$，送入Decoder的Cross-Attention层，其中$Q=W_Q H$，$K=W_K \bar{X}$，$V=W_V \bar{X}$，使答案生成过程同时感知两类知识。
- **损失函数**：标准自回归交叉熵 $\mathcal{L}_{CE} = -\sum_{t=1}^{n} \log p_\theta(y_t | y_{<t}, x^{exp}; x^{imp})$。

## 实验与结果
- **数据集**：OK-VQA（14,031张图像，14,055个问题，11个知识类别）。
- **基线对比**：涵盖纯视觉模型（VisualBERT/ViLBERT/LXMERT）、显式知识方法（KRISP/MAVEx/Vis-DPR）、大模型Few-shot方法（PICa）等。
- **主结果**：KAT-large（3次随机种子集成）达54.41%；仅用显式知识44.25%（超MAVEx 4.85%、KRISP 5.9%）；仅用隐式知识49.72%；联合推理比无推理模块（KAT w/o reasoning, 51.97%）提升2.43%。
- **消融结论**：① 模型容量越大（Large vs Base），隐式知识推理收益越明显；② 检索更多显式条目性能持续提升，说明细粒度实体能有效压缩搜索空间；③ 所有11类子任务均获显著提升（如Brands/Companies +10.2%，Science and Technology +8.3%）。
- **定性分析**：显式知识提供细粒度实体对齐，隐式知识补充常识逻辑链，二者联合可纠正单一来源的幻觉或遗漏（如Coca-Cola logo颜色推理示例）。

## 相关工作脉络
- **KRISP / MAVEx / Vis-DPR**：依赖结构化KB或搜索引擎检索显式知识，本文进一步引入冻结大模型隐式常识，并设计端到端联合推理替代独立检索-分类流水线。
- **PICa (Yang et al. 2022)**：将GPT-3作为隐式知识库进行Few-shot生成，本文在此基础上补充显式知识通路，并通过跨注意力实现异构知识的显式融合而非Prompt工程叠加。
- **DPR / REALM (ODQA)**：将文本检索与生成结合，本文将其范式扩展至视觉域，并区分显式/隐式两类知识源进行差异化处理。
- **VisualBERT / LXMERT / ViLBERT**：依赖参数内隐的跨模态表征，本文指出其不足以应对开放域知识问答，主张通过外部显式库+大模型隐式库双重增强提升推理上限。
- **CLIP / ALIGN**：无监督对比学习多模态对齐，本文直接复用其编码器作为显式知识检索 backbone，避免监督标注成本，提升实际部署灵活性。

## 局限性与未来方向
- 受算力限制，检索显式知识条目数仅设为40，未充分探索更大规模检索与动态阈值筛选的收益。
- 显式检索依赖CLIP无监督对齐，与监督预训练的物体检测器/场景分类器相比仍存性能差距，可作为后续改进方向。
- 多知识源的高效对齐（图像区域↔外部语义）及长尾实体的召回优化尚未系统解决。
- 未来可探索：① 多知识库融合架构；② 在线知识更新与增量检索机制；③ 将联合推理模块推广至其他开放域多模态任务（如Image Captioning、Referring Expression）。

## 研究启发与可借鉴点
- **分层Prompt挖掘大模型隐性知识**：先让GPT-3生成候选答案，再追问"This is because"提取支撑证据，形成结构化隐式知识对，该范式可直接迁移至其他知识密集型生成任务。
- **Sentinel Token + Cross-Attention融合策略**：用特殊标记区分问题与不同知识源，避免语义混淆，适用于多源信息（代码、文献、表格）统一的生成式推理器构建。
- **将分类VQA重构为生成任务**：放弃固定词表分类头，采用T5序列生成架构，天然支持开放式答案输出，提升模型在少样本/长尾类别上的泛化能力。
- **无监督对比学习替代监督检测器**：在资源受限场景下，直接使用CLIP等预训练模型进行区域-文本检索，可降低对昂贵标注数据的依赖，值得在开放世界视觉系统中复用。

## 关键术语表
- **OK-VQA**：要求模型调用图像外部显式与隐式知识才能正确回答的开放域视觉问答基准。
- **显式知识（Explicit Knowledge）**：来源于外部知识库（如Wikidata）的结构化/非结构化事实条目。
- **隐式知识（Implicit Knowledge）**：预训练大语言模型参数中内化的通用常识与语义关联。
- **KAT（Knowledge Augmented Transformer）**：本文提出的融合显式检索与隐式生成、支持端到端联合推理的多模态编码器-解码器架构。
- **CLIP**：基于对比学习的多模态预训练模型，本文用作图像区域与文本实体的语义对齐与向量检索 backbone。
- **Sentinel Token**：插入于Prompt首部的特殊标记（如`question:`、`entity:`），用于在Encoder中显式区分不同知识来源。
- **联合推理（Joint Reasoning）**：Decoder在自回归生成过程中通过Cross-Attention同时自适应加权多源知识表示的机制。

## 可复现要素
- **数据集**：OK-VQA（公开可用，遵循VQA Challenge标准评估协议）。
- **代码/权重**：已开源，见 https://github.com/guilk/KAT；提供预训练模型权重。
- **关键超参**：学习率3e-5（Warmup 2K步，共训练10K步），Batch size 32，16×V100 32GB GPU；检索实体数40；基座为T5-base（220M）与T5-large（770M）；正式结果取3次随机种子平均值。
