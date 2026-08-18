---
title: "Go-Back-in-Time-Generating-Flashbacks-in-Stories-with-Event"
source: https://aclanthology.org/2022.naacl-main.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:56"
field: "叙事生成与事件时序推理"
keywords: ["flashback", "story generation", "temporal prompt", "structured storyline", "reinforcement learning", "event temporal relation", "plan-and-write"]
innovations: ["在结构化故事线中嵌入时序提示（before/after/vague）引导闪回生成", "用RL端到端联合训练故事线和故事模型以缓解训练-推理差距"]
benchmarks: ["ROCStories", "WritingPrompts"]
---

# 论文速读：Go-Back-in-Time-Generating-Flashbacks-in-Stories-with-Event

## 一句话总结
本文提出了在开放域故事生成中通过结构化故事线中嵌入事件时序提示（temporal prompts）来生成闪回（flashback）的方法，结合端到端强化学习训练，使模型能够生成更具趣味性和时序多样性的故事，同时保持文本流畅性和时序常识。这是神经故事生成领域的首个专门研究闪回生成的开创性工作。

## 研究问题与动机
1. **时序偏差问题**：现有预训练语料和故事数据集中，相邻事件的before关系占比高达60-70%，简单语言模型会放大这一偏差，生成超过80%的before关系，导致故事单调。
2. **缺乏显式引导机制**：当前主流故事生成系统（如MEGATRON、CONTENTPLAN-NING）均假设事件时序遵循叙事顺序，没有显式提示帮助模型判断何时应使用闪回。
3. **闪回生成挑战**：生成闪回需要模型具备两方面能力——对事件时序关系的扎实理解（如先感到饥饿再进食），以及创造性地安排叙事顺序（早期事件不必先出现）。
4. **训练-推理差距**：既有Plan-and-Write方法在训练时使用gold故事线，推理时用预测故事线，导致两级模型之间性能不一致。

## 核心贡献（创新点）
1. **结构化故事线中的时序提示编码**：将事件表示（通过SRL提取触发词和论元）与成对时序关系（before/after/vague）组合为结构化故事线，使模型能够根据预设的时序提示决定事件序列的时序展开方式，区别于此前假设时序等于叙事顺序的工作。
2. **端到端强化学习联合训练框架**：利用REINFORCE算法将故事线模型和故事模型联合训练，以故事模型的负损失作为奖励信号反向更新故事线模型，有效缓解训练-推理差距，使模型更能充分利用时序提示。
3. **自标注预训练策略**：利用BookCorpus数据集通过SRL工具自我标注提取故事线进行预训练，帮助模型捕捉更多样化的事件序列，为下游故事生成提供更强的初始化。
4. **首个闪回故事生成基准评估**：在ROCStories和WritingPrompts两个开放域数据集上系统评估了时序多样性、准确性、时序连贯性和趣味性，证明该方法能生成更多闪回且提升故事趣味水平。

## 方法详解
**整体框架**：采用Plan-and-Write两阶段范式，首先生成故事线，再基于故事线生成完整故事，通过端到端强化学习联合优化两级模型。

**事件表示构建**：使用语义角色标注（SRL）工具（基于Shi & Lin, 2019的简单BERT模型）从每个故事句子中提取一个事件触发词（trigger）和两个论元（arguments），格式为"trigger ; arg1 ; arg2 <eoe>"，如"grabbed ; she ; the dog eoe"。缺失事件用"mask ; mask ; mask ; eoe"表示。

**时序提示定义**：定义三个成对时序关系{before, after, vague}，其中vague表示因上下文模糊导致时序未定的情况。使用时序关系抽取工具ECONET（在MATRES数据集上fine-tune）预测相邻事件间的时序关系，采用三模型投票共识机制。

**结构化故事线**：将事件表示与时序提示交替编码为$S_i^r = \{e_{i,1}, r_{i,1}, e_{i,2}, r_{i,2}, ..., e_{i,n}\}$，其中$r_{i,k}$作为预定义提示固定不变，仅预测事件表示部分。用时序提示替换故事线中的<eoe>标记（最后一个事件除外）。

**预训练策略**：使用BookCorpus的100万个五句文本片段，通过SRL提取故事线进行自我标注预训练，以获取更丰富的故事线表示能力。

**RL端到端训练**：故事模型参数$\theta$通过标准cross-entropy损失$\mathcal{L}_\theta$更新；故事线模型参数$\alpha$通过REINFORCE算法更新，梯度公式为$\nabla J_\alpha = R_i \cdot \nabla \log(p(S_i^r|x_i, r_i, \alpha))$，奖励信号定义为$R = -\mathcal{L}_\theta$（故事模型损失越小奖励越大）。

**损失函数**：
- 故事线模型损失：$\mathcal{L}_\alpha = -\log p(S_i^r|x_i, r_i, \alpha)$
- 故事模型损失：$\mathcal{L}_\theta = -\log p(\mathcal{V}_i|x_i, \hat{S}_i^r, \theta)$

## 实验与结果
**数据集**：ROCStories（5句故事，测试集4,909条）和WritingPrompts（更长故事，测试集1,000条），预训练数据为BookCorpus。

**评估指标**：自动指标包括Ref. PPL、Gen. PPL（GPT-2评分）、Distinct Ratio、BLEU-3、ROUGE-L；人工评估包括时序多样性（熵）、准确性、时序连贯性、趣味性（意外性）。

**主要结果（ROCStories）**：
| 模型 | Ref. PPL ↓ | Gen. PPL ↓ | Distinct Ratio ↑ | BLEU ↑ | ROUGE-L ↑ | 时序多样性 ↑ | 准确率 ↑ | 连贯性 ↑ | 趣味性 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| VANILLA-GEN | 27.30 | 19.29 | 3.99 | 5.13 | 19.29 | 0.88 | — | 0.88 | 2.95 |
| + STRUCTURED-PROMPT | 22.85 | 19.94 | 4.09 | 5.07 | 19.39 | 1.09 | 55.75 | 0.82 | 3.03 |
| + PRETRAINED | 21.16 | 19.25 | 4.01 | 5.06 | 19.44 | 1.07 | 52.21 | 0.84 | 2.96 |
| **+ RL** | **15.45** | **19.42** | **4.17** | **5.20** | **19.49** | **1.14** | **56.64** | **0.86** | **3.06** |

**主要结果（WritingPrompts）**：
| 模型 | Gen. PPL ↓ | Distinct Ratio ↑ | BLEU ↑ | ROUGE-L ↑ | 时序相关性 ↑ | 连贯性 ↑ | 趣味性 ↑ |
|---|---|---|---|---|---|---|---|
| CONTENTPLAN-NING | 25.52 | 1.80 | 3.46 | 14.40 | 0.04 | 0.57 | 2.20 |
| VANILLA-GEN | 11.17 | 3.50 | 0.67 | 9.43 | 0.09 | 0.52 | 2.49 |
| **+ RL** | **9.50** | **2.83** | **1.39** | **10.78** | **0.57** | **0.55** | **2.62** |

**关键结论**：
- RL模型在ROCStories上趣味性达到3.06，较VANILLA-GEN提升0.11分；时序多样性从0.88提升至1.14（提升约30%）。
- WritingPrompts上时序相关性从接近0（CONTENTPLAN-NING为0.04）大幅提升至0.57，证明时序提示能有效引导闪回生成。
- OLS回归分析显示：时序连贯性每增加1单位，趣味性提升0.609；after提示数量每增加1单位，趣味性提升0.387（p<0.001），而时序多样性本身对趣味性有轻微负面影响（因引入了过多vague关系）。

## 相关工作脉络
1. **闪回生成早期工作**：Bae & Young (2008)提出基于规划的方法生成闪回以引发读者惊喜；Wu et al. (2016)提出认知模型寻找原始故事中的最佳插入点。本文区别在于使用预训练语言模型结合时序提示生成整合式闪回，而非后期插入。
2. **Plan-and-Write框架**：Yao et al. (2019)首次引入该范式生成关键词再写故事；Xu et al. (2020)的MEGATRON和Goldfarb-Tarrant et al. (2020)的CONTENTPLAN均在此基础上发展。本文区别在于明确在事件规划中编码时序提示以控制闪回。
3. **结构化表示在故事生成中的应用**：Guan et al. (2021)的语篇结构、Peng et al. (2018)的故事关键词、Ammanabrolu et al. (2019)的事件/情节图。本文定位差异在于专注于时序控制而非泛化的结构化生成。
4. **强化学习在故事生成中的应用**：Xu et al. (2018)和Tambwekar et al. (2019)探索了RL在两阶段生成中的应用。本文区别在于RL的目标是增强时序提示的有效利用，而非泛化的生成优化。
5. **事件时序推理**：Ning et al. (2018a, b)提出MATRES数据集和CaTeRS标注体系；Han et al. (2021b)提出ECONET模型。本文作为开创性工作将时序推理成果首次应用于闪回故事生成。

## 局限性与未来方向
1. **少数模式学习不足**：时序提示中包含after的关系在数据中占少数，模型未能完美学习此类时序关系，部分生成故事存在时序不连贯（如Table 5所示的反例）。
2. **事件覆盖率与 perplexity 的权衡**：端到端训练中生成的事件不一定出现在最终故事中（覆盖率低于两阶段模型），混合训练方法（Table 4附录）需要手动调优超参数$\mu$。
3. **Reward设计简单**：当前使用故事模型的负损失作为奖励，可能不够精细，未来可设计更强大的reward机制。
4. **vague关系的过度使用**：时序多样性的提升部分源于vague关系的增加，但vague关系可能导致故事逻辑混乱，降低趣味性。
5. **开放域生成风险**：如论文所述，模型可能产生偏见、恶意语言或幻觉，需结合公平性和事实核查方法改进。

## 研究启发与可借鉴点
1. **结构化规划中嵌入控制信号的方法**：将显式时序/逻辑关系作为prompt嵌入故事线，为其他需要时序控制的生成任务（如剧本生成、视频叙事）提供了可复用的框架设计思路。
2. **端到端RL联合训练两阶段生成器**：用story模型的生成质量作为reward反向更新plan模型，解决了训练-推理差距问题，该思路可直接迁移至摘要、对话等Plan-and-Write类任务。
3. **自我标注预训练策略**：利用无标注语料通过NLP工具（如SRL）自动生成结构化中间表示进行预训练，可推广至其他需要结构化规划的生成任务。
4. **多维度评估设计**：将自动指标（PPL、多样性）与人工多维评估（时序连贯性、趣味性、意外性）结合，尤其是用OLS回归量化各因素对趣味性的贡献，为故事生成评估提供了方法论参考。
5. **时序关系扩展潜力**：当前仅用三种时序关系（before/after/vague），可考虑扩展至因果（cause）、目的（purpose）、对比（contrast）等更丰富的关系类型，进一步提升故事叙事深度。

## 关键术语表
**Flashback（闪回）**：一种叙事技巧，在故事当前情节中插入过去发生的事件，以提供背景或制造悬念。

**Structured Storyline（结构化故事线）**：由事件表示（trigger和arguments）和成对时序关系交替组成的中间结构化表示。

**Temporal Prompt（时序提示）**：嵌入在故事线中的预定义时序关系标记（before/after/vague），用于指导模型按期望的时序顺序生成事件。

**Plan-and-Write（规划-写作框架）**：两阶段故事生成范式，先生成故事线（plan），再基于故事线生成完整故事（write）。

**Semantic Role Labeling（SRL，语义角色标注）**：识别句子中动词的论元及其语义角色（如agent、patient）的NLP任务。

**ECONET**：作者团队提出的事件时序推理模型，在MATRES数据集上fine-tune，用于预测相邻事件间的时序关系。

**REINFORCE（强化学习算法）**：基于策略梯度的蒙特卡洛强化学习算法，本文用于端到端联合训练故事线模型和故事模型。

**Temporal Coherence（时序连贯性）**：生成故事中事件序列是否符合人类时序常识的判断指标。

## 可复现要素
- **数据集**：ROCStories（Mostafazadeh et al., 2016a）和WritingPrompts（Fan et al., 2018）均为公开数据集；BookCorpus（Zhu et al., 2015）亦公开。
- **代码开源**：论文声明"separately submitted code"（附录E），表明代码已开源。
- **预训练数据**：BookCorpus的100万个五句文本片段，通过SRL工具自我标注提取。
- **关键超参（ROCStories）**：学习率5e-5，batch size 10，训练10个epoch，使用3个随机种子（5, 9998, 20016）。
- **关键超参（WritingPrompts）**：学习率1e-4，batch size 64，gradient accumulation 8，训练10个epoch。
- **基座模型**：BART-base（Lewis et al., 2020）。
- **硬件**：ROCStories用单卡Nvidia GTX 2080（11G显存），WritingPrompts用Nvidia A100（40G显存）。
