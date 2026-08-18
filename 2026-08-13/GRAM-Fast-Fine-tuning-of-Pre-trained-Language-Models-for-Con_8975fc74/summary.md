---
title: "GRAM-Fast-Fine-tuning-of-Pre-trained-Language-Models-for-Con"
source: https://aclanthology.org/2022.naacl-main.61.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:37:19"
field: "推荐系统高效训练"
keywords: ["协同过滤", "预训练语言模型", "内容编码", "高效微调", "知识追踪", "新闻推荐"]
innovations: ["提出梯度累积方法GRAM替代CCF中PLM的端到端训练", "Single-step GRAM理论等价于E2E训练", "Multi-step GRAM支持更大batch与更低显存占用"]
benchmarks: ["TOEIC", "POJ", "Duolingo Spanish", "Duolingo French", "MIND"]
---

# 论文速读：GRAM-Fast-Fine-tuning-of-Pre-trained-Language-Models-for-Con

## 一句话总结
论文提出 **GRAM**（Gradient Accumulation for Multi-modality in CCF），通过**累积重复 item 的梯度**来替代 PLM-based CCF 中的端到端训练，实现最高 **146 倍**训练加速与显著 GPU 显存节省，同时保持与 E2E 训练近乎一致的性能。

## 研究问题与动机
- **核心问题**：基于内容的协同过滤（CCF）需将 PLM 作为内容编码器（CE），但端到端（E2E）训练时需对每个交互序列中的 item 反复前向/反向传播，计算开销巨大。
- **重复计算的浪费**：同一个 item 会在 batch 的不同用户历史中多次出现，但其 CE 编码被反复计算，造成冗余。
- **多模态导致复杂度立方级增长**：当序列长度 $l_t \approx l_I$ 时，CE 的计算复杂度约为 $O(|B_u| \cdot l_I (l_t^2 d + l_t d^2))$，GPU 显存需求极高。
- **现有微调方法不适用**：SpeedyFeed 等方法依赖特定工程实现与领域假设（如自回归建模、负样本），无法通用迁移到 KT 等场景。

## 核心贡献（创新点）
1. **Single-step GRAM**：提出交替训练 CF 与 CE 模块，通过伪目标（pseudo-target）累积同一 item 的梯度，**理论等价于 E2E 训练**，加速约 4 倍。
2. **Multi-step GRAM**：将梯度累积扩展到多个训练步，**进一步提速且显存降至 E2E 的不到 40%**，适用于大规模数据。
3. **理论等价性证明**：给出 Proposition 证明 pseudo-loss 下的梯度与 E2E 梯度严格相等（Appendix A）。
4. **双任务领域验证**：在知识追踪（4 个数据集）与新闻推荐（1 个数据集）上全面验证，TOEIC 上获得 **146 倍**加速。
5. **代码开源**：代码已发布在 GitHub，便于后续复用与扩展。

## 方法详解

### 整体框架
CCF 由内容编码器（CE）与协同过滤器（CF）组成，通常包含：
- **CE 模块**：BERT/TinyBERT，负责将 item 文本内容编码为嵌入 $e_n^u = CE(c_n^u)$
- **CF 模块**：LSTM/MHSA，负责基于用户历史交互序列预测响应

### Single-step GRAM
1. **前向**：对 mini-batch 中的**唯一 item** 各跑一次 CE，得到 $h_i^{(t)}$
2. **更新 CF**：用标准损失 $\mathcal{L}$ 反向传播更新 $\theta_{CF}$
3. **构造伪目标**：
   $$\tilde{h}_i^{(t)} \leftarrow h_i^{(t)} - \nabla \mathcal{L}(h_i^{(t)})$$
   即把 CE 的输出当作可训练 embedding，沿损失梯度方向走一步（学习率为 1 的 SGD）
4. **更新 CE**：最小化伪损失：
   $$\tilde{\mathcal{L}} := \frac{1}{2} \sum_{i \in B_I^{(t)}} \left(\tilde{h}_i^{(t)} - CE(c_i; \theta_{CE}^{(t)})\right)^2$$
   用 Adam 优化器更新 $\theta_{CE}$

**理论保证**：Proposition 证明 $\frac{\partial L}{\partial \theta_f} = \frac{\partial \tilde{L}}{\partial \theta_f'}$，即与 E2E 梯度完全等价。

### Multi-step GRAM
将交替频率设为 $N > 1$（如 10 步、半 epoch、全 epoch）：
- CF 和 CE 在多个 step 内分别独立更新多次
- 不再严格等价于 E2E，但可通过增大 $N$ 换取更大加速与更低显存

### 加速比分析
$$\mathcal{R} = \frac{\sum_{u \in B_u} |I^u|}{|B_I|} = \frac{\text{batch 中交互数}}{\text{batch 中唯一 item 数}}$$
- batch 越大，$\mathcal{R}$ 越高
- 数据集层面：$\mathcal{R} = \frac{\text{总交互数}}{\text{总 item 数}}$，Multi-step 可达 **146×**

## 实验与结果

### 数据集
| 数据集 | 任务 | 用户数 | Item 数 | 交互数 | 平均 $l_t$ |
|--------|------|--------|---------|--------|------------|
| Spanish | KT | 2,643 | 4,628 | 279,747 | 5.32 |
| French | KT | 1,202 | 4,078 | 174,749 | 5.24 |
| POJ | KT | 22,110 | 2,597 | 898,384 | 271.34 |
| TOEIC | KT | 1,240,955 | 9,336 | 94,264,845 | 147.47 |
| MIND | NR | 750,434 | 104,150 | 3,760,125 | 11.52 |

### 主要结果（KT 领域）
- **TOEIC 数据集**：GRAM 1E 较 E2E 加速 **146 倍**，AUC 仅下降 0.4（75.7 → 75.3），CSAUC 持平
- **GRAM 1S**：在所有 KT 数据集上接近 E2E 性能，平均加速约 **4–6 倍**
- **对比 NoContent / NoFinetune**：均有显著性能下降（NoContent AUC 仅 ~50）
- **对比 EERNN / LM-KT / CRCF**：GRAM 1S 在 AUC/CSAUC 上均优于这些基线

### 主要结果（NR 领域）
- **MIND 标题仅**：GRAM 1E 较 E2E 加速 **17.3 倍**，AUC 几乎持平（68.7 vs 68.9）
- **MIND 标题+正文**：GRAM 1E 用 56 倍加速、训练时间从 202h 降至 3.6h，AUC 达 69.3，超过所有仅用标题的基线
- ** leaderboard**：Single-step + Multi-step 集成排名 **MIND 官方排行榜第 4**

### 显存消耗（Table 6）
| 方法 | TOEIC 显存 | MIND 显存 |
|------|-----------|----------|
| E2E | 38.6 GB (95.2%) | 38.4 GB (95.1%) |
| GRAM 1E (CE batch=8) | 4.8 GB (12.1%) | 5.1 GB (12.5%) |
| GRAM 1E (CE batch=32) | 6.5 GB (16.0%) | 14.1 GB (34.9%) |

### 训练成本对比
- TOEIC E2E：\$3,161 → GRAM 1E：\$21
- MIND（标题+正文）E2E 估算 \$4,735 → GRAM 1E：\$84

## 相关工作脉络
1. **NRMS / NRMS-PLM**（Wu et al., 2019, 2021a）：新闻推荐中 CE-CF pipeline 的代表工作，GRAM 在其基础上引入高效训练机制。
2. **EERNN / DKT**（Su et al., 2018; Piech et al., 2015）：知识追踪领域经典 CE-CF 方法，GRAM 使用 BERT 替代其 Word2Vec+LSTM 方案。
3. **TARMF / CRCF**（Lu et al., 2018）：通过 CE 输出正则化 item 表示，GRAM 证明交替训练等价且更高效。
4. **LM-KT**（Jiao et al., 2020）：自回归建模方式，序列长度 $O(l_t \times l_I)$ 带来瓶颈，GRAM 避免此问题。
5. **SpeedyFeed**（Xiao et al., 2021）：工程型加速方案，依赖缓存与自回归，无法泛化到无负样本的 MIND；GRAM 提供正交且通用的训练范式。
6. **NoFinetune / NoContent**：消融基线，证明 CE 微调与内容特征均必要。

## 局限性与未来方向
- **额外超参数**：Multi-step GRAM 需设置梯度累积步数 $N$，不同数据集最佳值不同（论文未给出自动选择方案）
- **收敛稳定性**：交替训练在部分数据集上速度提升非单调，可能因 CE/CF 优化切换带来方差波动
- **未覆盖多模态**：目前仅处理文本内容，未扩展到图像/视频等高维输入
- **大规模扩展**：论文未讨论分布式环境下 GRAM 的通信开销与同步策略

## 研究启发与可借鉴点
1. **梯度累积替代重复计算**：当同一 entity 在 batch 中多次出现时，可考虑梯度累积而非重复前向/反向传播，思路可迁移至推荐系统、多实例学习等场景。
2. **伪目标技巧**：通过构造 $\tilde{h} = h - \nabla_h \mathcal{L}$ 将 CE 的输出视为可训练 embedding，实现模块间交替训练并保持梯度等价性，是一种优雅的理论工具。
3. **实验设计借鉴**：使用相同架构分别跑 E2E 与 GRAM 进行公平对比，并报告 wall-clock 时间与 GPU 成本，值得在多模态训练效率研究中借鉴。
4. **可结合 LoRA / Adapter**：GRAM 与参数高效微调方法正交，未来可组合使用进一步提升训练效率。
5. **工业落地价值**：论文已在 Santa 平台（400 万用户英语学习平台）部署，证明方法在真实业务中的可行性。

## 关键术语表
- **CCF（Content-based Collaborative Filtering）**：融合 item 内容信息的协同过滤方法，解决冷启动问题
- **CE（Content Encoder）**：将 item 文本内容编码为向量表示的模块，通常基于 PLM
- **CF（Collaborative Filter）**：基于用户交互历史预测用户响应的模块，通常为 LSTM/MHSA
- **E2E（End-to-End）**：同时训练 CE 与 CF 模块的标准方法，需要联合反向传播
- **Pseudo-target / Pseudo-loss**：GRAM 中的中间变量，使 CE 的训练梯度与 E2E 等价
- **Multi-step GRAM**：梯度累积步数 $N > 1$ 的变体，以轻微性能损失换取更大加速与显存节省
- **CSAUC（Cold-start AUC）**：在未见过的 item（冷启动）上的 AUC 指标
- **MIND（Microsoft News Dataset）**：大规模英文新闻推荐数据集

## 可复现要素
- **数据集**：Duolingo（Spanish/French）、POJ、TOEIC、MIND；部分数据集可公开获取，TOEIC 互动数据需申请研究用途
- **代码**：已开源，https://github.com/yoonseok312/GRAM
- **关键超参**：
  - CE 模块：TinyBERT（6层 MHSA，dim=768）
  - CF 模块（KT）：2层 LSTM
  - CF 模块（NR）：单层 MHSA
  - 学习率：CF 和 CE 均为 1e-4
  - 优化器：Adam + Noam scheduler
  - Batch size：KT 为 32，MIND 为 256
  - Early stopping patience：10 epochs
- **硬件**：NVIDIA A100 GPU
