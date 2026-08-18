---
title: "KroneckerBERT-Significant-Compression-of-Pre-trained-Languag"
source: https://aclanthology.org/2022.naacl-main.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:39:21"
---

# 论文速读：KroneckerBERT-Significant-Compression-of-Pre-trained-Languag

## 一句话总结
本文提出 KroneckerBERT，通过 Kronecker 分解对 BERT_BASE 的 Embedding 层及 Transformer 内的 MHA 与 FFN 线性映射进行联合压缩，并结合仅使用 10% 英文维基百科的两阶段轻量知识蒸馏，在 7.8× 与 21× 压缩比下显著超越现有 SOTA 方法，以不到教师模型 13%（或 5%）的参数保留了主干模型绝大部分性能。

## 研究问题与动机
1. **过参数化模型难以部署于边缘设备：** Transformer 类预训练语言模型（PLMs）参数量庞大，内存、延迟与能耗约束使其无法直接落地。
2. **中等/极端压缩性能滑坡严重：** 现有低秩/SVD 分解仅能在矩阵的水平与垂直维度挖掘冗余，灵活性不足；当压缩因子 >5× 或 >10× 时，精度往往大幅下降。
3. **主流蒸馏训练开销过高：** TinyBERT、MobileBERT 等方法需从头训练专用教师模型或使用完整语料+强数据增强，计算与数据成本高昂
