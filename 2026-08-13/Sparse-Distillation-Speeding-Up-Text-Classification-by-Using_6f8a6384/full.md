# Sparse Distillation: Speeding Up Text Classification by Using Bigger Student Models

Qinyuan Ye<sup>1</sup>† Madian Khabsa<sup>2</sup> Mike Lewis<sup>2</sup> Sinong Wang<sup>2</sup> Xiang Ren<sup>1</sup> Aaron Jaech<sup>2</sup> <sup>1</sup>University of Southern California <sup>2</sup>Meta AI {qinyuany,xiangren}@usc.edu   
{mkhabsa,mikelewis,sinongwang,ajaech}@fb.com

## Abstract

Distilling state-of-the-art transformer models into lightweight student models is an effective way to reduce computation cost at inference time. The student models are typically compact transformers with fewer parameters, while expensive operations such as selfattention persist. Therefore, the improved inference speed may still be unsatisfactory for real-time or high-volume use cases. In this paper, we aim to further push the limit of inference speed by distilling teacher models into bigger, sparser student models – bigger in that they scale up to billions of parameters; sparser in that most of the model parameters are ngram embeddings. Our experiments on six single-sentence text classification tasks show that these student models retain 97% of the RoBERTa-Large teacher performance on average, and meanwhile achieve up to 600x speedup on both GPUs and CPUs at inference time. Further investigation reveals that our pipeline is also helpful for sentence-pair classification tasks, and in domain generalization settings.<sup>1</sup>

## 1 Introduction

Large pre-trained Transformers (Devlin et al., 2019; Liu et al., 2019) are highly successful, but their large inference costs mean that people who host low-latency applications, or who are simply concerned with their cloud computing costs have looked for ways to reduce the costs. Prior work mainly achieves this by leveraging knowledge distillation (Hinton et al., 2015), which allows for the capabilities of a large well-performing model known as the teacher to be transferred to a smaller student model. For example, DistillBERT (Sanh et al., 2019) is a smaller transformer model distilled from BERT (Devlin et al., 2019), which reduces BERT’s size by 40% and becomes 60% faster during inference. However, such speed-up may be still insufficient for high-volume or low-latency inference tasks. In this paper, we aim to further push the limit of inference speed, by introducing Sparse Distillation, a framework that distills the power of state-of-the-art transformer models into a shallow, sparsely-activated, and richly-parameterized student model.

![](images/a2c6e669bfb611725eefc169c1bf0ab000bd78323abb3e2df15bb37eccb53f6d.jpg)  
Figure 1: Performance vs. Inference Speed. With Deep Averaging Network (DAN; Iyyer et al. 2015) and knowledge distillation, we obtain a student model with competitive performance on IMDB dataset, while being 607x faster than RoBERTa-Large, and 20x faster than bi-directional LSTMs at inference time.

Counter to the convention of using “smaller, faster, [and] cheaper” (Sanh et al., 2019) student models, our work explores a new area of the design space, where our fast and cheap student model is actually several times larger than the teacher. The student model we use is modified from Deep Averaging Network (DAN) in Iyyer et al. (2015). DANs take a simple architecture by mapping the n-grams in the input sentence into embeddings, aggregating the embeddings with average pooling, and then using multiple linear layers to perform classification (see Fig. 2). This architecture is reminiscent of the high expressive power of billion-parameter ngram models (Buck et al., 2014; Brants et al., 2007) from before the existence of pre-trained language models. By selecting the n-gram vocabulary and the embedding dimension, DANs also scale up to billions of parameters. Meanwhile, the inference costs are kept low as DANs are sparsely-activated.

One weakness of DANs is that they are restricted in modeling high-level meanings in long-range contexts, as compared to the self-attention operator in Transformers. However, recent studies have shown that large pre-trained Transformers are rather insensitive to word order (Sinha et al., 2021) and that they still work well when the learned selfattention is replaced with hard-coded localized attention (You et al., 2020) or convolution blocks (Tay et al., 2021). Taken together, these studies suggest that on some tasks it may be possible to get competitive results without computationally expensive operations such as self-attention.

To verify our hypothesis, we use six singlesentence text classification tasks<sup>2</sup> and apply knowledge distillation to DANs. We observe that the resulting student models retain 97% of the RoBERTa-Large teacher performance on average. We also show that our method falls outside of the Pareto frontier of existing methods; compared to a baseline of distilling to a LSTM student, our method gives comparable accuracy at less than 1/20 the inference cost (see Fig. 1). Based on our empirical results, we conclude that faster and larger student models provide a valuable benefit over existing methods. We further examine our method (1) with QQP, a sentence-pair task, (2) in privacy-preserving settings (i.e., no access to task-specific data during distillation), and (3) in domain generalization and adaptation settings (i.e., student models are applied and adapted to new data domains), where we find our method continues to bring improvements over non-distillation baselines.

## 2 Sparse Distillation with DANs

## 2.1 Problem Definition

Our goal is to train an efficient text classification model M for a given task T. In a n-way classification problem, the model M takes input text x, and produces $\hat { \mathbf { y } } \in \mathbb { R } ^ { n }$ , where $\hat { y } _ { i }$ indicates the likelihood that the input x belongs to category i. The task $T$ has a train set $D _ { t r a i n }$ and a validation/development set $D _ { d e v }$ . Additionally, we assume access to a large unlabeled corpus C which is supposedly in a domain relevant to task T. We comprehensively evaluate the efficiency of the model M by reporting: (1) accuracy on $D _ { d e v } ,$ (2) inference speed, and (3) the number of parameters in the model.

![](images/ce5aa647ca7ac5bf3c70e4bf0c30b75ae6f1c68c4542a11f1882926842c6a193.jpg)  
Figure 2: We primarily use a modified Deep Averaging Network (DAN; Iyyer et al. 2015) as the student model in this paper. DAN contains a sparse n-gram embedding table and two linear layers. Embedding dimension $d _ { e }$ is set to 3 in this figure for illustration purpose.

## 2.2 Method Overview

To train a text classifier that is both efficient and powerful, we employ knowledge distillation (Hinton et al., 2015), by having a powerful teacher model provide the supervision signal to an efficient student model. In particular, we are interested in using sparse n-gram based models as our student model. We explain the teacher and student model we use in §2.3, the training pipeline in §2.4, and implementation details in §2.5

## 2.3 Models

Teacher Model. Fine-tuning a pre-trained transformer model is the predominant recipe for obtaining state-of-the-art results on various text classification tasks. Our teacher model is a RoBERTa-Large model (Liu et al., 2019) fine-tuned on the training set $D _ { t r a i n }$ of task T.

Student Model. Our student model is based on the Deep Averaging Network (DAN, Iyyer et al. 2015) with the modification that we operate on ngrams instead of just words. See Fig. 2 for an illustration of the model architecture. Specifically, for an input sentence x, a list of n-grams g<sub>1</sub>, g<sub>2</sub>, ..., g<sub>n</sub> are extracted from the sentence. These n-gram indices are converted into their embeddings (with dimension $d _ { e } )$ using an embedding layer Emb(.). The sentence representation h will be computed as the average of all n-gram embeddings, i.e., h = <sup>Mean(Emb(g</sup>1<sup>),</sup> <sup>Emb(g</sup>2<sup>),</sup> <sup>...,</sup> <sup>Emb(g</sup>n<sup>))</sup> ∈ <sup>Rde.</sup> The sentence representation then goes through two fully connected layers, $( \mathbf { W } _ { 1 } , \mathbf { b } _ { 1 } )$ and $( \mathbf { W } _ { 2 } , \mathbf { b } _ { 2 } )$ to produces the final logits zˆ, i.e., $\hat { \mathbf { z } } = M _ { s } ( \mathbf { x } ) =$ $\mathbf { W } _ { 2 } ( \mathrm { R e L u } ( \mathbf { W } _ { 1 } \mathbf { h } + \mathbf { b } _ { 1 } ) ) + \mathbf { b } _ { 2 } \in \mathbb { R } ^ { n }$ . The logits are transformed into probabilities with the Softmax function, i.e., $\hat { \mathbf { y } } = \operatorname { S o f t m a x } ( \hat { \mathbf { z } } ) \in \mathbb { R } ^ { n }$

![](images/c55d4d4629a64937b0f777d171a4810147740b059d3482f8cfde7a881ab9d5e9.jpg)  
Figure 3: We adopt a three-stage pipeline for Sparse Distillation: (1) We fine-tune a RoBERTa-Large model on $D _ { t r a i n }$ to get the teacher model. (2) We apply teacher model to the unlabeled corpus C and $D _ { t r a i n }$ , and train the student model (DAN) to mimic the predictions of the teacher. This model is denoted as “DAN (KD)” (3) We further fine-tune the student model with $D _ { t r a i n }$ . This model is denoted as “DAN (KD+FT)”.

Remarks on Computation Complexity. Multiheaded self-attention is considered the most expensive operation in the teacher transformers, where the computation complexity is $O ( m ^ { 2 } )$ for a sequence with m sub-word tokens. The student model, Deep Averaging Network (DAN), can be considered as pre-computing and storing phrase representations in a large embedding table. By doing so, the computation complexity is reduced to $O ( m )$ . However, unlike the teacher, the context is limited to a small range, and no long-range information (beyond n-gram) is taken into account by the student model.

## 2.4 Training Pipeline

Our training pipeline is illustrated in Fig. 3. It has three stages: (1) We first fine-tune a RoBERTa-Large model on the train set $D _ { t r a i n }$ of task T, and use the resulting model as the teacher model. (2) We train the student model by aligning the predictions of the teacher (y˜) and the predictions of the student (yˆ) on the union of unlabeled corpus C and the train set $D _ { t r a i n }$ . We align the predictions by minimizing the KL divergence between the two distributions, i.e., $\begin{array} { r } { L = \sum _ { j = 1 } ^ { n } \tilde { y } _ { j } } \end{array}$ log $\frac { \tilde { y } _ { j } } { \hat { y } _ { j } }$ . The resulting student model is denoted as “DAN (KD)”. (3) We further fine-tune the student model from step (2) with the task train set $D _ { t r a i n } .$ , and get a new student model. This model is denoted as “DAN (KD+FT)”. This third stage is optional.

## 2.5 Implementation Details

Determine N-gram Vocabulary. Our student model takes in n-grams as input. We determine the n-gram vocabulary by selecting the top V frequent n-grams in $D _ { t r a i n }$ and C. For each downstream dataset, we compute the vocabulary separately. We use CountVectorizer with default whitespace tokenization in sklearn (Pedregosa et al., 2011) to perform this task. We set ngram range to be (1, 4) and set $\vert V \vert = 1 , 0 0 0 , 0 0 0$ $d _ { e } = 1 , 0 0 0$ , unless specified otherwise.

Optimization. The architecture of DAN is sparsely-activated, and thus can be sparselyoptimized to reduce memory footprint. To facilitate this, we design a hybrid Adam optimizer, where we use SparseAdam<sup>3</sup> for the sparse parameters (i.e., the embedding layer), and regular Adam for dense parameters. This implementation helps to improve speed and reduce memory usage greatly – we can train a 1-billion parameter DAN with the batch size of 2048 at the speed of 8 batches/second, on one single GPU with 32 GB memory.

Additional Details. Due to space limit, we defer details such as hyper-parameters settings and hardware configurations in Appendix A.

## 3 Experiment Settings

## 3.1 Data

Downstream Datasets. Following Tay et al. (2021), we mainly use six single-sentence classification datasets as the testbed for our experiments and analysis. These datasets cover a wide range of NLP applications. We use IMDB (Maas et al., 2011) and SST-2 (Socher et al., 2013) for sentiment analysis, TREC (Li and Roth, 2002) for question classification, AGNews (Zhang et al., 2015) for news classification. We use Civil Comments (Borkan et al., 2019) and Wiki Toxic (Wulczyn et al., 2017) dataset for toxicity detection.

<table><tr><td>Dataset D</td><td> $| D _ { t r a i n } |$ </td><td> $| D _ { d e v } |$ </td><td>Avg. l</td><td>Distillation Corpus C</td><td> $| C |$ </td></tr><tr><td>IMDB</td><td>25,000</td><td>25,000</td><td>300</td><td>Amazon Reviews and *</td><td>75m</td></tr><tr><td>SST-2</td><td>67,349</td><td>872</td><td>11</td><td>Amazon Reviews</td><td>75m</td></tr><tr><td>TREC</td><td>5,452</td><td>500</td><td>11</td><td>PAQ</td><td>65m</td></tr><tr><td>AGNews</td><td>120,000</td><td>7,600</td><td>55</td><td>CC-News</td><td>418m</td></tr><tr><td>CCom</td><td>1,804,874</td><td>97,320</td><td>67</td><td>Reddit News and *</td><td>60m</td></tr><tr><td>WToxic</td><td>159,571</td><td>63,978</td><td>92</td><td>★</td><td>37m</td></tr></table>

Table 1: Datasets and Distillation Corpus Used in Our Study. . represents the size of a dataset. “Avg. l” represents the average number of tokens in the input sentence. ? represents the unlabeled data released with the original dataset.

Knowledge Distillation Corpora. We manually select a relevant unlabeled corpus C based on the task characteristics and text domain.<sup>4</sup> For example, the IMDB and SST-2 models, which are tasked with classifying the sentiment of movie reviews, are paired with a corpus of unlabeled Amazon product reviews (Ni et al., 2019). TREC, a question classification task, is paired with PAQ (Lewis et al., 2021), a collection of 65 million questions. AG-News, a news classification task, is paired with CC-News corpus (Nagel, 2016). For Civil Comments, a dataset for detecting toxic news comments, we select the News subreddit corpus from ConvoKit (Chang et al., 2020), which is built from a previously existing dataset extracted and obtained by a third party and hosted by pushshift.io. Details of all datasets and corpora are listed in Table 1.

## 3.2 Compared Methods

To comprehensively evaluate and analyze the n-gram student models, we additionally experiment with (1) training a randomly-initialized DAN model with $D _ { t r a i n }$ , without knowledge distillation (“from scratch”); (2) directly fine-tuning generalpurpose compact transformers, e.g., DistilBERT (Sanh et al., 2019), MobileBERT (Sun et al., 2020); (3) using other lightweight architectures for the student model, such as DistilRoBERTa (Sanh et al., 2019), Bi-LSTM (Tang et al., 2019) and Convolution Neural Networks (Chia et al., 2019), in taskspecific distillation setting. We also quote performance from (Tay et al., 2021) when applicable.

## 4 Results and Analysis

## 4.1 Main Results

How well can DANs emulate the performance of the teacher? In Table 2, we present the results on 6 single-sentence classification datasets. Firstly, we find that in 5 out of the 6 datasets, the gap between the teacher and the student model is within 3%. This suggests the power of simple ngram models may be underestimated previously, as they are typically trained from scratch, without modern techniques such as pre-training and knowledge distillation. This also echoes with a series of recent work that questions the necessity of word order information (Sinha et al., 2021) and self-attention (You et al., 2020), in prevalent transformer architectures. Secondly, we observe that knowledge distillation help close more than half the gap between the teacher model and the student model trained from scratch. The effect is more significant with TREC dataset (13% improvement), a 46-way classification problem, whose train set has a small size of 5,452. It is hard to estimate parameters of a large sparse model with merely 5,452 examples; however, supervising it with large-scale corpus and distillation target effectively densified the supervision signals and help address the sparsity issues during model training.

How fast are DANs? We have previously hypothesized that DANs will have superior inference speed due to its simple and sparse architecture. In this section we quantify this advantage by comparing the student model with the RoBERTa-Large teacher model. We also include the baselines listed in §3.2 for a comprehensive comparison. For simplicity, we use BPE tokenizer and re-use the embedding table from RoBERTa-Large for our student Bi-LSTM and CNN model. We use 2-layer Bi-LSTM with hidden dimension of 4, 64, 256 and 512. For the CNN model, we use one 1D convolution layer with hidden dimension of 128 and context window of 7.

We provide speed comparison across all datasets in Table 3. We provide more fine-grained comparison on IMDB dataset in Table 4 and Fig. 1. DAN achieves competitive performance and the fastest inference efficiency among all different student model architectures. The speed-up differs across datasets, ranges from 4x to 1091x. It is most significant on Civil Comments (1091x), Wiki Toxic (668x) and IMDB dataset (607x), as they have longer input sequences, and the complexity grows quadratically with sequence length in transformer models. Moreover, as shown in Table 4, DAN has an acceptable CPU inference speed, which greatly reduce the hardware cost for inference. We believe all these characteristics makes student DAN model as an ideal option for production or real-time use on single-sentence classification tasks.

<table><tr><td>Model</td><td>IMDB</td><td>SST-2</td><td>TREC</td><td>AGNews</td><td>CCom</td><td>WToxic</td><td>QQP</td></tr><tr><td>DAN (from scratch)</td><td>88.3</td><td>79.5</td><td>78.4</td><td>91.1</td><td>95.7</td><td>92.2</td><td>82.0</td></tr><tr><td>DAN (KD)†</td><td>92.0</td><td>87.0</td><td>91.8</td><td>90.0</td><td>96.2</td><td>93.9</td><td>63.2</td></tr><tr><td>DAN (KD)</td><td>93.2</td><td>86.4</td><td>91.8</td><td>90.6</td><td>96.3</td><td>94.0</td><td>84.1</td></tr><tr><td>DAN (KD+FT)</td><td>93.5</td><td>88.5</td><td>92.6</td><td>93.0</td><td>96.3</td><td>92.5</td><td>84.2</td></tr><tr><td>DistilBERT (Sanh et al., 2019)</td><td>92.2</td><td>90.8</td><td>92.8</td><td>94.5</td><td>96.9</td><td>93.1</td><td>89.4</td></tr><tr><td>MobileBERT (Sun et al., 2020)</td><td>93.6</td><td>90.9</td><td>91.0</td><td>94.6</td><td>97.0</td><td>93.5</td><td>90.5</td></tr><tr><td>Transformer-Base (Tay et al., 2021)</td><td>94.2</td><td>92.1</td><td>93.6</td><td>93.5</td><td>$</td><td>91.5</td><td>-</td></tr><tr><td>ConvNet (Tay et al., 2021)</td><td>93.9</td><td>92.2</td><td>94.2</td><td>93.9</td><td>_#</td><td>93.8</td><td></td></tr><tr><td>RoBERTa-Large (Liu et al., 2019)</td><td>96.3</td><td>96.2</td><td>94.8</td><td>95.4</td><td>96.3</td><td>94.1</td><td>92.1</td></tr></table>

Table 2: Performance Comparison on 6 Single-sentence Tasks and 1 Sentence-pair Task. We report accuracy for all datasets. For single-sentence tasks, the gap between the teacher model (RoBERTa-Large) and the n-gram based student model (DAN(KD)/DAN(KD+FT)) is within 3% in most cases. Also, we observe that knowledge distillation help close more than half the gap between the teacher model and the n-gram model trained from scratch. †Knowledge distillation is performed without task data $( D _ { t r a i n } ) .$ , assuming that the task data is private (see §4.3). ‡The dataset we obtain from public sources differs from the one in Tay et al. (2021).
<table><tr><td>Model</td><td>IMDB</td><td>SST-2</td><td>TREC</td><td>AGNews</td><td>CCom</td><td>WToxic</td><td>QQP</td></tr><tr><td>RoBERTa-Large</td><td>29 (1x)</td><td>298 (1x)</td><td>549 (1x)</td><td>147 (1x)</td><td>35 (1x)</td><td>72 (1x)</td><td>240 (1x)</td></tr><tr><td>DistilBERT</td><td>176 (6x)</td><td>1055 (4x)</td><td>930 (2x)</td><td>740 (5x)</td><td>188 (5x)</td><td>426 (6x)</td><td>1201 (5x)</td></tr><tr><td>MobileBERT</td><td>158 (5x)</td><td>736 (3x)</td><td>402 (1x)</td><td>751 (5x)</td><td>187 (5x)</td><td>400 (6x)</td><td>943 (4x)</td></tr><tr><td>DANs</td><td>17557 (607x)</td><td>3020 (10x)</td><td>2236 (4x)</td><td>24084 (164x)</td><td>38024 (1091x)</td><td>48133 (668x)</td><td>35708 (149x)</td></tr></table>

Table 3: Inference Speed Comparison (Unit: samples per second). DANs greatly improves inference speed, with the speed-up ranging from 4x to 1091x. Speed-up is most significant with classification tasks with long sequences as input, e.g., Civil Comment, Wiki Toxic, and IMDB.

<table><tr><td rowspan="2"></td><td rowspan="2">Parameter Count Total/Sparse/Dense</td><td colspan="3">IMDB</td></tr><tr><td>Acc.</td><td>GPU Speed</td><td>CPU Speed</td></tr><tr><td>RoBERTa-Large</td><td>355M/51M/304M</td><td>96.3</td><td>29 (1x)</td><td>1 (1x)</td></tr><tr><td>DistilBERT</td><td>66M/23M/43M</td><td>92.2</td><td>176 (6x)</td><td>11 (8x)</td></tr><tr><td>MobileBERT</td><td>25M/4M/21M</td><td>93.6</td><td>158 (5x)</td><td>8 (6x)</td></tr><tr><td>*DistilRoBERTa</td><td>83M/39M/44M</td><td>95.9</td><td>176 (6x)</td><td>8 (6x)</td></tr><tr><td>*LSTM (2l-512d)</td><td>62M/51M/11M</td><td>95.9</td><td>362 (12x)</td><td>31 (22x)</td></tr><tr><td>*LSTM (21-256d)</td><td>56M/51M/5M</td><td>95.8</td><td>665 (23x)</td><td>52 (37x)</td></tr><tr><td>*LSTM (21-64d)</td><td>53M/51M/2M</td><td>95.3</td><td>818 (28x)</td><td>101 (73x)</td></tr><tr><td>*LSTM (21-4d)</td><td>52M/51M/&lt;1M</td><td>93.1</td><td>813 (28x)</td><td>146 (105x)</td></tr><tr><td>*CNN (11-256d)</td><td>53M/51M/2M</td><td>89.2</td><td>3411 (109x)</td><td>251 (181x)</td></tr><tr><td>*DAN (this work)</td><td>1001M/1000M/1M</td><td>93.5</td><td>17558 (607x)</td><td>923 (663x)</td></tr></table>

Table 4: Detailed Inference Speed Comparison on IMDB. DANs achieves better accuracy and inference speed compared to other lightweight architectures such as LSTMs and CNNs. Moreover, DANs achieves acceptable inference speed on CPUs. ? indicates the model is trained with task-specific distillation; no ? indicates the model is trained with direct fine-tuning.

Simplest is the best: Exploring different design choices for DAN. We try several modifications to our current experiment pipeline, including (1) replace average pooling with max pooling, attentive pooling, or taking sum in the DAN model; (2) pre-compute a n-gram representation by feeding the raw n-gram text to a RoBERTa-Large model, and using the representations to initialize the embedding table of the student model; (3) attach more dense layers in the DAN; (4) use even larger student models by leveraging parallel training across multiple GPUs. More details about these variations are in Appendix B.1. We experiment with IMDB dataset and list the performance in Table 5. In general, we do not observe significant performance improvements brought by these variations. Thus, we keep the simplest design of DAN for all other experiments.

<table><tr><td>Variations</td><td>Acc.</td><td>Variations</td><td>Acc.</td></tr><tr><td>1. Pooling Methods</td><td></td><td>2. Dense Layers</td><td></td></tr><tr><td>Mean Pooling (*)</td><td>93.2</td><td> $1 0 0 0 \to 1 0 0 0 \to 2 \left( \star \right)$ </td><td>93.2</td></tr><tr><td>Max Pooling</td><td>91.8</td><td> $1 0 0 0 \to 1 0 0 0 \to 2 5 6 \to 2$ </td><td>93.1</td></tr><tr><td>Attentive Pooling</td><td>93.0</td><td> $1 0 0 0 \to 1 0 0 0 \to 2 5 6 \to 6 4 \to 2$ </td><td>93.0</td></tr><tr><td>Sum</td><td>92.9</td><td></td><td></td></tr><tr><td>3. Embedding Initialization</td><td></td><td>4. Parallel Training</td><td></td></tr><tr><td>Without initialization (*)</td><td>93.2</td><td>1 GPU, param. 1b (*)</td><td>93.2</td></tr><tr><td>With initialization</td><td>93.2</td><td> $2 \mathrm { G P U s } , \mathrm { p a r a m } . 2 \mathrm { b }$ </td><td>93.1</td></tr></table>

Table 5: Variations made to the student model and the performance on IMDB. ? represents the design we adopt in our main experiments.

![](images/ead7bbbf82eb22ffe163b00f810b0d0fcb308e08d485928135f3b913ea590363.jpg)  
Figure 4: Trade-off between the vocabulary size and the embedding dimension. Given a fixed parameter budget, empirical results suggest that a larger embedding dimension and a smaller vocabulary size should be selected.

## 4.2 Controlling the Parameter Budget

Given a fixed parameter budget, how to allocate it wisely to achieve optimal performance? We discuss this question in two scenarios: the users wish to control the parameter budget (1) during knowledge distillation (KD), or (2) during inference.

During KD: Trade-off between vocabulary size and embedding dimension. We explore how the configuration of vocabulary size and embedding dimension influence the student model performance. We train student models on the IMDB dataset with 19 configurations, and show the results graphically in Figure 4. Detailed results are deferred in Table 8 in Appendix C. All else being equal, having more parameters in the student model is beneficial to the performance. For a fixed parameter budget, higher accuracy was achieved by increasing the embedding dimension and making a corresponding reduction in the vocabulary size. Our best performing model has $\vert V \vert = 1 , 0 0 0 , 0 0 0$ and $d _ { e } = 1 , 0 0 0$ . We keep this configuration for the main experiments in previous sections.

During inference: Reduce the model size with n-gram pruning. The model size of DANs is flexible even after training, by excluding the least frequent n-grams in the vocabulary. We test this idea on IMDB and AGNews dataset and plot the performance in Fig. 5. We try two ways to estimate n-gram frequency: (1) using distillation corpus C and the training set $D _ { t r a i n } ; ( 2 )$ using $D _ { t r a i n }$ only. We observe that: (1) n-gram frequencies estimated on $D _ { t r a i n }$ are more reliable, as $D _ { d e v }$ has a n-gram distribution more similar to $D _ { t r a i n }$ compared to $C + D _ { t r a i n } ;$ (2) DANs maintain decent accuracy (>90%) even when the model size is cut to 3% of its original size. In this case, users of DANs can customize the model flexibly based on their needs and available computational resources.

![](images/1ecd730c98bb93b4ed79b935f99cea31b6808618416e224f5264a23278a7344f.jpg)

![](images/8c5ba2ce40075d623e1eea672a39895753fd1ba05da6e1c986bbd5d3b1fefa35.jpg)  
Figure 5: Post-hoc pruning according to n-gram frequency. We disable the least frequent n-grams during inference to further reduce model size. When the ngram frequencies are estimated appropriately, DANs maintain decent performance (acc.>90%) even when model is 3% of its original size. $C + D _ { t r a i n } / D _ { t r a i n }$ represent different ways to estimate n-gram frequencies.

## 4.3 Privacy-preserving Settings

NLP datasets sometimes involve user generated text or sensitive information; therefore, data privacy can be a concern when training and deploying models with certain NLP datasets. In this section, we modify our experiment setting to a practical and privacy-preserving one. We assume the user has access to a public teacher model that is trained on private train dataset $( D _ { t r a i n } )$ , but does not has access to $D _ { t r a i n }$ itself. This is realistic nowadays with the growth of public model hubs such as TensorFlow Hub<sup>5</sup> and Hugging Face $\mathrm { M o d e l s } ^ { 6 }$ . After downloading the model, the user may wish to deploy a faster version of this model, or adapt the model to the user’s own application domain.

Knowledge Distillation without $D _ { t r a i n } .$ To simulate the privacy-preserving setting, we remove $D _ { t r a i n }$ from the knowledge distillation stage in our experiment pipeline and only use the unlabeled corpus C. We use $\mathrm { { ^ { * } D A N } ( K D ) ^ { \dagger } { } ^ { * } }$ to denote this model in Table 2. By comparing $\mathrm { \Sigma ^ { * } D A N } \left( \mathrm { K D } \right) ^ { , }$ and “DAN $( \mathrm { K D } ) ^ { \dag \dag , \dag }$ , we found that the performance difference brought by task specific data $D _ { t r a i n }$ is small for all single-sentence tasks, with the largest gap being 1.2% on IMDB dataset. This suggests that the proposed pipeline is still useful in the absence of task-specific data.

<table><tr><td>Source Target</td><td>IMDB SST-2</td><td>SST-2 IMDB</td></tr><tr><td>DAN (from scratch, tar)</td><td>79.5</td><td>88.3</td></tr><tr><td>(1) RoBERTa-Large (src)</td><td>90.0</td><td>94.1</td></tr><tr><td>(2) DAN (KD)</td><td>81.9</td><td>92.0</td></tr><tr><td>(3) DAN (KD+FT)</td><td>88.4</td><td>93.0</td></tr><tr><td>(4) DAN (KD+FT w. re-init.)</td><td>86.7</td><td>92.8</td></tr><tr><td>RoBERTa-Large (tar)</td><td>96.2</td><td>96.3</td></tr></table>

Table 6: Domain Generalization and Adaptation Results. (1) We take the teacher model trained on the source dataset and evluate it on the target dataset. (2) We obtain the student model “DAN $\left( \mathrm { K D } \right) ^ { \ast }$ with unlabeled corpus C and knowledge distillation. (3) We further fine-tune the student model on the target dataset to obtain “DAN (KD+FT)”. (4) The classification head is re-initialized before further fine-tuning.

Domain Generalization and Adaptation. We select the two sentiment analysis tasks: IMDB and SST-2, and further explore the domain generalization/adaptation setting. Specifically, during stage 1 of our three-stage pipeline (§2.4), we fine-tune the RoBERTa-Large model on a source dataset; during stage 2, we apply knowledge distillation with unlabeled corpus C only and get the student model; during stage 3, we further fine-tune the student model on the target dataset. The last step is optional and serves to simulate the situation where the user collects additional data for domain adaptation. We list the results in Table 6. With weakened assumptions about the teacher model and distillation supervision, we still have observations similar to those in our main experiments (§4.1): Performance of the final student model is significantly improved compared to DANs trained from scratch.

## 4.4 Limitations and Discussions

Extension to sentence-pair tasks. So far we have limited the scope to single-sentence classification tasks. We consider extending our sparse distillation framework to a sentence-pair task, Quora Question Pair $( \mathrm { Q Q P } ) ^ { 7 }$ , which aims to identify duplicated questions. We create pseudo sentence-pair data for knowledge distillation by randomly sampling 10 million question pairs from PAQ. To better model the relation between a pair of sentence, we modify DANs by introducing a concatenatecompare operator (Wang and Jiang, 2017), following the practice in (Tang et al., 2019). More specifically, the two input sentences, $\mathbf { x } _ { 1 }$ and $\mathbf { x } _ { 2 } ,$ go through the embedding layer and average pooling independently, resulting in two sentence representations, $\mathbf { h } _ { 1 }$ and $\mathbf { h } _ { 2 }$ . We then apply the concatenatecompare operator, i.e., $f ( \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } ) = [ \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } , \mathbf { h } _ { 1 }$ O h<sub>2</sub>, $| \mathbf { h } _ { 1 } - \mathbf { h } _ { 2 } | ]$ , where represents element-wise multiplication. Finally, $f ( \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } )$ go through two fully connected layers for classification, the same as DANs for single-sentence tasks.

![](images/55806bfa4b64799810468c22a647eb284dbafd32456bdb39b5b58d0afd6a1cac.jpg)

![](images/f6d32440931eba33975ef6c77ddacff0830d9e1dae883e34757a007ca2378f41.jpg)  
Figure 6: Analysis on N-gram Coverage. Left: Relation between n-gram coverage and cross-entropy loss w.r.t. ground truth labels. Each blue line represents the median loss in that n-gram coverage bucket. Right: Distribution of n-gram coverage.

The results on QQP dataset is listed in the rightmost column in Table 2. Firstly, knowledge distillation still helps close the gap between RoBERTa-Large and DANs trained from scratch (2% improvement) and leads to a decent accruacy of 84.2%; however the benefit brought by KD is not as strong as with single-sentence tasks. Secondly, the performance of $\mathrm { D A N } ( \mathrm { K D } ) ^ { \dagger }$ (i.e., without access to $D _ { t r a i n }$ during KD) is much worse than the performance of DAN(KD). We hypothesize that this is due to the quality and distribution of knowledge distillation corpus. We randomly sample questions pairs as the knowledge distillation examples, which may not carry sufficient supervision signals – more than 99% of them are negative (“not duplicated”) examples. Creating more suitable distillation corpus for sentence-pair tasks is beyond the scope of our work, and we leave this as future work.

Impact of N-gram Coverage. One potential drawback of n-grams (based on white-space tokenization) is that they cannot directly handle outof-vocabulary words, while WordPiece/BPE tokenization together with contextualization can better handle this issue. In Fig. 6, we quantify the influence of n-gram coverage on IMDB dev set. Here, n-gram coverage for an input sentence is defined as $| G \cap V | / | V |$ , where G represents the set of n-grams in the sentence and V is the n-gram vocabulary (§2.5). We first group the instances into buckets of n-gram coverage (e.g., [40%, 50%), [50%, 60%)) and then compute the statistics of cross-entropy loss in each bucket. We observe that performance is worse on sentences with more out-of-vocabulary words. Future work may build upon this observation and improve DANs performance by addressing out-of-vocabulary words. For example, BPE-based n-grams may be used for creating the vocabulary.

<table><tr><td>Teacher</td><td>Student</td><td>Label</td><td>Sentence</td></tr><tr><td>Negative</td><td>Positive</td><td>Negative</td><td>I really wanted to love this film. . . .</td></tr><tr><td>Negative</td><td>Positive</td><td>Negative</td><td>This movie is a great movie ONLY if you need something to sit and laugh at the stupidity of it. . ..</td></tr><tr><td>Positive</td><td>Negative</td><td>Positive</td><td>.. . They are such bad actors and it made this movie so much funnier to watch. . ..</td></tr></table>

Table 7: Case study on IMDB predictions. In these cases, the model can only make the correct predictions by understanding long contexts. Performance of DAN models are still limited as they only look at local n-grams.

Case study: What are DANs still not capable of? We take a closer look at the predictions made by our DAN model (student) and the RoBERTa-Large model (teacher) on the IMDB dataset. We list several representative cases in Table 7. These cases typically require understating of complex language phenomena, such as irony, conditional clauses, and slang. In addition, these phenomena typically occur in contexts longer than 4 words, which DANs are not capable of modeling by design. For example, “bad actors” can mean “good actors” based on the later context “much funnier to watch”. We conclude that sparse distillation is not suitable to cases where modeling complex language phenomena has a higher priority than improving inference speed.

Understanding the performance gaps. Tay et al. (2021) advocate that architectural advances should not be conflated with pre-training. Our experiments further support this claim, if we consider knowledge distillation as a “substitute” for pre-training that provides the student model with stronger inductive biases, and interpret the remaining teacher-student performance gap as the difference brought by architectural advances. On the other hand, we believe the power of DANs are previously undermined due to the challenges in optimizing large sparse models with limited supervision. Our experiments show that knowledge distillation effectively densify the supervision and greatly improve the performance of DANs.

Additional Analysis and Specifications. Due to space limit, we leave some additional analysis and specifications in Appendix C. We discuss tokenization speed (Table 9) and impact of n in n-grams (Table 10). We provide more detailed speed comparison in Table 12, model storage and memory usage information in Table 11. We provide fine-grained n-gram coverage information in Table 13.

## 5 Related Work

Efficient Transformers. Recent work attempts to improve computation or memory efficiency of transformer models mainly from the following perspectives: (1) Proposing efficient architectures or self-attention variants, e.g., Linformer (Wang et al., 2020a), Longformer (Beltagy et al., 2020). Tay et al. (2020) provide a detailed survey along this line of work. (2) Model compression using knowledge distillation, e.g., DistillBERT (Sanh et al., 2019), MobileBERT (Sun et al., 2020), MiniLM (Wang et al., 2020b). These compressed models are typically task-agnostic and general-purpose, while in this work we focus on task-specific knowledge distillation. (3) Weight quantization and pruning, e.g., Gordon et al. (2020); Li et al. (2020); Kundu and Sundaresan (2021).

Task-specific Knowledge Distillation in NLP. Researchers explored distilling a fine-tuned transformer into the following lightweight architectures, including smaller transformers (Turc et al., 2019; Jiao et al., 2020), LSTMs (Tang et al., 2019; Adhikari et al., 2020) and CNNs (Chia et al., 2019). Wasserblat et al. (2020) distill BERT into an architecture similar to DAN, however they restrict the model to only take unigrams (thus having small student models), and adopt a non-standard lowresource setting. To summarize, existing work typically focuses on reducing both number of parameter and the amount of computation, while in the paper we study an under-explored area in the design space, where the amount of computation is reduced by training a larger student model.

Reducing Contextualized Representations to Static Embeddings. Related to our work, Ethayarajh (2019) and Bommasani et al. (2020) show how static word embeddings can be computed from BERT-style transformer models. Ethayarajh (2019) suggest that less than 5% of the variance in a word’s contextualized representation can be explained by a static embedding, justifying the necessity of contextualized representation. Bommasani et al. (2020) found that static embeddings obtained from BERT outperforms Word2Vec and GloVe in intrinsic evaluation. These two papers mainly focus on post-hoc interpretation of pre-trained transformer models using static embeddings. In our work we opt to use knowledge distillation to learn n-gram embeddings. Meanwhile we acknowledge that the technique in Ethayarajh (2019) and Bommasani et al. (2020) could be used as an alternative method to convert transformer models to fast text classifiers.

Sparse Architectures. In our work we aggressively cut off computation cost by compensating it with more parameters in the student model. Alternatively, one could fix the computational cost at the same level as a transformer while greatly expanding the parameter count, as explored in the Switch Transformer (Fedus et al., 2021). Both their work and ours agree in the conclusion that scaling up parameter count allows the model to memorize additional useful information.

## 6 Conclusions & Future Work

We investigated a new way of using knowledge distillation to produce a faster student model by reversing the standard practice of having the student be smaller than the teacher and instead allowed the student to have a large table of sparsely-activated embeddings. This enabled the student model to essentially memorize task-related information that if an alternate architecture were used would have had to be computed. We tested this method on six single-sentence classification tasks with models that were up to 1 billion parameters in size, approximately 3x as big as the RoBERTa-Large teacher model, and found that the student model was blazing fast and performed favorably.

We hope that our work can lead to further exploration of sparse architectures in knowledge distillation. There are multiple directions for future work, including extending the DAN architecture to better support tasks with long range dependencies like natural language inference or multiple inputs like text similarity. Additionally, more work is needed to test the idea on non-English languages where n-gram statistics can be different from English.

## Acknowledgments

We would like to thank Robin Jia, Christina Sauper, and USC INK Lab members for the insightful discussions. We also thank anonymous reviewers for their valuable feedback. Qinyuan Ye and Xiang Ren are supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via Contract No. 2019-19051600007, the DARPA MCS program under Contract No. N660011924033, the Defense Advanced Research Projects Agency with award W911NF-19-20271, NSF IIS 2048211, NSF SMA 1829268.

## References

Ashutosh Adhikari, Achyudh Ram, Raphael Tang, William L. Hamilton, and Jimmy Lin. 2020. Exploring the limits of simple learners in knowledge distillation for document classification with DocBERT. In 5th Workshop on Representation Learning for NLP, pages 72–77, Online. ACL.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. ArXiv preprint, abs/2004.05150.

Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 4349–4357.

Rishi Bommasani, Kelly Davis, and Claire Cardie. 2020. Interpreting Pretrained Contextualized Representations via Reductions to Static Embeddings. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4758– 4781, Online. Association for Computational Linguistics.

Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion proceedings of the 2019 world wide web conference, pages 491–500.

Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, and Jeffrey Dean. 2007. Large language models in machine translation. In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), pages 858–867, Prague, Czech Republic. Association for Computational Linguistics.

Christian Buck, Kenneth Heafield, and Bas van Ooyen. 2014. N-gram counts and language models from

the Common Crawl. In Proceedings ofthe Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 3579–3584, Reykjavik, Iceland. European Language Resources Association (ELRA).

Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, and Cristian Danescu-Niculescu-Mizil. 2020. ConvoKit: A toolkit for the analysis of conversations. In Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 57–60, 1st virtual meeting. Association for Computational Linguistics.

Yew Ken Chia, Sam Witteveen, and Martin Andrews. 2019. Transformer to CNN: Label-scarce distillation for efficient text classification. ArXiv preprint, abs/1909.03508.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Kawin Ethayarajh. 2019. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 55–65, Hong Kong, China. Association for Computational Linguistics.

William Fedus, Barret Zoph, and Noam Shazeer. 2021. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv preprint, abs/2101.03961.

Mitchell Gordon, Kevin Duh, and Nicholas Andrews. 2020. Compressing BERT: Studying the effects of weight pruning on transfer learning. In Proceedings of the 5th Workshop on Representation Learning for NLP, pages 143–155, Online. Association for Computational Linguistics.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. ArXiv preprint, abs/1503.02531.

Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, and Hal Daumé III. 2015. Deep unordered composition rivals syntactic methods for text classification. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1681–1691, Beijing, China. Association for Computational Linguistics.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020. TinyBERT: Distilling BERT for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4163–4174, Online. Association for Computational Linguistics.

Brendan Kennedy, Xisen Jin, Aida Mostafazadeh Davani, Morteza Dehghani, and Xiang Ren. 2020. Contextualizing hate speech classifiers with post-hoc explanation. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5435–5442, Online. Association for Computational Linguistics.

Souvik Kundu and Sairam Sundaresan. 2021. Attentionlite: Towards efficient self-attention models for vision. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2225–2229.

Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler, Aleksandra Piktus, Pontus Stenetorp, and Sebastian Riedel. 2021. PAQ: 65 million probably-asked questions and what you can do with them. Transactions of the Association for Computational Linguistics, 9:1098–1115.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Clément Delangue, Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, François Lagunas, Alexander Rush, and Thomas Wolf. 2021. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Xin Li and Dan Roth. 2002. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics.

Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and Joseph E Gonzalez. 2020. Train large, then compress: Rethinking model size for efficient training and inference of transformers. ArXiv preprint, abs/2002.11794.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. ArXiv preprint, abs/1907.11692.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.

2011. Learning word vectors for sentiment analysis. In ACL, pages 142–150, Portland, Oregon, USA.

Sebastian Nagel. 2016. Cc-news. URL: http://web. archive. org/save/http://commoncrawl. org/2016/10/newsdatasetavailable.

Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 188–197, Hong Kong, China. Association for Computational Linguistics.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics (Demonstrations), pages 48–53, Minneapolis, Minnesota. Association for Computational Linguistics.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv preprint, abs/1910.01108.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053.

Koustuv Sinha, Robin Jia, Dieuwke Hupkes, Joelle Pineau, Adina Williams, and Douwe Kiela. 2021. Masked language modeling and the distributional hypothesis: Order word matters pre-training for little. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2888–2913, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

Xinying Song, Alex Salcianu, Yang Song, Dave Dopson, and Denny Zhou. 2021. Fast WordPiece tokenization. In Proceedings ofthe 2021 Conference on

Empirical Methods in Natural Language Processing, pages 2089–2103, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. 2020. MobileBERT: a compact task-agnostic BERT for resource-limited devices. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2158–2170, Online. Association for Computational Linguistics.

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin. 2019. Distilling taskspecific knowledge from BERT into simple neural networks. ArXiv preprint, abs/1903.12136.

Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. 2020. Efficient transformers: A survey. ArXiv preprint, abs/2009.06732.

Yi Tay, Mostafa Dehghani, Jai Prakash Gupta, Vamsi Aribandi, Dara Bahri, Zhen Qin, and Donald Metzler. 2021. Are pretrained convolutions better than pretrained transformers? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4349–4359, Online. Association for Computational Linguistics.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Well-read students learn better: On the importance of pre-training compact models. ArXiv preprint, abs/1908.08962.

Shuohang Wang and Jing Jiang. 2017. A compareaggregate model for matching text sequences. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. 2020a. Linformer: Self-attention with linear complexity. ArXiv preprint, abs/2006.04768.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020b. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Moshe Wasserblat, Oren Pereg, and Peter Izsak. 2020. Exploring the boundaries of low-resource BERT distillation. In Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing, pages 35–40, Online. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,

Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Ellery Wulczyn, Nithum Thain, and Lucas Dixon. 2017. Ex machina: Personal attacks seen at scale. In WWWW, WWW ’17, page 1391–1399, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

Weiqiu You, Simeng Sun, and Mohit Iyyer. 2020. Hard-coded Gaussian attention for neural machine translation. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 7689–7700, Online. Association for Computational Linguistics.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7- 12, 2015, Montreal, Quebec, Canada, pages 649– 657.

Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D. Manning. 2017. Positionaware attention and supervised data improve slot filling. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 35–45, Copenhagen, Denmark. Association for Computational Linguistics.

## A Reproducibility

## A.1 Datasets

Datasets and corpora used, and their specifications are previously listed in Table 1. Here we provide links to download these data.

• IMDB: https://ai.stanford.edu/\~ama as/data/sentiment/

• SST-2: https://huggingface.co/dataset s/glue

• AGNews: https://huggingface.co/dat asets/ag\_news

• TREC: https://huggingface.co/dataset s/trec

• CivilComments: https://huggingface.co /datasets/civil\_comments

• WikiToxic: https://www.tensorflow.org /datasets/catalog/wikipedia\_toxicit y\_subtypes and https://meta.m.wikimed ia.org/wiki/Research:Detox/Data\_Rel ease

• QQP: https://huggingface.co/dataset s/glue

• Amazon Reviews: https://nijianmo.git hub.io/amazon/index.html

• PAQ: https://github.com/facebookres earch/PAQ

• Reddit News: https://zissou.infosci.c ornell.edu/convokit/datasets/subredd it-corpus/corpus-zipped/newreddits \_nsfw\~-\~news/news.corpus.zip

QQP dataset has 363,846 training instances and 40,430 development instances. The average input length is 13 tokens. We thank huggingface dataset team (Lhoest et al., 2021) for providing easy access to these datasets.

Licensing. For WikiToxic, the dataset is licensed under CC0, with the underlying comment text being governed by Wikipedia’s CC-SA-3.0. The PAQ QA-pairs and metadata is licensed under CC-BY-SA. The licensing information of other datasets are unknown to us.

## A.2 Implementation Details

N-gram pre-processing are implemented with scikit-learn (Pedregosa et al., 2011). DistilBERT (Sanh et al., 2019) and MobileBERT baselines are implemented in huggingface transformers (Wolf et al., 2020). RoBERTa-Large, BiLSTM, CNN, and DAN experiments are implemented with fairseq (Ott et al., 2019).

## A.3 Hyperparameters

For fine-tuning in stage 1, we select batch size from {16, 32} and learning rate from 1e-5, 2e-5, 5e-5 following the recommendations in (Liu et al., 2019). We train the model for 10 epochs on $D _ { t r a i n }$ . For knowledge distillation in stage 2, we set the batch size to be 2048, learning rate to be 5e-4, and total number of updates to be 1,000,000, as they work well in our preliminary experiments. The embedding table is randomly initialized and the embedding dimension $d _ { e }$ is set to 1,000, unless specified otherwise. For further fine-tuning in stage 3, we set the batch size to be 32 and select the learning rate from 3e-4, 1e-4, 3e-5 . We train the model for 10 epochs on $D _ { t r a i n }$ . For all training procedures, we validate the model at the end of each epoch in the case of fine-tuning, or every 100,000 steps in the case of knowledge distillation. We save the best checkpoint based on dev accuracy. Due to the analysis nature of this work and the scale of experiments, performance are computed using dev set and based on one single run.

## A.4 Hardware

Model Training. Except for the parallel training attempt in Table 5, all experiments are done on one single GPU. We train DAN models on either A100 40GB PCIe or Quadro RTX 8000 depending on availability. Knowledge distillation (Stage 2) with 1,000,000 updates typically finishes within 36 hours.

Inference Speed Tests. All inference speed tests are done with the batch size of 32. GPU inference is performed with one Quadro RTX 8000 GPU, and CPU inference is performed with 56 Intel Xeon CPU E5-2690 v4 CPUs.

## B Additional Details

## B.1 DAN Variations

Due to space limits we have omitted the details for the DAN variations we studied in §4.1. We introduce these variations in the following.

Attentive Pooling. We consider adding attentive pooling to the DAN model to capture more complicated relations in the input. Our attention layer is modified from the one in (Zhang et al., 2017). we use the representation h after mean pooling as query, and each n-gram embedding $\mathbf { e } _ { i } = \operatorname { E m b } ( g _ { i } )$ as key. More specifically, for each n-gram g<sub>i</sub> we calculate an attention weight $a _ { i }$ as:

$$
u _ { i } = \mathbf { v } ^ { \top } \operatorname { t a n h } ( \mathbf { W } _ { g } \mathbf { e } _ { i } + \mathbf { W } _ { h } \mathbf { h } )\tag{1}
$$

$$
a _ { i } = \frac { \exp ( u _ { i } ) } { \sum _ { j = 1 } ^ { n } \exp ( u _ { j } ) }\tag{2}
$$

Here $\mathbf { W } _ { g } , \mathbf { W } _ { h } \in \mathbb { R } ^ { d _ { e } \times d _ { a } }$ and $\mathbf { v } \in \mathbb { R } ^ { d _ { a } }$ are learnable parameters. $d _ { e }$ is the dimension of the embedding table, and $d _ { a }$ is the size of the attention layer. To maintain an acceptable training speed, for attentive pooling, we use a batch size of 512 during knowledge distillation.

Parallel Training We try further scaling up the student model by splitting the gigantic embedding table to different GPUs and enable parallel training, as implemented in Megatron-LM (Shoeybi et al., 2019). We train a 2-billion parameter model in parallel on two GPUs. The embedding dimension is set to be 2, 000 in total, and each GPU handles an embedding table of hidden dimension 1, 000. The vocabulary size is 1 million.

## B.2 Comments on SparseAdam

SparseAdam is a modified version of the regular Adam optimizer. For Adam, the first and second moment for each parameter is updated at every step. This can be costly, especially for DAN, as most parameters in the embedding layer are not used during the forward pass. SparseAdam computes gradients and updates the moments only for parameters used in the forward pass.

## C Additional Results

Speed Comparison. Table 12 is an extended version of Table 4 which contains inference speed comparison on IMDB and SST-2 dataset, in three different settings (GPU-FP32, GPU-FP16, CPU-FP32). Our major conclusion remains the same: DANs achieve excellent inference speed in various settings.

Vocabulary Size vs. Embedding Dimension Trade-off. Table 8 contains original results that were visualized in Fig. 4.

<table><tr><td></td><td>Param. 500m</td><td></td><td>Param. 1b</td><td>Param. 2b</td><td></td></tr><tr><td>|V||</td><td> $d _ { e }$ </td><td>Acc</td><td> $d _ { e }$  Acc.</td><td> $d _ { e }$ </td><td>Acc.</td></tr><tr><td>1m</td><td>500</td><td>93.0</td><td>1000 93.2</td><td></td><td></td></tr><tr><td>2m</td><td>250</td><td>92.8</td><td>500</td><td>93.0 900</td><td>93.1</td></tr><tr><td>4m</td><td>125</td><td>92.7</td><td>250</td><td>92.9 500</td><td>93.1</td></tr><tr><td>5m</td><td>100</td><td>92.6</td><td>200</td><td>92.9 400</td><td>93.1</td></tr><tr><td>10m</td><td>50</td><td>92.3</td><td>100</td><td>92.5</td><td>200 92.9</td></tr><tr><td>20m</td><td>25</td><td>92.0</td><td>50</td><td>92.2</td><td>100 92.7</td></tr><tr><td>40m</td><td>一</td><td>一</td><td>25</td><td>92</td><td>50 92.4</td></tr></table>

Table 8: IMDB dev accuracy with different configurations of vocabulary size $( | V | )$ and embedding table dimension $( d _ { e } )$ . Performance grows with larger embedding tables, and the best performing model has V = 1m and d<sub>e</sub> = 1, 000.

N-gram Coverage Statistics. In our work, we opt to determine the n-gram vocabulary with the training set $D _ { t r a i n }$ and the corpus C, by selecting the top 1 million n-grams according to frequency.

N-gram range is set to be within 1 to 4. For reference, we list statistics about the n-gram vocabulary in Table 13. It is possible that adjustments to this pre-processing step (e.g., up-weighting n-grams in $D _ { t r a i n }$ and down-weighting n-grams in C) will further improve performance, however we stop further investigation.

Tokenization Speed. The speed comparison in our work does not take pre-processing process into account. When the inference speed is at millisecond level (e.g., with our DAN model), preprocessing time can become non-negligible. For reference, in Table 9 we report the tokenization time on the 25,000 training instances in the IMDB dataset with (1) n-gram tokenization (used by DAN, implemented with scikit-learn); (2) BPE tokenization (used by RoBERTa/DistilRoBERTa, implemented with fairseq); (3) WordPiece tokenization (used by DistilBERT, implemented with huggingface transformers).

<table><tr><td>Tokenization Method</td><td>Time</td><td>Complexity</td></tr><tr><td>BPE</td><td>26.46</td><td>O(n lg n) or O(|V|n) (Song et al., 2021)</td></tr><tr><td>WordPiece</td><td>20.60</td><td>O(n2) or O(mn) (Song et al., 2021)</td></tr><tr><td>N-gram</td><td>16.45</td><td>O(n)</td></tr></table>

Table 9: Comparison of tokenization speed and complexity. Time is computed for tokenization the train set of IMDB dataset (25,000 instances) with one single worker. Time is averaged across 5 runs. n represents input length.

First of all, by setting the number of workers to be equal to the batch size (32) we use in the speed test, the tokenization speed will be 48632 instances/sec (=25000/16.45\*32), which is roughly 3x faster than the inference speed. Tokenization speed is non-negligible in this case. Still, the main conclusion from the speed comparison remains the same: DANs are typically 10x-100x faster than the compared models.

Secondly, DAN models still have better tokenization speed than transformer models that use BPE/WordPiece tokenization. This is because our DAN model computes n-grams based on whitespace tokenization, which can be done in linear time when the n-gram to id mapping is implemented with a hashmap, i.e., O(n) where n is the input length. BPE/WordPiece tokenization has higher complexity according to Song et al. (2021).

We would also like to emphasize that this part is also highly dependent on the design choice and implementation. For example, the user could implement a DAN model with BPE tokenzation. The choice and optimization of tokenization is beyond the scope of this work.

Impact of n in n-grams. Similar to the post-hoc pruning experiments in §4.2, we gradually disable the usage of four-grams, trigrams and bigrams at inference time, and report the performance in Table 10.

<table><tr><td rowspan="2"></td><td colspan="2">IMDB</td><td colspan="2">AGNews</td></tr><tr><td>|V|</td><td>Acc.</td><td>|V|</td><td>Acc.</td></tr><tr><td>n = 1</td><td>54,089</td><td>74.86</td><td>81,796</td><td>91.32</td></tr><tr><td>n ≤ 2</td><td>446,793</td><td>92.09</td><td>541,431</td><td>92.93</td></tr><tr><td>n ≤ 3</td><td>835,403</td><td>93.33</td><td>882,489</td><td>93.03</td></tr><tr><td>n ≤ 4 (all)</td><td>1,000,000</td><td>93.47</td><td>1,000,000</td><td>92.99</td></tr></table>

Table 10: Impact of n in n-grams. We disable usage of longer n-grams in the DAN(KD+FT) model. V  is the size of the vocabulary after disabling.

Model Storage. In Table 11 we provide more details about the disk space and memory required for using DAN models and the baseline models. Note that the GPU memory listed below is the memory used to load the static model. During training, more memory will be dynamically allocated during forward and backward passes. DAN uses smaller memory during training because only a small portion of the parameters are activated and trained (see the last row in Table 13). In this way we are able to use batch sizes as large as 2048 to train DANs on one single GPU, which is not possible for transformer based models.

<table><tr><td></td><td>#Param</td><td>GPU Memory</td><td>Disk Space</td><td>Source</td></tr><tr><td>RoBERTa-Large</td><td>355M</td><td>2199MB</td><td>711MB</td><td>fairseq (fp16)</td></tr><tr><td>RoBERTa-Large</td><td>355M</td><td>2199MB</td><td>1.33GB</td><td>HF transformers (fp32)</td></tr><tr><td>DistilBERT</td><td>66M</td><td>1123MB</td><td>256MB</td><td>HF transformers</td></tr><tr><td>MobileBERT</td><td>25M</td><td>973MB</td><td>140MB</td><td>HF transformers</td></tr><tr><td>DistilRoBERTa</td><td>85M</td><td>1181MB</td><td>316MB</td><td>HF transformers</td></tr><tr><td>LSTM (2l-128d)</td><td>53M</td><td>1051MB</td><td>212MB</td><td>fairseq (fp32)</td></tr><tr><td>CNN (11-256d)</td><td>53M</td><td>1119MB</td><td>213MB</td><td>fairseq (fp32)</td></tr><tr><td>DAN</td><td>1001M</td><td>4655MB</td><td>3.99GB</td><td>fairseq (fp32)</td></tr></table>

Table 11: Disk space and GPU memory required for each model.

## D Potential Risks

It is risky to deploy DAN models to high-stakes applications (e.g., medical decisions) as the model lacks the ability of understanding long context (see case study in §4.4). DANs may raise fairness concerns: it lacks ability to understand the meaning of words in context, so it may learn spurious correlations such as overemphasis on group identifiers.

<table><tr><td rowspan="2"></td><td colspan="3">Parameter Count</td><td colspan="4">IMDB</td><td colspan="4">SST-2</td></tr><tr><td>Total</td><td>Sparse</td><td>Dense</td><td>Acc.</td><td>GPU-fp32</td><td>GPU-fp16</td><td>CPU-fp32</td><td>Acc.</td><td>GPU-fp32</td><td>GPU-fp16</td><td>CPU-fp32</td></tr><tr><td>RoBERTa-Large</td><td>355M</td><td>51M</td><td>304M</td><td>96.3</td><td>28.9 (1x)</td><td>92.3 (1x)</td><td>1.4 (1x)</td><td>96.2</td><td>267.3 (1x)</td><td>610.2 (1x)</td><td>22.2 (1x)</td></tr><tr><td>DistillBERT</td><td>66M</td><td>23M</td><td>43M</td><td>92.2</td><td>175.8 (6x)</td><td>334.7 (4x)</td><td>10.7 (8x)</td><td>90.8</td><td>828.5 (3x)</td><td>1117.3 (2x)</td><td>60.6 (3x)</td></tr><tr><td>MobileBERT</td><td>25M</td><td>4M</td><td>21M</td><td>93.6</td><td>157.7 (5x)</td><td>200.3 (2x)</td><td>7.7 (6x)</td><td>90.9</td><td>574.5 (2x)</td><td>545.8 (1x)</td><td>89.4 (4x)</td></tr><tr><td>*DistillRoBERTa</td><td>83M</td><td>39M</td><td>44M</td><td>95.9</td><td>176.4 (6x)</td><td>569.8 (6x)</td><td>7.8 (6x)</td><td>94.2</td><td>636.5 (2x)</td><td>771.7 (1x)</td><td>185.9 (8x)</td></tr><tr><td>*LSTM (2l-512d)</td><td>62M</td><td>51M</td><td>11M</td><td>95.9</td><td>361.5 (12x)</td><td>594.5 (6x)</td><td>30.6 (22x)</td><td>93.9</td><td>4222.1 (14x)</td><td>6281.4 (8x)</td><td>394.3 (18x)</td></tr><tr><td>*LSTM (21-256d)</td><td>56M</td><td>51M</td><td>5M</td><td>95.8</td><td>665.2 (23x)</td><td>788.0 (9x)</td><td>51.9 (37x)</td><td>93.3</td><td>6361.5 (21x)</td><td>7080.6 (9x)</td><td>678.5 (31x)</td></tr><tr><td>*LSTM (21-64d)</td><td>53M</td><td>51M</td><td>2M</td><td>94.0</td><td>818.5 (28x)</td><td>808.5 (9x)</td><td>101.4 (73x)</td><td>92.8</td><td>7075.8 (24x)</td><td>7384.1 (9x)</td><td>1378.5 (62x)</td></tr><tr><td>*LSTM (21-4d)</td><td>52M</td><td>51M</td><td>&lt;1M</td><td>93.1</td><td>812.9 (28x)</td><td>817.0 (9x)</td><td>146.4 (105x)</td><td>88.3</td><td>7026.3 (24x)</td><td>7521 (9x)</td><td>2014.6 (91x)</td></tr><tr><td>*CNN (11-256d)</td><td>53M</td><td>51M</td><td>2M</td><td>89.2</td><td>3410.7 (109x)</td><td>8427.1 (91x)</td><td>251.2 (181x)</td><td>82.8</td><td>1323.5 (5x)</td><td>1563.9 (3x)</td><td>3820.4 (172x)</td></tr><tr><td>*DAN (ours)</td><td>1001M</td><td>1000M</td><td>1M</td><td>93.5</td><td>17557.9 (607x)</td><td>20888.1 (226x)</td><td>922.6 (663x)</td><td>88.5</td><td>1745.5 (7x)</td><td>1865.9 (3x)</td><td>16478.6 (741x)</td></tr></table>

Table 12: Model Size and Inference Speed Comparison. We report accuracy, inference speed (unit: samples per second) and relative speed compared to the teacher model (RoBERTa-Large). Our DAN model achieves competitive accuracy while achieving significant inference speed-up in various settings. ? indicates the model is trained with task-specific distillation; no ? indicates the model is trained with direct fine-tuning.
<table><tr><td>Notation</td><td>Description</td><td>IMDB</td><td>SST-2</td><td>TREC</td><td>AGNews</td><td>CCom</td><td>WToxic</td></tr><tr><td> $\mathcal { V } _ { \mathrm { 0 } }$ </td><td>Top 1 million n-grams in C and  $D _ { t r a i n }$  一</td><td colspan="6">1,000,000</td></tr><tr><td> $\nu _ { 1 }$   $\nu _ { 2 }$ </td><td>All n-grams in  $D _ { t r a i n }$  All n-grams in  $D _ { d e v }$ </td><td>10,109,522 9,843,369</td><td>262,417 39,666</td><td>89,358 5,995</td><td>7,156,063 662,665</td><td>116,143,462 8,987,055</td><td>15,805,923 6,958,457</td></tr><tr><td> $\nu _ { 3 }$ </td><td>V0∩V1</td><td>805,360</td><td>76,370</td><td>31,770</td><td>486,438</td><td>983,843</td><td>828,302</td></tr><tr><td></td><td>|ν0∩ν1|/|νo| (%) |Vo ∩ νi|/|ν1| (%)</td><td>80.54% 7.97%</td><td>7.64% 29.10%</td><td>3.18% 35.56%</td><td>48.44% 6.80%</td><td>98.38% 0.85%</td><td>82.83% 5.24%</td></tr><tr><td> $\nu _ { 4 }$ </td><td>V0∩V2</td><td>792,251</td><td>15,395</td><td>3,461</td><td>123,247</td><td>740,286</td><td>671,985</td></tr><tr><td></td><td>|V0∩V2|/|Vo| (%) |V0∩ V2|/|ν2| (%)</td><td>79.22% 8.05%</td><td>1.54% 38.81%</td><td>0.35% 57.73%</td><td>12.32% 18.60%</td><td>74.03% 8.23%</td><td>67.20% 9.18%</td></tr><tr><td> $\nu _ { 5 }$ </td><td>V0∩V1∩V2</td><td>690,790</td><td>8,804</td><td>1,840</td><td>113,311</td><td>739,920</td><td>638,833</td></tr><tr><td></td><td>|V₀∩V₁∩V2|/|V2|(%)</td><td>7.01%</td><td>22.20%</td><td>30.69%</td><td>17.10%</td><td>8.23%</td><td>9.18%</td></tr><tr><td></td><td>Average # activated n-grams per instance</td><td>496</td><td>16</td><td>17</td><td>68</td><td>103</td><td>144</td></tr></table>

Table 13: Size of different sets of n-gram and their statistics of n-gram coverage.

We believe a thorough analysis is needed and bias mitigation methods such as (Bolukbasi et al., 2016; Kennedy et al., 2020) are necessary for combating these issues.