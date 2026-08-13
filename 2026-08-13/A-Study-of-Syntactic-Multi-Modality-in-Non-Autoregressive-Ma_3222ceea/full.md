# A Study of Syntactic Multi-Modality in Non-Autoregressive Machine Translation

Kexun Zhang<sup>1</sup>∗, Rui Wang<sup>2</sup>†, Xu Tan<sup>2</sup>, Junliang Guo<sup>2</sup>, Yi Ren<sup>1</sup>, Tao Qin<sup>2</sup>, Tie-Yan Liu<sup>2</sup> <sup>1</sup>Zhejiang University, <sup>2</sup>Microsoft Research Asia

<sup>1</sup>{kexunz,rayeren}@zju.edu.cn

<sup>2</sup>{ruiwa,xuta,junliangguo,taoqin,tyliu}@microsoft.com

## Abstract

It is difficult for non-autoregressive translation (NAT) models to capture the multi-modal distribution of target translations due to their conditional independence assumption, which is known as the “multi-modality problem”, including the lexical multi-modality and the syntactic multi-modality. While the first one has been well studied, the syntactic multi-modality brings severe challenge to the standard cross entropy (XE) loss in NAT and is under studied. In this paper, we conduct a systematic study on the syntactic multi-modality problem. Specifically, we decompose it into short- and longrange syntactic multi-modalities and evaluate several recent NAT algorithms with advanced loss functions on both carefully designed synthesized datasets and real datasets. We find that the Connectionist Temporal Classification (CTC) loss and the Order-Agnostic Cross Entropy (OAXE) loss can better handle short- and long-range syntactic multi-modalities respectively. Furthermore, we take the best of both and design a new loss function to better handle the complicated syntactic multi-modality in real-world datasets. To facilitate practical usage, we provide a guide to use different loss functions for different kinds of syntactic multimodality.

## 1 Introduction

Traditional Neural Machine Translation (NMT) models predict each target token conditioned on previous generated tokens in an autoregressive way (Vaswani et al., 2017), resulting in high latency in inference. Non-Autoregressive Translation (NAT) models generate all the target tokens in parallel (Gu et al., 2018), significantly reducing inference latency. A disadvantage of NAT is that it suffers from the multi-modality problem (Gu et al., 2018) when a source sentence corresponds to multiple correct translations (Ott et al., 2018).

There are two types of multi-modalities: the lexical one and the syntactic one. The former one has been adequately studied (Gu et al., 2018; Zhou et al., 2020; Ding et al., 2021), while the latter one brings severe challenges to the widely used cross entropy (XE) loss in NAT. With standard XE loss, the generated tokens are required to be strictly aligned with ground truth tokens in the same positions, which fails to provide positive feedback for correctly predicted words on different positions as shown in Fig. 1a. Therefore, advanced loss functions are introduced to provide better feedback for NAT training: Connectionist Temporal Classification (CTC) loss (Libovický and Helcl, 2018) considers all possible monotonic alignments between a generated sequence and the ground truth; Aligned Cross-Entropy (AXE) loss (Ghazvininejad et al., 2020) selects the best monotonic alignment; and Order-Agnostic cross entropy (OAXE) loss (Du et al., 2021) calculates the XE loss with the best alignment based on maximum bipartite matching algorithm.

Even if with those advanced loss functions, we find they do not perform consistently across datasets and languages. In addition, diverse grammar rules in natural language (Comrie, 1989) implies the existence of different kinds of syntactic multi-modality. Inspired by Odlin (2008); Jing and Liu (2015); Liu (2007, 2010), we categorize the syntactic multi-modality into two sub types: the long-range and short-range ones. The long-range multi-modality is mainly caused by long-range word order diversity (e.g., an adverbial of place may appear at the beginning or the end of a sentence). The short-range multi-modality is mainly caused by short-range word order diversity (e.g., an adverb may appear either in front of or behind the corresponding verb) and optional words (e.g., in some languages, determiners and prepositions may be optional (Ott et al., 2018)). Based on the above categorization of syntactic multi-modality, we further ask two research questions: (1) Which kinds of syntactic multi-modality do these loss functions excel at respectively? (2) How to better address this problem by taking advantage of different loss functions?

In this paper, we conduct a systematic study to answer these questions:

• Since the short-range and long-range syntactic multi-modalities are usually entangled together in real-world datasets, we first design synthesized datasets to decouple them to better evaluate existing NAT algorithms (§3). We find that the CTC loss (Libovický and Helcl, 2018) can better handle the short-range syntactic multi-modality while the OAXE loss (Du et al., 2021) is good at the long-range one. Though carefully designed, the synthesized datasets are still different from the real-world datasets. Accordingly, we further conduct analyses on real-world datasets (§4), which also show consistent findings with that in synthesized datasets.

• We design a new loss function that takes the best of both CTC and OAXE, and performs better to handle the short- and long-range syntactic multimodalities simultaneously (§5), as verified by experiments on benchmark datasets including WMT14 EN-DE, WMT17 EN-FI, and WMT14 EN-RU. Moreover, we further provide a practical guide to use different loss functions for different kinds of syntactic multi-modality (§5).

## 2 Background

Non-Autoregressive Translation Given the source sentence $x = ( x _ { 1 } , x _ { 2 } , . . . , x _ { T _ { x } } )$ , traditional NMT model generates the target sentence $y \_ =$ $( y _ { 1 } , y _ { 2 } , . . . , y _ { T _ { y } } )$ from left to right and token by token: $\begin{array} { r } { P ( y | x ) ~ = ~ \prod _ { t = 1 } ^ { T _ { y } } P ( y _ { t } | y _ { < t } , x ; \theta _ { \mathrm { e n c } } , \theta _ { \mathrm { d e c } } ) } \end{array}$ where $y _ { < t }$ indicates the target tokens generated before the t-th timestep, $T _ { x }$ and $T _ { y }$ denote the length of source and target sentence, $\theta _ { \mathrm { e n c } }$ and $\theta _ { \mathrm { d e c } }$ denote the encoder and decoder parameters respectively. This autoregressive way suffers from high latency during inference. Non-Autoregressive Translation (NAT) (Gu et al., 2018) is proposed to reduce the inference time by generating the whole sequence in parallel, $P ( y | x ) = P ( T _ { y } | x )$ $\begin{array} { r l } {  { \prod _ { t = 1 } ^ { T _ { y } } P ( y _ { t } | x ; \theta _ { \mathrm { e n c } } , \theta _ { \mathrm { d e c } } ) } \quad } & { { } } \end{array}$ , where $P ( T _ { y } | x )$ indicates the length prediction function. While the inference speed is boosted, the translation accuracy is sacrificed due to that target tokens are generated conditional independently.

![](images/abd2aa40246cc2c79e840edde947d2a9fba1c890f47219a42c446b40dba754c7.jpg)

(a) XE  
![](images/946875d43af0ea1a0614deb4223d0b436999da75442c4831e740d2f175d28609.jpg)

(b) AXE  
![](images/d2745c909d9476206340a9adaa74bc2b195e5bf15eb12d62b4c6d1f86351c80f.jpg)

(c) CTC, where solid, dash, and dot dash lines illustrate three possible alignments respectively.  
![](images/bfde0dc7b2cea93e9d76d3ede03555a24900975704df77aaed73d208ad8d733b.jpg)  
(d) OAXE  
Figure 1: The illustration of different loss functions, where $\mathbf { \ddot { G } T } ^ { 5 }$ stands for ground truth, “PRED” stands for predicted sequence, the green check indicates that credit is provided to the token.

Multi-Modality Problem The multi-modality problem (Gu et al., 2018; Zhou et al., 2020) indicates that one source sentence may have multiple correct target translations, which brings challenges to NAT models as they generate each target token independently. Specifically, we categorize the multi-modality problem into two sub-problems, i.e., lexical and syntactic multi-modalities. The lexical multi-modality indicates that a source token can be translated into different target synonym tokens (i.e., “thank you” in English can be translated into both “Danke” or “Vielen Dank” in German), while the syntactic multi-modality indicates the inconsistency of word-orders (e.g., an adverb may appear either in front of or behind the corresponding verb) and the existence of optional words between source and target languages (e.g., in some languages, determiners and prepositions may be optional) (Ott et al., 2018). The lexical multi-modality problem has been adequately studied in recent works. Sequence-level knowledge distillation (Gu et al., 2018; Zhou et al., 2020) is shown capable to reduce the lexical diversity of the dataset and thus alleviate the problem. Some works also introduce extra loss functions such as KL-divergence (Ding et al., 2021) and bag-of-ngram (Shao et al., 2020) to alleviate the lexical multi-modality problem.

On the contrary, there still lacks a systematic study on the syntactic multi-modality problem. Generally, it is difficult to solve this problem because the order and optional words vary across different languages. For example, the word order of Russian is quite flexible (Kallestinova, 2007), thus the syntactic multi-modality may exist more frequently in Russian corpora. In contrast, the structure of English sentences is mostly subject–verb–object (SVO) (Givón, 1983), which results in less variation on word order. In this paper, we categorize the syntactic multi-modality problem into short-range and long-range instances, and provide detailed analyses accordingly.

Loss Functions in NAT Standard crossentropy (XE) loss requires the predicted tokens to be strictly aligned with ground truth tokens, which fails to deal with the syntactic multi-modality problem. Different loss functions are proposed to solve the problem, and here we consider some most recent works. The CTC loss sums XE losses of all possible monotonic alignments and has been widely used in speech recognition (Graves et al., 2006, 2013), and the effectiveness of the CTC loss in NAT has been validated (Libovický and Helcl, 2018; Gu and Kong, 2021). AXE (Ghazvininejad et al., 2020) selects the monotonic alignment between the predicted sequence and the ground truth with the minimum XE loss. OAXE (Du et al., 2021) further relaxes the position constraint and only considers the best alignment. The illustration for each loss function is provided in Fig. 1. Though effective in different datasets, these works ignore fine-grain features of the multi-modality problem such as short/long syntactic multi-modalities. In this work, we analyse the performance of these loss functions in different syntactic scenarios, and provide a practical guide to use appropriate loss functions for different kinds of syntactic multi-modality.

## 3 Analyses on Synthesized Datasets

To make fine-grained analyses on the syntactic multi-modality problem, we first categorize it into long-range and short-range types, where the longrange one is mainly caused by long-range word order diversity, and the short-range one is mainly caused by short-range word order diversity and optional words. Then, we would like to evaluate the accuracy of different losses on different types of syntactic multi-modality. However, in real-world corpora, the different types are usually entangled, making it difficult to control and analyse one aspect without changing the other. Thus, we construct synthesized datasets based on phrase structure rules (Chomsky, 1959) to manually control the degree of syntactic multi-modality in different aspects, and evaluate the performance of different existing techniques.

![](images/89e54ac9b614bad4e7bbbadaeb9e99aa684f9b3cf11e0e54b59056cafe30fa8c.jpg)  
Figure 2: An illustration of generating a syntax tree for a source sentence. In the first iteration, “Sen” consists of $( ^ { 6 6 } \mathrm { N P ^ { 3 } , ^ { 6 6 } V P ^ { 3 } } )$ as the solid lines. In the second iteration, “NP” consists of (“DT”, “RB”, “JJ”, “N”) and “VP” consists of (“V”, “NP”, “RB”) as the dash lines. In the third iteration, “NP” consists of (“DT”, “JJ”, “N”) as the dot-and-dash lines.

## 3.1 Synthesized Datasets

We first employ phrase structure rules (Chomsky, 1959) to synthesize the source sentences, where the rules are based on the syntax of languages. Considering that translation can be decomposed to word reordering and word translation (Bangalore and Riccardi, 2001; Sudoh et al., 2011), we then “translate” the synthesized source sentences to synthesized target sentences in two steps: 1) word reordering by changing its syntax tree; 2) and word translation by substituting the source words into target words.

Source Sentence Synthesis. We first generate the syntax tree of the source sentence. Specifically, we use the notations of the constituents in syntax tree according to the Penn Treebank syntactic and part of speech (POS) tag sets<sup>1</sup> (Marcus et al., 1993), and generate the syntax tree of a source sentence as following (Rosenbaum, 1967):

$$
\cdot \ \mathrm { S e n }  \mathrm { N P } \ \mathrm { V P } ,
$$

![](images/a89d176e545dbdcfac25c1f5730748c74d84744737d2437499dc86754e3a6230.jpg)  
Figure 3: An illustration of “translation”, where the constituent order of $\mathbf { \cdots } \mathbf { S e n } ^ { \mathbf { \prime } \mathbf { \prime } }$ is changed to ${ } ^ { \mathrm { 6 6 } } \mathrm { V P } ~ \mathrm { N P } ^ { \mathrm { 3 } }$ with probability $1 - P ^ { l o }$ , the constituent order of $\mathbf { \omega ^ { 6 6 } V P } ^ { \prime }$ is changed to “RB V NP” with probability $1 - P _ { 1 } ^ { s o } - P _ { 2 } ^ { s o }$ , and the circled $\mathbf { \ddot { \mu } } ^ { 6 6 } \mathbf { D } \mathbf { T } ^ { 5 }$ is removed with probability $P ^ { o p }$ . Meanwhile, the numbers in the source sentence are replaced with the ones in the target sentence based on mappings.

$$
\begin{array} { r l } & { \bullet \mathrm {  ~ N P  ( D T ) } ( \mathrm { \bf R B } ) ^ { \ast } ( \mathrm { \bf J J } ) ^ { \ast } \mathrm {  ~ N } , } \\ & { } \\ & { \bullet \mathrm {  ~ V P  V } ( \mathrm { \bf N P } ) ( \mathrm { \bf R B } ) ^ { \ast } , } \end{array}
$$

where the constituent on the left side of the arrow consists of the constituents on the right side in sequence, $^ { 6 6 } ( \cdot ) ^ { 5 }$ means that the constituent is optional, and $^ { 6 6 } ( \cdot ) ^ { * 9 }$ denotes that the constituent is not only optional but can also be repetitive. For each sentence, we start with a single constituent Sen and iteratively decompose “Sen”, “NP”, and $\mathbf { \tilde { \mu } } ^ { 6 6 } \mathbf { V } \mathbf { P } ^ { \prime }$ according to the rules until all the constituents are decomposed to “DT”, “JJ”, “RB”, “V”, and $\mathbf { \bar { \Psi } } ^ { \bullet } \mathbf { N } ^ { \mathbf { \vec { \mathbf { \Lambda } } } \mathbf { \vec { \mathbf { \Lambda } } } }$ An illustration of generating a syntax tree is depicted in Fig. 2. To synthesize the source sentence according to the syntax tree, we use numbers as the words in the synthesized source sentences and use different ranges of numbers to represent words with different POS, where the details of the ranges are provided in Appendix A. Then, a number is randomly sampled in the corresponding range for each word in the syntax tree.

Word Reordering. To introduce syntactic multimodality, we consider multiple possible rules for “Sen”, “NP”, and $\mathbf { \tilde { \mu } } ^ { 6 6 } \mathbf { V } \mathbf { P } ^ { \prime }$ in the target sentences. Dependency distance is defined as the linear distance between two words with syntactical relationship (Liu et al., 2017), which can be used as a guide to select typical rules to introduce long- and short-range word order diversity. Specifically, we consider three options: 1) The word order of “Sen” is with probability $P ^ { l o }$ to be the same with the source sentence (i.e., NP VP) and with probability $\left( 1 - P ^ { l o } \right)$ to swap the $\bf \Pi ^ { 6 6 } N P ^ { , 9 }$ and $\mathbf { \tilde { \mu } } ^ { 6 6 } \mathbf { V } \mathbf { P } ^ { \prime }$ (i.e., VP NP), which has long dependency distance and represents for the long-range word order; 2) For the word order in $\mathbf { \tilde { \mu } } ^ { 6 6 } \mathbf { V } \mathbf { P } ^ { \prime }$ , it is considered to be the same with the source sentence with probability $P _ { 1 } ^ { s o }$ place “RB” between “V” and “NP” with probability $P _ { 2 } ^ { s o }$ , and place “RB” before $\mathbf { \hat { \Pi } } ^ { 6 6 } \mathbf { V } ^ { 5 }$ with probability $( 1 - P _ { 1 } ^ { s o } - P _ { 2 } ^ { s o } )$ , which has short dependency distance and represents for the short-range word order; 3) To introduce the syntactic multi-modality of optional words, we change the existence of $\bf \Pi ^ { * } \bf { D } \bf { T } ^ { * }$ in each “NP” of the source sentence with probability $P ^ { o p }$ (i.e, remove “DT” if it exists in the source sentence and add “DT” if it does not exist in the source sentence).

<table><tr><td rowspan=1 colspan=2>Probability |Default</td><td rowspan=1 colspan=1>Effect</td></tr><tr><td rowspan=1 colspan=1> $P ^ { l o }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>long-range word order</td></tr><tr><td rowspan=1 colspan=1> $P _ { 1 } ^ { s o }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>short-range word order</td></tr><tr><td rowspan=1 colspan=1> $P _ { 2 } ^ { s o }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>short-range word order</td></tr><tr><td rowspan=1 colspan=1> $P ^ { o p }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>optional words</td></tr></table>

Table 1: Default values of the probabilities to adjust the syntactic multi-modality.

Word Translation. Same as in the source sentences, we use different range of numbers to represent words with different POS in target sentences. To do the word translation, we first randomly build mappings between the source and target words with different POS respectively. Since we focus on studying the syntactic multi-modality, we consider each source word is mapped to a single target word to eliminate the lexical multi-modality. Then, we replace the words in the source sentence based on the mappings to generate the target sentence. An illustration of “translation” is shown in Fig. 3.

## 3.2 Experiments and Analyses

We conduct experiments to compare existing loss functions on different kinds of syntactic multimodality on the synthesized datasets, by changing the probabilities $( \mathrm { i . e . , } P ^ { o p } , P _ { 1 } ^ { s o } , P _ { 2 } ^ { s o }$ , and $P ^ { l o } )$ as listed in Table 1. In the following, we first provide the experimental settings, then show the results on the long-range and short-range syntactic multimodalities, and finally conclude the key findings.

Experimental Settings. We consider two separate vocabularies for the source and target sentences, each containing 15K words. 0.3M, 5K, and 5K synthesized sentence pairs are generated as training, validation, and test sets respectively. We use the same hyper-parameters in the transformerbase model (Vaswani et al., 2017), which is commonly used in the NAT models (Gu et al., 2018; Du et al., 2021; Saharia et al., 2020). All settings are trained on 4 Nvidia V100 GPUs with 16k tokens in a batch. For the model with OAXE loss, we train the first 50K updates with XE loss and the next 50K updates with OAXE loss (Du et al., 2021). For the other losses, we train the model for 100K updates. The length of the decoder input is set as twice the length of the source sequence for CTC loss (Saharia et al., 2020), while the golden target length is used for OAXE, AXE, and XE. To evaluate the accuracy of the predicted sequence, we first calculate the longest common sub-sequence between the predicted and the golden sequences, and then the accuracy is defined as the ratio between the length of the longest common sub-sequence and the golden sequence. The accuracy on the test set is calculated as the average accuracy among all the predicted sentences.

Long-Range Syntactic Multi-modality. To consider the effect of long-range diversity, we change the corresponding probability $P ^ { l o }$ , while keeping the others unchanged to eliminate the short-range syntactic multi-modality. It is observed in Fig. 4a that CTC loss always performs better than AXE, and OAXE is the best with different degree of longrange multi-modality.

Short-Range Syntactic Multi-modality. Similarly, we only change the probabilities $P _ { 1 } ^ { s o }$ and $P _ { 2 } ^ { s o }$ to adjust the degree of short-range word order diversity. The results are shown in Fig. 4b, where OAXE loss performs better than AXE loss, and CTC loss outperforms all the other losses with varied degree of short-range word order diversity. In order to study the effect of optional words, we vary the probability $P ^ { o p }$ to change the existence of $\mathbf { \ddot { \mu } } ^ { 6 6 } \mathbf { D } \mathbf { T } ^ { 5 }$ . As shown in Fig. 4c, OAXE loss is slightly better than AXE loss, and CTC loss performs the best, indicating that CTC loss is superior in the syntactic multi-modality problem caused by optional words.

![](images/806657d8313285bf1c471e1a5de943f94076b8a17d647529a6921f9de2a31669.jpg)

(a) Effect of long-range word order.  
![](images/cefca16067a0f53e4f001aafcdae083a42e22f9144d098ad2d422bccb3c05b13.jpg)

(b) Effect of short-range word order.  
![](images/37679ebd5ec85641d882d19098266e7e135c0c4f55283de574a3b6871d9f8757.jpg)  
(c) Effect of optional words.  
Figure 4: The accuracy of different loss functions on synthesized datasets.

Analyses and Discussions. Based on the results in Fig. 4, we can get the following observations:

• OAXE loss is superior in handling the long-range syntactic multi-modality (i.e., long-range word order). OAXE loss is order-agnostic, which is able to provide fully positive feedback to the word in different positions compared to the ground truth sequence. Accordingly, OAXE is suitable for datasets with long-range word order diversity. Though it can deal with high diversity of word order, it may also incur wrong predictions on word order, which may be why OAXE is not suitable when the diversity only exists in short-range.

• CTC loss is the best choice for dealing with shortrange syntactic multi-modality (i.e., short-range word order and optional words). CTC loss is generally considered to handle monotonic matching, which seems not effective in handling the multi-modality caused by word order (Saharia et al., 2020). However, it is observed in Fig. 4a and 4b that CTC loss outperforms AXE and XE when dealing with long-range word order diversity and performs the best on the multi-modality caused by short-range word order. Since CTC considers all the monotonic alignments, it can partially provide positive feedback to the words with different order through multiple monotonic alignments. As shown in Fig. 1c, all the words can be considered in the three alignments.

Considering that AXE loss does not show superiority on any type of the syntactic multi-modality, we will only focus on CTC and OAXE losses in the following.

## 4 Analyses on Real Datasets

Though carefully designed, the synthesized sentence pairs consisting of numbers are still different from the real sentence pairs. Therefore, in this section, we validate the findings in Section 3 based on real datasets. Considering that different types of syntactic multi-modality are highly coupled in the real corpus, we conduct experiments on carefully selected sub-datasets from a corpus, to approximately decompose the syntactic multi-modality. In the following, we first show the details of the approach to decompose the syntactic multi-modality, and then provide the analytical results based on the real datasets.

Analytical Approach. In order to decompose the long-range and short-range types of syntactic multi-modality, we select sentences that only contain subject and verb phrases from a corpus, and divide them into two sub-datasets according to the relative order of subject and verb (i.e., subject first that denoted as “SV”, or verb first that denoted as “VS”). Meanwhile, we only consider the declarative sentence pairs in the corpus to eliminate the word order difference caused by mood. Following this method, the long-range multi-modality is eliminated in each sub-dataset (i.e., “SV” and “VS”), which can be used to evaluate the effect of short-range multi-modality. To analyse the longrange multi-modality, we can adjust the degree of long-range word order diversity by sampling data from the two sub-datasets with varied ratios, while roughly keeping the degree of short-range word order diversity unchanged. Specifically, considering that Russian is flexible on word order (Kallestinova, 2007) and it is feasible to select sentences on both the “SV” and “VS” order, we use an English-Russian (EN-RU) corpus from Yandex<sup>2</sup> that contains 1M EN-RU sentence pairs, from which we get 0.2M and 0.1M sentence pairs data with “SV” order and “VS” order respectively. To select the sentence pairs with different word orders, we use spaCy(Honnibal et al., 2020) to parse the dependency of Russian sentences. For the models with CTC loss, we train for 300K updates. For the models with OAXE loss, we train with XE loss for 100K updates and then train with OAXE loss for 200K updates.

Table 2: BLEU scores of models with CTC and OAXE losses, where the models are evaluated on the WMT’19 EN-RU test set. The percentage of sentences with “RB V” among the sentences with both “RB V” and “V RB” orders are shown in column “RB V”. The percentage of sentences with “JJ N” among the sentences with both “JJ N” and “N JJ” orders are shown in column “JJ N”.
<table><tr><td>“SV”:“VS”</td><td>CTC</td><td>OAXE</td><td>“RB V”</td><td>“JJN”</td></tr><tr><td>100%: 0%</td><td>17.7</td><td>16.5</td><td>68%</td><td>84%</td></tr><tr><td>75%:25%</td><td>17.2</td><td>16.9</td><td>63%</td><td>82%</td></tr><tr><td>50%:50%</td><td>16.8</td><td>17.3</td><td>70%</td><td>79%</td></tr></table>

Analytical Results. We keep the total number of sentence pairs in the training set as 0.2M (i.e., the number of Russian sentences in the “VS” subdataset), and change the ratio of sentence pairs sampled from two sub-datasets (i.e., “SV” and “VS”). The results are shown in Table 2, where the training parameters are the same as that used in Section 3. It is observed that CTC loss outperforms OAXE loss when all data samples are from the “SV” subdataset, which indicates that CTC loss performs better on short-range syntactic multi-modality problem. When the ratio of the data sizes on the two subdatasets is changed to 75% : 25%, the gap between the performance of CTC and OAXE losses diminished, while CTC loss still performs slightly better than OAXE loss. When the ratio changed to 50% : 50%, model with OAXE loss becomes better than that with CTC loss. In summary, OAXE loss is better at handling long-range syntactic multi-modality while CTC loss is better on short-range syntactic multi-modality, which validates the key observations we obtained on the synthesized datasets in Section 3.

In order to demonstrate whether we have decomposed the long- and short-range syntactic multimodalities, we verify whether the degree of shortrange multi-modality remains almost the same when varying the degree of long-range multimodality. We evaluate the short-range syntactic diversity based on the relative order between: 1) adverb and verb (“RB V”); 2) adjective and noun (“JJ $\mathbf { N } ^ { \prime \prime } )$ . As shown in Table 2, when the ratio of the data sizes on the two sub-datasets varied from 100% : 0% to 50% : 50% (i.e., the ratio between “SV” and “VS” changes), the relative order on “RB $\mathbf { V } ^ { \ast }$ and “RB $\mathbf { V } ^ { \ast }$ (which can represent the degree of short-range word order diversity) does not vary much. These analyses verify the rationality of our analytical approach in this section.

## 5 Better Solving the Syntactic Multi-Modality Problem

As shown in previous sections, the CTC and the OAXE loss functions are good at dealing with shortand long-range syntactic multi-modalities respectively. While in real-world corpora, different types of multi-modalities usually occur together and vary in different languages. Accordingly, it may be better to use different loss functions for different languages. In this section, we first introduce a new loss function named Combined CTC and OAXE (CoCO), which takes advantage of both CTC and OAXE to better handle the long-range and shortrange syntactic multi-modalities simultaneously, and then provide a guideline on how to choose the appropriate loss function for different scenarios.

## 5.1 CoCO Loss

To obtain a general loss that performs well at both types of multi-modalities, it is natural to combine the two loss functions studied above. However, the output length is mismatched between the models using CTC and OAXE, where the output length is required to be longer than the target sequence with CTC loss, and is required to be the same as the target sequence with OAXE loss. To solve this length mismatch problem, we consider using the same output length as in CTC loss, and modify the OAXE loss to make it suitable on this output length by allowing consecutive tokens in the output to be aligned with the same token in the reference sequence. The details of the modified OAXE loss are provided in Appendix B. Then, the proposed CoCO loss is defined as a linear combination of the

![](images/d8685c18f3c2a8f8ffc857b4aa7bd27d9836384f67c56dbda233ff04e512d571.jpg)  
Figure 5: Comparing different losses on different language pairs.

CTC and modified OAXE losses as:

$$
\mathcal { L } _ { C o C O } = \lambda \mathcal { L } _ { C T C } + ( 1 - \lambda ) \mathcal { L } _ { M - O A X E } ,\tag{1}
$$

where $\mathcal { L } _ { M - O A X E }$ denotes the modified OAXE loss and λ is a hyper-parameter that balances the two losses.

## 5.2 Choosing Appropriate Loss Function

The degree of different types of multi-modalities varies among different languages. In order to find the insight to choose the appropriate loss function for different languages, we conduct experiments on several languages including Russian (RU), Finnish (FI), German (DE), Romanian (RO), and English (EN). These languages have different requirements on the positions of subject (S), verb (V), and object (O), which is one major influence factor on the large-range syntactic multi-modality. Specifically, the order in RU and FI is quite flexible, where all the 6 possible orders of “S”, “V”, and $\mathbf { \tilde { \Delta } } ^ { 6 6 } \mathbf { O } ^ { 9 }$ are valid. In DE, the verb is required to be placed on the second position, which is called verb-second word order. Meanwhile, in RO and EN, the order is restricted to “SVO”.

We evaluate the accuracy of different loss functions (i.e., CTC, OAXE, and CoCO) on WMT’14 EN-RU, WMT’17 EN-FI, WMT’14 EN-DE, and WMT’16 EN-RO datasets with 1.5M, 2M, 4M, and 610K sentence pairs, respectively. The λ in COCO loss is set as 0.1 so that $\lambda \mathcal { L } _ { C T C }$ and $( 1 - \lambda ) \mathcal { L } _ { M - O A X E }$ are in the same order of magnitude. Following Du et al. (2021), for the models with OAXE and CoCO loss, we first train with XE or CTC loss for 100K updates and then train with OAXE or CoCO loss for 200K updates, respectively. For CTC loss, we train for 300K updates. For decoding, we follow Gu and Kong (2021); Huang et al. (2021) to use beam search with language model scoring<sup>3</sup> for CTC and CoCO. The other training settings are the same as used in Section 3. We report the tokenized BLEU score to keep consistent with previous works. We show the difference values of BLEU score in Fig. 5 and provide the corresponding absolute BLEU scores in Appendix C. According to Fig. 5, we have several observations: 1) The proposed CoCO loss consistently improves the translation accuracy on all the language pairs compared to OAXE loss; 2) The CoCO loss outperforms CTC loss when the target language is with flexible word order or verb-second word order (i.e., EN-RU, EN-FI, and EN-DE); 3) CTC loss performs the best when the target language is “SVO” language (i.e., DE-EN, RO-EN, and EN-RO).

Table 3: BLEU scores of NAT models.
<table><tr><td>Model</td><td colspan="2">WMT14 EN-DE DE-EN</td><td>WMT16 EN-RO</td><td>WMT14 EN-RU</td><td>WMT17</td><td></td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td>EN-FI</td><td>Speedup</td></tr><tr><td>Transformer</td><td>27.48</td><td>31.39</td><td>33.70</td><td>27.2</td><td>28.12</td><td>1.0 ×</td></tr><tr><td>Non-Autoregressive</td><td></td><td>21.47</td><td>27.29</td><td></td><td></td><td>15.0 ×</td></tr><tr><td>Vanilla NAT (Gu et al., 2018) BoN (Shao et al., 2020)</td><td>17.69 20.90</td><td>24.60</td><td>28.30</td><td></td><td></td><td>10.0 ×</td></tr><tr><td>AXE (Ghazvininejad et al., 2020)</td><td>23.53</td><td>27.90</td><td>30.75</td><td></td><td></td><td>15.3 ×</td></tr><tr><td>Imputer (Saharia et al., 2020)</td><td>25.80</td><td>28.40</td><td>32.30</td><td></td><td></td><td>18.6 ×</td></tr><tr><td>OAXE (CMLM) (Qian et al., 2021)</td><td>26.10</td><td>30.20</td><td>32.40</td><td></td><td></td><td>15.6 ×</td></tr><tr><td>GLAT (Qian et al., 2021)</td><td>26.39</td><td>29.84</td><td>32.79</td><td></td><td></td><td>14.6 ×</td></tr><tr><td>CTC (VAE) (Gu and Kong, 2021)</td><td>27.49</td><td>30.46</td><td>33.79</td><td></td><td></td><td>16.5 ×</td></tr><tr><td>CTC (GLAT) (Gu and Kong, 2021)</td><td>27.20</td><td>31.39</td><td>33.71</td><td></td><td></td><td>16.8 ×</td></tr><tr><td>CTC (DSLP) (Huang et al., 2021)</td><td>27.02</td><td>31.61</td><td>34.17</td><td>21.38</td><td>22.83</td><td>14.8 ×</td></tr><tr><td>CoCO (DSLP)</td><td>27.41</td><td>31.37</td><td>34.32</td><td>21.82</td><td>23.25</td><td>14.2 ×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

We would also like to evaluate the CoCO loss based on SOTA NAT models. Though the proposed CoCO loss can be used in both iterative and noniterative models, we only show the results on noniterative models in this paper and leave the iterative models as future work. We use CoCO loss on a recently proposed Deeply Supervised, Layer-wise Prediction-aware (DSLP) transformer (Huang et al., 2021), which achieves competitive results. The details of how CoCO loss is applied on DSLP are provided in Appendix D. The results are shown in Table 3. Compared to DSLP with CTC loss (Huang et al., 2021), DSLP with CoCO loss consistently improves the BLEU scores on three language pairs, including EN-RU, EN-FI, and EN-DE. On the contrary, DSLP with CTC loss is better or comparable to DSLP with CoCO loss when the target language is restricted to the “SVO” word order, including EN-RO and DE-EN.

According to the experiments on language pairs with different kinds of requirements on word order, we suggest to: 1) use CoCO loss when the word order of the target language is relatively flexible ( e.g., RU and FI, where word order on “S” “V” “O” is free, and DE, where the verb is required to be placed on the second position); 2) use CTC loss when the target language is with relatively strict word order (e.g., RO and EN, which are “SVO” languages).

## 6 Conclusion

In this paper, we conduct a systematic study on the syntactic multi-modality problem in nonautoregressive machine translation. We first categorize this problem into long-range and short-range types and study the effectiveness of different loss functions on each type. Considering the different types are usually entangled in real-world datasets, we design and construct synthesized datasets to control the degree of one type of multi-modality without changing another for analyses. We find that CTC loss is good at short-range syntactic multimodality while OAXE loss is better at the longrange one. These findings are further verified on real-world datasets with our designed analytical approach. Based on these analyses, we propose a CoCO loss that can better handle the complicated syntactic multi-modality in real-world datasets, and a practical guide to use different loss functions for different kinds of syntactic multi-modality: CoCO loss is preferred when the word order of target language is relatively flexible while CTC loss is preferred when target language is with strict word order. Our study in this paper can facilitate better understanding of the multi-modality problem and provide insights to better solve this problem in non-autoregressive translation. Besides, there still remain some open problems that need future investigation. For example, we generally consider longrange and short-range types for syntactic multimodality, while there may be more fine-gained categorizations on the syntactic multi-modality due to the complex syntax of natural language.

## References

Srinivas Bangalore and Giuseppe Riccardi. 2001. A finite-state approach to machine translation. In IEEE Workshop on Automatic Speech Recognition and Understanding, 2001. ASRU’01., pages 381–388. IEEE.

Noam Chomsky. 1959. On certain formal properties of grammars. Information and control, 2(2):137–167.

Bernard Comrie. 1989. Language universals and linguistic typology: Syntax and morphology. University of Chicago press.

Liang Ding, Longyue Wang, Xuebo Liu, Derek F. Wong, Dacheng Tao, and Zhaopeng Tu. 2021. Understanding and improving lexical choice in nonautoregressive translation. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Cunxiao Du, Zhaopeng Tu, and Jing Jiang. 2021. Orderagnostic cross entropy for non-autoregressive machine translation. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 2849–2859. PMLR.

Marjan Ghazvininejad, Vladimir Karpukhin, Luke Zettlemoyer, and Omer Levy. 2020. Aligned cross entropy for non-autoregressive machine translation. In International Conference on Machine Learning, pages 3515–3523. PMLR.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Mask-predict: Parallel decoding of conditional masked language models. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 6111–6120. Association for Computational Linguistics.

Talmy Givón. 1983. Topic continuity in spoken english. Topic continuity in discourse: A quantitative crosslanguage study, 3:347–363.

Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings ofthe 23rd international conference on Machine learning, pages 369–376.

Alex Graves, Navdeep Jaitly, and Abdel-rahman Mohamed. 2013. Hybrid speech recognition with deep bidirectional lstm. In 2013 IEEE workshop on automatic speech recognition and understanding, pages 273–278. IEEE.

Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. 2018. Non-autoregressive neural machine translation. In International Conference on Learning Representations.

Jiatao Gu and Xiang Kong. 2021. Fully nonautoregressive neural machine translation: Tricks of the trade. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 120–133. Association for Computational Linguistics.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrialstrength Natural Language Processing in Python.

Chenyang Huang, Hao Zhou, Osmar R. Zaïane, Lili Mou, and Lei Li. 2021. Non-autoregressive translation with layer-wise prediction and deep supervision. CoRR, abs/2110.07515.

Yingqi Jing and Haitao Liu. 2015. Mean hierarchical distance augmenting mean dependency distance. In Proceedings ofthe third international conference on dependency linguistics (Depling 2015), pages 161– 170.

Elena Dmitrievna Kallestinova. 2007. Aspects ofword order in Russian. The University of Iowa.

Harold W Kuhn. 1955. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97.

Jindrich Libovický and Jindrich Helcl. 2018. End-toend non-autoregressive neural machine translation with connectionist temporal classification. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 3016– 3021. Association for Computational Linguistics.

Haitao Liu. 2007. Probability distribution of dependency distance. Glottometrics.

Haitao Liu. 2010. Dependency direction as a means of word-order typology: A method based on dependency treebanks. Lingua, 120(6):1567–1578.

Haitao Liu, Chunshan Xu, and Junying Liang. 2017. Dependency distance: A new perspective on syntactic patterns in natural languages. Physics oflife reviews, 21:171–193.

Mitchell Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a large annotated corpus of english: The penn treebank.

Terence Odlin. 2008. A handbook of varieties of english. Language, 84(1):193–196.

Myle Ott, Michael Auli, David Grangier, and Marc’Aurelio Ranzato. 2018. Analyzing uncertainty in neural machine translation. In International Conference on Machine Learning, pages 3956–3965. PMLR.

Lihua Qian, Hao Zhou, Yu Bao, Mingxuan Wang, Lin Qiu, Weinan Zhang, Yong Yu, and Lei Li. 2021. Glancing transformer for non-autoregressive neural machine translation. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 1993–2003. Association for Computational Linguistics.

Peter S Rosenbaum. 1967. Phrase structure principles of english complex sentence formation. Journal of Linguistics, 3(1):103–118.

Chitwan Saharia, William Chan, Saurabh Saxena, and Mohammad Norouzi. 2020. Non-autoregressive machine translation with latent alignments. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 1098–1108. Association for Computational Linguistics.

Chenze Shao, Jinchao Zhang, Yang Feng, Fandong Meng, and Jie Zhou. 2020. Minimizing the bagof-ngrams difference for non-autoregressive neural machine translation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 198–205.

Katsuhito Sudoh, Xianchao Wu, Kevin Duh, Hajime Tsukada, and Masaaki Nagata. 2011. Post-ordering in statistical machine translation. In Proc. MT Summit.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008.

Chunting Zhou, Jiatao Gu, and Graham Neubig. 2020. Understanding knowledge distillation in nonautoregressive machine translation. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

## Appendix

## A Number Ranges to Synthesis the Source and Target Sentences

We use [1, 5000], [5001, 10000], [10001, 12500], [12501, 15000], and 15001, 15002, 15003 to represent “N” “V” “JJ” “RB” “DT” in the source sentences, and [15004, 20003], [20004, 25003], [25004, 27503], [27504, 30003], and 30004, 30005, 30006 to represent “N” “V” “JJ” “RB” “DT” in the target sentences.

## B Modified OAXE Loss

![](images/d3c25cff43442c5ceaaa59667d9abf33dd372ac7f634c8ece9b11e63b206d3d7.jpg)  
Figure 6: An illustration of the modified OAXE loss.

Specifically, we consider one training pair $( X , Y )$ , where there are n tokens in the ground truth sequence, denoted as $Y = ( y _ { 1 } , y _ { 2 } , \dots y _ { n } )$ . The corresponding output sequence has m tokens with probability distributions $P _ { 1 } , P _ { 2 } , \dots , P _ { m }$ , where $m \ > \ n$ . According to OAXE, we first get the alignment between the ground truth sequence and the output sequence that minimizes the cross entropy loss based on maximum bipartite matching algorithm (Kuhn, 1955):

$$
\alpha ^ { \star } = \underset { \alpha \in \gamma ( \alpha ) } { \arg \operatorname* { m i n } } \left( - \sum _ { i } \log P _ { \alpha ( i ) } ( y _ { i } | X , \theta ) \right) ,\tag{2}
$$

where α denotes the alignment from the ground truth sequence to the output sequence, $\gamma ( \alpha )$ is the set of all possible alignments, and y<sub>i</sub> is aligned with the α(i)-th token of the output. We consider each output token can only be aligned to one ground truth token $( \mathrm { i . e . , } \alpha ( i ) \neq \alpha ( j ) \mathrm { i f } i \neq j )$ . Then, we can get the alignment from the output sequence to the ground truth sequence, based on $\alpha ^ { \star }$ :

Table 4: BLEU scores of models with different losses on different language pairs.
<table><tr><td>Loss</td><td>EN-RU</td><td>EN-FI</td><td>EN-DE</td><td>DE-EN</td><td>RO-EN</td><td>EN-RO</td></tr><tr><td>CTC</td><td>20.84</td><td>22.86</td><td>26.10</td><td>30.36</td><td>33.68</td><td>33.06</td></tr><tr><td>OAXE</td><td>21.23</td><td>23.13</td><td>26.16</td><td>30.07</td><td>33.25</td><td>32.31</td></tr><tr><td>CoCO</td><td>21.45</td><td>23.27</td><td>26.25</td><td>30.19</td><td>33.31</td><td>32.67</td></tr></table>

$$
\beta ( k ) = \left\{ { \begin{array} { r l } { i } & { { \mathrm { i f } } \alpha ^ { \star } ( i ) = k , } \\ { - 1 } & { { \mathrm { i f } } \forall i \in [ 1 , n ] , \alpha ^ { \star } ( i ) \neq k , } \end{array} } \right.\tag{3}
$$

where the k-th token of the output is aligned to $y _ { \beta ( k ) }$ and $\beta ( k ) = - 1$ denotes the token has not been aligned. We provide an illustration as the “step $1 ^ { \circ }$ in Fig. 6, where we consider 3 tokens in the target sequence and 6 tokens in the output and the best alignment is $\mathrm { ^ { 6 4 } A ^ { 9 - } } \mathrm { ^ { 4 6 } } P _ { 6 } \mathrm { ^ { 9 , } } \mathrm { ^ { 6 4 } B ^ { 9 - } } \mathrm { ^ { 6 4 } } P _ { 4 } \mathrm { ^ { 9 } }$ , and $^ { 6 6 } \mathrm { C } ^ { 3 9 } - ^ { 6 6 } \mathrm { P } _ { 1 } ^ { 3 9 }$ . Since consecutive repetitive tokens are merged when decoding with CTC loss, we consider that consecutive tokens in the output can be aligned to the same ground truth token. Accordingly, we enumerate the end of each ground truth token in the output sequence respectively, and select the one that minimize the cross entropy loss. For example, given $\beta ( k _ { 1 } ) = i , \beta ( k _ { 2 } ) = j$ and $\beta ( k ) = - 1$ when $k _ { 1 } \leq k \leq k _ { 2 }$ , we select $k ^ { \star }$ according to:

$$
\begin{array} { r } { \boldsymbol { k } ^ { \star } = \underset { k _ { 1 } \leq k ^ { \prime } < k _ { 2 } } { \arg \operatorname* { m i n } } \bigg ( - \underset { k _ { 1 } \leq k \leq k ^ { \prime } } { \sum } \log P _ { k } ( \boldsymbol { y } _ { i } | \boldsymbol { X } , \boldsymbol { \theta } ) } \\ { - \underset { k ^ { \prime } < k \leq k _ { 2 } } { \sum } \log P _ { k } ( \boldsymbol { y } _ { j } | \boldsymbol { X } , \boldsymbol { \theta } ) \bigg ) , } \end{array}\tag{4}
$$

and align the $( k _ { 1 } , k ^ { \star } ]$ -th output token to i and the $( k ^ { \star } , k _ { 2 } )$ -th output token to j as:

$$
\beta ( k ) = { \left\{ \begin{array} { l l } { i } & { { \mathrm { i f ~ } } k \in ( k _ { 1 } , k ^ { \star } ] } \\ { j } & { { \mathrm { i f ~ } } k \in ( k ^ { \star } , k _ { 2 } ) . } \end{array} \right. }\tag{5}
$$

As the illustration in Fig. 6, we enumerate all the possible end tokens of $\ ' \mathrm { A } '$ and $\because \mathbf { B } ^ { \prime }$ to find the best one. Then, we get the modified OAXE loss as:

$$
\mathcal { L } _ { M - O A X E } = - \sum _ { 1 \leq k \leq m } \log P _ { k } \left( y _ { \beta ( k ) } \vert X , \theta \right) .\tag{6}
$$

## C BLEU Scores of Different Losses on Different Language Pairs.

The BLEU scores of models with CTC, OAXE and CoCO loss on different languages pairs are shown

in Table 4.

## D Use CoCO Loss in DSLP

Partially feeding ground truth tokens to the decoder during training shows promising performance in NAT (Ghazvininejad et al., 2019; Saharia et al., 2020; Qian et al., 2021; Huang et al., 2021). For the models training with golden length of the ground truth sentence using XE loss, the ground truth token embedding is placed to the position of the corresponding input (Qian et al., 2021). When using CTC loss, the inputs of the decoder are always longer than the ground truth sentences, where Gu and Kong (2021) proposes to use the best monotonic alignment between the ground truth and output sequences, and provides the ground truth to the corresponding input position of the decoder. With the proposed CoCO loss, we use the best alignment which is not required to be monotonous. In addition, DSLP requires deep supervision on each layer of the decoder. We find that only replacing CTC loss with CoCO loss on the first layer is better than using CoCO loss on all layers. Accordingly, when using CoCO loss in DSLP transformer, we use CoCO loss in the first layer and CTC loss for all the other layers in the DSLP transformer.