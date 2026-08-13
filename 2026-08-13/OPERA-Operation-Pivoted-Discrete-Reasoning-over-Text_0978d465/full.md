# OPERA: Operation-Pivoted Discrete Reasoning over Text

Yongwei Zhou<sup>1</sup>, Junwei Bao<sup>2</sup>∗, Chaoqun Duan<sup>2</sup>, Haipeng Sun<sup>2</sup>, Jiahui Liang<sup>2</sup>, Yifan Wang<sup>2</sup>, Jing Zhao<sup>2</sup>, Youzheng Wu<sup>2</sup>, Xiaodong He<sup>2</sup>, Tiejun Zhao<sup>1</sup>† <sup>1</sup>Harbin Institute of Technology <sup>2</sup>JD AI Research ywzhou@hit-mtlab.net baojunwei@jd.com tjzhao@hit.edu.cn

## Abstract

Machine reading comprehension (MRC) that requires discrete reasoning involving symbolic operations, e.g., addition, sorting, and counting, is a challenging task. According to this nature, semantic parsing-based methods predict interpretable but complex logical forms. However, logical form generation is nontrivial and even a little perturbation in a logical form will lead to wrong answers. To alleviate this issue, multi-predictor -based methods are proposed to directly predict different types of answers and achieve improvements. However, they ignore the utilization of symbolic operations and encounter a lack of reasoning ability and interpretability. To inherit the advantages of these two types of methods, we propose OPERA, an operation-pivoted discrete reasoning framework, where lightweight symbolic operations (compared with logical forms) as neural modules are utilized to facilitate the reasoning ability and interpretability. Specifically, operations are first selected and then softly executed to simulate the answer reasoning procedure. Extensive experiments on both DROP<sup>1</sup> and RACENum datasets show the reasoning ability of OPERA. Moreover, further analysis verifies its interpretability. <sup>2</sup>

## 1 Introduction

Machine reading comprehension (MRC) that requires discrete reasoning is a valuable and challenging task (Dua et al., 2019), especially involving symbolic operations such as addition, sorting, and counting. The examples in Table 1 illustrate the task. To answer the question “Who threw the longest pass”, it requires a model to choose the person with the longest pass from all the people and

Passage: Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return...

Question: Who threw the longest pass? Answer: Russell

Table 1: An example of question-answer pair along with a passage from DROP dataset (Dua et al., 2019). Question words in color indicate the potential operations for reasoning, i.e., ARGMAX and KEY\_VALUE.

corresponding distance pairs based on the given passage. This task has various application scenarios in the real world, such as analyzing financial reports and sports news.

Existing approaches for this task can be roughly divided into two categories: the semantic-parsingbased (Chen et al., 2020b; Gupta et al., 2020) and the multi-predictor-based methods (Dua et al., 2019; Ran et al., 2019; Hu et al., 2019; Chen et al., 2020a; Zhou et al., 2021). The former maps the natural language utterances into logical forms and then executes them to derive the answers. For example, Chen et al. (2020b) propose NeRd. It includes a reader to encode the passage and question, and a programmer to generate a logical form for multi-step reasoning. Intuitively, this method has an advantage in interpretability. However, semantic parsing over text is nontrivial and even a little perturbation will result in wrong answers, which hinders the MRC performance.

To alleviate the heavy dependence on logical forms in the first category, the latter directly employs multiple predictors to derive different types of answers. For example, Dua et al. (2019) and Chen et al. (2020a) divide instances of the DROP dataset into several types and design a model with multi-predictors to deal with different answer types, i.e., question/passage span(s), count, and arithmetic expression. It is the capability of deriving different types of answers that improves the performance of models. However, such methods are lack of the necessary components to imitate discrete reasoning, which leads to inadequacy in reasoning ability and interpretability.

To alleviate the shortcomings of the above methods and preserve their advantages, we attempt to summarize reasoning steps into a set of operations and adopt them as the pivot to connect the question and the answer, which makes it possible to perform discrete reasoning. For example, to answer the question in Table 1, it needs two steps: (1) finding all persons and the corresponding distance of touchdown pass, and (2) choosing the one with the longest pass among them. We attempt to convert them into two operations, KEY\_VALUE and ARGMAX, respectively. We then use them to produce the answer. Specifically, we design a set of lightweight symbolic operations (compared with logical forms) to cover all of the questions in the datasets and utilize them as neural modules to facilitate reasoning ability and interpretability. We denote this method as OPERA, an operation-pivoted discrete reasoning MRC framework. To utilize the operations, we propose an operation-pivoted reasoning mechanism composed of an operation selector and an operation executor. Specifically, the operation selector automatically identifies relevant operations based on the input. To enhance the performance of this sub-mechanism, we further design an auxiliary task to learn the alignment from a question to operations according to a set of heuristics rules. The operation executor softly integrates the selected operations to perform discrete reasoning over text via an attention mechanism (Vaswani et al., 2017).

To verify the effectiveness of the proposed method, comprehensive experiments are conducted on both the DROP and RACENum datasets, where RACENum used in this paper is a subset of the RACE dataset following (Chen et al., 2020a). Experimental results indicate that our method outperforms strong baselines and achieves the stateof-the-art on both datasets under the single model setting. We further analyze the interpretability of OPERA. Overall, this paper primarily makes the following contributions:

(1) We propose OPERA, an operation-pivoted discrete reasoning MRC framework, improving both the reasoning ability and interpretability.

(2) Extensive experiments on DROP and RACENum dataset demonstrate the reasoning ability of OPERA. Moreover, statistic analysis and visualization indicate the interpretability of OPERA.

(3) We systematically design operations and heuristic rules to map questions to operations, aiming to facilitate research on symbolic reasoning.

## 2 Related Work

Recently, machine reading comprehension (MRC) methods tend to deal with more practical problems (Yang et al., 2018; Dua et al., 2019; Zhao et al., 2021), for example, answering complex questions that require discrete reasoning (Dua et al., 2019) such as arithmetic computing, sorting, and counting. Intuitively, semantic parsing-based methods, which are well explored to deal with discrete reasoning in question answering with structured knowledge graphs (Bao et al., 2016) or tables, have potential to address the discrete reasoning MRC problem. Therefore, semantic parsing-based methods for discrete reasoning over text are proposed to firstly convert the unstructured text into a table, and then answer questions over the structured table with a grammar-constrained semantic parser (Krishnamurthy et al., 2017). NeRd (Chen et al., 2020b) is a generative model that consists of a reader and a programmer, which are responsible for encoding the context into vector representation and generating grammar-constrained logical forms, respectively. NMNs (Gupta et al., 2020) learned to parse compositional questions as executable logical forms. However, it only adapts to limited question types matched with a few pre-defined templates.

Multi-predictor-based methods employ multiple predictors to derive different types of answers. NAQANET (Dua et al., 2019), a number-aware framework, employed multiple predictors to produce corresponding answer types, including a span, count, and arithmetic expression. Based on NAQANET, MTMSN(Hu et al., 2019) added a negation predictor to solve the negative question and re-ranked arithmetic expression candidates. To aggregate the relative magnitude relation between two numbers, NumNet (Ran et al., 2019) and Num-Net+ leveraged a graph convolution network to perform multi-step reasoning over a number graph. QDGAT (Chen et al., 2020a) proposed a questiondirected graph attention network for reasoning over a heterogeneous graph composed of entity and number nodes. EviDR (Zhou et al., 2021), an evidenceemphasized MRC method, performed reasoning over a multi-grained evidence graph. Compared with these existing methods, our proposed OPERA focuses on bridging the gap from questions to answers with operations and integrating them to simulate discrete reasoning.

<table><tr><td>Operations</td><td>Description</td><td>Examples</td></tr><tr><td>ADDITION/DIFF</td><td>Addition or subtraction</td><td>How many more yards was Kris Browns&#x27;s first field goal over his second?</td></tr><tr><td>MAX/MIN</td><td>Select the maximum/minimum one from given numbers</td><td>How many yards was the longest field goal in the game?</td></tr><tr><td>ARGMAX/ARGMIN</td><td>Select key with highest/lowest value from key-value pairs</td><td>Which player had the longest touchdown reception?</td></tr><tr><td>ARGMORE/ARGLESS</td><td>Select key with higher/lower value from two key-value pairs</td><td>Who scored more field goals, David Akers or John Potter?</td></tr><tr><td>COUNT</td><td>Count the number of spans</td><td>How many field goals did Kris Brown kick?</td></tr><tr><td>KEY_VALUE</td><td>Extract key-value pairs</td><td>How many percent of Forth Worth households owned a car?</td></tr><tr><td>SPAN</td><td>Select spans from input sequence</td><td>Which team scored the final TD of the game?</td></tr></table>

Table 2: All the operations, descriptions and the corresponding examples.

## 3 Approach

## 3.1 Task and Model Overview

Given a question Q and a passage P, MRC that requires discrete reasoning aims to predict an answer $\hat { A }$ with the maximum probability over the candidate space Ω as follows:

$$
\hat { A } = \arg \operatorname* { m a x } _ { A \in \Omega } p ( A | Q , P )\tag{1}
$$

where the answer $\hat { A }$ in this task could not only be span(s) extracted from context but also a number calculated with some numbers in context. To handle this task, it generally requires not only natural language understanding but also performing discrete reasoning over text, such as comparison, sorting and arithmetic computing.

To address the aforementioned challenges in this task, we propose OPERA, an operation-pivoted discrete reasoning MRC framework and it is briefly illustrated in Figure 1. In our framework, a set of operations $O P .$ , defined in Table $^ { 2 , }$ are introduced to support the modeling of answer probability $p ( A | Q , P )$ as follows:

$$
p ( A | Q , P ) = \sum _ { O \in O P } p ( A | Q , P , O ) p ( O | Q , P ) ,\tag{2}
$$

where $O \in { \mathcal { O P } }$ represents one of the operations. Concretely, in our framework, we first design an operation selector $p ( O | P , Q )$ for choosing the correct question-related operations. These selected operations are then softly executed over the given context. Eventually, answer predictor $p ( A | Q , P , O )$ utilizes the execution result to predict the final answer.

## 3.2 Definition of Operations

To imitate discrete reasoning, we design a set of operations as shown in Table 2. The set contains 11 operations and each one represents a reasoning unit. Specifically, for questions that need to be answered by calculation, we design operations ADDITION/DIFF to represent addition and subtraction. For questions which need to be answered by counting or sorting, we also design operations COUNT, MAX/MIN, ARGMAX/ARGMIN, and ARGMORE/ARGLESS. The rest operations KEY\_VALUE and SPAN are used to extract spans from the question and the passage. To incorporate them into OPERA, each operation is denoted as a tuple. Formally, i-th operation $O P _ { i }$ is $\langle \mathbf { E } _ { i } ^ { O P } , f _ { i } ( \cdot ) \rangle$ , where $i \in \{ 1 , 2 , . . . , n \}$ and n is the numbers of operations. $\mathbf { E } _ { i } ^ { O P } \in \mathbb { R } ^ { d _ { h } }$ represents the learnable embedding of the i-th operation. $f _ { i } ( \cdot )$ is a neural executor parameterized with trainable matrices $\mathbf { W } _ { q , i } ^ { O P } , \mathbf { W } _ { k , i } ^ { O P }$ and $\mathbf { W } _ { v , i } ^ { O P } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ . The neural executor $f _ { i } ( \cdot )$ is capable of performing execution of $O P _ { i }$ on the given context. Specifically, it takes the representation of context as input and outputs the executed representation as ${ \bf m } _ { i } ^ { { \dot { O P } } }$ (§ 3.3.2).

## 3.3 Architecture of OPERA

## 3.3.1 Context Encoder

The context encoder aims to learn the contextual representation of the input. Formally, given a question $Q$ and a passage P, we concatenate them into a sequence and feed it into a pre-trained language model (Liu et al., 2019; Clark et al., 2020; Lan et al., 2020) to obtain their whole representation H $\in \mathbb { R } ^ { l \times d _ { h } }$ . After that, we split H into the question and passage representations, which are respectively denoted as $\bar { \mathbf { H } ^ { Q } } \in \mathbb { R } ^ { l _ { q } \times d _ { h } }$ and $\mathbf { H } ^ { P } \in \mathbb { R } ^ { l _ { p } \times d _ { h } }$ $l _ { q } , l _ { p } ,$ , and l are the number of tokens in question, passage and concatenation of them. $d _ { h }$ is the dimension of the representations.

![](images/157f244d9819551a67c5a1c52399d83866c4f64ea29075084ce22d7d34c60b38.jpg)  
Figure 1: The architecture of OPERA. It consists of a context encoder, an operation-pivoted reasoning module, and a prediction module. The prediction module supports five types of answers, including question span, passage span, arithmetic expression, count, and multi-spans. MHA means a multi-head attention mechanism.

## 3.3.2 Operation-pivoted Discrete Reasoning

The operation-pivoted reasoning module is composed of an operation selector and an operation executor. The operation selector is adopted to select operations related to the given question. The operation executor is responsible for imitating the execution of the selected operations with an attention mechanism.

Operation Selector To imitate discrete reasoning, existing methods usually adopt a logical form generated by a semantic parser to address this task. However, these methods suffer severely from the cascade error, where a little perturbation in the logical form may result in wrong answers. Therefore, we propose to map each question into an operation set, instead of logical forms. Namely, we intend to select relevant operations from the $\mathcal { O P }$ . To this end, we adopt a bilinear function to compute the similarity between each operation and the question and normalize them with a softmax as follows:

$$
p ( O | Q , P ) = { \tt s o f t m a x } ( { \bf E } ^ { O P } { \bf W } { \bf h } ^ { Q } ) ,\tag{3}
$$

where $\mathbf { E } ^ { O P } \in \mathbb { R } ^ { n \times d _ { h } }$ is a learnable parameter, which demotes the operation embedding matrix. $\mathbf { h } ^ { Q } \in \mathbb { R } ^ { d _ { h } }$ is the representation of the question, which is obtained by executing weighted pooling on the $\mathbf { H } ^ { Q }$ $\mathbf { W } \in \dot { \mathbb { R } } ^ { d _ { h } \times d _ { h } }$ is a parameter matrix and $p ( O | Q , P )$ is the distribution over operations.

Operation Executor The operation executor is responsible for performing the execution of the selected operations over the given context. Inspired by previous studies (Andreas et al., 2016; Gupta et al., 2020), we implement the operation executor based on the neural module network, which takes advantage of neural network in fitting and generalization, and the composition characteristics of symbolic processing. Specifically, for each operation $\mathcal { O P } _ { i } = \langle \mathbf { E } _ { i } ^ { O P } , f _ { i } ( \cdot ) \rangle , i = \{ 1 , 2 , . . . , n \}$ , we use a multi-head cross attention mechanism (Vaswani et al., 2017) to implement $f _ { i } ( \cdot )$ . In detail, we leverage the embedding of each operation $\mathbf { E } _ { i } ^ { O P }$ as query and the representations of the whole input sequence H as keys and values, respectively, to model $f _ { i } ( \cdot )$ as follows:

$$
\alpha _ { i } ^ { O P } = \mathbf { s o f t m a x } ( \frac { ( \mathbf { E } _ { i } ^ { O P } \mathbf { W } _ { q , i } ^ { O P } ) ( \mathbf { H } \mathbf { W } _ { k , i } ^ { O P } ) ^ { T } } { \sqrt { d _ { h } } } ) ,
$$

$$
\mathbf { m } _ { i } ^ { O P } = \alpha _ { i } ^ { O P } ( \mathbf { H } \mathbf { W } _ { v , i } ^ { O P } ) ,\tag{4}
$$

(5)

where ${ \mathbf W } _ { q , i } ^ { O P } , { \mathbf W } _ { k , i } ^ { O P } , { \mathbf W } _ { v , i } ^ { O P } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ are the parameter matrices in executor of operation $O P _ { i }$ $\mathbf { m } _ { i } ^ { O P } \in \mathbb { R } ^ { d _ { h } }$ is the representation of the execution result of the i-th operation.

Finally, we softly integrate all of the execution results as the final output $\mathbf { h } ^ { O P } \in \mathbb { R } ^ { d _ { h } }$ with the distribution $p ( O | Q , P )$ as follows:

$$
\mathbf { h } ^ { O P } = \sum _ { i = 1 } ^ { n } p ( O = \mathcal { O P } _ { i } | Q , P ) \mathbf { m } _ { i } ^ { O P } .\tag{6}
$$

The operation-aware semantic representation $\mathbf { h } ^ { O P }$ is further fed into the prediction module to reason the final answer (§ 3.3.3).

As described above, OPERA introduces operations that assist in understanding questions and integrates them into the model to perform discrete reasoning. Therefore, it achieves an advantage in the reasoning capability and interpretability over the previous multi-predictor-based methods (Hu et al., 2019; Chen et al., 2020a; Zhou et al., 2021). Moreover, soft execution and composition of operations in OPERA alleviate the cascaded error that the semantic parsing methods (Ran et al., 2019; Chen et al., 2020b) suffer from. More experiments and analyses about reasoning ability and interpretability are illustrated in § 4.4 and § 4.5.

## 3.3.3 Prediction Module

In this section, we introduce the prediction module to derive different types of answers via multipredictors. Each predictor first reasons out a derivation and then performs execution to obtain the final answer. This answer prediction procedure is formalized as follows:

$$
p ( A | Q , P , O ) = \sum _ { D \in D } \mathbb { I } ( g ( D ) = A ) p ( D | Q , P , O ) ,\tag{7}
$$

where $\mathbb { I } ( g ( D ) = A )$ is an indicator function with value 1 if the answer A can be derived from a derivation executor $g ( \cdot )$ based on D, and 0 otherwise. $p ( D | Q , P , O )$ models the derivation prediction. Specifically, a derivation $D \ = \ \langle T , L \rangle$ includes an answer type $T$ and a corresponding label L. For example, in Table 3, the textual answer A of the question “how many yards was the longestfield goals in the game” is “80”. The possible derivations  to this answer include a span Span, (100, 102) , and an arithmetic expression $\langle \mathtt { A E } , ( 0 * 2 9 ) + ( 1 * 8 0 ) \rangle$ . Inspired by previous studies (Chen et al., 2020a; Zhou et al., 2021), the derivation predictor

$$
p ( D | Q , P , O ) = \sum _ { T \in \mathcal { T } } p _ { T } ( L | Q , P , O ) p ( T | Q , P , O )\tag{8}
$$

is decomposed into an answer type predictor $p ( T | Q , P , O )$ and corresponding label predictors $p _ { T \in \mathcal { T } } ( L | Q , P , O )$ where  = Question Span, Passage Span, Count, Arithmetic Expression, Multispans includes all the answer types defined in this paper. Each label predictor takes question-passage representation H and the operation-pivo representation $\mathbf { h } ^ { O P }$ as input and calculates the probability of label L. Specifically, these label predictors are specified as follows and more details are shown in Appendix A.3.

Question / Passage Span The probability of a question/passage span is the product of the probabilities of the start index and the end index. Following

MTMSN (Hu et al., 2019), we use a question-aware decoding strategy to predict the start and end index across the input sequence, respectively.

Count As indicated in QDGAT (Chen et al., 2020a), questions with 0-9 as answers account for 97% in all the count questions. Hence, such questions are modeled as a 10-class (0-9) classification problem.

Arithmetic Expression Similar to NAQANet (Dua et al., 2019), we first assign a sign (positive, negative, or zero) to each number in the context and then compute the answer by summing them.

Multi-spans Inspired by Segal et al. (2020), the multi-span answer (a set of non-contiguous spans) is derived with a sequence labeling method, in which each token of the input is tagged with BIO labels. Finally, each span which is tagged with continuous B and I is taken as a candidate span.

## 3.4 Learning with Weak Supervision

## 3.4.1 Training Instance Construction

Each training instance is originally composed of a passage $P ,$ , a question Q, and answer text A. Since the derivations (i.e., labels for the spans, arithmetic expressions, and count) are not provided, weak supervision is adopted in OPERA. Specifically, for each training instance, given the golden textual answer A, we heuristically search all the possible derivations as supervision signals, each of which can derive the correct answer A. Table 3 shows an example of .

In addition, we propose heuristic rules to map a question to its related operations denoted as $\mathcal { O } \subseteq \mathcal { O } \mathcal { P }$ . For example, to detect the operations intimated by the question $Q$ in Table 3, we design a question template “how many yards [Slot] longest $I S l o t { j } ^ { \mathrm { v } }$ which maps matched questions to the operation MAX. Overall, a training instance can be constructed as a tuple $\langle P , Q , A , \mathcal { O } , \mathcal { D } \rangle$ . The one-shot heuristic rules to obtain operation labels reduce the cost of human annotations. Moreover, when applying OPERA to other discrete reasoning MRC tasks, both the operations and the heuristic rules can be extended and adjusted if necessary. Fortunately, there is no need to construct strict logical forms in our architecture, but only the set of lightweight operations involved in the question. It tremendously reduces the difficulty of adapting OPERA to other discrete reasoning MRC tasks.

Meanwhile, we analyze the distribution of operations in the training set. More details about the heuristic rules for mapping questions to operations and the operation distribution in the dataset are respectively given in the Appendix A.1 and A.2.

P ...Oakland would take the lead in the third quarter   
with wide receiver Johnnie Lee Higgins catching a   
29-yard touchdown passfrom Russell,followed up   
by an 80-yard punt returnfor a touchdown ...   
<sup>Q</sup> <sub>A</sub> How many yards was the longestfield goals   
80   
O MAX How many yards [Slot] longest [Slot]   
D Span, (100, 102) ; AE, (0 29) + (1 80)  
Table 3: An example of building training instances.

## 3.4.2 Joint Training

The training objective consists of two parts, including the loss for answer prediction and operation selection. The loss for answer prediction $\mathcal { L } _ { a }$ is

$$
{ \mathcal { L } } _ { a } = - \log p ( A | Q , P ) .\tag{9}
$$

Note that the calculation of loss $\mathcal { L } _ { a }$ takes all possible derivations that can obtain the correct answer A into account, which means that OPERA does not require labeling answer types for training. In addition, to learn better alignment from a question to operations, we introduce auxiliary supervision for the operation selector and calculate the loss

$$
\mathcal { L } _ { o p } = - \sum _ { O \in \mathcal { O } } \log p ( O | Q , P ) ,\tag{10}
$$

where indicates the operations provided by the heuristic rules. Finally, OPERA is optimized by minimizing the loss $\mathcal { L } = \mathcal { L } _ { a } + \lambda \mathcal { L } _ { o p }$ where λ is a hyperparameter as a trade-off of the two objectives.

## 4 Experiment

## 4.1 Dataset and Evaluation

We conduct experiments on the following two MRC datasets to examine the discrete reasoning capability of our model. We employ Exact Match (EM) and F1 score as the evaluation metrics.

DROP Question-answer pairs in DROP dataset (Dua et al., 2019) are crowd-sourced based on passages collected from Wikipedia. In detail, it contains 96.6K question-answer pairs, where 77400/9536/9615 samples are for training/development/test. Three kinds of answers are involved in the raw dataset, i.e., NUMBER (60.69%), SPANS (37.72%), and DATE (1.59%).

RACENum To investigate the generalization capability of OPERA, we compare OPERA to other strong baselines on samples of RACE (Lai et al., 2017). Following Chen et al. (2020a), we sample instances from RACE, denoted as RACENum, where the question of each instance starts with “how many”. To conveniently evaluate the models on RACENum, we convert the format of instances in RACENum into the same as DROP, since RACE is a multi-choice MRC dataset. RACENum is divided into two categories, i.e., middle/high school exam (RACENum-M/H). They respectively contain 609 and 565 questions, where the scale is a bit different from that reported in Chen et al. (2020a)<sup>3</sup>.

## 4.2 Baselines

We compare OPERA with various prior systems in terms of reasoning capability and interpretability. w/o Pre-trained Language Model: NAQANET (Dua et al., 2019) leverages several answer predictors to produce corresponding types of answers, including a span, count, and arithmetic expression. NumNet (Ran et al., 2019) leverages a graph convolution network to reason over a number graph aggregated relative magnitude among numbers.

w/ Pre-trained Language Model: GenBERT (Geva et al., 2020) injects reasoning capability into BERT by pre-training with large-scale numerical data. Based on NAQANET, MTMSN (Hu et al., 2019) adds a negation predictor to solve the negative question and re-rank arithmetic expression candidates. NeRd (Chen et al., 2020b) is essentially a generative semantic parser that maps questions and passages into executable logical forms. ALBERT-Calc was proposed for DROP by combining ALBERT with several predefined answer predictors (Andor et al., 2019). NumNet+ employs a pre-trained model to further boost the performance of NumNet. QDGAT (Chen et al., 2020a) builds a heterogeneous graph composed of entity and value nodes upon RoBERTa and utilizes a questiondirected graph attention network to reason over the graph. EviDR (Zhou et al., 2021), an evidenceemphasized MRC model, performs reasoning over a multi-grained evidence graph based on ELEC-TRA.

## 4.3 Implementation Details

We utilize adam optimizer (Kingma and Ba, 2015) with a cosine warmup mechanism and set the weight of loss $\lambda = 0 . 3$ to train the model. The hyper-parameters are listed in Table 4, where BLR, LR, BWD, WD, BS, and $d _ { h }$ respectively represent the learning rate of the encoder, the learning rate of other parts of the model, the weight decay of the encoder, the weight decay of other parts of the model, batch size and hidden size of the model. Each operation is neutralized with a multi-head attention layer with $n _ { h }$ heads and $d _ { h }$ dimension.

<table><tr><td></td><td>BLR</td><td>LR</td><td>BWD</td><td>WD</td><td>Epochs</td><td>BS</td><td> $n _ { h }$ </td><td> $d _ { h }$ </td></tr><tr><td>RoBERTa</td><td>1.5e-5</td><td>5e-4</td><td>0.01</td><td>5e-5</td><td>12</td><td>16</td><td>16</td><td>1024</td></tr><tr><td>ELECTRA</td><td>1.5e-5</td><td>5e-4</td><td>0.01</td><td>5e-5</td><td>12</td><td>16</td><td>16</td><td>1024</td></tr><tr><td>ALBERT</td><td>3e-5</td><td>1e-4</td><td>0.01</td><td>5e-5</td><td>8</td><td>128</td><td>64</td><td>4096</td></tr></table>

Table 4: Hyperparameters settings for training OPERA.

## 4.4 Main Results

## 4.4.1 Results on DROP and Analysis

Table 5 shows the overall results of OPERA and all the baselines on the DROP dataset. OPERA achieves comparable and even higher performance than the recently available methods. Specifically, OPERA(RoBERTa) achieves comparable performance to QDGAT with advantages of 0.32 EM and 0.42 F1. OPERA(ELECTRA) exceeds EviDR by 0.89 EM and 0.90 F1 and OPERA(ALBERT) outperforms ALBERT-Calc by 4.84 EM and 4.24 F1. Moreover, the voting strategy is employed to ensemble 7 OPERA(ALBERT) models with different random seeds, achieving 86.26 EM and 89.12 F1 scores. We think the better performance comes from the modeling of discrete reasoning over text via operations, which mines more semantic information of context and explicitly integrates them into the answer prediction.

## 4.4.2 Results on RACENum

To investigate the generalization of OPERA for discrete reasoning, we additionally compare OPERA with QDGAT and NumNet+ on the RACENum dataset. We directly evaluate the three models without fine-tuning on RACENum due to its small scale. As Table 6 shows, the scores of models on the RACENum dataset are generally lower than that on the DROP dataset, which is attributed to the lack of in-domain training data. Nevertheless, the performance of OPERA significantly outperforms NumNet+ and QDGAT by a large margin of more than 3.49 EM and 3.53 F1 score on average. It indicates that OPERA has better generalization ability.

<table><tr><td rowspan="2">Method</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>w/o Pre-trained Models NAQANet</td><td></td><td>49.24</td><td>44.07</td><td>47.01</td></tr><tr><td>NumNet</td><td>46.20 64.92</td><td>68.31</td><td>64.56</td><td>67.97</td></tr><tr><td>w/ Pre-trained Models GenBERT</td><td>68.80</td><td>72.30</td><td>68.6</td><td>72.35</td></tr><tr><td>MTMSN</td><td>76.68</td><td>80.54</td><td>75.88</td><td>79.99</td></tr><tr><td>NeRd</td><td>78.55</td><td>81.85</td><td>78.33</td><td>81.71</td></tr><tr><td>ALBERT-Calc</td><td>80.22</td><td>83.98</td><td>79.85</td><td>83.56</td></tr><tr><td>NumNet+ EviDR</td><td>81.07</td><td>84.42</td><td>81.52</td><td>84.84</td></tr><tr><td>QDGAT</td><td>82.09 82.74</td><td>85.14 85.85</td><td>82.55 83.23</td><td>85.80 86.38</td></tr><tr><td>Single Model Results OPERA(RoBERTa)</td><td>83.74</td><td>86.52</td><td>83.55</td><td>86.80</td></tr><tr><td>OPERA(ELECTRA) OPERA(ALBERT)</td><td>83.86 84.86</td><td>86.66 87.54</td><td>83.46 84.69</td><td>86.70 87.80</td></tr><tr><td>Ensemble Results OPERA</td><td>86.79</td><td>89.41</td><td>86.26</td><td>89.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Human</td><td></td><td></td><td>94.90</td><td>96.42</td></tr></table>

Table 5: Results on the DROP dataset. We solely compare with QDGAT, but leaving $\mathrm { Q D G A T } _ { p }$ alone, since we focus on the reasoning mechanism in this work, while $\mathrm { Q D G A T } _ { p }$ is a variant of QDGAT with data augmentation (Chen et al., 2020a).
<table><tr><td rowspan="2">Method</td><td colspan="4">RACENum-M RACENum-H</td><td colspan="2">Avg</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>NumNet+</td><td>41.71</td><td>41.82</td><td>29.73</td><td>29.73</td><td>35.94</td><td>36.00</td></tr><tr><td>QDGAT</td><td>44.01</td><td>44.01</td><td>28.85</td><td>28.85</td><td>36.71</td><td>36.71</td></tr><tr><td>OPERA</td><td>47.62</td><td>47.62</td><td>32.21</td><td>32.30</td><td>40.20</td><td>40.24</td></tr></table>

Table 6: The performance of RoBERTa-based models on the RACENum dataset without finetuning.

## 4.5 Interpretability Analysis

Interpretability is an essential property for evaluating an MRC model. We analyze the interpretability of OPERA from the following two stages: (1) mapping from questions to operations, and (2) mapping from operations to answers.

Mapping from Question to Operation To explicitly show the correlations between questions and related operations, we manually evaluate the performance of the operation selection on 50 samples on the development set of DROP. Specifically, precision@n (P@n) is used as the evaluation metric, i.e., judging whether the top n predicted operations contain the correct ones according to questions. We finally achieve 0.88 on P@1 and 0.98 on P@2 for our model OPERA, which indicates that the operation selection module can accurately predict interpretable operations.

![](images/1ac754c5bce717a3c30cb51bdf87708096be4b0110ebc4cb83533054c5967e46.jpg)  
Figure 2: Statistical correlations between operations and answer types. The horizontal axis indicates answer types, and the vertical axis means operations. PS, CT, AE, MS and QS respectively means passage span, count, arithmetic expression, multi-spans and question span.

Mapping from Operation to Answer We explore the relations between operations and final answer types based on model predictions on the development set of DROP. Specifically, for each type of answer, the predicted operation distributions are summed over all samples and normalized, which derives a relation matrix as shown in Figure 2. We can observe that obvious correlations between operations and answer types exist. ADDITION, DIFF, MAX and MIN obviously correspond to Arithmetic Expression. The top 3 answer types for KEY\_VALUE and SPAN are Passage Span, Multiple Spans, and Question Span. COUNT probably maps to Count answer type. ARGMORE, ARGLESS, ARGMAX, and ARGMIN are usually used for both Passage Span and Arithmetic Expression.

## 4.6 Ablation Study

Effect of Operations for OPERA As shown in Table 7, we first remove the loss of operation selection (w/o $\mathcal { L } _ { o p } )$ by setting λ = 0, resulting in the performance degradation of 0.74 EM / 0.52 F1 points and 0.14 EM / 0.18 F1 points for OPERA based on RoBERTa and ALBERT, respectively. It indicates that the supervision for explicitly learning the alignment from a question to operations contributes to the reasoning capability of OPERA. Yet our approach is somewhat limited by the fact that the operation selector needs an auxiliary training task to work better, and heuristics rules are required to map questions into an operation set as training labels. Furthermore, we remove the total operation-pivoted reasoning module (w/o OP), the performance respectively declines by 1.06 EM /

![](images/9bd1598b857cc35e57a9edeed7a210cdc7f15358f8612cdcdcd280d5f2758f69.jpg)

Figure 3: Ablation study on all the subsets of DROP containing some specific operations.
<table><tr><td rowspan="2">Method</td><td colspan="2">NUM (60.69%) SPANS (37.72%)</td><td rowspan="2">Overall</td></tr><tr><td>EM/F1</td><td>EM/F1</td></tr><tr><td>OPERA(RB)</td><td>85.91 / 86.14</td><td>83.89 / 88.90</td><td>83.74 / 86.52</td></tr><tr><td>w/o  $\mathcal { L } _ { o p }$ </td><td>85.83 / 86.00</td><td>82.68 / 88.02</td><td>83.00 / 86.00</td></tr><tr><td>w/o OP</td><td>85.36 / 85.53</td><td>82.53 / 87.06</td><td>82.68 / 85.73</td></tr><tr><td>OPERA(AB)</td><td>86.39 / 86.58</td><td>86.30 / 90.99</td><td>84.86 / 87.54</td></tr><tr><td>w/o  $\mathcal { L } _ { o p }$ </td><td>86.08 / 86.24</td><td>86.39 / 90.96</td><td>84.72 / 87.36</td></tr><tr><td>w/o OP</td><td>85.62 / 85.89</td><td>84.95 / 89.83</td><td>84.01 / 86.79</td></tr></table>

Table 7: Ablation study on the dev set of DROP. RB and AB mean RoBERTa and ALBERT, respectively.

0.79 F1 points and 0.85 EM / 0.75 F1 points for OPERA(RoBERTa) and OPERA(ALBERT). We also conduct the ablation study on the subsets containing a specific operation. As shown in Figure 3, OPERA achieves better performance than OPERA w/o OP on the majority of subsets. Overall, it confirms that integrating the operation-pivoted discrete reasoning mechanism contributes to the reasoning ability of the model.

Probe on Answer Types and Language Models As reported in Table 7, we observe that the performance on the NUMBER(NUM) and SPANS questions, which together account for 98.4% of the total, respectively declines by 0.55 EM / 0.61 F1 and 1.36 EM / 1.84 F1 when removing operation-pivoted reasoning mechanism from OPERA(RB). It demonstrates that this mechanism contributes to various answer types. Also, we respectively evaluate the performance of OPERA based on RoBERTa and ALBERT. We observe that OPERA(ALBERT) outperforms OPERA(RoBERTa) due to the stronger capability of semantic representation. Furthermore, integrating this mechanism consistently contributes to the performance of OPERA no matter it is based on RoBERTa or ALBERT. It indicates that OPERA could compensate for the discrete reasoning capability of language models.

<table><tr><td>Question-Answer</td><td>Passage</td><td>NumNet+</td><td>QDGAT</td><td>OPERA</td></tr><tr><td>Q: How many total yards of touchdown passes were there? A: 73</td><td>... receiver Johnny Knox on a 23-yard touchdown pass. Afterwards, the Falcons took the lead as quar- terback Matt Ryan completed a 40-yard touchdown pass to wide receiver Roddy White and a 10-yard touchdown pass to tight end Tony Gonzalez ...</td><td>AnswerType: Count Answer: 0</td><td>AnswerType: Count Answer: 0</td><td>Top-1 OP: ADDITION AnswerType: Arithmetic Expression Answer: 23+40+10=73</td></tr><tr><td>Q: Which period was Wolf ex- ecutive and player personnel di- rector with the Oakland Raiders longer for, 1963-1974 or 1979-1989? A: 1963-1974</td><td>Wolf only had a brief stint with the Jets between 1990 and 1991, while most of his major contribu- tions occurred as an executive and player personnel director with the Oakland Raiders (1963-1974, 1979- 1989), and later as General Manager...</td><td>AnswerType: Passage Span Answer: 1979-1989</td><td>AnswerType: Passage Span Answer: 1979-1989</td><td>Top-1 OP: SPAN AnswerType: Passage Span Answer: 1963-1974</td></tr></table>

Table 8: The cases from the development set of DROP. The predictions from the state-of-the-art model NumNet+ and QDGAT are shown. The last column indicates our predicted answers and Top-1 operations.

## 4.7 Case Study

We show two examples from the development set of DROP to illustrate the effectiveness of our model by comparing the results of different models in Table 8. The first example shows that operation is essential for the prediction of answer type. Num-Net+ and QDGAT fail to predict the correct answer since the answer type of “how many” questions are wrongly predicted to Count. In contrast, OPERA can capture the ADDITION operation, which prompts the model to answer it with an arithmetic expression predictor. The second example shows that OPERA has stronger reasoning capability. In the example, though NumNet+ and QDGAT correctly predict the answer type, the final answer is wrong. OPERA can utilize more semantic information for answer prediction with the help of the operation-pivoted discrete reasoning mechanism.

## 5 Conclusion

We propose a novel framework OPERA for machine reading comprehension requiring discrete reasoning. Lightweight and one-shot operations and heuristic rules to map questions to an operation set are systematically designed. OPERA can leverage the operations to enhance the model’s reasoning capability and interpretability. Experiments on DROP and RACENum demonstrate that OPERA achieves remarkable performance. Further visualization and analysis verify its interpretability.

## 6 Acknowledge

We would like to thank all the anonymous reviewers for their useful feedback. This work is supported by the project of the National Natural Science Foundation of China (No.U1908216) and the

National Key Research and Development Program of China (No. 2020AAA0108600).

## References

Daniel Andor, Luheng He, Kenton Lee, and Emily Pitler. 2019. Giving BERT a calculator: Finding operations and arguments with reading comprehension. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5947– 5952, Hong Kong, China. Association for Computational Linguistics.

Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. 2016. Neural module networks. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 39–48. IEEE Computer Society.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. ArXiv preprint, abs/1607.06450.

Junwei Bao, Nan Duan, Zhao Yan, Ming Zhou, and Tiejun Zhao. 2016. Constraint-based question answering with knowledge graph. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 2503–2514, Osaka, Japan. The COLING 2016 Organizing Committee.

Kunlong Chen, Weidi Xu, Xingyi Cheng, Zou Xiaochuan, Yuyu Zhang, Le Song, Taifeng Wang, Yuan Qi, and Wei Chu. 2020a. Question directed graph attention network for numerical reasoning over text. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6759–6768, Online. Association for Computational Linguistics.

Xinyun Chen, Chen Liang, Adams Wei Yu, Denny Zhou, Dawn Song, and Quoc V. Le. 2020b. Neural symbolic reader: Scalable integration of distributed and symbolic representations for reading comprehension. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. ELECTRA: pretraining text encoders as discriminators rather than generators. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics.

Mor Geva, Ankit Gupta, and Jonathan Berant. 2020. Injecting numerical reasoning skills into language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 946–958, Online. Association for Computational Linguistics.

Nitish Gupta, Kevin Lin, Dan Roth, Sameer Singh, and Matt Gardner. 2020. Neural module networks for reasoning over text. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Dan Hendrycks and Kevin Gimpel. 2016. Bridging nonlinearities and stochastic regularizers with gaussian error linear units.

Minghao Hu, Yuxing Peng, Zhen Huang, and Dongsheng Li. 2019. A multi-type multi-span network for reading comprehension that requires discrete reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1596–1606, Hong Kong, China. Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Jayant Krishnamurthy, Pradeep Dasigi, and Matt Gardner. 2017. Neural semantic parsing with type constraints for semi-structured tables. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1516–1526, Copenhagen, Denmark. Association for Computational Linguistics.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785– 794, Copenhagen, Denmark. Association for Computational Linguistics.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A lite BERT for self-supervised learning of language representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv preprint, abs/1907.11692.

Qiu Ran, Yankai Lin, Peng Li, Jie Zhou, and Zhiyuan Liu. 2019. NumNet: Machine reading comprehension with numerical reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2474–2484, Hong Kong, China. Association for Computational Linguistics.

Elad Segal, Avia Efrat, Mor Shoham, Amir Globerson, and Jonathan Berant. 2020. A simple and effective model for answering multi-span questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3074–3080, Online. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Jing Zhao, Junwei Bao, Yifan Wang, Yongwei Zhou, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2021. RoR: Read-over-read for long document machine reading comprehension. In Findings of the Associationfor Computational Linguistics: EMNLP 2021, pages 1862–1872, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yongwei Zhou, Junwei Bao, Haipeng Sun, Jiahui Liang, Youzheng Wu, Xiaodong He, Bowen Zhou, and Tiejun Zhao. 2021. Evidr: Evidence-emphasized discrete reasoning for reasoning machine reading comprehension. ArXiv preprint, abs/2108.07994.

## A Appendix

## A.1 Template-Based Heuristic Rules

<table><tr><td rowspan=1 colspan=1>Operations / Examples / Templates</td></tr><tr><td rowspan=1 colspan=1>ADDITION/DIFFHow many more yards was Kris Browns&#x27;s first field goal overhis second?How many [Slot] more/less [Slot] over [Slot]?</td></tr><tr><td rowspan=1 colspan=1>MAX/MINHow many yards was the longest field goal in the game?how many yards [Slot] longest/shortest [Slot]?</td></tr><tr><td rowspan=1 colspan=1>ARGMAX/ARGMINWhich player had the longest touchdown reception?Which player [Slot] longest/shortest [Slot]?</td></tr><tr><td rowspan=1 colspan=1>ARGMORE/ARGLESSWho scored more field goals, David Akers or John Potter?Who [Slot] more/less, [Slot] or [Slot]?</td></tr><tr><td rowspan=1 colspan=1>COUNTHow many field goals did Kris Brown kick?How many field goals [slot]?</td></tr><tr><td rowspan=1 colspan=1>KEY_VALUEHow many percent of Forth Worth households owned a car?How many percent of [Slot]</td></tr><tr><td rowspan=1 colspan=1>SPANWhich team scored the final TD of the game?Which team [Slot]</td></tr></table>

Table 9: All the operations, the corresponding examples and templates.

In this section, we propose some general template-based heuristic rules to illustrate mapping from questions to operations. For example, to detect the operations intimated by the question "how many yards was the longest field goals" in Table 9, we design a question template $O P _ { p a t }$ "how many yards [Slot] longest/shortest [Slot]" which maps matched questions to the operation MAX. Meanwhile, we humanly evaluate the quality of the heuristic rules. Specifically, we sample 50 instances from the training set and ask three annotators to label the required operations for each question manually. The final average F1 score of the three annotators is 86%, which indicates that the heuristic rules can correctly predict most of the operations, while still 14% to be noise for training.

## A.2 Distribution of the Operations

We analysis the distribution of operations in the training set of DROP, where ADDITION, DIFF and SPAN together accounts for more than 85%. For other questions with span answers that requires sorting or comparison, some specific operations are involved, such as ARGMAX / ARGMIN / ARGMORE / ARGLESS and KEY\_VALUE.

![](images/37c87863d80daae9750d2fa36c44fc943cada478ff727deb6fd016b16daccfce.jpg)  
Figure 4: The Distribution of operations in the training set of DROP.

## A.3 Details of Prediction Module

In this section, we reveal the architecture details of the prediction module, including a prediction module for answer type and five label predictors corresponding to different answer types. FFN( ) means a feed-forward network that consists of two linear projections with a GeLU activation function (Hendrycks and Gimpel, 2016) and a layer normalization mechanism (Ba et al., 2016).

Answer Type The probability distribution of answer type choices $p ( T | Q , P , O )$ is derived by a -classifier with $\mathbf { h } ^ { Q } , \dot { \mathbf { h } } ^ { P }$ and $\mathbf { h } ^ { E }$ as input:

$$
\begin{array} { r l } & { { \bf h } ^ { E } = \displaystyle \sum _ { O _ { i } \in \mathcal { O P } } p ( O _ { i } | Q , P ) { \bf E } _ { i } ^ { O P } , } \\ & { p ( T | Q , P , O ) \propto \mathrm { F F N } ( [ { \bf h } ^ { E } ; { \bf h } ^ { Q } ; { \bf h } ^ { P } ] ) , } \end{array}\tag{11}
$$

where $\mathbf { h } ^ { Q }$ and $\mathbf { h } ^ { P } \in \mathbb { R } ^ { d _ { h } }$ is the representation vector of question and passage calculated by weighted pooling with $\mathbf { H } ^ { Q }$ and $\bar { \mathbf { H } } ^ { P }$ , respectively. ${ \bf E } ^ { \bar { O } P }$ is the embedding matrix of operations.

Question/Passage Span Following MTMSN (Hu et al., 2019), we use a question-aware decoding strategy to predict the start and end indices of the answer span. Specifically, we first compute a question representation vector $\mathbf { g } ^ { Q }$ via weighted pooling. Then derive the probabilities of the start and end indices of the answer span denoted as $p _ { s }$ and $p _ { e }$ :

$$
\begin{array} { r l } & { \mathbf { M } = [ \mathbf { h } ^ { O P } ; \mathbf { H } ; \mathbf { H } \odot \mathbf { g } ^ { Q } ] , } \\ & { p _ { s } , p _ { e } \propto \mathtt { F F N } ( \mathbf { M } ) , } \end{array}\tag{12}
$$

where denotes element-wised product. $\mathbf { h } ^ { O P }$ is derived by Eq. 6 and H is the representation of input sequence from context encoder.

Count The count predictor is a 10-classifier with the operation-aware representation, all the mentioned number representation, question and passage representations as input. Specifically, when $N$ numbers exists, we gather the representation of all numbers ${ \bf U } = ( \bar { \bf u } ^ { 1 } , \bar { \bf u } ^ { 2 } , . . . , \bar { \bf u } ^ { N } ) \in \mathbb { R } ^ { N \times d _ { h } }$ from H and compute a global representation vector of numbers as $\mathbf { h } ^ { U }$ . Then compute the probability distribution of count answer $p _ { c } \mathbf { . }$

$$
\begin{array} { r } { \boldsymbol \alpha ^ { U } \propto \mathbf { U } \mathbf { W } ^ { U } , ~ \mathbf { h } ^ { U } = \boldsymbol \alpha ^ { U } \mathbf { U } , } \\ { p _ { c } \propto \mathbf { F } \mathbf { F } \mathbb { N } ( [ \mathbf { h } ^ { O P } ; \mathbf { h } ^ { U } ; \mathbf { h } ^ { Q } ; \mathbf { h } ^ { P } ] ) , } \end{array}\tag{13}
$$

Arithmetic Expression Similar to NAQANet (Dua et al., 2019), we perform addition and subtraction over all the numbers mentioned in the context by assigning a sign (plus, minus, or zero) to each number. The probability $p _ { s i g n } ^ { i }$ of the i-th number’s sign is derived as below:

$$
p _ { s i g n } ^ { i } \propto \mathtt { F F N } ( [ \mathbf { h } ^ { O P } ; \mathbf { u } ^ { i } ; \mathbf { h } ^ { Q } ; \mathbf { h } ^ { P } ] ) .\tag{14}
$$

Multi-Spans Inspired by Segal et al. (2020), the multi-span answer is derived with a sequence role labeling method over the input token sequence, denoted as $\mathtt { S R L } ( \cdot ) . \ p _ { m s }$ is the probability distribution of token’s BIO tag:

$$
p _ { m s } = \mathtt { S R L } ( [ \mathbf { H } ; \mathbf { h } ^ { O P } ] ) .\tag{15}
$$