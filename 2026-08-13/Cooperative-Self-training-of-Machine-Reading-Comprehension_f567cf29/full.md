# Cooperative Self-training of Machine Reading Comprehension

Hongyin Luo<sup>1</sup> Shang-Wen Li<sup>2</sup> Mingye Gao<sup>3</sup> Seunghak Yu<sup>2</sup>∗ James Glass<sup>1</sup> <sup>1</sup>MIT CSAIL, <sup>2</sup>Amazon AI <sup>3</sup>MIT MTL

{<sup>hyluo,mingye,glass</sup>}<sup>@mit.edu,</sup>{<sup>shangwel,yuseungh</sup>}<sup>@amazon.com</sup>

## Abstract

Pretrained language models have significantly improved the performance of downstream language understanding tasks, including extractive question answering, by providing high-quality contextualized word embeddings. However, training question answering models still requires large amounts of annotated data for spe cific domains. In this work, we propose a cooperative self-training framework, RGX, for automatically generating more non-trivial questionanswer pairs to improve model performance. RGX is built upon a masked answer extraction task with an interactive learning environment containing an answer entity Recognizer, a question Generator, and an answer eXtractor. Given a passage with a masked entity, the generator generates a question around the entity, and the extractor is trained to extract the masked en tity with the generated question and raw texts. The framework allows the training of question generation and answering models on any text corpora without annotation. We further leverage a self-training technique to improve the performance of both question generation and answer extraction models. Experiment results show that RGX outperforms the stateof-the-art (SOTA) pretrained language models and transfer learning approaches on standard question-answering benchmarks, and yields the new SOTA performance under given model size and transfer learning settings.

## 1 Introduction

Recent studies have shown that language model pretraining provides high-quality text representations and significantly improves neural networks’ performance on a variety of natural language processing (NLP) tasks (Peters et al., 2018). Based on the popular Transformer architecture (Vaswani et al., 2017), various language models have been proposed (Devlin et al., 2018; Liu et al., 2019; Clark et al., 2020). These models are pretrained to predict a masked word in a given context from large corpora, and generate a contextual representation that encodes semantic and syntactic information. After finetuning, these representations significantly improve performance on downstream NLP tasks. Although masked language modeling is a powerful self-supervised learning technique, annotation on large-scaled data is still necessary for finetuning on difficult downstream tasks, including extractive question answering (QA)<sup>1</sup> where a large number of labeled question-answer pairs are required as a training corpora.

![](images/f4a3c301061729d803a423269be872dc8ae7bcfb053e48225cc14cd2f05b8074.jpg)  
Figure 1: The pipeline of semi-supervised question answering (machine reading comprehension) by RGX. AER (answer entity Recognition) agent recognizes answer entity from a given passage; QG (question Generation) generates a question based on the passage and entity; QAE (question-answering eXtractor) extracts answer from the question and passage.

Previous studies showed that the QA models can be improved by training on synthetic questionanswer pairs, namely self-training (Sachan and Xing, 2018; Puri et al., 2020; Shakeri et al., 2020; Bartolo et al., 2021). The core step of these work is pretraining a question-answer pair synthesis model on a seed corpus, and apply the generator on target domains to obtain synthetic training data. The QA model learns domain knowledge after finetuning on the synthetic data, and thus the domain adaptation is improved. However, the gap between the pretraining (i.e., seed) and the target corpus still exists, in terms of domain knowledge, question difficulty, and language style. The gap affects the quality of the synthetic training data.

We thus propose a framework that allows cooperative self-training for both QA pair synthesis and question answering to better adapt the synthesis models to the target domain and improve the learning of the QA models. In the framework, we construct a cooperative environment where a question generator and an answer extractor work together to solve a masked entity prediction problem. We first leverage an entity recognizer to mask out an entity in a provided passage. The question generator then outputs a question based on the masked passage. With the generated question and the original, unmasked passage, we train the answer extractor to select the correct answer spans, which are the masked entity. The extractor is also the final model used for extractive QA. To extract the spans accurately, the generator has to provide a good question, and the extractor should select the most likely tokens. We apply an expectationmaximization algorithm to select high-quality QA pairs and update both question generation and answer extraction models to improve the quality of synthetic data and the accuracy of the self-trained QA model based on synthetic QA pairs. We call our algorithm RGX since it incorporates an answer entity Recognizer, a question Generator, and a question-answering eXtractor. The RGX pipeline is illustrated in Figure 1.

With RGX, we can train a QA model for any unlabeled target domain given the corresponding text corpora and a labeled QA corpus in a seed domain (either the same or different from the target). By training QA models on synthetic QA data generated by RGX and evaluating the trained model on human-labeled evaluation data, we show that RGX outperforms SOTA approaches in QA benchmark datasets when domain specific human labels are not available during finetuning. In this work, we make the following contributions:

1. We propose a cooperative self-training framework, RGX, which contains an answer entity recognition, question generation, and answer span extraction to automatically generate nontrivial QA pairs on unlabeled corpora.

2. We design a expectation-maximization (EM) synthetic QA selection that identifies difficult but answerable questions without supervision to incrementally train the QA model with challenging examples, and an answer entity recognition (AER) based maximum mutual information (MMI) inference method for question answering.

3. Experiments show that our method significantly outperforms SOTA pretrained QA models and self-training QA baselines.

## 2 Related Work

Reinforcement learning and self-training have emerged recently for learning language generation in addition to maximum likelihood training. To optimize text generation models directly with non-differentiable objective functions, Rennie et al. (2017) proposed self-critical sequence training (SCST) using a policy gradient (Kakade, 2001; Silver et al., 2014). On the other hand, self-training has been shown to be effective in many tasks, such as machine translation (He et al., 2019), image classification (Xie et al., 2020), and structured databasegrounded question answering (Xu et al., 2020).

In the domain of question answering, a question generator can be used for joint answer prediction (Tang et al., 2017; Duan et al., 2017), and synthetic QA data are used for in-domain data augmentation (Sachan and Xing, 2018; Puri et al., 2020; Liu et al., 2020; Klein and Nabi, 2019) and outof-domain adaptation. Lewis et al. (2019b) and Lee et al. (2020) introduced models for question answering under unsupervised/zero-shot settings. Shakeri et al. (2020) proposed generating synthetic question-answer pairs with an end-to-end model simultaneously. Bartolo et al. (2021) improved the question synthesis by training with difficult QA cases from the AdversarialQA corpus (Bartolo et al., 2020) and fine-grained answer synthesis by multi-model voting. We include more related studies in Appendix A.

In this work, we mainly compare our method with latest baselines, Shakeri et al. (2020) and Bartolo et al. (2021) that reported results on outof-domain adaptation. Besides improved QA performance, our framework, RGX, differs from the previous work in the following aspects: (1) Our method features reinforced finetuning of the QA

![](images/e7218c70d2a634b1e30ad84838ebeb744b5539988c35cb1acd063a7d3103592d.jpg)  
Figure 2: The cooperative learning pipeline for question answering. The pipeline starts from a passage and follows the steps: (1) recognizing a potential answer entity, (2) generating a question asking about the answer entity, and (3) answering the question by extracting the answer span in the passage.

Synthesizer, (2) Our framework supports and improves maximize mutual information inference in test time, and (3) Our work did not use complicated data annotation, e.g. AdversarialQA.

## 3 RGX Framework

In this section, we first introduce (1) the QA synthesis pipeline, (2) cooperative self-training for both QA synthesis and question answering, and (3) an improved maximum mutual information inference strategy. The self-training pipeline of RGX is shown in Figure 2.

## 3.1 Data Synthesis

Given a passage p, our goal is generating a set of questions q and answers a for the self-training of the QA model. The RGX model first recognize potential answer entities (AE) in p with an answer entity recognition (AER) model, and then generate question based on the recognized AEs with a question generation (QG) model, and fine-grain the AEs with a pretrained question-answering extraction (QAE) model.

## 3.1.1 Answer Entity Recognition (AER)

Latest QA synthesis models, QAGen2S (Shakeri et al., 2020) and SynQA (Bartolo et al., 2021), directly generate questions from passages by modeling $P _ { q g } ( q | p )$ . In RGX, we first recognize all potential answer entities in a passage before generating questions for (1) increasing question diversity and coverage, and (2) modeling the mutual information between question generation and answering models in test time. The AER model in trained on the seed QA corpus.

We found that using an off-the-shelf named entity recognition (NER) model pretrained on the CONLL 2003 shared task (Bender et al., 2003) performs poorly as a AER model (shown in our experiments). To learn an effective recognizer, given a passage p and an annotated answer entity e, we select the sentence s containing e from p and train language models to recognize e in s. We tried two models for this task: a BIO sequence tagging model (AER-Tag) and a extractive AER model, which is similar to an extractive question answering model, for easier decoding. The model predicts the start and end positions of the answer entity e. With this method, we get potential answer entities by probabilities of all candidate spans.

## 3.1.2 Masked Question Generation

With AER, we replace the answer entity e in the passage p with a [MASK] token and obtain the masked passage $p ^ { * }$ . We then build a question generator Q (denoted as QG interchangeably) that outputs answerable questions $q$ in natural language with the concatenation of $p ^ { * }$ and e as input, i.e., $q = Q ( [ p ^ { * } , e ] )$ . We adopt the BART sequence-tosequence model (Lewis et al., 2019a) as the architecture of Q in our implementation, and we train Q on the question-answer pairs in the seed corpus by maximizing the likelihood of annotated questions.

## 3.1.3 Answer Extraction as Fine-grained AER

The answer extraction model A (denoted as QAE, question-answering extractor) takes generated question q and the original passage $p$ as inputs. Following the standard extractive QA method, we predict the answers by

$$
I _ { s t } , I _ { e d } = A ( [ q , p ] )\tag{1}
$$

where $I _ { s t }$ and $I _ { e d }$ stand for the start and end positions of e in p, respectively. We train the QAE model to predict $I _ { s t }$ and $I _ { e d }$ separately with cross entropy losses.

Besides being trained with synthetic QA pairs and evaluated for the final QA performance, the QAE model is also a part of the data synthesis pipeline. After generating questions with the QG model, we use a pretrained QAE model to answer the generated questions. The QAE model recognizes better answers spans than the AER model since it takes questions as additional inputs. As a result, the final synthetic dataset is constructed by selecting generated questions and their corresponding QAE outputs. However, we still found the AER model necessary for generating diverse questions.

## 3.2 Cooperative Self-training

Although the pretrained models can generate synthetic QA pairs from corpora in unseen domains, there is always a domain shift from the seed QA corpus for pretraining to the target. To efficiently adapt the pretrained models to the new domains, we propose a cooperative self-training algorithm that allows finetuning on the target corpora without additional annotations. The finetuning is based on a three-agent (AER, QG, QAE) cooperative framework, RGX. The pipeline is illustrated in Figure 2 and comprises the following steps:

1. Produce a masked passage by replacing an answer entity selected by AER with the ‘[MASK]’ token.

2. Generate a question asking about the masked entity.

3. Feed the generated question and the original passage into the QAE to predict an answer span.

4. Optimize the QAE model with selected QA pairs.

5. Optimize the QG model with selected QA pairs.

In the proposed pipeline, all the AER, QG, and QAE models need pretraining to provide a reasonable start point for the cooperative self-training. However, the domain gap between the pretraining and the target corpus causes performance degradation. To mitigate the gap, we propose to measure the quality of generated questions and incorporate the measurement in loss functions. The quality is defined in two folds, correctness and difficulty. Firstly, the question should be fluent and answerable, and secondly, it should not be too trivial. To automatically select high-quality generated QA pairs, we introduce a expectationmaximization (EM) method based on QAE losses that learns the question quality without supervision.

## 3.2.1 Synthetic QA Selection with EM

To select synthetic QA pairs for finetuning, we first divide the generated questions based on the QAE loss for each question into three groups: low-, medium-, and high- loss questions. We can interpret questions with low loss as simple ones that the QAE model can easily answer. Medium-loss questions are challenging for the QAE, while those with high loss usually contain noise (e.g., containing grammatical errors or asking about incorrect answers). If we train the answering model with all questions, the training signal would be very noisy due to the high-loss questions. If we only reward questions that are correctly answered, the generator will converge to a trivial local optima. Thus, we train the QG and QAE model with the low- and medium- loss questions, namely simple and challenging questions. For the entire pipeline to be fully-automatic, we classify a given QA pair into one of the three types described above. Note that simply setting the thresholds as hyper-parameters is difficult since the loss decreases as the QAE model varies with different passages and domains. In order to find the thresholds adaptively, we apply an expectation-maximization (EM) algorithm to bucket synthetic QA pairs for each passage.

We finetune both QG and QAE models with the selected simple and challenging QA pairs. After the training, re-running the RGX pipeline with the finetuned question generation model leads to improved data synthesis. Training the QAE model on the updated synthetic dataset can significant outperform the previous finetuned QAE model.

## 3.2.2 Maximum Mutual Information QA

Li and Jurafsky (2016) proposed a maximum mutual information (MMI) decoding method for machine translation, and Tang et al. (2017) proposed a MMI method for jointly learning question generation and answering models. There is no previous study to our knowledge that applies MMI inference in test time of question answering that improves the final performance, because (1) modeling $P ( q | p , a )$ for all possible answers (spans) a is too inefficient, and (2) Unlike the QAE model that receives loss signals from all words in a given passage, the QG model does not receive loss signal from the passage directly, so $P _ { q g } ( q | p , a )$ it is less accurate for ranking answer spans.

However, the AER and self-training strategy en-

able efficient MMI inference for QA,

$$
a = \underset { a } { \operatorname { a r g m a x } } [ \alpha \log P _ { q g } ( q | p , a ) + \beta \log P _ { q a } ( a | p , q ) ]
$$

In test time, we run the RGX pipeline for each passage without additional training to get fine-grained AEs and corresponding questions. On the other hand, we take the top span predicted by the QAE model, and the top-k answer entities spans recognized by the RGX pipeline. In practice, we fix $\beta = 1$ . We used an adaptive α value by comparing the synthetic question generated by the QG model and the input question. For each answer entity $^ { a , }$ we calculate

$$
\alpha = \mathfrak { m a x } ( 1 - \mathsf { a b s } ( \frac { q _ { i n p u t } } { q _ { g e n } } - 1 ) , 0 . 1 )
$$

This value normalizes the question probability $p ( q | p , a )$ estimated by the QG model, since generated questions from some answer entities is easier than other spans in the same passage, which makes the QG model assign all natural questions a relative low perplexity.

## 4 Experiments

In this work, we train three modules for building the cooperative self-training environment RGX, i.e., the answer entity recognizer (AER), the question generator (QG), and the question-answering extractor (QAE). We used a BERT (Devlin et al., 2018) model for AER, a BART (Lewis et al., 2019a) model for QG, and an ELECTRA (Clark et al., 2020) model for AER and QAE. To compare with the results reported in Shakeri et al. (2020) and Bartolo et al. (2021), we (1) pretrain question generation and answering models on the seed corpora, (2) generate synthetic QA data on the target domains, (3) finetune QA models with synthetic data, and (4) evaluate the finetuned QA model on humanlabeled evaluation sets. The source code and demo are publicly available<sup>2</sup>.

## 4.1 Data

In our experiment work, we leveraged Natural Questions (Kwiatkowski et al., 2019) and SQuAD v1.1 (Rajpurkar et al., 2016) as the seed corpora for pretraining all modules introduced above. To evaluate the performance of the proposed RGX on question answering tasks with different difficulty levels, we conduct experiments on both SQuAD v1.1 (Rajpurkar et al., 2016) and MRQA (Fisch et al., 2019)

out-of-domains (BioASQ, TextbookQA, RACE, RelationExtraction, DuoRC, and DROP). In the following sections, we use the term SQuAD to represent the SQuAD v1.1 corpus. For self-training, we sample 3000 passages from the training set of each corpus for data synthesis. More details about the data are provided in Appendix B

## 4.2 Implementation Details

Pretraining. We pretrain the AER, QG, and QAE models on NaturalQuestions and SQuAD (i.e., the seed) corpora. For NaturalQuestions, we only use the data points containing a short answer. For Cooperative training, we follow the steps described in Section 3.2 for the cooperative training phase.

Self-training. We apply self-training for QG and QAE by finetuning the models on selected synthetic QA pairs using the same method as pretraining. The AER model is fixed after pretraining. The QAE model is finetuned using the official Huggingface (Wolf et al., 2019) training scripts for question answering. We will open-source the RGX framework if the submission is accepted.

Hyperparameters. There are three phases of model training in this work: pretraining on the seed corpora, cooperative adaptation with selftraining on the target corpora, and final finetuning on the synthetic data. We adopt most of the hyper-parameters reported in the original BERT (Devlin et al., 2018), BART (Lewis et al., 2019a), and ELECTRA (Clark et al., 2020) papers. We select the final finetuning learning rates from $\{ 3 e - 5 , 4 e - 5 , 5 e - 5 \}$ and report the highest performance. All the other hyper-parameters are the same as reported in the corresponding papers. For all the phases, we fix $e p s = 1 e - 6$ and $s _ { w } = 2 0 0 0 .$ , where $s _ { w }$ is the number of warm-up steps, and we apply no weight decays. We use BART-large (406M parameters) and ELECTRAlarge (335M parameters) models for our experiments. We run our experiments on 2 Tesla V100 GPUs. Training the QAE models on augmented data takes about 4 hours.

## 4.3 Experiment Results

We assess the performance of RGX with both semiannotated and zero-annotated evaluation on unseen domains using exact match (EM) and F1 scores. The exact match metric assesses the percentage of predicted spans that are exactly the same as labeled answers, while the F1 score measure the overall token-level overlap between predicted and labeled answers. In our semi-annotated setting, we use the annotated answer entities in the target corpora but utilize QG to generate questions for obtaining the training question-answer pairs. The labeled questions are not used. We employ no annotation from the target corpora for the out-of-domain task but automatically construct the question-answer training pairs with entities and questions inferred by AER and QG on the corpora.

## 4.3.1 Semi-annotated Evaluation

The model performance with the pretrained QA model, RGX, and SOTA trained with fullsupervision is shown in Table 1.

<table><tr><td>Models</td><td>EM F1</td></tr><tr><td>Source domain: NQ, Target domain: SQuAD</td><td>80.3</td></tr><tr><td>ELECTRA-large (NaturalQuestions) RGX</td><td>67.8</td></tr><tr><td>83.1</td><td>90.7</td></tr><tr><td>-w/o Coop. ST 81.2</td><td>89.1</td></tr><tr><td>ELECTRA-large (SQuAD)</td><td>89.7 94.9</td></tr></table>

Table 1: The performance of the question answering models in the semi-annotated setting. RGX stands for our cooperative training approach, and Coop. ST stands for cooperative self-training.

Table 1 shows that RGX yields improvement over the pretrained model, approaching the SOTA performance of the fully trained ELECTRA-largediscriminator model. The experiment result suggests that the cooperative learning strategy improves the question generation model with humanannotated answer entities.

## 4.3.2 Out-of-domain Evaluation

We also evaluate the models in unseen domains, where we do not use any annotated QA for finetuning. We train the QAE models based on the synthetic training data and evaluate the models on the target domains. We compare RGX with latest selftraining QA methods, QAGen2S (Shakeri et al., 2020) and SynQA (Bartolo et al., 2021). Since QAGen2S did not report full MRQA results, we implemented our own version. We first present the RGX performance and the results reported by the authors QAGen2S and SynQA, and then conduct ablation study by training different language models on RGX synthetic QA data.

The full evaluation results on MRQA out-ofdomains are shown in Table 2, and the experiment setting comparison is shown in table 3. The results show that the models trained with the RGX framework achieve significantly higher EM and F1 scores on most domains, comparing to both pretrained QA models and self-training baselines. The results showed that the RGX model achieves 7.7 and 3.0 average F1 improvement over ELECTRA, the SOTA pretrained language model for QA, by pretraining on NQ and SQuAD respectively. The improvement over previous SOTA self-training QA methods, QAGen2S and SynQA, is also significant on both pretraining corpora, although SynQA applies complicated adversarial QA annotation. The largest gain we got is adapting NQ model to TextbookQA domain, increasing 18.0 EM and 19.4 F1 scores. Note that our model still outperforms all baselines without MMI. The performance on the DROP benchmark drops since DROP requires multi-step reasoning, but the synthetic generation model tends to generate safe question-answer pairs. We also found that without selecting harder questions with SEM in RGX, the performance is significantly lower. These facts indicate that the QA model needs hard training examples for better performance, and explains the good performance of SynQA on DROP. For the same reason, the performance drop led by removing EM from RGX is significantly larger when the QG model is pretrained on SQuAD, since SQuAD questions are more coherent with the context than NQ, and selecting simple questions for RGX training will encourage the model to generate trivial questions, which is harmful for the QA training.

## 4.4 Analysis

## 4.4.1 Answer Entity Recognition

We first compare the performance of different AER models and strategies by setting NQ as the source domain and SQuAD 1.1 as the target domain in Table 4. The results showed that the choice of AER model and strategy significantly influences the final QA performance. The low performance of the NER model trained on CONLL shared task suggests the importance of our AER module. We notice that the improvement from the cooperative learning over the pretrained models is higher in the zero-annotation setting than the semi-annotated task. The observation indicates that the model trained with RGX is more robust against the automatically recognized answer entities. More details about the AER methods are shown in Appendix C.

The AER method also enables and improves the maximum mutual information (MMI) inference in test time. Table 2 shows that MMI achieves the best performance, and we also show that the MMI accuracy is hurt without AER. Table 5 shows that MMI grounded on AER constantly outperform the ELECTRA model, but grounding on top-k seriously hurts the EM scores. Some invalid answer predictions leads to low question generation perplexities, which makes MMI inference noisy. Table 6 shows that the QG model generated more diverse questions based on the AER outputs.

<table><tr><td>Model Domain</td><td colspan="2">BioASQ Bio</td><td colspan="2">TextbookQA Book</td><td colspan="2">RACE Exam</td><td colspan="2">RelExt. Wiki</td><td colspan="2">DuoRC Movie</td><td colspan="2">DROP Wiki</td><td colspan="2">Avg</td></tr><tr><td></td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td></td><td></td><td></td><td>Source Domain:</td><td></td><td>NaturalQuestionSwiki,</td><td></td><td></td><td>Method: Extraction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ELECTRÃ</td><td>41.9</td><td>59.0</td><td>31.9</td><td>41.5</td><td>32.4</td><td>43.4</td><td>67.7</td><td>81.8</td><td>40.0</td><td>48.5</td><td>39.3</td><td>51.1</td><td>42.2</td><td>54.2</td></tr><tr><td>QAGen2S</td><td>43.2</td><td>64.1</td><td>39.9</td><td>51.7</td><td>33.7</td><td>45.5</td><td>71.6</td><td>84.4</td><td>43.8</td><td>53.2</td><td>24.2</td><td>37.1</td><td>42.7</td><td>56.0</td></tr><tr><td>RGX (Ours)</td><td>50.3</td><td>70.1</td><td>49.9</td><td>60.9</td><td>40.3</td><td>52.4</td><td>76.1</td><td>87.2</td><td>47.8</td><td>58.4</td><td>27.6</td><td>42.1</td><td>48.7</td><td>61.9</td></tr><tr><td>– w/o MMI</td><td>49.7</td><td>69.1</td><td>49.4</td><td>60.6</td><td>39.7</td><td>51.5</td><td>75.4</td><td>86.7</td><td>46.9</td><td>57.5</td><td>27.1</td><td>41.7</td><td>46.8</td><td>61.2</td></tr><tr><td>– w/o EM</td><td>48.2</td><td>67.9</td><td>47.4</td><td>59.8</td><td>38.3</td><td>50.5</td><td>74.1</td><td>86.2</td><td>46.6</td><td>56.9</td><td>26.1</td><td>40.9</td><td>46.8</td><td>60.4</td></tr><tr><td>– w/o CST</td><td>45.4</td><td>66.4</td><td>41.9</td><td>53.8</td><td>35.1</td><td>47.2</td><td>72.7</td><td>85.4</td><td>45.5</td><td>54.9</td><td>24.6</td><td>37.9</td><td>44.2</td><td>57.6</td></tr><tr><td>Source Domain: SQuADwiki</td><td></td><td></td><td></td><td></td><td>(SQuAD+AQA+Wiki for SynQA), Method: Extraction</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ÉLÉCTRA</td><td>58.7</td><td>73.1</td><td>43.0</td><td>53.6</td><td>38.3</td><td>52.5</td><td>79.0</td><td>88.4</td><td>53.1</td><td>64.2</td><td>48.3</td><td>60.8</td><td>53.4</td><td>65.4</td></tr><tr><td>QAGen2S</td><td>56.8</td><td>71.7</td><td>48.0</td><td>56.5</td><td>43.4</td><td>54.9</td><td>73.4</td><td>84.8</td><td>53.3</td><td>64.6</td><td>42.2</td><td>54.5</td><td>52.8</td><td>64.5</td></tr><tr><td>SynQA</td><td>55.1</td><td>68.7</td><td>41.4</td><td>50.2</td><td>40.2</td><td>54.2</td><td>78.9</td><td>88.6</td><td>51.7</td><td>62.1</td><td>64.9</td><td>73.0</td><td>55.3</td><td>66.1</td></tr><tr><td>RGX (Ours)</td><td>60.3</td><td>74.8</td><td>51.2</td><td>61.2</td><td>44.9</td><td>58.7</td><td>79.2</td><td>88.6</td><td>57.4</td><td>66.2</td><td>47.6</td><td>60.9</td><td>56.8</td><td>68.4</td></tr><tr><td>– w/o MMI</td><td>59.2</td><td>73.6</td><td>50.1</td><td>60.4</td><td>46.3</td><td>57.6</td><td>78.9</td><td>88.5</td><td>56.2</td><td>65.7</td><td>46.9</td><td>60.6</td><td>56.3</td><td>67.7</td></tr><tr><td>– w/o EM – w/o CST</td><td>52.1</td><td>64.0</td><td>50.6</td><td>58.9</td><td>35.4</td><td>48.3</td><td>75.6</td><td>85.9</td><td>55.6</td><td>64.9</td><td>40.7</td><td>53.2</td><td>51.7</td><td>62.5</td></tr><tr><td></td><td>57.5</td><td>72.1</td><td>48.6</td><td>57.0</td><td>43.8</td><td>55.2</td><td>74.3</td><td>85.3</td><td>53.9</td><td>65.3</td><td>43.0</td><td>55.1</td><td>53.5</td><td>65.0</td></tr><tr><td>Source Domain: SQuADwiki, Method: Prompt Tuning + Seq2seq Generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>54.6</td><td>71.1</td><td>37.9</td><td>61.9</td><td>15.0</td><td>53.1</td><td>74.5</td><td>86.5</td><td>48.2</td><td>65.2</td><td>40.4</td><td>51.9</td><td>45.1</td><td>64.9</td></tr><tr><td>T5  $\mathrm { T } 5 + \mathrm { R G X }$ </td><td>55.1</td><td>71.6</td><td>41.1</td><td>64.2</td><td>15.5</td><td>55.1</td><td>75.9</td><td>87.1</td><td>49.5</td><td>66.2</td><td>42.9</td><td>53.8</td><td>46.7</td><td>66.3</td></tr></table>

Table 2: The QA performance evaluation on the out-of-domains of the MRQA benchmark. All models used are pretrained on the human-labeled training set from the source domains, and the QA models are finetuned on synthetic data generated based on the unannotated passages of the target domains. The finetuned QA models are evaluated on human-generated evaluation data for each target domains with the exact match (EM) and F1 scores. MMI stands for maximum mutual information inference, EM stands for involving difficult questions with EM selection, and CST stands for cooperative self-training.

<table><tr><td></td><td>QAGen2S</td><td>SynQA</td><td>RGX</td></tr><tr><td>Pretraining</td><td>XQ</td><td>SQ+AQA</td><td>XQ</td></tr><tr><td>Synthesis</td><td>Target</td><td>Wikipedia</td><td>Target</td></tr><tr><td>Finetuning</td><td>XQ+Syn</td><td> $\mathrm { S Q + A Q A { + } S y n }$ </td><td>XQ+Syn</td></tr><tr><td>AER Model</td><td>None</td><td>None</td><td>ELECTRA</td></tr><tr><td>Coop. ST</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>QA Num.</td><td>1M</td><td>1.5M</td><td>0.3M</td></tr></table>

Table 3: Comparison of different self-training methods. XQ stands for “NaturalQuestions (NQ) or SQuAD (SQ)”. QA Num. stands for the number of synthetic QA pairs used for self-training.

## 4.4.2 Synthetic QA Selection with EM

Previous experiments showed that selecting nontrivial synthetic QA pairs is essential for RGX to achieve high performance. Table 2 shows that the performance of cooperative self-trained RGX is much lower than the pretrained baseline without

<table><tr><td>Models EM</td><td>F1</td></tr><tr><td>Source domain: NQ, Target domain: SQuAD</td></tr><tr><td>Pretrained NQ 67.8 80.3</td></tr><tr><td> $\mathrm { R G X + N E R }$  27.4 35.4</td></tr><tr><td> $\mathrm { R G X } + \mathrm { A E R } { \cdot } \mathrm { T a g }$  71.4 82.4</td></tr><tr><td> $\mathbf { R G X } + \mathbf { A E R - L M }$  72.7 85.9</td></tr><tr><td> $\mathrm { R G X + A E R \mathrm { - } E M }$  79.2 89.4 Supervised ELECTRA-large 89.7 94.9</td></tr><tr><td></td></tr></table>

Table 4: Comparison of different AER strategies. NER stands for the BERT named entity recognition model trained on the CONLL 2003 shared task. AER-Tag stands for a BIO-based tagging strategy, AER-LM means selecting synthetic QA pairs with lowest QAE losses. AER-EM is the EM-based QA selection strategy applied in our full model.

EM. If selecting QA pairs with low perplexities instead of EM, the QA diversity is significantly lower as shown in Table 6, thus makes the QAE model overfit to simple training cases and hurts the QA accuracy. We show questions about the same answer entity being classified into simple, challenging, and difficult types by EM in figure 3. The data points in the plot represents the losses of synthetic QA pairs and the predicted QA type. Based on the highlighted answer entity, question 1 and 2 are predicted as correct questions, while question 3, which has a relatively high QAE loss, is regarded as a wrong question. Note that we only generate one question for each span recognized by the AER model, but different questions might be re-directed to the same AE after QAE fine-graining.

<table><tr><td></td><td>ELECTRA</td><td></td><td>Top-k+MMI</td><td></td><td>AER+MMI</td></tr><tr><td></td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM F1</td></tr><tr><td>BioASQ</td><td>58.7</td><td>73.1</td><td>57.8</td><td>72.9</td><td>59.9 74.0</td></tr><tr><td>TextbookQA</td><td>43.0</td><td>54.6</td><td>44.6 54.9</td><td>45.3</td><td>55.4</td></tr><tr><td>RACE</td><td>38.3</td><td>52.5</td><td>38.1</td><td>52.4</td><td>39.7 54.1</td></tr><tr><td>RelExt</td><td>79.0</td><td>88.4</td><td>78.6</td><td>88.3</td><td>79.2 88.6</td></tr><tr><td>DuoRC</td><td>53.1</td><td>64.2</td><td>52.6</td><td>64.3</td><td>53.8 65.1</td></tr><tr><td>DROP</td><td>48.3</td><td>60.8</td><td>46.7</td><td>60.8</td><td>49.7 61.5</td></tr></table>

Table 5: Comparison between maximum mutual information inference performance grounded on AER results and top-k (k = 20) predictions of the QA model.
<table><tr><td>Models</td><td>Mean Len.</td><td>Std Len.</td><td>Vocab</td></tr><tr><td>Ground-truth</td><td>11.29</td><td>3.72</td><td>988703</td></tr><tr><td>Semi-anno. RGX</td><td>10.54</td><td>1.91</td><td>923191</td></tr><tr><td>-w/o Coop. ST</td><td>10.49</td><td>2.48</td><td>919105</td></tr><tr><td>Zero-anno. RGX</td><td>10.53</td><td>1.94</td><td>873300</td></tr><tr><td>-w/o Coop. ST</td><td>10.57</td><td>2.63</td><td>789924</td></tr><tr><td>-w/o AER</td><td>10.60</td><td>1.87</td><td>743454</td></tr><tr><td>-w/o EM</td><td>10.18</td><td>1.62</td><td>692301</td></tr></table>

Table 6: The vocabulary sizes and lengths of Annotated and generated questions on SQuAD under both semiand zero-annotated settings in unseen domains

## 4.4.3 Cooperative Self-training

We found that the cooperative self-training method improves domain adaptation ability of self-trained QA models by increasing both accuracy and diversity of QA synthesis.

Accuracy. We also evaluate the quality of the generated QA pairs without a downstream task by assessing the answer entity hit rate and the BLEU scores of generated questions using the evaluation sets of each domain. The results are shown in

<table><tr><td>Domain</td><td>RGX w/o Coop. ST</td><td>RGX</td></tr><tr><td>Hit</td><td>BLEU Hit</td><td>BLEU</td></tr><tr><td>BioASQ</td><td>68.1 5.9</td><td>75.8 12.7</td></tr><tr><td>TextbookQA</td><td>43.7 7.5</td><td>58.2 13.2</td></tr><tr><td>RACE</td><td>8.3 5.2</td><td>12.3 6.8</td></tr><tr><td>RelExt.</td><td>47.4 2.8</td><td>54.2 3.3</td></tr><tr><td>DuoRC</td><td>53.5 6.7</td><td>60.0 7.5</td></tr><tr><td>DROP</td><td>73.5 12.3</td><td>75.3 9.3</td></tr></table>

Table 7: Evaluation of the answer hit rates and question BLEU scores of the synthetic data. Hit rate stands for the percentage of human-labeled answer entities in the evaluation passages that are successfully covered by the selected synthetic data generated by RGX.

Context: Despite diferences in the spectrum of mutations in CN or CyN, type or localization of mutation only partially determine the clinical phenotype.

Q1: What determines the clinical phenotype of a person with a mutation? Q2: What determines the clinical phenotype of a mutation? Q3: What is the only way to determine the clinical phenotype of a mutation?

![](images/7e207b0cfa734ae8a823b2d5d55602241b97f15b33cc7f6f46ca504d349797ef.jpg)  
Figure 3: Generated questions about the same answer entity classified into different types by EM. Questions Q1 is answered by the QAE model confidently, while the Q2 is considered more challenging than Q1 since less information is provided. Q3 is an unanswerable questions given the context passage.

Table 7, indicating that RGX find mores humanannotated answer entities, and the generated questions have higher BLEU scores on all domains. The evaluation results show that the synthetic QA pars generated by RGX covers more human annotated answer entities, and the generated questions are more similar to human annotations than the pretrained question generation model. We also found that tuning the generation model for more than 1 iterations does not result in further improvement, since keeping training language models with their own outputs leads to difficult optimization.

Diversity. We compare the lengths and vocabulary sizes of the questions and summarize the statistics in Table 6, which shows that the ground-truth questions are longer and more diverse in vocabulary than the generated ones. However, the cooperative self-training, together with AER and EM, improves the vocabulary diversity. We observe a correlation between the vocabulary size and the QA performance reported in Table 1 and 4, presumably because the QAE model requires diverse knowledge for training. Thus, we believe generating more diverse QA pairs with good quality will be a critical next step to improve RGX.

Case Study. An example of a SQuAD passage is shown in Table 8. We list the annotated and generated question-answer pairs by different models. The table shows that the models can recognize rea-

Architecturally, the school has a Catholic character. Atop the Main Building’s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend ”Venite Ad Me Omnes”. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.

<table><tr><td colspan="2">Annotated Pretrained</td><td colspan="2">RGX</td></tr><tr><td colspan="2">Saint Bernadette Soubirous</td><td>a Marian place of prayer and reflection</td><td>a Marian place of prayer and reflection</td></tr><tr><td rowspan="2">To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?</td><td></td><td>what is the grotto at st bernadette&#x27;s?</td><td>what is the grotto in st bernadette</td></tr><tr><td></td><td>the grotto at Lourdes,</td><td>school?</td></tr><tr><td>a copper statue of Christ What is in front of the Notre Dame</td><td></td><td>France where is the grotto located at st</td><td>Venite Ad Me Omnes what is the message on the statue in</td></tr><tr><td>Main Building?</td><td></td><td>bernadette school? Immediately behind the</td><td>front of st bernadette school?</td></tr><tr><td>the Main Building The Basilica of the Sacred heart at</td><td></td><td>basilica is the Grotto</td><td>1858</td></tr><tr><td>Notre Dame is beside to which structure?</td><td></td><td>what is the grotto in st peter&#x27;s school?</td><td>when was the grotto at lourdes built?</td></tr><tr><td rowspan="2">a Marian place of prayer and reflection</td><td></td><td>copper statue of Christ with arms upraised</td><td>a simple, modern</td></tr><tr><td></td><td></td><td>stone statue of Mary what is the statue at st bernadette</td></tr><tr><td>What is the Grotto at Notre Dame? a golden statue of</td><td></td><td>what is it a statue of christ?</td><td>school?</td></tr><tr><td>the Virgin Mary</td><td></td><td>a replica</td><td>the grotto at Lourdes, France</td></tr><tr><td>What sits on top of the Main Building at Notre Dame?</td><td></td><td>is the grotto at st bernadette school in paris a replica of which European school in paris?</td><td>what is the replica of st bernadette&#x27;s</td></tr></table>

Table 8: An example of a passage in the training set of the SQuAD corpus. We list the annotated question-answer pairs, and the question-answer pairs generated by the models pretrained on NQ and finetuned by RGX. The bold texts are annotated or recognized answer entities. Adapting from NQ is difficult since the questions in NQ do not strictly coherent with a given context. More generation examples are shown in Appendix D.

sonable answer entities other than the annotated ones, and RGX generates more natural QAs.

## 5 Conclusion

We propose a cooperative self-training framework, RGX, consisting of an answer entity Recognizer, a question Generator, and an answer eXtractor, for question generation and answering. We also introduce in the framework an expectationmaximization method that measures the quality of generated questions for reinforced finetuning of the question generation models. Experiments show that RGX significantly outperforms pretrained and self-trained model baselines while adapted to unseen domains, suggesting that RGX is a promising framework for making extractive question answering methods more scalable and less dependent on human annotation.

## References

Max Bartolo, Alastair Roberts, Johannes Welbl, Sebastian Riedel, and Pontus Stenetorp. 2020. Beat the ai: Investigating adversarial human annotation for read-

ing comprehension. Transactions ofthe Association for Computational Linguistics, 8:662–678.

Max Bartolo, Tristan Thrush, Robin Jia, Sebastian Riedel, Pontus Stenetorp, and Douwe Kiela. 2021. Improving question answering model robustness with synthetic adversarial data generation. arXiv preprint arXiv:2104.08678.

Oliver Bender, Franz Josef Och, and Hermann Ney. 2003. Maximum entropy models for named entity recognition. In Proceedings of CoNLL-2003, pages 148–151. Edmonton, Canada.

Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and´ Christian Jauvin. 2003. A neural probabilistic language model. Journal of machine learning research, 3(Feb):1137–1155.

Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. 2020. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Nan Duan, Duyu Tang, Peng Chen, and Ming Zhou. 2017. Question generation for question answering.

In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 866– 874.

Hao Fei, Yafeng Ren, and Donghong Ji. 2020. Retrofitting structure-aware transformer language model for end tasks. arXiv preprint arXiv:2009.07408.

Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. 2019. Mrqa 2019 shared task: Evaluating generalization in reading comprehension. In Proceedings of the 2nd Workshop on Machine Reading for Question Answering, pages 1–13.

Michael Glass, Alfio Gliozzo, Rishav Chakravarti, Anthony Ferritto, Lin Pan, GP Bhargav, Dinesh Garg, and Avirup Sil. 2019. Span selection pretraining for question answering. arXiv preprint arXiv:1909.04120.

Ian Goodfellow. 2016. Nips 2016 tutorial: Generative adversarial networks. arXiv preprint arXiv:1701.00160.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems, 27:2672–2680.

Serhii Havrylov and Ivan Titov. 2017. Emergence of language with multi-agent games: Learning to communicate with sequences of symbols. In Advances in neural information processing systems, pages 2149– 2159.

Junxian He, Jiatao Gu, Jiajun Shen, and Marc’Aurelio Ranzato. 2019. Revisiting self-training for neural sequence generation. arXiv preprint arXiv:1909.13788.

Matthew Henderson, Inigo Casanueva, Nikola Mrk˜ siˇ c,´ Pei-Hao Su, Tsung-Hsien Wen, and Ivan Vulic.´ 2019. Convert: Efficient and accurate conversational representations from transformers. arXiv preprint arXiv:1911.03688.

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2019. Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring. arXiv preprint arXiv:1905.01969.

Robin Jia, Mike Lewis, and Luke Zettlemoyer. 2021. Question answering infused pre-training of generalpurpose contextualized representations. arXiv preprint arXiv:2106.08190.

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. 2020. Spanbert: Improving pre-training by representing and predicting spans. Transactions ofthe Associationfor Computational Linguistics, 8:64–77.

Sham M Kakade. 2001. A natural policy gradient. Advances in neural information processing systems, 14:1531–1538.

Tassilo Klein and Moin Nabi. 2019. Learning to answer by learning to ask: Getting the best of gpt-2 and bert worlds. arXiv preprint arXiv:1911.02365.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark for question answering research. Transactions ofthe Association for Computational Linguistics, 7:453– 466.

Angeliki Lazaridou, Alexander Peysakhovich, and Marco Baroni. 2016. Multi-agent cooperation and the emergence of (natural) language. arXiv preprint arXiv:1612.07182.

Dong Bok Lee, Seanie Lee, Woo Tae Jeong, Donghwan Kim, and Sung Ju Hwang. 2020. Generating diverse and consistent qa pairs from contexts with information-maximizing hierarchical conditional vaes. arXiv preprint arXiv:2005.13837.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019a. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461.

Patrick Lewis, Ludovic Denoyer, and Sebastian Riedel. 2019b. Unsupervised question answering by cloze translation. arXiv preprint arXiv:1906.04980.

Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Kuttler, Aleksandra Piktus, Pontus¨ Stenetorp, and Sebastian Riedel. 2021. Paq: 65 million probably-asked questions and what you can do with them. arXiv preprint arXiv:2102.07033.

Jiwei Li and Dan Jurafsky. 2016. Mutual information and diverse decoding improve neural machine translation. arXiv preprint arXiv:1601.00372.

Dayiheng Liu, Yeyun Gong, Jie Fu, Yu Yan, Jiusheng Chen, Jiancheng Lv, Nan Duan, and Ming Zhou. 2020. Tell me how to ask again: Question data augmentation with controllable rewriting in continuous space. arXiv preprint arXiv:2010.01475.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Hongyin Luo, Lan Jiang, Yonatan Belinkov, and James Glass. 2019. Improving neural language models by segmenting, attending, and predicting the future. arXiv preprint arXiv:1906.01702.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26:3111–3119.

Igor Mordatch and Pieter Abbeel. 2018. Emergence of grounded compositional language in multi-agent populations. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings ofthe 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543.

Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. arXiv preprint arXiv:1802.05365.

Raul Puri, Ryan Spring, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. 2020. Training question answering models from synthetic data. arXiv preprint arXiv:2002.09599.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250.

Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. 2017. Self-critical sequence training for image captioning. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 7008–7024.

Mrinmaya Sachan and Eric Xing. 2018. Self-training for jointly learning to ask and answer questions. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 629–640.

Siamak Shakeri, Cicero Nogueira dos Santos, Henry Zhu, Patrick Ng, Feng Nan, Zhiguo Wang, Ramesh Nallapati, and Bing Xiang. 2020. End-to-end synthetic data generation for domain adaptation of question answering systems. arXiv preprint arXiv:2010.06028.

Yikang Shen, Zhouhan Lin, Athul Paul Jacob, Alessandro Sordoni, Aaron Courville, and Yoshua Bengio. 2018. Straight to the tree: Constituency parsing with neural syntactic distance. arXiv preprint arXiv:1806.04168.

Yikang Shen, Yi Tay, Che Zheng, Dara Bahri, Donald Metzler, and Aaron Courville. 2020. Structformer: Joint unsupervised induction of dependency and constituency structure from masked language modeling. arXiv preprint arXiv:2012.00857.

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. 2014. Deterministic policy gradient algorithms.

Duyu Tang, Nan Duan, Tao Qin, Zhao Yan, and Ming Zhou. 2017. Question answering and question generation as dual tasks. arXiv preprint arXiv:1706.02027.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz,´ et al. 2019. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771.

Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. 2020. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10687–10698.

Silei Xu, Sina J Semnani, Giovanni Campagna, and Monica S Lam. 2020. Autoqa: From databases to qa semantic parsers with only synthetic training data. arXiv preprint arXiv:2010.04806.

Lantao Yu, Weinan Zhang, Jun Wang, and Yong Yu. 2017. Seqgan: Sequence generative adversarial nets with policy gradient. In Proceedings of the AAAI conference on artificial intelligence, volume 31.

## A More Related Work

Representation learning has been an important topic in NLP area since neural language models were proposed (Bengio et al., 2003). Based on word co-occurrence, Mikolov et al. (2013) and Pennington et al. (2014) proposed language embedding algorithms to model word-level semantics. Recent studies have focused on pretraining contextualized word representations with large-scaled corpora (Peters et al., 2018). State-of-the-art representation models are pretrained with the masked language modeling task (Devlin et al., 2018; Liu et al., 2019; Clark et al., 2020) using the Transformer architecture (Vaswani et al., 2017).

Different variants of masked language models have been investigated to improve performance in downstream tasks. Joshi et al. (2020) leveraged a masked span generation task instead of word prediction. Fei et al. (2020) and Shen et al. (2020) proposed models that learns better syntax knowledge with syntactic distances (Shen et al., 2018) and heights (Luo et al., 2019). Henderson et al. (2019) and Humeau et al. (2019) showed that pretraining language models on dialog corpora perform better on dialog-related downstream tasks, as compared to pretraining on Wikipedia. A span selection pretraining objective is proposed in Glass et al. (2019) to reduce the gap between the pretraining and downstream finetuning stages and to improve the performance on the QA task. Some applications of generated questions are shown in (Lewis et al., 2021; Jia et al., 2021).

In contrast to self-training methods that usually adopt a teacher-student learning strategy, cooperative learning pipelines contain several agents working together to learn as much knowledge as possible. A typical cooperative learning framework is generative adversarial networks (GAN) (Goodfellow, 2016; Goodfellow et al., 2014), where a generator is optimized to confuse a discriminator, and a discriminator is trained to distinguish real examples from generated ones. Sequence GAN is further designed for learning diverse text generation (Yu et al., 2017). Unlike the adversarial learning method where two networks work for opposite goals, other studies proposed learning environments in which different agents learn the same objective functions for language emergence (Lazaridou et al., 2016; Mordatch and Abbeel, 2018; Havrylov and Titov, 2017), including simple natural language, compositional language, and

<table><tr><td>Dataset</td><td>Num. Synthetic QA</td></tr><tr><td>BioASQ</td><td>123121</td></tr><tr><td>TextbookQA</td><td>133773</td></tr><tr><td>RACE</td><td>115847</td></tr><tr><td>RelExt.</td><td>52142</td></tr><tr><td>DuoRC</td><td>250698</td></tr><tr><td>DROP</td><td>100394</td></tr></table>

Table 9: Number of synthetic QA of each MRQA domain.

symbolic language.

## B Data

The SQuAD v1.1 is the easiest QA corpus used in this paper. The dataset contains 107, 785 questionanswer pairs on 536 articles, which are split into passages. Each question is labeled with an answer that can be extracted from the given passage.

The Natural Questions dataset is a large-scale corpus designed for open-domain question answering. The dataset is more challenging than SQuAD. All questions are collected from human search queries and are annotated with long and abstractive answers. Some of the questions are also labeled with a short answer for learning answer-span extraction or reading comprehension. Focusing on the machine reading comprehension task, we select 106, 926 questions labeled with both long and short answers from the dataset for experiments.

For each target domain in MRQA, we collect the corresponding training data and sample 3000 passages for QA synthesis. The number of synthetic QAs varies based on the length of input passages, and is shown in Table 9.

## C Answer Entity Recognition Details

In this section, we describe details of the AER methods, which are not covered in detail in previous sections. All AER models are pretrained on the Natural Questions corpus. To solve the sparsity problem, in other words, the passages are long but not all potential question-answer pairs are annotated, we train all following AER models by using the sentence containing the annotated answer entities as inputs, instead of the whole passage. If a sentence in the passage does not contain an annotated answer entity, we do not use it for training.

In this work, we introduce two types of AER methods, tagging based AER (AER-tag) and extraction based AER (AER-Search and AER-Coop). We describe their training and how we use the trained model to recognize answer entities in our experiments.

## C.1 AER-Tag

## C.1.1 Training

We apply a BIO tagging model for answer entity recognition in the AER-Tab method. We train the model to classify all tokens in the input sentence into three classes,

• B(egin) - the first token of the annotated answer entity

• I(nsize) - other tokens of the annotated answer entity

• O(utside) - tokens that are not a part of the annotated answer entity

## C.1.2 Evaluation

Given an input passage, we run the trained BIO tagging model on each of its sentences and greedily predict answer entities. There might be more than one answer entities predicted in each sentence, and we only use the answer entities start with a predicted B tag.

## C.2 AER-LM

## C.2.1 Training

For AER-LM method, we need to pretrain an extraction-based AER model. We also take a sentence of L tokens containing an annotated answer entity as an example. Using an extraction model, which is similar as our question answering model, we train the model to predict the start and end location of the annotated answer entity. The model outputs a start score and an end score for each token, and predicts the start/end locations by selecting the tokens that are assigned with highest scores. The model is trained with cross-entropy loss, by regarding the extraction task as two L-class classification tasks.

## C.2.2 Evaluation

In evaluation, we first run the model on each sentence of the input passages and calculate the start and end scores for each token. For each span $( x _ { i } , x _ { i + 1 } , \ldots , x _ { j } )$ that is not longer than $L _ { s p a n }$ tokens, we calculate the span score with

$$
s _ { i j } = s _ { s t } ^ { i } + s _ { e d } ^ { j }\tag{2}
$$

where $s _ { s t } ^ { i }$ is the start score of the first token of span $( i , j )$ , and $s _ { e d } ^ { j }$ is the end score of the last token of the span. In practice, we set $L _ { s p a n } = 1 0$

To re-rank all possible answer entities, we select top $N _ { 0 } = 4 0$ spans according to $s _ { i j }$ for each passage. For all selected answer entities, we generated questions with a pretrained question generator and collect the generation perplexity of the questions. We select $N _ { s e a r c h } = 5$ question-answer pairs with lowest perplexities for the final question-answering finetuning.

## C.3 AER-Coop

In AER-Coop, we use the same extraction training method applied in AER-Search, and we also use the $s _ { i j }$ scores to select the top $N _ { 0 } = 4 0$ preliminary answer entities for further search. The difference is that we search for final answer entities cooperatively with the pretrained question generator and question answering extractor.

With the question generator and question answering extractor, we re-rank the recognized answer entities with the following score

$$
s _ { i j } ^ { c } = \gamma \cdot I _ { c } - p\tag{3}
$$

where $\gamma$ is a large, positive coefficient, $p$ is the perplexity of generated question based on span $( i , j )$ and $I _ { c } = 1$ if the generated question is correctly answered, and otherwise $I _ { c } = 0$

## C.4 Answer Entity Overlapping

We found the extraction-based AER model leads to overlapping problems, since a large start or end score assigned to a token leads to many candidate answer entities start or end at the token. In practice, if an answer entity is selected by the AER-Search and AER-Coop method, we no longer consider any other answer entities that overlap with the selected ones.

## D RGX Examples

In this section, we show some examples of our full model. The examples are contained in Table 10.

<table><tr><td rowspan=1 colspan=1>The National History Museum of Montevideo is located in the historical residence of General Fructuoso Rivera. It exhi-bits artifacts related to the history of Uruguay. In a process begun in 1998, the National Museum of Natural History (1837)and the National Museum of Anthropology (1981), merged in 2001, becoming the National Museum of Natural Historyand Anthropology. In July 2009, the two institutions again became independent. The Historical Museum has annexed eighthistorical houses in the city, five of which are located in the Ciudad Vieja. One of them, on the same block with the mainbuilding, is the historic residence of Antonio Montero, which houses the Museo Romantico.</td></tr><tr><td rowspan=1 colspan=1>When was the national history museum of montevideo founded?</td></tr><tr><td rowspan=1 colspan=1>In the 1920s, John Maynard Keynes prompted a division between microeconomics and macroeconomics. Under Keynesianeconomics macroeconomic trends can overwhelm economic choices made by individuals. Governments should promoteaggregate demand for goods as a means to encourage economic expansion. Following World War II, Milton Friedmancreated the concept of monetarism. Monetarism focuses on using the supply and demand of money as a method for con-trolling economic activity. In the 1970s, monetarism has adapted into supply-side economics which advocates reducingtaxes as a means to increase the amount of money available for economic expansion.</td></tr><tr><td rowspan=1 colspan=1>Monarism focuses on the relationship between the?</td></tr><tr><td rowspan=2 colspan=1>Starting in 2006, Apple&#x27;s industrial design shifted to favor aluminum, which was used in the construction of the first Mac-Book Pro. Glass was added in 2008 with the introduction of the unibody MacBook Pro. These materials are billed as env-ironmentally friendly. The iMac, MacBook Pro, MacBook Air, and Mac Mini lines currently all use aluminum enclosuresand are now made of a single unibody. Chief designer Jonathan Ive continues to guide products towards a minimalist andsimple feel, including eliminating of replaceable batteries in notebooks. Multi-touch gestures from the iPhone&#x27;s interfacehave been applied to the Mac line in the form of touch pads on notebooks and the Magic Mouse and Magic Trackpad fordesktops.</td></tr><tr><td rowspan=1 colspan=1>ironmentally frien</td></tr><tr><td rowspan=1 colspan=1>Who is the designer of the macbook pro?</td></tr><tr><td rowspan=1 colspan=1>The city&#x27;s total area is 468.9 square miles (1,214 km2). 164.1 sq mi (425 km2) of this is water and 304.8 sq mi (789 km2) island. The highest point in the city is Todt Hill on Staten Island, which, at 409.8 feet (124.9 m) above sea level, is thehighest point on the Eastern Seaboard south of Maine. The summit of the ridge is mostly covered in woodlands as partof the Staten Island Greenbelt.</td></tr><tr><td rowspan=1 colspan=1>Where is the highest point in new york city?</td></tr><tr><td rowspan=1 colspan=1>In 1922, the number of supporters had surpassed 20,000 and by lending money to the club, Barça was able to build thelarger Camp de Les Corts, which had an initial capacity of 20,000 spectators. After the Spanish Civil War the club startedattracting more members and a larger number of spectators at matches. This led to several expansion projects: thegrandstand in 1944, the southern stand in 1946, and finally the northern stand in 1950. After the last expansion, Les Cortscould hold 60,000 spectators.</td></tr><tr><td rowspan=1 colspan=1>What is the capacity of barcelona&#x27;s stadium?</td></tr><tr><td rowspan=1 colspan=1>On 1 November 2013, international postal services for Somalia officially resumed. The Universal Postal Union is nowassisting the Somali Postal Service to develop its capacity, including providing technical assistance and basic mailprocessing equipment.</td></tr><tr><td rowspan=1 colspan=1>Who is responsible for supporting the somali postal service?</td></tr><tr><td rowspan=1 colspan=1>In addition to membership, as of 2010[update] there are 1,335 officially registered fan clubs, called penyes, around theworld. The fan clubs promote Barcelona in their locality and receive beneficial offers when visiting Barcelona. Amongthe best supported teams globally, Barcelona has the highest social media following in the world among sports teams,with over 90 million Facebook fans as of February 2016. The club has had many prominent people among its support-ers, including Pope John Paul II, who was an honorary member, and former prime minister of Spain José LuisRodríguez Zapatero. FC Barcelona has the second highest average attendance of European football clubs only behindBorussia Dortmund.</td></tr><tr><td rowspan=1 colspan=1>Who was an honorary member of barcelona football club?</td></tr><tr><td rowspan=1 colspan=1>In April 1758, the British concluded the Anglo-Prussian Convention with Frederick in which they committed to pay himan annual subsidy of £670,000. Britain also dispatched 9,000 troops to reinforce Ferdinand&#x27;s Hanoverian army, the firstBritish troop commitment on the continent and a reversal in the policy of Pitt. Ferdinand had succeeded in driving theFrench from Hanover and Westphalia and re-captured the port of Emden in March 1758 before crossing the Rhine withhis own forces, which caused alarm in France. Despite Ferdinand&#x27;s victory over the French at the Battle of Krefeld andthe brief occupation of Düsseldorf, he was compelled by the successful manoeuvering of larger French forces to with-draw across the Rhine.</td></tr><tr><td rowspan=1 colspan=1>What did france pay to the prussian monarchy?</td></tr><tr><td rowspan=1 colspan=1>Executives at Trump Entertainment Resorts, whose sole remaining property will be the Trump Taj Mahal, said in 2013that they were considering the option of selling the Taj and winding down and exiting the gaming and hotel business</td></tr><tr><td rowspan=1 colspan=1>What is the future of the trump taj mahal?</td></tr><tr><td rowspan=1 colspan=1>Vehicles typically include headlamps and tail lights. Headlamps are white or selective yellow lights placed in the front ofthe vehicle, designed to illuminate the upcoming road and to make the vehicle more visible. Many manufactures are turn-ing to LED headlights as an energy-efficient alternative to traditional headlamps. Tail and brake lights are red and emitlight to the rear so as to reveal the vehicle&#x27;s direction of travel to following drivers. White rear-facing reversing lamps in-dicate that the vehicle&#x27;s transmission has been placed in the reverse gear, warning anyone behind the vehicle that it ismoving backwards, or about to do so. Flashing turn signals on the front, side, and rear of the vehicle indicate an intendedchange of position or direction. In the late 1950s, some automakers began to use electroluminescent technology to back-light their cars&#x27; speedometers and other gauges or to draw attention to logos or other decorative elements.</td></tr><tr><td rowspan=1 colspan=1>When did they start putting back up lights in cars?</td></tr></table>

257 Table 10: Examples of recognized answer entities and generated questions with the full RGX model