# Locally Aggregated Feature Attribution on Natural Language Model Understanding

Sheng Zhang<sup>1</sup> Jin Wang<sup>2</sup> Haitao Jiang<sup>3</sup> Rui Song<sup>2,3</sup>

<sup>1</sup>AWS AI Labs, <sup>2</sup>Amazon Core AI

<sup>3</sup>Department of Statistics, North Carolina State University {zshe, jiwngn}@amazon.com, {hjiang24, rsong}@ncsu.edu

## Abstract

With the growing popularity of deep-learning models, model understanding becomes more important. Much effort has been devoted to demystify deep neural networks for better interpretability. Some feature attribution methods have shown promising results in computer vision, especially the gradient-based methods where effectively smoothing the gradients with reference data is key to a robust and faithful result. However, direct application of these gradient-based methods to NLP tasks is not trivial due to the fact that the input consists of discrete tokens and the “reference” tokens are not explicitly defined. In this work, we propose Locally Aggregated Feature Attribution (LAFA), a novel gradient-based feature attribution method for NLP models. Instead of relying on obscure reference tokens, it smooths gradients by aggregating similar reference texts derived from language model embeddings. For evaluation purpose, we also design experi ments on different NLP tasks including Entity Recognition and Sentiment Analysis on public datasets as well as key feature detection on a constructed Amazon catalogue dataset. The su perior performance of the proposed method is demonstrated through experiments.

## 1 Introduction

With the growing popularity of deep-learning models, model understanding becomes more and more critical in many folds. In one aspect, model understanding helps us understand what the model is doing by identifying crucial features among unstructured raw data. For example, Shrikumar et al., 2017 utilized the model explainability technique to discover motifs in regulatory DNA elements from distinct molecular signatures in the field of Genomics. In another aspect, model understanding helps people audit or debug the deep models. An interesting example is that Ribeiro et al. (Ribeiro et al., 2016) found that their image classification model sometimes misclassifies a husky as a wolf.

The model explainability tool reveals that their model relies on the snow in the background rather than the appearance when distinguishing the two animals. More importantly, model understanding helps gain trust when making important decisions based on the model. In the NLP domain, deep language models are quickly evolving and show superior performance in various benchmark tasks. However, even experts struggle to understand the mechanism of complex language models.

Much effort has been devoted to demystifying the “black box” of deep models. A natural idea is through feature attribution, explaining the model by attributing the prediction to each input feature according to how much it affects the model output, of which two main directions emerge. One is model agnostic approaches including Shapley regression values (Shapley, 1953) and LIME (Ribeiro et al., 2016). We can apply these methods regardless of the model structure, however, they could suffer from computational inefficiency in the scenario of high dimensional input space and complex deep models when making inferences across all possible permutations or with small perturbations in the local neighborhood.

Another direction is model-specific approaches which look into the internal model mechanism to understand specific models. Gradient-based feature attribution models are often adopted to explain neural networks since gradients can be easily accessed through back-propagation, which gives a great computational advantage over model-agnostic methods. Since the gradient map itself is often noisy and challenging to interpret, most gradient-based methods aim to stabilize the feature attribution score by smoothing the gradients or learning from the reference data (Sundararajan et al., 2017; Smilkov et al., 2017; Lundberg and Lee, 2017). However, direct application of these gradient-based methods to NLP problems is not trivial, due to the fact that the input consists of discrete tokens and the “reference”

tokens are not explicitly defined.

In this paper, we propose Locally Aggregated Feature Attribution (LAFA), a novel gradientbased approach that leverages sentence-level embedding as a smoothing space for the gradients, motivated by the observation that the feature attribution is often shared by similar text inputs. For example, key features in product descriptions on an online marketplace are often shared by similar products. We implement a neighbor-searching method to ensure the quality of neighboring sentences.

Furthermore, to evaluate feature attribution methods in NLP, we consider two situations. For datasets with golden labels of feature score, we use the Area Under Curve (AUC) or Pearson correlation as the performance metric. As for datasets without golden labels, we conduct a similar evaluation task following prior works (Shrikumar et al., 2017; Lundberg and Lee, 2017) by masking tokens with high importance scores and find the change in the predicted log-odds.

In summary, our contributions are threefold: First, we build a novel context-level smooth gradient approach for feature attribution in NLP. The key ingredients of our method are constructing an appropriate aggregation function over the smoothing space. Second, to the best of our knowledge, this is the first proposal to conduct numerical studies on multiple NLP tasks, including Entity Recognition and Sentiment Analysis, for feature attribution. Third, our method achieves superior performance compared with the state-of-the-art feature attribution methods.

The paper is organized as follows. Section 2 elaborates the current challenges of feature attribution in NLP and recaps the preliminaries about gradient-based feature attribution approaches. The proposed feature attribution method is described in section 3, followed by a review of other existing approaches in Section 4. The evaluation tasks and the application results on NLP are presented in Section 5.

## 2 Feature Attribution in NLP

Challenge Direct application of gradient-based methods to NLP problems is not trivia. There are three main challenges. First, NLP models consist of non-differentiable discrete input tokens, hence the gradient hook can only reach out to the embedding space and gradient-based feature attribution methods are not directly applicable to word tokens.

Second, the reference data in NLP are difficult to define. It is studied by Sundararajan. et al (Sundararajan et al., 2017) that using the gradient as the feature attribution may suffer from the problems of model saturation or thresholding. Model saturation means the perturbation of some elements in the input cannot change the output, and the thresholding problem indicates discontinuous gradients can produce misleading importance scores. Such problems can be addressed by comparing the difference between the gradient of input and reference data. The guiding principle to select reference data is to ask ourselves that “what am I interested in measuring differences against?” For example, in the tasks of binary classification on DNA sequence inputs, the reference data are chosen as the expected frequencies of DNA sequence or randomly shuffling the original sequence. However, in NLP tasks, randomly shuffling texts as reference may not be grammatically sensible.

Lastly, we note that the evaluation of the language model is much more challenging than the explanations of the images. In the image application, the important features of an image obtained from feature attribution methods can be visually validated by checking the composition of objects. However, the detected important features in language may require more domain knowledge to validate.

Problem Definition Feature attribution task can be formally formulated as follows. A deep model $\mathcal { F }$ is provided to be explained, which is fine-tuned on dataset . The input sentence is denoted as $X _ { 0 } = ( w _ { 1 } , w _ { 2 } , . . , w _ { T } ) ^ { \bar { T } }$ where $w _ { i }$ represents i-th word. The goal for feature attribution is to determine function $\mathcal { M } ( \cdot )$ by quantifying the importance score of each word $\pmb { \mathcal { M } } ( x ) = ( m _ { 1 } , m _ { 2 } , . . , m _ { T } ) ^ { T }$ where $m _ { i }$ denotes the importance score for $w _ { i }$

Simple Gradient as Feature Attribution As illustrated in the first challenge above, in NLP models, directly taking derivative on each word is infeasible due to the non-differentiable embedding layer. We can resolve the challenge as follows.

The fist layer of the NLP model usually maps input discrete tokens to embedding from a predefined dictionary.

$$
h _ { 0 , i } = e m b ( w _ { i } ) , i = 1 , 2 , . . , T ,\tag{1}
$$

where $h _ { 0 , i } \in \mathbb { R } ^ { d }$ represents the word embedding for $w _ { i } .$ . This step is non-derivative. But we can

![](images/f93801fc8e08871979f589edd56dde62eeaf803e7253e90cba6b499606472d6f.jpg)  
Figure 1: Upper panel shows the overview of LAFA methods; Lower panel provides a motivating example of LAFA method. In this motivating example, the input text is a description of computer. The key features of the computer should include “Brand”, “CPU type” and “RAM size”. The simple gradient method may miss certain feature, such as “RAM size” in the example, while the gradients on similar texts can provide more contexts. The proposed method is constructed to aggregate the information from similar texts summarized in Algorithm 1.

obtain the derivative of output with respect to the word embedding:

$$
\begin{array} { r } { S ( H _ { 0 } ) = \partial \mathcal { F } / \partial H _ { 0 } \in \mathbb { R } ^ { T \times d } , } \end{array}\tag{2}
$$

where $H _ { 0 } = ( h _ { 0 , 1 } , h _ { 0 , 2 } , . . , h _ { 0 , T } ) ^ { T } \in \mathbb { R } ^ { T \times d }$ . Then, we consider the feature attribution score of a token $\mathcal { M } ( X ) \in \mathbb { R } ^ { T }$ as the sum of squares of the gradients with regard to each word embedding dimension:

$$
\mathcal { M } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } S ( H _ { 0 } ) _ { i , j } ^ { 2 } , \ i = 1 , 2 , . . , T .\tag{3}
$$

However, simply using the gradients of one token as feature attribution would lead to noisy results (Sundararajan et al., 2017). The next section describes a novel feature attribution approach that smoothes the gradients by leveraging similar input texts.

## 3 The Proposed Framework: LAFA

The proposed method contains three steps: (1) find the appropriate neighbors of the input text for gradient smoothing; (2) calculate the gradients of texts as well as neighbors; (3) aggregation of the gradients. The proposed framework is summarized in the upper panel of Figure 1. One motivating example is shown in the lower panel of Figure

1. In this motivating example, the input text is a description of computer. The key features of the computer should include “Brand”, “CPU type” and “RAM size”. The simple gradient method may miss certain feature, such as “RAM size” in the example, while the gradients on similar texts can provide more contexts. The proposed method is constructed to aggregate the information from similar texts.

Step I: Context-level Localization Given the input text $X _ { 0 } ~ \in ~ { \mathcal { X } }$ , where denotes the input datasets, the goal is to find similar texts $\mathscr { X } _ { s i m } =$ $\{ X _ { 1 } , X _ { 2 } , . . , X _ { M } \} \subset \mathcal { X }$ such that the feature attributions of $X _ { 0 }$ and $X _ { j } \in \mathcal { X } _ { s i m }$ are similar under a defined similarity metric.

To obtain similar texts $\mathcal { X } _ { s i m }$ , we first define an encoder that maps the text with discrete word tokens to a continuous embedding vector; then, in the embedding space, similar texts are found in the neighbor of $X _ { 0 }$ . To be specific, let $\mathcal { H } _ { e n c o d e r }$ denote the mapping from input to one of the hidden layers in deep model . $\mathcal { X } _ { s i m }$ can be obtained by choosing closest texts in the dataset as follows:

$$
\begin{array} { r } { X \in \mathcal { X } \qquad } \\ { s . t . \ : | | \mathcal { H } _ { e n c o d e r } ( X ) - \mathcal { H } _ { e n c o d e r } ( X _ { 0 } ) | | _ { 2 } < \epsilon } \end{array}\tag{4}
$$

where $| | \cdot | | _ { 2 }$ represents $L ^ { 2 }$ norm. ϵ is a threshold score to guarantee that founded neighbors are similar to the center text $X _ { 0 }$ to improve the faithfulness of aggregation. In our application, a fixed quantile served as the cut-off rate of L2 distance for all possible pairs is chosen as the threshold score to filter the nearest-neighbor result. During inference time, we apply the hidden layer encoder $\mathcal { H } _ { e n c o d e r }$ to all the input datasets and index, then using FAISS <sup>1</sup> (Johnson et al., 2017) offline. FAISS is an efficient, open-source library for similarity search and clustering on dense vectors, which can be applied to large-scale vectors.

The output of this step, $\mathcal { X } _ { s i m }$ can be viewed as the reference data to smooth the feature attribution of $X _ { 0 }$ , which addresses the second challenge listed in Section 2.

Step II: Taking Gradients According to Equation (3), the gradient of $X _ { i }$ can be denoted as ${ \mathcal M } ( X _ { i } ) \ : = \ ( m _ { i , 0 } , m _ { i , 1 } , . . , m _ { i , T _ { i } } ) ^ { T }$ for $\begin{array} { r l } { i } & { { } = } \end{array}$ $0 , 1 , . . , , M$ where $T _ { i }$ represent the token length of $X _ { i }$ . To be noticed that our proposed method can be easily extended to variants of simple gradient including smooth gradient or integrated gradient methods (Smilkov et al., 2017; Sundararajan et al., 2017) in Step II.

Step III: Aggregation over Multiple Feature Attribution Our goal is to smooth the gradient ${ \mathcal { M } } ( X _ { 0 } )$ by aggregating the gradients of similar text inputs:

$$
\begin{array} { r } { \mathcal { M } _ { L A F A } ( X _ { 0 } ) = A G G R E G A T E ( \mathcal { M } ( X _ { 0 } ) ; } \\ { \mathcal { M } ( X _ { 1 } ) , . . , \mathcal { M } ( X _ { M } ) ) \quad } \end{array}\tag{5}
$$

Since the lengths of $X _ { 0 } , X _ { 1 } , . . , X _ { M }$ may vary, the lengths of gradients $\mathcal { M } ( X _ { 0 } ) , \mathcal { M } ( X _ { 1 } ) , . . . , \mathcal { M } ( X _ { M } )$ are different as well. Consequently, aggregation by simply taking the average is infeasible. Following the intuition that, the tokens with high gradients in $\mathcal { X } _ { s i m }$ should be important in $X _ { 0 } ,$ , we propose the following aggregation function:

$$
\begin{array} { r } { \mathcal { M } _ { L A F A } ( X _ { 0 } ) = \mathcal { M } ( X _ { 0 } ) + \lambda ( \mathcal { E } ( w _ { 0 , 1 } ; \mathcal { X } _ { s i m } ) , . . , } \\ { \mathcal { E } ( w _ { 0 , T } ; \mathcal { X } _ { s i m } ) ) ^ { T } , \qquad } \end{array}\tag{6}
$$

where λ is a hyper-parameter for leveraging the feature attribution from similar inputs. $\mathcal { E } ( w ; \mathcal { X } _ { s i m } )$ is a scalar representing the importance of token w obtained from the neighbor inputs $\mathcal { X } _ { s i m }$ . Formally,

it can be defined as

$$
\mathcal { E } ( w ; \mathcal { X } _ { s i m } ) = \frac { 1 } { | \mathcal { X } _ { s i m } | } \sum _ { i = 1 } ^ { | \mathcal { X } _ { s i m } | } \sum _ { k = 1 } ^ { T _ { i } } \frac { m _ { i , k } \times k ( h , h _ { i , k } ) } { T _ { i } } ,\tag{7}
$$

where $h , h _ { i , k }$ are the word embedding of w and $w _ { i , k }$ as in Equation (1) respectively, and $k ( \cdot , \cdot )$ is a kernel function (Hofmann et al., 2008) (examples of kernel function are listed in the Appendix E. ). According to Equation (7), if word w and $w _ { i , k }$ have a high similarity, then inner product between the embeddings h and $h _ { i , k }$ in the kernel space would be large, which would assign a large weight to the corresponding importance score $m _ { i , k }$ . On the contrary, dissimilar word $w _ { i , k }$ in $\mathcal { X } _ { s i m }$ has little effect to the word w in $\mathcal { E } ( w ; \mathcal { X } _ { s i m } )$ The whole process is summarized in Algorithm 1.

Algorithm 1 Feature attribution method with   
smoothing over similar inputs.   
1: Input: Text of interest $X _ { 0 } ,$ input datasets $x ,$   
and fine-tuned deep model ${ \mathcal F } .$   
2: Output: Feature attribution of $X _ { 0 }$   
3: Step I: Localization   
4: Construct encoder which maps from input   
space to the space of hidden layer in ${ \mathcal F } .$   
5: Obtain the similar texts set $\mathcal { X } _ { s i m }$ =   
$\{ X _ { 1 } , X _ { 2 } , . . , X _ { M } \}$ of $X _ { 0 }$ according to Equa  
tion (4).   
6: Step II: Taking Gradient   
7: Calculate the gradient of texts $X _ { 0 } , X _ { 1 } , . . , X _ { M }$   
according to Equation (3).   
8: Step III: Aggregation   
9: Smooth the gradient of text of interest over the   
gradient of similar texts according to Equation   
(6).   
10: Output the aggregated gradient $\mathcal { M } _ { L A F A } ( X _ { 0 } )$   
as the feature attribution.

Discussion of Faithfulness One important criteria for model explainability method is “faithfulness”, which refers to how accurately it reflects the true reasoning process of the model (Jacovi and Goldberg, 2020). In our proposed method, the original input $X _ { 0 }$ is infused with similar texts in the input dataset for better interpretation. Since the deep model  is also trained on $\mathcal { X } ,$ , using similar texts $\mathcal { X } _ { s i m } \subset \mathcal { X }$ to facilitate explanation will not violate the faithfulness.

In the localization step (Step I), out of the consideration about faithfulness, we do not use popular bi-encoder frameworks, such as S-BERT (Reimers and Gurevych, 2019) or DenseRetrival (Karpukhin et al., 2020), to obtain similar neighbors. Because it will involve an extra black box model when explaining deep model ${ \mathcal F } .$

## 4 Related Work

In NLP, transformer-based models yield great successfulness and some works focus on explaining the attention mechanism. For example, Serrano and Smith, 2019 and Jain and Wallace, 2019 inspected a single attention layer and found out that attention weights only weakly and inconsistently correspond to feature importance; Wiegreffe and Pinter, 2019 argued that we cannot separate the attention layer and should view the entire model as a whole. In this section, We mainly review the gradient-based methods for feature attribution.

Feature Attribution on Single Input Simonyan et al (Simonyan et al., 2013) computed the “saliency map” denoted as Simple Gradient from the derivative of the output with respect to the input in an image classification task. In the NLP application, “saliency map” is obtained as the derivative of the output with respect to the word embedding as in Equation (2). However, “saliency map” can be visually noisy. Several methods are proposed to improve the gradient method from different perspectives. Gradient\*Input method (Shrikumar et al., 2017) improves the visual sharpness of the “saliency map” by multiplying gradient with the input itself. In NLP, we can write it as:

$$
\begin{array} { r l } & { \displaystyle \mathcal { S } _ { G r a d * I n p u t } ( H _ { 0 } ) = H _ { 0 } \times \mathcal { S } ( H _ { 0 } ) } \\ & { \displaystyle \mathcal { M } _ { G r a d * I n p u t } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } \mathcal { S } _ { G r a d * I n p u t } ( H _ { 0 } ) _ { i , j } ^ { 2 } . } \end{array}
$$

Layerwise Relevance Propagation method (Bach et al., 2015) is shown to be equivalent to the Gradient\*Input method up to a scaling factor. Smooth Gradient method (Smilkov et al., 2017) smoothes the feature attribution score by adding random noises to the input and taking average of the gradients from noisy inputs, formally:

$$
\begin{array} { l } { { \displaystyle \left. S _ { S m o o t h G r a d } ( H _ { 0 } ) \approx \frac { 1 } { N } \sum _ { k = 1 } ^ { N } S ( H _ { 0 } + \epsilon _ { k } ) , \right. } } \\ { { \displaystyle \left. \epsilon _ { k } \sim N ( 0 , \sigma ^ { 2 } ) , \right. } } \\ { { \displaystyle \left. \mathcal { M } _ { S m o o t h G r a d } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } S _ { S m o o t h G r a d } ( H _ { 0 } ) _ { i , j } ^ { 2 } . \right. } } \end{array}
$$

Guided Backpropagation method (Springenberg et al., 2014) modifies the back-propagation to preserve negative gradients in the ReLU activation layer which also sharpens the “saliency map” visually. Other methods, such as Grad-CAM or Guided-CAM (Selvaraju et al., 2017), are applicable to specific architecture of neural networks in the field of computer vision.

Since language models like BERT do not contain specific architecture utilized in Guided Backpropagation or Grad-CAM method, we ignore the mathematical formulation here.

Feature Attribution on Input with Reference Data Integrated Gradient method computes the feature score by integrating the gradients from single pre-determined reference input to the target input (Sundararajan et al., 2017). In computer vision problems, black image is usually considered as the reference data, and integrating gradients from the black image to the input image represents the feature attribution of the input image. In NLP problems, we can define the i-th element of feature attribution as:

$$
\begin{array} { l } { { \displaystyle \mathcal { S } _ { I n t e G r a d } ( H _ { 0 } ) _ { i j } } } \\ { { \displaystyle ~ \approx \frac { H _ { 0 , i j } - H _ { i j } ^ { \prime } } { N } \sum _ { k = 1 } ^ { N } S ( H ^ { \prime } + k \frac { H _ { 0 } - H ^ { \prime } } { N } ) _ { i j } } , } \\ { { \displaystyle \mathcal { M } _ { I n t e G r a d } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } S _ { I n t e G r a d } ( H _ { 0 } ) _ { i , j } ^ { 2 } } . } \end{array}
$$

where $H ^ { \prime }$ denotes the embedding of reference text.

SHAP-Gradient method which combines ideas from Integrated Gradient and Smooth Gradient into a single expected value equation (Lundberg and Lee, 2017) . To be specific, the feature attribution is defined from:

$$
\begin{array} { l } { \displaystyle \mathcal { S } _ { S h a p G r a d } ( H _ { 0 } ) } \\ { \displaystyle \approx \frac { 1 } { N } \sum _ { k = 1 } ^ { N } { S ( \alpha _ { k } H _ { 0 } + ( 1 - \alpha _ { k } ) H _ { k } ) } , } \\ { \displaystyle M _ { S h a p G r a d } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } { S _ { S h a p G r a d } ( H _ { 0 } ) _ { i , j } ^ { 2 } } . } \end{array}
$$

where $\alpha _ { k } \sim U ( 0 , 1 )$ denotes uniform distribution from zero to one, $H _ { k } \in \mathcal { H } _ { r e f }$ denotes the embedding of reference text.

DeepLIFT (Shrikumar et al., 2017) assigns the feature score by comparing the difference of contribution between input and some reference inputs via gradient. As discussed in (Lundberg and Lee, 2017), DeepLIFT can be considered as an approximation of Shapley Value estimation. Specifically, as in the application of $\mathrm { S H A P } ^ { 2 }$ , the feature attribution of SHAP-Deep as a variant of DeepLIFT is defined as:

$$
\begin{array} { l } { { \displaystyle S _ { S h a p D e e p } ( H _ { 0 } ) \approx \frac { 1 } { N } \sum _ { k = 1 } ^ { N } S ( H _ { k } ) \times ( H _ { 0 } - H _ { k } ) . } } \\ { { \displaystyle M _ { S h a p D e e p } ( X ) _ { i } = \sum _ { j = 1 } ^ { d } S _ { S h a p D e e p } ( H _ { 0 } ) _ { i , j } ^ { 2 } . } } \end{array}
$$

## 5 Experiments

In this section, we compare the proposed method to the state-of-the-art feature attribution methods under different use cases.

## 5.1 Case I: Feature Attribution on Relation Classification Model

<table><tr><td>Dataset</td><td>Precison</td><td>Recall</td><td>F1</td></tr><tr><td>NYT10</td><td>94.8</td><td>93.3</td><td>94.1</td></tr><tr><td>Webnlg</td><td>93.6</td><td>82.5</td><td>87.7</td></tr></table>

Table 1: Fine-tuned result on multi-label relation classification task.

Motivation Relation Classification is beneficial to downstream problems, including question answering and knowledge graph (KG) construction tasks (Wen et al., 2016; Dhingra et al., 2016; Dong et al., 2020). With the development of deep language model, existing relation extraction methods have achieved significant performance in relation classification task (Soares et al., 2019; Wei et al., 2019). We hope to better understand the features in the text that help deep language model to classify the relations. In this use case, we fine-tune a deep language model with relation as labels. With the fine-tuned model, the feature attribution technique is applied to identify the entities in the text as important features.

Data We use the public available datasets NYT10 (Riedel et al., 2010) and Webnlg (Gardent et al., 2017) for numerical study. Zeng et al., 2018 adapted the original dataset for relation extraction task. We follow the same setting as in Zeng et al., 2018, i.e. NYT10 dataset contains

56,196/5,000/5,000 plain texts in train/val/test set, 24 relation type, averaged 2.01 relational triples in each text. Webnlg dataset contains 5,019/500/703 plain texts in train/val/test set, 211 relation type, averaged 2.78 relation triples in each text.

Language Model We fine-tuned BERT-base models to classify the relations for NYT10 and Webnlg datasets, respectively. We use the plain text as input X, and relations as multi-class label Y in the model fine-tuning. Since multiple relations may exist in single text, we use the Sigmoid activation in the output layer. Mean Square Error (MSE) is used as loss objective and Adam (Kingma and Ba, 2014) is adopted as the optimizer. The micro Precision, Recall and F1 results are reported in Table 1 with 0.5 threshold of output score. From the result, the F1 scores are high for both NYT10 and Webnlg dataset, hence we can apply feature attribution methods to the fine-tuned models and identify the important features in the text which help to classify the relations.

<table><tr><td rowspan="2">Method</td><td colspan="2">AUC</td></tr><tr><td>NYT</td><td>Webnlg</td></tr><tr><td>Rand</td><td>0.498 (0.143)</td><td>0.501 (0.121)</td></tr><tr><td>SimpleGrad</td><td>0.949 (0.071)</td><td>0.670 (0.135)</td></tr><tr><td>InputGrad</td><td>0.953 (0.061)</td><td>0.713 (0.120)</td></tr><tr><td>InteGrad</td><td>0.948 (0.077)</td><td>0.663 (0.126)</td></tr><tr><td>SmoothGrad</td><td>0.960 (0.064)</td><td>0.664 (0.142)</td></tr><tr><td>SHAP + Zero</td><td>0.805 (0.213)</td><td>0.670 (0.133)</td></tr><tr><td>SHAP + Ref.</td><td>0.872 (0.169)</td><td>0.675 (0.133)</td></tr><tr><td>LAFA</td><td>0.958 (0.060)</td><td>0.724 (0.115)</td></tr></table>

Table 2: Feature attribution result on Relation Classification model (Case I). Top two results are highlighted in bold.

Evaluation Metric In datasets, NYT10 and Webnlg, the positions of entities in triples are provided. Therefore, we can constructed the golden feature attribution label as follow. For text X = $( w _ { 1 } , w _ { 2 } , . . , w _ { T } ) ^ { T }$ and triple $( s , r , o )$ , where subject $s = ( w _ { i } , . . , w _ { j } )$ and object $o = ( w _ { k } , . . , w _ { s } )$ are words shown in the text from positions i to j and k to s, respectively. The gold labels of feature attribution for relation r is constructed as

$$
\mathcal { M } _ { g o l d } ( X ) = ( 0 , . . , 0 , 1 , . . , 1 , 0 , . . . , 0 , 1 , . . , 1 , . . 0 ) ^ { T }
$$

where we set 1 from positions i to j as well as k to s and set 0 on other positions.

We use the evaluation metric Area under Curve (AUC) to compare the feature attribution (X)

<table><tr><td rowspan="2">Method</td><td colspan="2">Pearson Correlation</td></tr><tr><td>SST-2</td><td>SST</td></tr><tr><td>Rand SimpleGrad</td><td>0.039(0.074)</td><td>0.040(0.072)</td></tr><tr><td>InputGrad</td><td>0.441(0.083) 0.456(0.080)</td><td>0.430(0.081) 0.448(0.078)</td></tr><tr><td>InteGrad</td><td>0.468(0.071)</td><td>0.454(0.072)</td></tr><tr><td>SmoothGrad</td><td></td><td></td></tr><tr><td> $\mathrm { S H A P + Z e r o }$ </td><td>0.484(0.073)</td><td>0.471(0.073)</td></tr><tr><td></td><td>0.400(0.087)</td><td>0.392(0.085)</td></tr><tr><td> $\mathrm { S H A P + R e f . }$ </td><td>0.279(0.093)</td><td>0.278(0.091)</td></tr><tr><td>LAFA</td><td>0.494(0.070)</td><td>0.481(0.070)</td></tr></table>

Table 3: Feature attribution result on Sentiment Analysis Model (Case II).

and ${ \mathcal { M } } _ { g o l d } ( X )$ for the test dataset. AUC ranges from 0 to 1, higher AUC represents the the feature attribution result is closer to the gold feature attribution.

Main Results The results of AUC under different methods are summarized in Table 2. The popular feature attribution methods are listed and compared. More introduction about the competitors can be found in Section 4. “Rand”, as a baseline method, denotes that the feature score is randomly assigned, therefore, the AUC score is about 0.5. InputGrad method performs better than the Simple-Grad, showing the effeteness of Taylor approximation of layer-wise relevance propagation. “SHAP + Zero” means zero references are used in SHAP and $\mathrm { ^ { * } S H A P + R e f . ^ { \prime } }$ means $\mathcal { X } _ { s i m }$ is used as references. SHAP-based methods show low AUC values, because such methods aggregate the gradients of input and reference by simply taking average aggregation (see details in Section 4), which is not meaningful in NLP tasks. From the result, our method LAFA achieves a superior performance in Webnlg dataset and comparable performance in NYT dataset, which indicates that our feature attribution method can identify entities well.

## 5.2 Case II: Feature Attribution on Sentiment Analysis

Motivation The goal of the sentiment classification task is to classify a text into a sentiment categories such as positive or negative sentiment (Aghajanyan et al., 2021; Raffel et al., 2019; Jiang et al., 2020). In this use case, we hope to explain the deep sentiment classification model and obtain sentiment factors that drive the model to identify the sentiment.

Data The Stanford Sentiment Treebank (SST) (Socher et al., 2013) is a sentiment analysis dataset collected from English movie reviews (Pang and Lee, 2005). For all 9, 645 sentences in SST, Amazon Mechanical Turk labeled the sentiment for words/phrases/sentences yielded from the Stanford Parser (Manning et al., 2014) on a scale between 1 and 25. SST-2 is first introduced by GLUE (Wang et al., 2018), a famous multi-task benchmark and analysis platform for natrual language understanding, which took a subset from the SST and applied a two-way split (positive or negative) on sentence-level labels. Owing to the fact that the train/validation/test split are aligned between SST and SST-2, we can run gradient-based methods on the either one of them. Note that we are only working with the test split for both data sets, which contains 2210 and 1821 sentences respectively.

Language Model We use a popular and publicly available Distill-BERT (Sanh et al., 2019) model which is fine-tuned on SST-2 <sup>3</sup>. The accuracy of the Distill-BERT model on SST and SST-2 is 86.6% and 92.4% respectively.

Evaluation Metric We extract word-level sentiments from the phrase structure tree (PTB) in SST dataset. We take an absolute value after centralization to yield the golden label ${ \mathcal { M } } _ { g o l d } ( X )$ . Pearson correlation coefficient, $\rho$ , is the evaluation metric for feature attribution ${ \mathcal { M } } _ { g o l d } ( X )$ and (X). The correlation $\rho$ takes value from the range from 1 to 1, and a higher $\rho$ means better feature attribution result.

Main Results The main results of the correlation are summarized in Table 3. Popular feature attribution methods are listed and compared. To leverage the problem that some words can have opposite meaning when their sentiment are different, we only limited the sentences neighbor for same category. Based on the preliminary experiment, we choose the second layer with 10 neighbors and 0.39 cut-off rate, more details about preliminary experiment can be found in Appendix D that using all layers of DistilBERT as the encoder will improve the performance.

From the result Table 3, it is interesting to point out that the DistilBERT model fine-tuned on SST-2 does not perform equally well on the remaining sentences in SST, so the explanation we yield also has lower correlation for all methods compared with the SST-2. Some example and analysis when LAFA works and fails in this dataset by showing neighbor sentences can be found in Appendix B.

We can find out that the InputGrad method outperform SimpleGrad on SST/SST2 as well. SmoothGrad method achieves a good result by introducing random noise. From our observation, Sharpley-Value based methods, $\mathrm { ^ { * } S H A P + Z e r o ^ { * } }$ and $\mathrm { ^ { * } S H A P + R e f . ^ { \prime } }$ can identify important features with a good chance but may include several irrelevant tokens leading to higher variance. Our method LAFA achieves a superior performance in larger average correlation and smaller variance on both SST-2 and SST data sets.

## 5.3 Case III: Feature Attribution on Regression Model

Motivation Amazon’s online stores contain rich information about millions of products in product title, brand and description. We hope to better understand the trendy features that affecting price directly from such unstructured raw data, without the need for human labelers / data cleaning. In this application, we fine-tuned a deep language model with price as labels and aim to understand important factors from product descriptions with the given language model.

Data We collected the product catalog data of about one million products in personal computer category on Amazon’s online store. We concatenate product’s title, brand, bullet points and description as the input X, and use product price as the label Y.

Language Model We use BERT-base model and fine-tuned on collected catalog data for price regression.

Evaluation Metric To evaluate the performance of feature attribution methods without golden labels, we follow a similar idea as in work (Shrikumar et al., 2017; Lundberg and Lee, 2017) where the difference of prediction log-odds are measured by deleting pixels with highest importance scores. In our application, we first randomly select 200 input texts within a threshold of 1% prediction error as evaluation set. For each input text, we then mask p% of the tokens with highest feature attribution scores according to different feature attribution methods. Then we obtain new prediction result from the masked text denoted as $\hat { y } _ { m a s k e d }$ and calculate the new mean absolute percentage error (MAPE). Higher value of MAPE means that the corresponding method excels in picking important features.

![](images/9fddfe203c8c9af67736a7cc6b324b60cce9afe3f96538b28e7fc8572500d789.jpg)  
Figure 2: Feature attribution result on Case III. Comparisons of MAPE under different mask proportion.

Main Results The results are shown in Figure 2 where x-axis is the mask proportion p, and yaxis is MAPE. We observe that the random method has very low MAPE, because randomly masking the input texts will not affect the predicted result as much as the other feature attribution methods. ShapDeep and ShapGrad also have low MAPE values since simply taking average as aggregation is meaningless in NLP tasks. Other competing methods have similar performances on this case study and non of these performs better than others in a wide range of mask ratio. The proposed LAFA method outperforms other methods by significant margin with masking proportion from 5% to 50%, which demonstrates that smoothing over contextlevel neighbors helps to highlight the important features in similar type of products.

## 6 Conclusion

This paper presents a novel locally aggregated feature attribution method in NLP, which efficiently captures the important features by leveraging similar input texts in the embedding space. We focused on feature attribution of single input based on a fine-tuned model instead of training a language model, henceforth the computation time is of less concern.

One limitation of the LAFA model is that it requires informative neighbor sentences that carry similar information. Otherwise, aggregating information from other sentences could be misleading. Experiments in our datasets show that our method is effective, but the improvements gained from the

LAFA varies among different datasets based on the information that neighbor carries.

There are several future directions worthy of study. Firstly, labeling feature attribution result in the NLP requires massive human labor, and few datasets are available with golden feature attribution label. Developing new evaluation techniques to further measure model performance is interesting to investigate. Also, readable feature attribution results could help human beings to develop more business applications. For example, developing a key-value pair like processor-i5 as important feature can provide a more plausible feature attribution result to customers.

## References

Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. 2021. Muppet: Massive multi-task representations with pre-finetuning. arXiv preprint arXiv:2101.11038.

Sebastian Bach, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller, and Wojciech Samek. 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7):e0130140.

Bhuwan Dhingra, Lihong Li, Xiujun Li, Jianfeng Gao, Yun-Nung Chen, Faisal Ahmed, and Li Deng. 2016. Towards end-to-end reinforcement learning of dialogue agents for information access. arXiv preprint arXiv:1609.00777.

Xin Luna Dong, Xiang He, Andrey Kan, Xian Li, Yan Liang, Jun Ma, Yifan Ethan Xu, Chenwei Zhang, Tong Zhao, Gabriel Blanco Saldana, et al. 2020. Autoknow: Self-driving knowledge collection for products of thousands of types. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2724–2734.

Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. Creating training corpora for nlg micro-planning. In 55th annual meeting of the Association for Computational Linguistics (ACL).

Thomas Hofmann, Bernhard Schölkopf, and Alexander J Smola. 2008. Kernel methods in machine learning. The annals ofstatistics, pages 1171–1220.

Alon Jacovi and Yoav Goldberg. 2020. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? arXiv preprint arXiv:2004.03685.

Sarthak Jain and Byron C Wallace. 2019. Attention is not explanation. arXiv preprint arXiv:1902.10186.

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao. 2020. SMART: Robust and efficient fine-tuning for pretrained natural language models through principled regularized optimization. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2177–2190, Online. Association for Computational Linguistics.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with gpus. arXiv preprint arXiv:1702.08734.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Scott Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. arXiv preprint arXiv:1705.07874.

Christopher D Manning, Mihai Surdeanu, John Bauer, Jenny Rose Finkel, Steven Bethard, and David Mc-Closky. 2014. The stanford corenlp natural language processing toolkit. In Proceedings of 52nd annual meeting ofthe associationfor computational linguistics: system demonstrations, pages 55–60.

Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. arXiv preprint cs/0506075.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135– 1144.

Sebastian Riedel, Limin Yao, and Andrew McCallum. 2010. Modeling relations and their mentions without labeled text. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 148–163. Springer.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings ofthe IEEE international conference on computer vision, pages 618–626.

Sofia Serrano and Noah A Smith. 2019. Is attention interpretable? arXiv preprint arXiv:1906.03731.

Lloyd S Shapley. 1953. A value for n-person games. Contributions to the Theory of Games, 2(28):307– 317.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. 2017. Learning important features through propagating activation differences. In International Conference on Machine Learning, pages 3145–3153. PMLR.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2013. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. 2017. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, and Tom Kwiatkowski. 2019. Matching the blanks: Distributional similarity for relation learning. arXiv preprint arXiv:1906.03158.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642.

Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. 2014. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461.

Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, and Yi Chang. 2019. A novel cascade binary tagging framework for relational triple extraction. arXiv preprint arXiv:1909.03227.

Tsung-Hsien Wen, David Vandyke, Nikola Mrksic, Milica Gasic, Lina M Rojas-Barahona, Pei-Hao Su, Stefan Ultes, and Steve Young. 2016. A network-based end-to-end trainable task-oriented dialogue system. arXiv preprint arXiv:1604.04562.

Sarah Wiegreffe and Yuval Pinter. 2019. Attention is not not explanation. arXiv preprint arXiv:1908.04626.

Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu, and Jun Zhao. 2018. Extracting relational facts by an end-to-end neural model with copy mechanism. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 506–514.

## Appendix:

## A Model Implementation Detail

All experiments are conducted with eight NVIDIA Tesla V100 GPUs with 2.5 GHz (base) and 3.1 GHz (sustained all-core turbo) Intel Xeon 8175M processors.

Case I For LAFA, we adopt the cosine function as the kernel function and hyper-parameter with λ = 1 and SimpleGrad is implemented and aggregated by (x) in Equation (5).

Case II For LAFA, we adopt the Polynomial function as the kernel function $k ( \cdot , \cdot ) = \mathbb { I } ( \cdot , \cdot )$ and hyper-parameter with $\lambda = 0 . 4 4$ and SmoothGrad is chosen and aggregated by $\mathcal M ( x )$ in Equation (5).

For gradient-based model with hyperparameters, we tuned them on the first 100 sentences in the test set. From the grid [10, 25, 50], we choose 25 as the integral iteration and the smooth candidates.

Case III The indicator function as the kernel function $k ( \cdot , \cdot ) ~ = ~ \mathbb { I } ( \cdot , \cdot )$ with λ = 1 as hyperparameter is adopted for LAFA. Neighbor information is aggregated by (x) in Equation (5) from the SimpleGrad.

## B Example of Neighbor Sentences found by Case Studies

## B.1 Relation Extraction

In Figure 3, we show two examples in NYT and Webnlg with their neighbors. We can observe that detected neighbor sentences have a similar meaning, which can be utilized as a reference to help extract the key features from the original sentence.

![](images/872b002c99739c323f7205e2553318d2419d03f99135c26564d37be060a1f567.jpg)  
Figure 3: Example of neighbors for NYT and Webnlg. The head and tail entities are highlighted with red color.

## B.2 Sentiment Analysis

In SST-2, finding informative neighbors for every sentence is difficult because top sentences may not contain similar tokens, thus does not help. For this reason, we used a cut-off value for this data set to filter out non-informative sentences. In figure 4 we can find two examples from SST, one with “informative” good neighbors but another without them. Here for the word “informative” we use a quote because we are judging them based on our human understanding.

![](images/b06e369c7e6da6ee1ace14803bf0f67b97b182c74e3fd02110855e52118b232e.jpg)  
Figure 4: Example of neighbors for the SST data set. Sentiment factors found by SimpleGrad are highlighted with red color.

## C Examples of Different Feature Attribution Methods under Multiple Cases

Here we provide an example in Cases I and II. In Figure 5, LAFA identified locations and “lived” as the important factors for relation extraction, and the importance of the “Atlantic City” and “Bader Field” is stronger than the backbone SimGrad because of aggregation.

## D Experiment on Different Layer as Neighbor Encoder

Denote the size of $\mathcal { X } _ { s i m }$ as M, the choice of which can be a critical and challenging task. Intuitively, an overly small M would lead to under-smoothing because the target text cannot incorporate enough information from the neighbors. On the contrary, an overly large M would cause over-smoothing by introducing too much noise.

![](images/0a4d95c74ccd25dbcc6f932d1d96c8f88388362deb05afab4282af0f05495f30.jpg)  
Figure 5: Examples of Case Study I. Important factors are highlighted with red color.

![](images/a9b680e0ea2a52d9f7fe514f0e103aa2826cbb34202de18e1722f43a0d7dea22.jpg)  
Figure 6: Examples of Case Study III. Important features are highlighted with red color.

To clarify the neighbor searching process and the difference in the result using different layers, we show some experiments below.

Admittedly, we can directly use the WordPiece embedding as the encoder, which is the input of BERT-based models and enable us to find neighbors in the sense of “Word Similarity”. However, since the same word can have different meanings in different sentences, and thus different importance in yielded gradients, we might need to use another layer in the BERT model as the encoder to incorporate contextual information.

We separate the layer search process into two cases depending on the availability of a set of labels that categorizes similar contents into the same group. Generally speaking, both cases recommend the middle layer as the encoder based on our experience.

## D.1 When extra labels are not available

In the case of SST data, we do not have anything to group similar sentences, so we need to try for different possible layers and find the one that performs the best.

![](images/76086add891d22d59bd36165cd4f5fe60be49743e52b47810f5158b2d22c5fb1.jpg)  
Figure 7: Precision Result in Case Study III

Here we fixed the max number of neighbors as 10 and uses 0.05 quantile of sampled similarities as the cut-off rate to filter those neighbors that are not “actually close”. We use the SimpleGrad and the SmoothGrad as the baseline for comparison on the first 100 sentences in the test set.

From table A1 we can find out that LAFA is a generally good method that always beats the baseline when we use the smooth gradient as the basement method. Layer 2 performs the best among candidates. The combination of SmoothGrad and Layer w is the final choice and we showed the results on entire SST in the main result part. From here we can find out that for all seven layers, information from faithful neighbors can bring some useful information to an existing sentence.

<table><tr><td>Method</td><td>SST_first100</td><td>Method</td><td>SST_first100</td></tr><tr><td>SimpleGrad</td><td>0.457(0.074)</td><td>SmoothGrad</td><td>0.481(0.064)</td></tr><tr><td>SpG + LAFA + L1</td><td>0.457(0.073)</td><td>SmG + LAFA + L1</td><td>0.488(0.063)</td></tr><tr><td>SpG + LAFA + L2</td><td>0.458(0.072)</td><td>SmG + LAFA + L2</td><td>0.490(0.063)</td></tr><tr><td>SpG + LAFA + L3</td><td>0.456(0.072)</td><td>SmG + LAFA + L3</td><td>0.489(0.065)</td></tr><tr><td>SpG + LAFA + L4</td><td>0.456(0.072)</td><td>SmG + LAFA + L4</td><td>0.478(0.064)</td></tr><tr><td>SpG + LAFA + L5</td><td>0.454(0.072)</td><td>SmG + LAFA + L5</td><td>0.479(0.063)</td></tr><tr><td>SpG + LAFA + L6</td><td>0.454(0.073)</td><td>SmG + LAFA + L6</td><td>0.483(0.065)</td></tr><tr><td>SpG + LAFA + L7</td><td>0.456(0.074)</td><td>SmG + LAFA + L7</td><td>0.481(0.059)</td></tr></table>

Table A1: Feature Attribution Result on first 100 test cases in SST, using simple and smooth gradient as baselines

## D.2 When we have extra label

The performance of encoders can be evaluated by the similarity between text $X _ { 0 }$ and similar texts $\mathcal { X } _ { s i m }$ obtained from Equation (4). In this application, we use the product category or subcategory which is an additional source of labels produced by Amazon to construct a proxy metric to evaluate the similarity. Define the metric of precision as:

![](images/cacc2c50d23b83e9a02a49ff2cc4d3e46754c72ef57a515a95ff87f627916d70.jpg)  
Figure 8: Comparisons for different kernel functions in Case III.

$$
P r e c i s i o n = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } ( c ( X _ { j } ) = c ( X _ { 0 } ) ) ,\tag{8}
$$

where $c ( \cdot )$ denotes the category or subcategory of the corresponding product, $\mathbb { I } ( \cdot )$ is the indicator function. A high precision represents that the text found $\mathcal { X } _ { s i m }$ are similar to the text of interest $X _ { 0 }$

In the numerical study, we randomly sample 10, 000 inputs texts and obtain their corresponding neighbor texts from Equation (4) with $M = 1 0$ using each of the 12 hidden layers in BERT as the encoder $\mathcal { H } _ { e n c o d e r }$ under $L _ { 2 }$ norm. Figure 7 shows the precision result from different encoders, where we observe that the fifth hidden layer has the highest precision in terms of both category and subcategory, which is consistent with the intuition that the middle layer is a trade-off of token-alike and output-alike inputs. In the following experiment, we adopt the fifth layer as the encoder. In general, when no external labels are provided, we may choose a different encoder depending on the use case.

## E Ablation Study on Kernel Function

In Case III, we conduct an ablation study with different choices of kernel functions using different mask ratio to find out if different kernel yields different learning speed:

1. Radial basis function kernel (RBF) :

$$
k _ { R B F } ( a , b ) = e x p ( - | | a - b | | _ { 2 } / l ^ { 2 } ) ,
$$

where larger hyper-parameter l indicates lower impact from neighbors and vice versa. In the numerical study, we choose $l = 2$ based on the range of embedding a and b.

2. Cubic kernel (Cubic):

$$
k _ { C u b i c } ( a , b ) = ( \gamma a ^ { T } b + c _ { 0 } ) ^ { d } ,
$$

where $\gamma = 7 , c _ { 0 } = 0$ and $d = 3 ,$ smaller γ means lower impact from neighbors.

3. Cosine kernel (Cosine):

$$
k _ { C o s } ( a , b ) = a ^ { T } b / | | a | | | b | | |
$$

This kernel function havee no parameter.

4. Laplacian kernel (Laplacian):

$$
k _ { L a p l a c i a n } ( a , b ) = e x p ( - | | a - b | | _ { 1 } / l ^ { 2 } ) ,
$$

in the numerical study, we choose $l = 2$

5. L2 norm based similarity (L2):

$$
k _ { L 2 } ( a , b ) = 1 / c l i p ( | | a - b | | _ { 2 } , \lambda _ { l e f t } , \lambda _ { r i g h t } ) ,
$$

where $c l i p ( \cdot , \lambda _ { l e f t } , \lambda _ { r i g h t } )$ denotes clip function with $\bar { \lambda } _ { l e f t } \overset { \cdot } { = } 0 . 3$ and $\lambda _ { r i g h t } = 3$ as clip boundary in numerical study.

6. Indicator function based similarity (Indicator):

$$
k _ { I n d i c a t o r } ( a , b ) = \mathbb { I } ( a , b ) ,
$$

where $\mathbb { I } ( \cdot , \cdot )$ denotes indicator function.

The results are shown in Figure 8. We observe that no single kernel function outperforms all other kernel functions under all mask ratios in this study. Indicator function shows a good performance when the masked ratio is greater than 10%, while RBF kernel shows a good performance when the masked ratio is smaller than 5%. This can due to the reason that the indicator function only aggregates identical words and this conservative manner helps when we lost most important words.