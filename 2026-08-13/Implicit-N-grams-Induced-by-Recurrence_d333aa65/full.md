# Implicit N-grams Induced by Recurrence

Xiaobing Sun and Wei Lu StatNLP Research Group Singapore University of Technology and Design xiaobing\_sun@mymail.sutd.edu.sg, luwei@sutd.edu.sg

## Abstract

Although self-attention based models such as Transformers have achieved remarkable successes on natural language processing (NLP) tasks, recent studies reveal that they have limitations on modeling sequential transformations (Hahn, 2020), which may prompt re-examinations of recurrent neural networks (RNNs) that demonstrated impressive results on handling sequential data. Despite many prior attempts to interpret RNNs, their internal mechanisms have not been fully understood, and the question on how exactly they capture sequential features remains largely unclear. In this work, we present a study that shows there actually exist some explainable components that reside within the hidden states, which are reminiscent of the classical n-grams features. We evaluated such extracted explainable features from trained RNNs on downstream sentiment analysis tasks and found they could be used to model interesting linguistic phenomena such as negation and intensification. Furthermore, we examined the efficacy of using such n-gram components alone as encoders on tasks such as sentiment analysis and language modeling, revealing they could be playing important roles in contributing to the overall performance of RNNs. We hope our findings could add in terpretability to RNN architectures, and also provide inspirations for proposing new architectures for sequential data.

## 1 Introduction

Modern recurrent neural networks (RNNs), including Long Short-Term Memory (LSTM) (Hochreiter and Schmidhuber, 1997) and Gated Recurrent Units (GRU) (Cho et al., 2014), have demonstrated impressive results on tasks involving sequential data. They have proven to be capable of modeling formal languages (Weiss et al., 2018; Merrill, 2019; Merrill et al., 2020) and capturing structural features (Li et al., 2015a,b, 2016; Linzen et al., 2016; Belinkov et al., 2017; Liu et al., 2019) on NLP tasks. Although Transformers (Vaswani et al., 2017) have achieved remarkable performances on

![](images/b2683463063e8aa17dc917594f1dcd198a45781a7dabee6108150652543f0ddc.jpg)  
Figure 1: An RNN hidden state may encode a linear combination of all the n-grams ending at the current time step.

NLP tasks such as machine translation, it is argued that they may have limitations on modeling hierarchical structure (Tran et al., 2018; Hahn, 2020) and cannot handle functions requiring sequential processing of input well (Dehghani et al., 2019; Hao et al., 2019; Bhattamishra et al., 2020; Yao et al., 2021). Furthermore, a recent work shows that combining recurrence and attention (Lei, 2021) can result in strong modeling capacity. Another recent work incorporating recurrent cells into Transformers (Hutchins et al., 2022) substantially improved performance on language modeling involving very long sequences, prompting re-investigations of RNNs. On the other hand, it was observed in prior work that RNNs were able to capture linguistic phenomena such as negation and intensification (Li et al., 2016), but the question why they could achieve so still largely remains unanswered.

In this work, we focus on better understanding RNNs from a more theoretical perspective. We demonstrate that the recurrence mechanism of RNNs may induce a linear combination of interpretable components. These components reside in their hidden states in the form of the iterated matrix-vector multiplication that is based on the representations of tokens in the (reverse) order they appear in the sequence. Such components, solely depending on inputs and learned parameters, can be conveniently interpreted and are reminiscent of those compositional features used in classical ngram models (Jurafsky and Martin, 2009). They may also provide us with insights on how RNNs compose semantics from basic linguistic units. Our analysis further shows that, the hidden state at each time step includes a weighted combination of components that represent all the “n-grams” ending at that specific position in the sequence as shown in Figure 1. We gave specific representations for the n-gram components in Elman RNNs (Elman, 1990), GRUs and LSTMs.

We investigated the interpretability of those ngram components on trained RNN models, and found they could explain phenomena such as negation and intensification and reflect the overall polarity on downstream sentiment analysis tasks, where such linguistic phenomena are prevalent. Our experiments also revealed that the GRU and LSTM models are able to yield better capabilities in modeling such linguistic phenomena than the Elman RNN model, partly attributed to the gating mechanisms they employed which resulted in more expressive n-gram components. We further show that the linear combination of such components yields effective context representations. We explored the effectiveness of such n-gram components (along with the corresponding context representations) as alternatives to standard RNNs, and found they can generally yield better results than the baseline compositional methods on several tasks, including sentiment analysis, relation classification, named entity recognition, and language modeling.

We hope that our work could give inspirations to our community, serving as a useful step towards proposing new architectures for capturing contextual information within sequences.<sup>1</sup>

## 2 Related Work

Interpretability of RNNs: A line of work focuses on the relationship between RNNs and finitestate machines (Weiss et al., 2018; Merrill, 2019; Suzgun et al., 2019; Merrill et al., 2020; Eyraud and Ayache, 2020; Rabusseau et al., 2019), providing explanation and prediction on the expressive power and limitations of RNNs on formal languages both empirically and theoretically. Kanai et al. (2017) investigated conditions that could prevent gradient explosions for GRU based on dynamics. Maheswaranathan et al. (2019) and Maheswaranathan and Sussillo (2020) linearized the dynamics of RNNs around fixed points of hidden states and elucidated contextual processing. Our work focuses on studying a possible mechanism of RNNs that handles exact linguistic features.

Another line of work aims to detect linguistic features captured by RNNs. Visualization approaches (Karpathy et al., 2015; Li et al., 2016) were initially used to examine compositional information in RNN outputs. Linzen et al. (2016) assessed LSTMs’ ability to learn syntactic structure and

Emami et al. (2021) gave rigorous explanations on the standard RNNs’ ability to capture long-range dependencies. Decomposition methods (Murdoch and Szlam, 2017; Murdoch et al., 2018; Singh et al., 2019; Arras et al., 2017, 2019; Chen et al., 2020) were proposed to produce importance scores for hierarchical interactions in RNN outputs. Our work can be viewed as an investigation on how those interaction came about.

Compositional Models: A variety of compositional functions based on vector spaces have been proposed in the literature to compose semantic meanings of phrases, including simple compositions of adjective-noun phrases represented as matrix-vector multiplication (Mitchell and Lapata, 2008; Baroni and Zamparelli, 2010) and a matrix-space model (Rudolph and Giesbrecht, 2010; Yessenalina and Cardie, 2011) based on matrix multiplication. Socher et al. (2012, 2013) introduced a recursive neural network model that assigns every word and longer phrase in a parse tree both a vector and a matrix, and represents composition of a non-terminal node with matrix-vector multiplication. Kalchbrenner and Blunsom (2013) employed convolutional and recurrent neural networks to model compositionality at the sentence and discourse levels respectively. Those models are designed in an intuitive manner based on the nature of languages thus being interpretable. We can show that RNNs may process contextual information in a way bearing a resemblance to those early models.

## 3 A Theory on N-gram Representation

First, let us spend some time to discuss how to represent n-grams. Various approaches to representing n-grams have been proposed in the literature (Mitchell and Lapata, 2008; Bengio et al., 2003; Mitchell and Lapata, 2008; Mnih and Teh, 2012; Ganguli et al., 2008; Orhan and Pitkow, 2020; Emami et al., 2021; Rudolph and Giesbrecht, 2010; Yessenalina and Cardie, 2011; Baroni and Zamparelli, 2010). We summarize in Table 1 different approaches for representing n-grams.

Although empirically it has been shown that different approaches can lead to different levels of effectiveness, the rationales underlying many of the design choices remain unclear. In this section, we establish a small theory on representing n-grams, which leads to a new formulation on capturing the semantic information within n-grams.

Let us assume we have a vocabulary V that consists of all possible word tokens. The set of ngrams can be denoted as $\mathbb { V } ^ { * }$ (including the special n-gram which is the empty string ϵ). Consider three n-grams $a , b ,$ and c from $\mathbb { V } ^ { * }$ , with their semantic representations $r ( a ) , r ( b )$ , and $r ( c )$ respectively. Similarly, we may have $r ( a b )$ which return the semantic representations of the concatenated n-grams ab. It is desirable for our representations to be compositional in some sense. Specifically, a longer n-gram may be semantically related to those shorter n-grams it contains in some way.

<table><tr><td>Model</td><td>N-gram Representation</td><td>Context Representation</td><td>L</td><td>Representative Work</td></tr><tr><td>Vector Multiplicative (VM)</td><td> $\pmb { v } _ { i : j } = g ( x _ { i } ) \odot \cdots \odot g ( x _ { j } )$ </td><td> ${ \boldsymbol { v } } _ { 1 : t }$ </td><td>t</td><td>Mitchell and Lapata (2008)</td></tr><tr><td>Matrix Multiplicative (MM)</td><td> $\begin{array} { r } { M _ { i : j } = \prod _ { k = i } ^ { j } A ( x _ { k } ) } \end{array}$ </td><td> $M _ { 1 : t }$ </td><td>t</td><td>Yessenalina and Cardie (2011)</td></tr><tr><td>Vector Additive (weighted) (VA-W)</td><td> $\pmb { v } _ { i : j } = \pmb { C } _ { j - i } \pmb { g } ( \pmb { x } _ { i } )$ </td><td> $\textstyle \sum _ { i = t - m + 1 } ^ { t } { \boldsymbol { v } } _ { i : t }$ </td><td>m</td><td>Bengio et al. (2003)</td></tr><tr><td>Vector Additive (exponentially weighted) (VA-EW)</td><td> $\pmb { v } _ { i : j } = \pmb { C } ^ { j - i } \pmb { g } ( \pmb { x } _ { i } )$ </td><td> $\textstyle \sum _ { i = 1 } ^ { t } v _ { i : t }$ </td><td>t</td><td>Emami et al. (2021)</td></tr><tr><td>Matrix-Vector Multiplicative (restricted) (MVM-R)</td><td> $v _ { i - 1 : i } = A ( x _ { i - 1 } ) g ( x _ { i } )$ </td><td> $v _ { t - 1 : t }$ </td><td>2</td><td>Baroni and Zamparelli (2010)</td></tr><tr><td>Matrix-Vector Multiplicative (MVM)</td><td> $\begin{array} { r } { \pmb { v } _ { i : j } = \left( \prod _ { k = j } ^ { i + 1 } A ( x _ { k } ) \right) \ b { g } ( x _ { i } ) } \end{array}$ </td><td>v1:t</td><td>t</td><td></td></tr><tr><td>Matrix-Vector Multiplicative-Additive (MVMA)</td><td> $\begin{array} { r } { \pmb { v } _ { i : j } = \left( \prod _ { k = j } ^ { i + 1 } A ( x _ { k } ) \right) \ b { g } ( x _ { i } ) } \end{array}$ </td><td> $\textstyle \sum _ { i = 1 } ^ { t } v _ { i : t }$ </td><td>t</td><td>This work</td></tr></table>

Table 1: Different models for defining representations for n-grams within the phrase $x _ { 1 } , x _ { 2 } , \ldots , x _ { t - 1 } , x _ { t }$ and constructing the context representation out of the n-grams during learning. $L \colon$ the maximum length allowed for the context representation. $C$ is a weight matrix, and $C _ { k }$ is a (relative) position-specific weight matrix. A and $g$ are functions that return a matrix and a vector respectively.

Under some mild compositional assumptions related to the principle of compositionality (Frege, $1 9 4 8 ) ^ { 2 } \ /$ , it is reasonable to expect that there exists some sort of rule or operation that allows us to compose semantics of longer n-grams out of shorter ones. Let us use  to denote such an operation. We believe a good representation system for ngrams shall satisfy several key properties. First, the semantics of the n-gram abc shall be determined through either combining the semantics of the two n-grams a and bc or through combining the semantics of ab and c. The semantics of abc is unique, regardless of which of these two ways we use. Second, for the empty string ϵ, it should not convey any semantics. Formally, we can write them as:<sup>3</sup>

Associativity: $\forall a , b , c \in \mathbb { V } ^ { * } , ( r ( a ) \otimes r ( b ) )$ ⊗ $r ( c ) = r ( a ) \otimes ( r ( b ) \otimes r ( c ) )$

• <sup>Identity:</sup> $\forall a \in \mathbb { V } ^ { * } , r ( a ) \otimes r ( \epsilon ) = r ( a )$ , and $r ( \epsilon ) \otimes r ( a ) = r ( a )$

This essentially shows that the representation space for all n-grams under the operation $\bigotimes ,$ denoted as $( \mathbb { V } ^ { \ast } , \otimes )$ , forms a monoid, an important concept in abstract algebra (Lallement, 1979), with significance in theoretical computer science (Meseguer and Montanari, 1990; Rozenberg and Salomaa, 2012).

On the other hand, it can be easily verified that the space of all $d \times d$ (where d is an integer) real square matrices under matrix multiplication, denoted as $( \mathbb { R } ^ { d \times d } , \cdot )$ , also strictly forms a monoid (i.e., it is associative and has an identity, but is not commutative). We can therefore establish a homomorphism from $\mathbb { V } ^ { * }$ to $\mathbb { R } ^ { d \times d }$ , resulting in the function $\boldsymbol { r } ( \cdot ) \in \mathbb { V } ^ { * }  \mathbb { R } ^ { d \times d } ,$

This essentially means that we may be able to rely on a sub-space within $\mathbb { R } ^ { d \times d }$ as our mathematical object to represent the space of n-grams, where the matrix multiplication operation can be used to compose representations for longer n-grams from shorter ones. Thus, for a unigram x (a single word in the vocabulary), we have:

$$
r ( x ) : = A _ { x }\tag{1}
$$

where $A _ { x } \ \in \ \mathbb { R } ^ { d \times d }$ is the representation for the word x (how to learn such a matrix is a separate question to be discussed later). Note that the empty string ϵ comes with a unique representation which is the d  d identity matrix I.

We can either use matrix left-multiplication or right-multiplication as our operator . Assume the language under consideration employs the left-toright writing system. It is reasonable to believe that a human reader processes the text left-to-right, and the semantics of the text gets evolved each time the reader sees a new word. We may use the matrix left-multiplication as the preferred operator in this case. The system will left-multiply (modify) an existing n-gram representation with a matrix associated with the new word that appears right after the existing n-gram, forming the representation of the new n-gram. Such an operation essentially performs a transform that simulates the process of yielding new semantics when appending a new word at the end of an existing phrase. With this, for a general n-gram $x _ { i } , x _ { i + 1 } , \ldots , x _ { t } \left( i \leq t \right)$ , we have:

$$
r ( x _ { i } , x _ { i + 1 } , \ldots , x _ { t } ) = \prod _ { k = t } ^ { i } A _ { x _ { k } }\tag{2}
$$

However, the conventional wisdom in NLP has been to use vectors to represent basic linguistic units such as words, phrases or sentences (Mikolov et al., 2013a,b; Pennington et al., 2014; Kiros et al., 2015). This can be achieved by a transform:

$$
\left( \prod _ { k = t } ^ { i } { A _ { x _ { k } } } \right) u\tag{3}
$$

where $\pmb { u } \in \mathbb { R } ^ { d }$ is a vector that maps the resulting matrix representation into a vector representation.

Next, we will embark on our journey to examine the internal representations of RNNs. As we will see, interestingly, our developed n-gram representations can emerge within such models.

## 4 Interpretable Components in RNNs

An RNN is a parameterized function whose hidden state can be written recursively as:

$$
h _ { t } = f ( x _ { t } , h _ { t - 1 } ) ,\tag{4}
$$

where $x _ { t }$ is the input token at time step t and $\pmb { h } _ { t - 1 } \in \mathbb { R } ^ { d }$ is the previous hidden state. Assume $f$ is differentiable at any point, with the Taylor expansion, $h _ { t }$ can be rewritten as:

$$
\begin{array} { r } { h _ { t } = f ( x _ { t } , 0 ) + \nabla f ( x _ { t } , 0 ) h _ { t - 1 } + o ( h _ { t - 1 } ) , } \end{array}\tag{5}
$$

where $\begin{array} { r } { \nabla f ( x _ { t } , \mathbf { 0 } ) = \frac { \partial f } { \partial h _ { t - 1 } } | _ { h _ { t - 1 } = 0 } } \end{array}$ is the Jacobian matrix, and o is the remainder of the Taylor series.

Let $g ( x _ { t } ) = f ( x _ { t } , \mathbf { 0 } )$ and $A ( x _ { t } ) = \nabla f ( x _ { t } , \mathbf { 0 } )$ Note that $g ( x _ { t } ) \in \mathbb R ^ { d }$ and $A ( x _ { t } ) \in \mathbb { R } ^ { d \times d }$ are both functions of $x _ { t }$ . Therefore, the equation above can be written as:

$$
\pmb { h } _ { t } = g ( \pmb { x } _ { t } ) + A ( \pmb { x } _ { t } ) \pmb { h } _ { t - 1 } + o ( \pmb { h } _ { t - 1 } ) .\tag{6}
$$

If the hidden state has a sufficiently small norm, it can be approximated by the first-order Taylor expansion as follows<sup>4</sup>:

$$
\begin{array} { r } { \pmb { h } _ { t } \approx g ( \pmb { x } _ { t } ) + \pmb { A } ( \pmb { x } _ { t } ) \pmb { h } _ { t - 1 } . } \end{array}\tag{7}
$$

Next we illustrate how this recurrence relation can help us identify some salient components.

## 4.1 Emergence of N-grams

Consider the simplified RNN with the following recurrence relation,

$$
\pmb { h } _ { t } = g ( \pmb { x } _ { t } ) + \pmb { A } ( \pmb { x } _ { t } ) \pmb { h } _ { t - 1 } ,\tag{8}
$$

where $\pmb { h } \in \mathbb { R } ^ { d }$ , and $g ( x _ { t } ) \in \mathbb R ^ { d }$ and $A ( x _ { t } ) \in \mathbb { R } ^ { d \times d }$ are functions of $x _ { t }$ . This recurrence relation can be expanded repeatedly as follows,

$$
\begin{array} { r l } & { \displaystyle h _ { t } = g ( x _ { t } ) + A ( x _ { t } ) g ( x _ { t - 1 } ) + A ( x _ { t } ) A ( x _ { t - 1 } ) h _ { t - 2 } } \\ & { \quad = \cdots = \displaystyle \sum _ { i = 1 } ^ { t } A ( x _ { t } ) \ldots A ( x _ { i + 1 } ) g ( x _ { i } ) } \\ & { \quad = \displaystyle \sum _ { i = 1 } ^ { t } \underbrace { \left( \prod _ { j = t } ^ { i + 1 } A ( x _ { j } ) \right) g ( x _ { i } ) } _ { v _ { i + 1 } } , } \end{array}
$$

We can see that ${ \boldsymbol { v } } _ { i : t }$ bear some resemblance to the term in Equation 3, which can be rewritten as:

$$
\left( \prod _ { j = t } ^ { i + 1 } \underbrace { A _ { x _ { j } } } _ { A ( x _ { j } ) } \right) \underbrace { \left( A _ { x _ { i } } u \right) } _ { g ( x _ { i } ) } ,\tag{9}
$$

With the definition $A ( x _ { j } ) : = A _ { x _ { j } }$ and $g ( x _ { i } ) : =$ $A ( x _ { i } ) { \mathbf { \em u } }$ , we can see ${ \boldsymbol { v } } _ { i : t }$ can be interpreted as an “n-gram representation” that we developed in the previous section. It is important to note that, however, the use of function $g ( x _ { i } )$ in RNNs may lead to greater expressive power than the original formulation based on $A _ { x _ { i } } u . ^ { 5 }$

This interesting result shows that the hidden state of a simple RNN (characterized by Equation 8) is the sum of the representations of all the n-grams ending at time step t. Such salient components within RNN also show that the standard RNN may actually have a mechanism that is able to capture implicit n-gram information as described above. This leads to the following definition:

Definition 1 (N-gram Representation) For the n-gram $x _ { i } , x _ { i + 1 } , \ldots , x _ { t } .$ , its representation is:

$$
{ \pmb v } _ { i : t } = \left( \prod _ { j = t } ^ { i + 1 } A ( x _ { j } ) \right) g ( x _ { i } ) ,\tag{10}
$$

where $A ( x _ { j } ) \in \mathbb { R } ^ { d \times d } a n d g ( x _ { i } ) \in \mathbb { R } ^ { d } .$

## 4.2 Context Representation

With the above definition, we may want to consider how to perform learning. The learning task involves identifying the functions A and $g -$ in other words, learning representations for word tokens.

A typical learning setup that we may consider here is the task of language modeling. Such a task can be defined as predicting the next word $x _ { t + 1 }$ based on the representation of preceding words $x _ { 1 } , x _ { 2 } , \ldots , x _ { t }$ which serves as its left context. This is an unsupervised learning task, where the underlying assumption involved is the distributional hypothesis (Harris, 1954). Specifically, the model learns how to “reconstruct” the current word $x _ { t + 1 }$ out of $x _ { 1 } , x _ { 2 } , \ldots , x _ { t }$ which serves as its context.

Now the research question is how to define the representation for this specific context. As this left context is also an $n { \mathrm { - g r a m } }$ , it might be tempting to directly use its n-gram representation defined above to characterize such a left context. However, we show such an approach is not desirable.

The n-gram representation for this context can be written in the following alternative form:

$$
{ \pmb v } _ { 1 : t } = \left( \prod _ { j = t } ^ { 2 } A ( x _ { j } ) \right) g ( x _ { i } ) = W ( x _ { 2 : t } ) g ( x _ { 1 } ) ,\tag{11}
$$

This shows that the n-gram representation of $x _ { 1 } , x _ { 2 } , \ldots , x _ { t }$ could be interpreted as a “weighted” representation of the word $x _ { 1 }$ (where the weight matrix is derived from the words between $x _ { 1 }$ and $x _ { t + 1 }$ , measuring the strength of the connection between them). However, ideally, the context representation shall not just take $x _ { 1 }$ but other adjacent words preceding $x _ { t + 1 }$ into account, where each word contributes towards the final context representation based on the connection between them. This leads to the following way of defining the context:

$$
\begin{array} { r } { \displaystyle \sum _ { i = 1 } ^ { t } \pmb { v } _ { i : t } = \sum _ { i = 1 } ^ { t } \left( \prod _ { j = t } ^ { i + 1 } A ( x _ { j } ) \right) g ( x _ { i } ) } \\ { \displaystyle = \sum _ { i = 1 } ^ { t } W ( x _ { i : t } ) g ( x _ { i } ) , } \end{array}\tag{12}
$$

In fact, such an idea of defining the context as a weighted combination of surrounding words is not new – it recurs in the literature of language modeling (Bengio et al., 2003; Mnih and Teh, 2012), word embedding learning (Mikolov et al., 2013a,b), and graph representation learning (Cao et al., 2016).

Interestingly, the hidden states in the RNNs, as shown in Equation 9, also suggest exactly the same way of defining this left context. Indeed, when using RNNs for language modeling, each hidden state is exactly serving as the context representation for predicting the next word in the sequence.

The above gives rise to the following definition: Definition 2 (Context Representation) For the n-gram $x _ { 1 } , x _ { 2 } , \ldots , x _ { t } ,$ , its representation when serving as the $( l e f t )$ context is:

$$
\pmb { c } _ { 1 : t } = \sum _ { i = 1 } ^ { t } \pmb { v } _ { i : t } = \sum _ { i = 1 } ^ { t } \left( \prod _ { j = t } ^ { i + 1 } A ( x _ { j } ) \right) g ( x _ { i } ) ,\tag{13}
$$

where $A ( x _ { j } ) \in \mathbb { R } ^ { d \times d }$ and $g ( x _ { i } ) \in \mathbb { R } ^ { d } .$

## 4.3 Model Parameterization

With the above understandings on such salient components within RNNs, we can now look into how different variants of RNNs parameterize the functions A and $g .$ . The definition of Elman RNN, GRU and LSTM together with the corresponding Jacobian matrix $A ( x _ { t } )$ and vector function $g ( x _ { t } )$ functions are listed in Table $2 ^ { 6 }$ . We discuss how such different parameterizations may lead to different expressive power when they are used in practice.

We can see the ways GRU or LSTM parameterize $A ( x _ { t } )$ and $g ( x _ { t } )$ appear to be more complex compared to Elman RNN. This can partially be attributed to their gating mechanisms. Although the original main motivation of introducing such mechanisms may be to alleviate the exploding gradient and vanishing gradient issues (Hochreiter and Schmidhuber, $1 9 9 7 ;$ Cho et al., 2014), we could see such designs also result in terms describing gates and intermediate representations. A and $g$ are then independently derived based on certain rich interactions between such terms. We believe such interactions may likely increase the expressive power of the resulting n-gram representations. We will validate these points and discuss more in our experiments.

## 5 Experiments

In our experiments, we focus on the following aspects: 1) understanding the effectiveness of the proposed n-gram (and context) representations when used in practice, as compared to baseline models; 2) examining the significance of the choice of context representation; 3) interpreting the proposed representations by examining how well they could be used to capture certain linguistic phenomena.

We employ the sentiment analysis, relation classification, named entity recognition (NER) and language modeling tasks as testbeds. The first task is often used in investigating n-gram phenomena (Yessenalina and Cardie, 2011; Li et al., 2016) while the others are often used in examining how capable an encoder is when extracting features from texts (Grave et al., 2018; Zhou et al., 2016; Lample et al., 2016).

Datasets For sentiment analysis, we considered the Stanford Sentiment Treebank (SST) (Socher et al., 2013), the IMDB (Maas et al., 2011), and the AG-news topic classification<sup>7</sup> (Zhang et al.,

<table><tr><td>Definition</td><td></td><td colspan="2">Parameterization</td></tr><tr><td>Ean</td><td> $h _ { t } = \operatorname { t a n h } ( W _ { i n } \mathbf { x } _ { t } + W _ { i h } { h _ { t - 1 } } )$ </td><td> $A ( x _ { t } ) = \mathrm { d i a g } [ \operatorname { t a n h } ^ { \prime } ( W _ { i n } \mathbf { x } _ { t } ) ] W _ { i h }$ </td><td> $g ( x _ { t } ) = \operatorname { t a n h } ( W _ { i n } \pmb { x } _ { t } ) .$ </td></tr><tr><td>GRU</td><td> $\pmb { r } _ { t } = \sigma ( \pmb { W } _ { i r } \pmb { x } _ { t } + \pmb { W } _ { h r } \pmb { h } _ { t - 1 } )$   $z _ { t } = \sigma \big ( W _ { i z } \pmb { x } _ { t } + W _ { h z } \pmb { h } _ { t - 1 } \big )$   ${ \pmb n } _ { t } = \operatorname { t a n h } ( { \pmb W } _ { i n } { \pmb x } _ { t } + { \pmb r } _ { t } \odot { \pmb W } _ { h n } { \pmb h } _ { t - 1 } )$   $\pmb { h } _ { t } = \left( 1 - \pmb { z } _ { t } \right) \odot \pmb { n } _ { t } + \pmb { z } _ { t } \odot \pmb { h } _ { t - 1 }$ </td><td> $A ( x _ { t } ) = \mathrm { d i a g } [ f _ { n } ( x _ { t } ) \odot [ 1 - g _ { z } ( x _ { t } ) ] \odot g _ { r } ( x _ { t } ) ] W _ { h n }$   $- \mathrm { d i a g } [ g _ { n } ( x _ { t } ) \odot f _ { z } ( x _ { t } ) ] W _ { h z }$   $+ \mathrm { d i a g } [ g _ { z } ( x _ { t } ) ]$   $g ( x _ { t } ) = [ 1 - \overset { \vartriangle } { g _ { z } ( \ v x _ { t } ) } ] \odot g _ { n } ( \ v x _ { t } )$ </td><td>where:  $g _ { r } ( x _ { t } ) = \sigma ( W _ { i r } { \pmb x } _ { t } ) ,$   $f _ { r } ( t ) = g _ { r } ^ { \prime } ( x _ { t } ) ,$   $g _ { z } ( x _ { t } ) = \sigma ( W _ { i z } { \mathbf x } _ { t } ) , \qquad f _ { z } ( x _ { t } ) = g _ { z } ^ { \prime } ( x _ { t } ) ,$   $g _ { n } ( x _ { t } ) = \operatorname { t a n h } ( W _ { i n } x _ { t } ) , \ f _ { n } ( x _ { t } ) = g _ { n } ^ { \prime } ( x _ { t } ) .$ </td></tr><tr><td>LMM</td><td> $i _ { t } = \sigma ( W _ { i i } x _ { t } + W _ { h i } h _ { t - 1 } )$   $\pmb { f } _ { t } = \sigma ( \pmb { W } _ { i f } \pmb { x } _ { t } + \pmb { W } _ { h f } \pmb { h } _ { t - 1 } )$   $\pmb { o } _ { t } = \sigma ( \pmb { W } _ { i o } \pmb { x } _ { t } + \pmb { W } _ { h o } \pmb { h } _ { t - 1 } )$   $\pmb { c } _ { t } ^ { m } = \operatorname { t a n h } ( \pmb { W } _ { i c } \pmb { x } _ { t } + \pmb { W } _ { h c } \pmb { h } _ { t - 1 } )$   $\pmb { c } _ { t } = \pmb { f } _ { t } \odot \pmb { c } _ { t - 1 } + i _ { t } \odot \pmb { c } _ { t } ^ { m }$   $\pmb { h } _ { t } = \pmb { \sigma } _ { t } \odot \operatorname { t a n h } ( \pmb { c } _ { t } )$ </td><td> $A ( x _ { t } ) = { \bigg [ } { B _ { t } } D _ { t } { \bigg ] } , g ( x _ { t } ) = { \bigg [ } g _ { c } ( x _ { t } ) { \bigg ] }$  Bt = diag[gf(xt)]  $\overline { { E _ { t } } } = \mathrm { d i a g } \left[ g _ { o } ( x _ { t } ) \odot \mathrm { t a n h } ^ { \prime } [ g _ { c } ( x _ { t } ) ] \right] B _ { t }$   $D _ { t } = \mathrm { d i a g } [ g _ { c } ^ { m } ( x _ { t } ) \odot f _ { i } ( x _ { t } ) ] \bar { W } _ { h i }$   $+ \exp [ g _ { i } ( x _ { t } ) \odot f _ { c } ^ { m } ( t ) ] \dot { W } _ { h c }$   $F _ { t } = \mathrm { d i a g } \left[ g _ { o } ( x _ { t } ) \odot \mathrm { t a n h } ^ { \prime } [ g _ { c } ( x _ { t } ) ] \right] D _ { t }$ </td><td> $\begin{array} { r } { g _ { c } ( x _ { t } ) = g _ { i } ( x _ { t } ) \odot g _ { c } ^ { m } ( x _ { t } ) , } \end{array}$   $g _ { h } ( x _ { t } ) = g _ { o } ( x _ { t } ) \odot \operatorname { t a n h } [ g _ { c } ( x _ { t } ) ] ,$   $g _ { i } ( x _ { t } ) = \sigma ( W _ { i i } \pmb { x } _ { t } ) ,$   $f _ { i } ( x _ { t } ) = g _ { i } ^ { \prime } ( x _ { t } ) ,$   $g _ { f } ( x _ { t } ) = \sigma ( W _ { i f } { \pmb x } _ { t } ) ,$   $f _ { f } ( x _ { t } ) = g _ { f } ^ { \prime } ( x _ { t } ) ,$   $g _ { o } ( x _ { t } ) = \sigma ( W _ { i o } { \pmb x } _ { t } ) ,$   $f _ { o } ( x _ { t } ) = \mathring { g _ { o } ^ { \prime } } ( x _ { t } ) ,$ </td></tr></table>

Table 2: Parameterization of A and g by Elman RNN, GRU, and LSTM. $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is the representation of the input token $x _ { t }$ and $W _ { \ast }$ refers to a weight matrix. σ and tanh are the element-wise sigmoid and tanh functions respectively. $g ^ { \prime } ,$ tanh′ and $f ^ { \prime }$ refer to the element-wise derivative. The diag operation converts a vector into a diagonal matrix.

2015) datasets. The first dataset has sufficient labels for phrase-level analysis, the second dataset has instances with relatively longer lengths, and the third one is multi-class. For relation classification and NER, we considered the SemEval 2010 Task 8 (Hendrickx et al., 2010) and CoNLL-2003 (Tjong Kim Sang and De Meulder, 2003) datasets respectively. For language modeling, we considered the Penn Treebank (PTB) dataset (Marcus et al., 1993), the Wikitext-2 (Wiki2) dataset and the Wikitext-103 (Wiki103) dataset (Merity et al., 2016). PTB is relatively small while Wiki103 is large. The statistics are shown in Tables 6 and 7 in the appendix.

Baselines The n-gram representations (together with their corresponding context representations) discussed in the literature are considered as baselines, which are listed in Table 1 along with the MVMA and MVM models. MVM(A)-G/L/E refers to the MVM(A) model created with the A and g functions derived from GRU/LSTM/Elman, but are trained directly from data. The A and g functions for GRU, LSTM and Elman are listed in Table 2.

Additionally, to understand whether the complexity of A affects the expressive power, we created a new model called MVMA-ME, which comes with an A function that is slightly more complex than that of MVMA-E but less complex than those of MVMA-G and MVMA-L: $A ( x _ { t } ) =$ 0.25 diag[tanh(W x<sub>t</sub>)]M + 0.5I and $g ( x _ { t } ) \ =$ tanh $\left( W ^ { \prime } x _ { t } \right)$ (here, W, M and $W ^ { \prime }$ are learnable weight matrices). The g function is the same as that of MVMA-E.

Setup For sentiment analysis, relation classification and language modeling, models consist of one embedding layer, one RNN layer, and one fullyconnected layer. The Adagrad optimizer (Duchi et al., 2011) was used along with dropout (Srivastava et al., 2014) for sentiment analysis<sup>8</sup> and relation classification. For language modeling, models were trained with the Adam optimizer (Kingma and Ba, 2014). We ran word-level models with truncated backpropagation through time (Williams and Peng, 1990) where the truncated length was set to 35. Adaptive softmax (Joulin et al., 2017) was used for Wiki103. For NER, models consist of one embedding layer, one bidirectional RNN layer, one projection layer and one conditional random field (CRF) layer. The SGD optimizer was used. Final models were chosen based on the best validation results. More implementation details can be found in the appendix.

<table><tr><td rowspan="2">Model</td><td colspan="2">SST-2</td><td colspan="2">AG-news</td><td colspan="2">IMDB</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>MM</td><td>86.0±1.3</td><td>85.6±0.4</td><td></td><td></td><td></td><td></td></tr><tr><td>VA-W</td><td>80.6±1.6</td><td>80.4±1.4</td><td>90.3±0.4</td><td>90.0±0.3</td><td>88.0±0.6</td><td>88.0±0.4</td></tr><tr><td>VA-EW</td><td>82.6±0.3</td><td>82.0±0.3</td><td></td><td></td><td></td><td></td></tr><tr><td>MVM-G</td><td>84.9±0.5</td><td>85.0±1.0</td><td>84.9±4.0</td><td>84.4±4.0</td><td>50.9±0.0</td><td>50.2±0.1</td></tr><tr><td>MVM-L</td><td>85.4±0.4</td><td>84.9±0.8</td><td>86.9±1.7</td><td>86.5±1.7</td><td>51.0±0.1</td><td>50.2±0.1</td></tr><tr><td>MVM-E</td><td>59.6±1.6</td><td>59.5±1.1</td><td></td><td></td><td></td><td></td></tr><tr><td>MVMA-G</td><td>87.0±0.4</td><td>85.3±0.5</td><td>91.6±0.5</td><td>91.3±0.3</td><td>90.5±0.5</td><td>89.6±0.7</td></tr><tr><td>MVMA-L</td><td>86.7±1.0</td><td>85.4±1.0</td><td>91.4±0.5</td><td>91.3±0.5</td><td>89.4±0.6</td><td>89.2±0.6</td></tr><tr><td>MVMA-E</td><td>81.4±1.1</td><td>80.8±1.5</td><td></td><td></td><td></td><td></td></tr><tr><td>MVMA-ME</td><td>83.2±0.5</td><td>81.9±0.3</td><td>90.6±0.5</td><td>90.2±0.3</td><td>80.6±0.5</td><td>80.1±1.1</td></tr><tr><td>GRU</td><td>84.9±0.9</td><td>84.9±0.5</td><td>92.1±0.1</td><td>91.6±0.3</td><td>87.7±0.2</td><td>87.2±0.3</td></tr><tr><td>LSTM</td><td>84.3±0.8</td><td>84.4±0.3</td><td>91.9±0.4</td><td>91.5±0.5</td><td>89.0±0.1</td><td>88.7±0.4</td></tr><tr><td>Elman</td><td>79.1±0.3</td><td>79.7±1.4</td><td>87.5±0.5</td><td>87.5±0.6</td><td>67.0±1.9</td><td>66.7±0.9</td></tr></table>

Table 3: Accuracy percentage ( ) on sentiment analysis (text classification) datasets (averaged over 3 runs). “-” means the model failed to converge.

## 5.1 Comparison of Representation Models

We investigate how baseline n-gram representation models<sup>9</sup>, the MVM model, and the MVMA model perform on the aforementioned testbeds. We also compare with the standard RNN models.

Sentiment Analysis Apart from the GRU and LSTM models, it can be observed that our MVMA-G and MVMA-L models are also able to achieve competitive results on three sentiment analysis datasets, as we can see from Table 3, demonstrating the efficacy of those recurrence-induced n-gram representations. Although Elman RNN and its corresponding MVMA-E and MVM-E models also have a mechanism for capturing n-gram information (similar to GRU and LSTM), they did not perform well, which may be attributed to a limited expressive power of their A and g functions when used for defining n-grams as described previously.

Both MM and VA-EW fail to converge on AGnews and IMDB, showing challenges for them to handle long instances. This may be explained by the lengthy matrix multiplication involved in their representations, which may result in vanishing/exploding gradient issues. Interestingly, MVM-G and MVM-L, which solely rely on the longest ngram representation, are also able to achieve good results on SST-2, indicating a reasonable expressive power of such n-gram representations alone. However, they fail to catch up with MVMA-G and MVMA-L on IMDB which contains much longer instances, confirming the significance of the context representation, which captures n-grams of varying lengths.

Unlike MVMA-E, the MVMA-ME model does not suffer from loss stagnation on AG-news and IMDB but the performance on IMDB obviously falls behind MVMA-G and MVMA-L as shown in Table 3. This indicates a sufficiently expressive A(x<sub>t</sub>) (such as the Jacobian matrices of GRU and LSTM) may be needed to handle long instances.

Relation Classification & NER For relation classification, context representations (or final hidden states) are used for classification. For NER, we use the concatenated context representations (or hidden states) at each position of bidirectional models to predict entities and their types. Table 4 shows that MVMA-G and MVMA-L outperform the MVM-G and MVM-L models respectively on both tasks, again confirming the effectiveness of the context representations. MVM(A)-E did not perform as well as MVM(A)-G and MVM(A)-L, which demonstrates the significance of expressive power for the A and g functions. Similar to the results in sentiment analysis, MVMA-ME did not perform as well as MVMA-G and MVMA-L. However, to our surprise, MVMA-ME did not outperform VA-EW on NER, suggesting that a delicate choice of A can be important for this task. The poor performance of VA-W on NER might be explained by a weak expressive power of its n-gram representations. MM fails to converge on the relation classification task, which implies it is not robust across different datasets. Interestingly, it is remarkable that MVMA-G, MVMA-L and MVMA-E could yield competitive results compared to GRU, LSTM and Elman on NER, implying such n-gram representations could be crucial for our NER task.

<table><tr><td rowspan="2">Model</td><td colspan="2">Relation Classification</td><td colspan="2">NER</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>MM</td><td></td><td></td><td>33.9±0.6</td><td>30.8±0.4</td></tr><tr><td>VA-W</td><td>41.2±0.2</td><td>37.9±0.9</td><td>17.6±0.6</td><td>16.5±1.6</td></tr><tr><td>VA-EW</td><td>39.7±1.1</td><td>38.3±0.7</td><td>70.8±0.7</td><td>63.4±1.0</td></tr><tr><td>MVM-G</td><td>51.2±0.5</td><td>52.6±0.7</td><td>54.2±1.6</td><td>47.6±2.2</td></tr><tr><td>MVM-L</td><td>48.8±1.3</td><td>50.5±1.5</td><td>53.8±1.7</td><td>46.6±1.6</td></tr><tr><td>MVM-E</td><td></td><td></td><td>27.8±0.9</td><td>25.6±0.9</td></tr><tr><td>MVMA-G</td><td>62.2±1.0</td><td>59.7±0.1</td><td>75.0±0.4</td><td>67.7±0.5</td></tr><tr><td>MVMA-L</td><td>57.5±0.3</td><td>56.2±0.8</td><td>75.6±0.2</td><td>67.9±0.3</td></tr><tr><td>MVMA-E</td><td>27.8±0.9</td><td>25.6±0.9</td><td>69.0±0.4</td><td>61.7±0.1</td></tr><tr><td>MVMA-ME</td><td>46.3±0.9</td><td>46.2±0.6</td><td>67.0±0.5</td><td>57.6±0.8</td></tr><tr><td>GRU</td><td>67.2±0.6</td><td>62.2±0.2</td><td>75.6±0.5</td><td>67.9±0.5</td></tr><tr><td>LSTM</td><td>65.2±0.9</td><td>61.3±1.4</td><td>76.3±0.5</td><td>68.1±0.5</td></tr><tr><td>Elman</td><td>27.8±0.9</td><td>25.6±0.9</td><td>67.1±0.9</td><td>58.6±0.6</td></tr></table>

Table 4: F1 scores ( ) (averaged over 3 runs) on the relation classification and NER tasks. “-” means the model failed to converge.
<table><tr><td>Model</td><td></td><td>PTB</td><td>Wiki2</td><td>Wiki103</td></tr><tr><td rowspan="2">GRU</td><td>dev</td><td>118.4±0.4</td><td>146.1±0.4</td><td>109.4±0.6</td></tr><tr><td>test</td><td>110.1±0.4</td><td>136.8±0.1</td><td>113.3±0.8</td></tr><tr><td rowspan="2">MVMA-G</td><td>dev</td><td>119.8±0.4</td><td>150.3±0.8</td><td>111.8±0.5</td></tr><tr><td>test</td><td>111.1±0.2</td><td>140.2±1.0</td><td>115.2±0.5</td></tr><tr><td rowspan="2">MVM-G</td><td>dev</td><td>146.5±1.3</td><td>170.1±2.8</td><td>-</td></tr><tr><td>test</td><td>138.8 ±1.0</td><td>160.0±2.6</td><td></td></tr><tr><td rowspan="2">LSTM</td><td>valid</td><td>118.6±0.4</td><td>150.6±0.6</td><td>108.3±0.6</td></tr><tr><td>test</td><td>109.8±0.4</td><td>140.4±0.8</td><td>112.4±0.8</td></tr><tr><td rowspan="2">MVMA-L</td><td>dev</td><td>121.5±0.5</td><td>152.0±0.5</td><td>109.1±0.6</td></tr><tr><td>test</td><td>113.2±0.5</td><td>142.5±0.7</td><td>112.6±0.6</td></tr><tr><td rowspan="2">MVM-L</td><td>dev</td><td>124.3±1.5</td><td>155.6±0.9</td><td>1</td></tr><tr><td>test</td><td>117.0±1.0</td><td>145.7±1.6</td><td></td></tr><tr><td rowspan="2">MVMA-ME</td><td>dev</td><td>140.7±0.9</td><td>169.0±1.0</td><td>153.1±4.2</td></tr><tr><td>test</td><td>134.0±1.0</td><td>158.4±1.4</td><td>157.4±4.3</td></tr></table>

Table 5: Perplexities ( ) on language modeling (averaged over 5 runs). $^ { 6 6 } - \ ' 2$ the model failed to converge.

Language Modeling For the language modeling task, we choose MVMA-G, MVMA-L, MVM-G and MVM-L for experiments. We also run MVMA-ME. As we can see from Table 5, there are performance gaps between the MVMA models and the standard RNNs – though the gaps often do not appear to be particularly large. This indicates there may be extra information within higher order terms of the standard RNN functions useful for such a task. Yet, such information cannot be captured by the MVMA models that employ simplified functions. The gaps between the MVM models and MVMA models are remarkable, which again indicates that the correct way of defining the left context representation can be crucial for the task of next word prediction. MVMA-ME did not perform well on the language modeling task, which might be attributed to the less expressive power of its functions A and g.

## 5.2 Interpretation Analysis

We conduct some further analysis to examine the interpretability of n-gram representations. Specifically, we examine whether the models are able to capture certain linguistic phenomena such as negation, which is important for sentiment analysis (Ribeiro et al., 2020). We also additionally made comparisons with the vanilla Transformer (Vaswani et al., 2017) here<sup>10</sup> despite the fact that it remains largely unclear how it precisely captures sequence features such as n-grams.

We could also obtain the n-gram representations and the corresponding context representations from the learned standard RNN models, based on their learned parameters. We denote such n-gram representations as $\mathrm { R N N } _ { n \cdot \mathrm { g r a m } } .$ , and the context representations as $\mathrm { R N N } _ { \mathrm { c o n t e x t } } .$ , where “RNN” can be GRU, LSTM or Elman. As n-gram representations are vectors, a common approach is to transform them into scalars with learnable parameters (Murdoch et al., 2018; Sun and Lu, 2020). We define the n-gram polarity score to quantify the polarity information as captured by an n-gram representation ${ \boldsymbol { v } } _ { i : t }$ from time step i to t, which is calculated as:

$$
\begin{array} { r } { s _ { i : t } ^ { v } = { w ^ { \top } } v _ { i : t } , } \end{array}\tag{14}
$$

where w is the learnable weight vector of the final fully-connected layer. We also define the context polarity score for the context as $\textstyle \sum _ { i = 1 } ^ { t } s _ { i : t } ^ { v }$

We trained RNNs and baseline models on SST-2 and automatically extracted 73 positive adjectives (e.g., “nice” and “enjoyable”) and 47 negative adjectives (e.g., “bad” and “tedious”) from the vocabulary<sup>11</sup>. N-gram polarity scores were calculated for those adjective unigrams and their negation bigrams formed by prepending “not” to them. For VA-EW and VA-W, their n-gram representations do not involve tokens other than the last token. Such limitations prevent them from capturing any negation information. We therefore calculate the context polarity scores using their context representations instead (which in this case is a bigram). This also applies to Transformer for the same reason.

We observed that, for the GRU and LSTM models along with their corresponding MVMA models, the n-gram representations are generally able to learn the negation for both the adjective and their negation bigrams as shown in Figures 2a and 2b<sup>12</sup>, prepending “not” to an adjective will likely reverse the polarity. This might be a reason why they could achieve relatively higher accuracy on the sentiment analysis tasks. Interestingly, MVM-G could also capture negation as shown in Figure 2c, again suggesting the impressive expressive power of such n-gram representations alone.

However, as shown in Figure 2, models such as VA-W, MVMA-E, and MM are struggling to capture negation for negative adjectives, again implying a weaker expressive power of their n-gram representations. Specifically, MVMA-E fails to capture negation for negative adjectives, which may be attributed to a relatively weaker Jacobian matrix function A (as compared to those of GRU and LSTM) preventing them from pursuing optimal conditions.

Figure 2e shows that the MVMA-ME model, which has a function A less complex than the ones from MVMA-G and MVMA-L but more complex than the one from MVMA-E, still can generally learn negation of negative adjectives better than the MVMA-E model. This demonstrates the necessity of choosing more expressive A and g functions.

Interestingly, both VA-W and Transformer are struggling with capturing the negation phenomenon for negative adjectives in our experiments as shown in Figures 2g and 2h, which suggests that they may have a weaker capability in modeling sequential features in our setup. However, we found they could still achieve good performances on the AGnews and IMDB datasets<sup>13</sup>. We hypothesize this is because the nature of SST-2 makes these two models suffer more on this dataset – it has rich linguistic phenomena such as negation cases while the other two datasets do not.

Additionally, we examined the ability for GRU, LSTM, MVMA-G and MVMA-L to capture both the negation and intensification phenomena. For such experiments, instead of using SST-2, we trained the models on SST-5, which comes with polarity intensity information. Polarity intensities were mapped into values of 2, 1, 0, +1, +2 , ranging from extremely negative to extremely positive. We conducted some experiments based on the same setup above for capturing negation on SST-2. To our surprise, our preliminary results show that all models were performing substantially worse in terms of capturing intensification than capturing negations. We hypothesize that this is caused by the imbalance between negation phrases and intensification phrases. Specifically, the intensification word “very” (1,729 times) was exposed less than the negation word “not” (4,601 times) in the training set of SST-5.

One approach proposed in the literature for sentence classification is to consider all the hidden states of an RNN in an instance (Bahdanau et al., 2015). We believe this may actually be able to alleviate the above issue as it allows more n-grams within an instance to be exposed to the label information. Thus, we followed their approach for training our MVMA and MVM models<sup>14</sup>.

![](images/a8735849f1e42167be669f09d111508d5d3ee8ba005b551279b0d81c05c6c1f0.jpg)  
(a) GRU<sub>n-gram</sub>

![](images/d0a1da5c876169d46a78f52a43fb73f5c3554996ed3bc2b6a08594b2621ef73f.jpg)

![](images/782e2cbeea462a3e5af83f62ef687bc860bbe5c2aa55ce3b9139c78e09fc1aa2.jpg)  
(e) MVMA-ME

(b) MVMA-G  
![](images/ba6831752066bf8aca1c368c68f613221f5f81d2349e7681806a4384bc85389b.jpg)  
(f) MM

![](images/cc3650749854e1a2ec9555e02ec50c2fcc36d7624c2963d5a3ff54194677ef51.jpg)  
(c) MVM-G

![](images/7744437904cef902bf50a78a84b37cb635d542a591bd5b37af5a61df01233eb4.jpg)  
(g) VA-W

![](images/6e6c3667f15dd2d9299f0c77e8bd1af54865fddc4b7b4698f614249c5ed2a5fd.jpg)  
(d) MVMA-E

![](images/9821984060cf58d2598de1999057318d09f86ff5f2d411c76fa329158feb96e2.jpg)  
(h) Transformer

Figure 2: Distribution of polarity scores for adjectives and their negation bigrams on SST-2. p-adj and n-adj refer to the positive and negative adjectives respectively. [-] refers to the negation operation (prepending the word “not”). Circles refer to outliers. More results can be found in the appendix.  
![](images/37c7289506bc01ca4bdff2b41acc47bd852ceb9e5a17c59e383fdb1ec5ed8211.jpg)

![](images/e0e24fd913410ca0e7ea0adc913d7dbe8387cb6856f85db43e0ff80b3c805a35.jpg)  
Figure 3: Context polarity scores (MVMA-G, SST-5) for positive (L) and negative (R) adjectives along with their negation and intensification bigrams.

We can see that the negation and intensification phenomena can be explained by both the context representations in Figure 3<sup>15</sup>. Specifically, prepending either positive or negative adjectives with “very” will likely strengthen their polarity while adding “not” will likely weaken their polarity. These results suggest that RNNs are able to capture information of linguistic significance within the sequence, and our identified n-gram representations within their hidden states appear to be playing a salient role.

## 5.3 Discussion

From the experiments above, we can see that our introduced n-gram representations, coupled with the corresponding context representations, are powerful in practice in capturing n-gram information better than the baseline compositional models introduced in the literature. We also found that RNNs can induce such representations due to their recurrence mechanism<sup>16</sup>.

However, there can be several factors that affect the efficacy of different representations. First, through comparisons with different variants of

MVMA, we can see that the precise way of parameterizing the functions $A ( x _ { t } )$ and g(x<sub>t</sub>) matter. Second, through the comparison between MVMA and MVM, we can see that defining an appropriate context representation that incorporates a correct set of n-grams is also important. Third, for models which do not capture such explicit n-gram features like ours, interestingly, they may still be able to yield good performances on certain tasks. For example, though VA-W and Transformer did not perform well on SST-2, they yielded results competitive to GRU and LSTM on AG-news and IMDB. This observation indicates there could be other useful features captured by such models that can contribute towards their overall modeling power.

Although in this work we did not aim to propose novel or more powerful architectures, we believe our work can be a step towards better understanding of RNN models. We also hope it can provide inspiration for our community to design more interpretable yet efficient architectures.

## 6 Conclusion

In this work, we focused on investigating the underlying mechanism of RNNs in terms of handling sequential information from a theoretical perspective. Our analysis reveals that RNNs contain a mechanism where each hidden state encodes a weighted combination of salient components, each of which can be interpreted as a representation of a classical n-gram. Through a series of comprehensive empirical studies on different tasks, we confirm our understandings on such interpretations of these components. With the analysis coupled with experiments, we provide findings on how RNNs learn to handle certain linguistic phenomena such as negation and intensification. Further investigations on understanding how the identified mechanism may capture a wider range of linguistic phenomena such as multiword expressions (Schneider et al., 2014) could an interesting future direction.

## Acknowledgements

We would like to thank the anonymous reviewers and our ARR action editor for their constructive comments. This research/project is supported by the Ministry of Education, Singapore, under its Tier 3 Programme (The Award No.: MOET32020- 0004). Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not reflect the views of the Ministry of Education, Singapore.

## References

Leila Arras, José Arjona-Medina, Michael Widrich, Gré- goire Montavon, Michael Gillhofer, Klaus-Robert Müller, Sepp Hochreiter, and Wojciech Samek. 2019. Explaining and interpreting lstms. In Explainable ai: Interpreting, explaining and visualizing deep learning, pages 211–238. Springer.

Leila Arras, Grégoire Montavon, Klaus-Robert Müller, and Wojciech Samek. 2017. Explaining recurrent neural network predictions in sentiment analysis. In Proceedings ofthe 8th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In Proceedings of ICLR.

Marco Baroni and Roberto Zamparelli. 2010. Nouns are vectors, adjectives are matrices: Representing adjective-noun constructions in semantic space. In Proceedings ofEMNLP.

Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass. 2017. What do neural machine translation models learn about morphology? In Proceedings of ACL.

Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. The journal of machine learning research, 3:1137–1155.

Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. 2020. On the Ability and Limitations of Transformers to Recognize Formal Languages. In Proceedings ofEMNLP.

Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2016. Deep neural networks for learning graph representations. In Proceedings of AAAI.

Hanjie Chen, Guangtao Zheng, and Yangfeng Ji. 2020. Generating hierarchical explanations on text classification via feature interaction detection. In Proceedings of ACL.

Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of EMNLP.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. 2019. Universal transformers. In Proceedings ofICLR.

John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal ofmachine learning research, 12(Jul):2121–2159.

J. Elman. 1990. Finding structure in time. Cogn. Sci., 14:179–211.

Melikasadat Emami, Mojtaba Sahraee-Ardakan, Parthe Pandit, Sundeep Rangan, and Alyson K Fletcher. 2021. Implicit bias of linear rnns. In Proceedings of ICML.

Rémi Eyraud and Stéphane Ayache. 2020. Distillation of weighted automata from recurrent neural networks using a spectral approach. https://arxiv.org/abs/2009.13101.

Gottlob Frege. 1948. Sense and reference. The philosophical review, 57(3):209–230.

Surya Ganguli, Dongsung Huh, and Haim Sompolinsky. 2008. Memory traces in dynamical systems. Proceedings of the National Academy of Sciences, 105:18970 – 18975.

Edouard Grave, Armand Joulin, and Nicolas Usunier. 2018. Improving neural language models with a continuous cache. In Proceedings ofICLR.

Pankaj Gupta and Hinrich Schütze. 2018. LISA: Explaining recurrent neural network judgments via layer-wIse semantic accumulation and example to pattern transformation. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networksfor NLP.

Michael Hahn. 2020. Theoretical limitations of selfattention in neural sequence models. Transactions of the Associationfor Computational Linguistics.

Jie Hao, Xing Wang, Baosong Yang, Longyue Wang, Jinfeng Zhang, and Zhaopeng Tu. 2019. Modeling recurrence for transformer. In Proceedings ofNAACL.

Zellig S Harris. 1954. Distributional structure. Word, 10(2-3):146–162.

Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano, and Stan Szpakowicz. 2010. SemEval-2010 task 8: Multiway classification of semantic relations between pairs of nominals. In Proceedings ofthe 5th International Workshop on Semantic Evaluation.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735– 1780.

DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. 2022. Block-recurrent transformers. https://arxiv.org/abs/2203.07852.

Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou, et al. 2017. Efficient softmax approximation for gpus. In Proceedings ofICML.

Daniel Jurafsky and James H. Martin. 2009. Speech and Language Processing (2nd Edition). Prentice-Hall, Inc., USA.

Nal Kalchbrenner and Phil Blunsom. 2013. Recurrent convolutional neural networks for discourse compositionality. In Proceedings of the Workshop on Continuous Vector Space Models and their Compositionality.

Sekitoshi Kanai, Yasuhiro Fujiwara, and Sotetsu Iwamura. 2017. Preventing gradient explosions in gated recurrent units. In Proceedings ofNeurIPS.

Andrej Karpathy, Justin Johnson, and Li Fei-Fei. 2015. Visualizing and understanding recurrent networks. http://arxiv.org/abs/1506.02078.

Diederik P. Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. In Proceedings of ICLR.

Ryan Kiros, Yukun Zhu, Russ R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Skip-thought vectors. In Proceedings ofNeurIPS.

Gérard Lallement. 1979. Semigroups and combinatorial applications. John Wiley & Sons, Inc.

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In Proceedings ofNAACL.

Tao Lei. 2021. When attention meets fast recurrence: Training language models with reduced compute. In Proceedings ofEMNLP.

Jiwei Li, Xinlei Chen, Eduard Hovy, and Dan Jurafsky. 2016. Visualizing and understanding neural models in NLP. In Proceedings ofNAACL.

Jiwei Li, Thang Luong, and Dan Jurafsky. 2015a. A hierarchical neural autoencoder for paragraphs and documents. In Proceedings ofACL.

Jiwei Li, Thang Luong, Dan Jurafsky, and Eduard Hovy. 2015b. When are tree structures necessary for deep learning of representations? In Proceedings ofEMNLP.

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. Assessing the ability of LSTMs to learn syntaxsensitive dependencies. Transactions ofthe Associationfor Computational Linguistics, 4.

Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. 2019. Linguistic knowledge and transferability of contextual representations. In Proceedings ofNAACL.

Yi Luan, Dave Wadden, Luheng He, Amy Shah, Mari Ostendorf, and Hannaneh Hajishirzi. 2019. A general framework for information extraction using dynamic span graphs. In Proceedings ofNAACL.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings ofACL.

Niru Maheswaranathan and David Sussillo. 2020. How recurrent networks implement contextual processing in sentiment analysis. In Proceedings ofICML.

Niru Maheswaranathan, Alex H. Williams, Matthew D. Golub, S. Ganguli, and David Sussillo. 2019. Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. In Proceedings ofNeurIPS.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. https://arxiv.org/abs/1609.07843.

William Merrill. 2019. Sequential neural networks as automata. In Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges.

William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020. A formal hierarchy of RNN architectures. In Proceedings ofACL.

José Meseguer and Ugo Montanari. 1990. Petri nets are monoids. Information and computation, 88(2):105– 155.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. https://arxiv.org/abs/1301.3781.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Proceedings ofNeurIPS.

Jeff Mitchell and Mirella Lapata. 2008. Vector-based models of semantic composition. In Proceedings of ACL.

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. 2018. Spectral normalization for generative adversarial networks. In Proceedings of ICLR.

Andriy Mnih and Yee Whye Teh. 2012. A fast and simple algorithm for training neural probabilistic language models. In Proceedings ofICML.

W. James Murdoch, Peter J. Liu, and Bin Yu. 2018. Beyond word importance: Contextual decomposition to extract interactions from LSTMs. In Proceedings of ICLR.

W. James Murdoch and Arthur Szlam. 2017. Automatic rule extraction from long short term memory networks. In Proceedings ofICLR.

Emin Orhan and Xaq Pitkow. 2020. Improved memory in recurrent neural networks with sequential nonnormal dynamics. In Proceedings ofICLR.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Proceedings ofEMNLP.

Guillaume Rabusseau, Tianyu Li, and Doina Precup. 2019. Connecting weighted automata and recurrent neural networks through spectral learning. In Proceedings ofAISTATS.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of ACL.

Grzegorz Rozenberg and Arto Salomaa. 2012. Handbook ofFormal Languages: Volume 3 Beyond Words. Springer Science & Business Media.

Sebastian Rudolph and Eugenie Giesbrecht. 2010. Compositional matrix-space models of language. In Proceedings of ACL.

Nathan Schneider, Emily Danchik, Chris Dyer, and Noah A. Smith. 2014. Discriminative lexical semantic segmentation with gaps: Running the MWE gamut. Transactions ofthe Associationfor Computational Linguistics, 2:193–206.

Chandan Singh, W James Murdoch, and Bin Yu. 2019. Hierarchical interpretations for neural network predictions. In Proceedings ofICLR.

Richard Socher, Brody Huval, Christopher D. Manning, and Andrew Y. Ng. 2012. Semantic compositionality through recursive matrix-vector spaces. In Proceedings ofEMNLP.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings ofENMLP.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958.

Xiaobing Sun and Wei Lu. 2020. Understanding attention for text classification. In Proceedings of ACL.

Mirac Suzgun, Yonatan Belinkov, Stuart Shieber, and Sebastian Gehrmann. 2019. LSTM networks can perform dynamic counting. In Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges.

Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings ofCoNLL.

Ke Tran, Arianna Bisazza, and Christof Monz. 2018. The importance of being recurrent for modeling hierarchical structure. In Proceedings ofEMNLP.

Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of NeurIPS.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On the practical computational power of finite precision RNNs for language recognition. In Proceedings of ACL.

Ronald J. Williams and Jing Peng. 1990. An efficient gradient-based algorithm for on-line training of recurrent network trajectories. Neural Computation, 2(4):490–501.

Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. 2021. Self-attention networks can process bounded hierarchical languages. In Proceedings of ACL-IJCNLP.

Ainur Yessenalina and Claire Cardie. 2011. Compositional matrix-space models for sentiment analysis. In Proceedings ofEMNLP.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Proceedings ofNeurIPS.

Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. 2016. Attention-based bidirectional long short-term memory networks for relation classification. In Proceedings ofACL.

## A Dataset Statistics

The statistics of the sentiment analysis, relation classification and NER datasets are shown in Table 6. The language modeling datasets are obtained from Einstein.ai and the statistics are shown in Table 7.

<table><tr><td>Data</td><td>Train</td><td>Dev</td><td>Test</td><td>V.size</td><td>Max.len</td><td>Class</td></tr><tr><td>SST-2</td><td>98,794</td><td>872</td><td>1,821</td><td>17,404</td><td>54</td><td>2</td></tr><tr><td>IMDB</td><td>17,212</td><td>4,304</td><td>4,363</td><td>63,311</td><td>437</td><td>2</td></tr><tr><td>AG-news</td><td>110,000</td><td>10,000</td><td>7,600</td><td>85,568</td><td>212</td><td>4</td></tr><tr><td>SST-5</td><td>318,582</td><td>41,447</td><td>82,600</td><td>18,025</td><td>54</td><td>5</td></tr><tr><td>SemEval</td><td>7,000</td><td>1,000</td><td>2,717</td><td>27,115</td><td>91</td><td>10</td></tr><tr><td>CoNLL-2003</td><td>14,987</td><td>3,466</td><td>3,684</td><td>26,873</td><td>113</td><td>20</td></tr></table>

Table 6: Statistics of the sentiment analysis, relation classification and NER datasets. “V.size” indicates the vocabulary size and “Max.len” indicates the maximum length of the instances. “SemEval” refers to the SemEval 2010 Task 8 dataset for relation classification. For CoNLL-2003, “class” refers to the tag size.

We created the binary dataset SST-2 by extracting instances (including phrases) with polarity from the constituency parse trees in the original SST dataset (Socher et al., 2013). We merged the labels extremely positive and positive as positive and the labels extremely negative and negative as negative. We also extracted all the phrases in the constituency parse trees from the original dataset and created the 5-class dataset SST-5. The labels extremely positive, positive, neutral, negative and extremely negative were mapped into +2, +1, 0, -1, and -2 respectively.

## B More Result from the SST datasets

## B.1 Negation and Intensification

Figure 4 shows that the n-gram representations from the LSTM model together with its corresponding MVMA-L and MVM-L models can also capture negation on the extracted adjectives from SST-2. However, VA-EW fails to capture the negation phenomenon for the negative adjectives, which may be explained by that: the n-gram representation of VA-EW solely involves the current token, thus being less expressive compared to the one from models such as MVMA-L and MVMA-G. Moreover, the MVMA-G model can also capture the negation and intensification phenomena on SST-5 as shown in Figure 5. The intensification token will generally strengthen the polarity of an adjective while the negation token will generally weaken the polarity of it.

<table><tr><td colspan="2">Dataset</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>PTB</td><td>Token Num Vocab Size</td><td>887,521</td><td>70,390 10,000</td><td>78,669</td></tr><tr><td>Wiki2</td><td>Token Num Vocab Size</td><td>2,088,628</td><td>217,646 33,278</td><td>245,569</td></tr><tr><td>Wiki103</td><td>Token Num Vocab Size</td><td>103,227,021</td><td>217,646 267,735</td><td>245,569</td></tr></table>

Table 7: Statistics of the language modeling datasets.

![](images/20eea914583ff71cafde5a987ed84780c9975515ae7410e7d98c82be6ec234b0.jpg)  
(a) VA-EW

![](images/470e4dbe77edbc62255ea64c7c7311d3c23a17514cde8cde0b6a945a8d267833.jpg)  
(b) LSTM<sub>n-gram</sub>

![](images/b2fd6d08011107cbf5212f4991fb327be8c36e1fddb7d676941f8ab00e6d0e3b.jpg)  
(c) MVM-L

![](images/ea1ccb550f7df196a376a2f57f24b45073a75b31915f955117d808918f3f528f.jpg)  
(d) MVMA-L

Figure 4: Distribution of polarity scores for adjectives and their negation bigrams. p-adj and n-adj refer to the positive and negative adjectives respectively. [-] refers to the negation operation (prepending the word “not”). Circles refer to outliers.  
![](images/3bd5e6300ba7528acb308147eb7403cf10fa654c48bb71b8ec709d38d224194a.jpg)  
(a) GRU<sub>context</sub>, positive adjectives

![](images/c1cecb0a5ef5adce0e9787846f75aeeae51725c1d8eb7492586c3de73ce8a2c2.jpg)  
(b) GRU<sub>context</sub>, negative adjectives  
Figure 5: Context polarity scores for positive adjectives (a) and negative adjectives (b) along with their corresponding negation and intensification bigrams from SST-5.

We also visualized the polarity score of each ngram within a sentence. Two examples are shown in Figures 6a and 6b, where a warmer color indicates a higher polarity score (i.e., the n-gram is more positive). For example, “never” itself has a remarkably negative polarity score while “loses” has a remarkably positive one. Consequently, the ngrams starting from “never” (while ending with another word) generally have positive polarity scores.

![](images/d2ca1687036ba82181bcd40d8ee06240b5dbd30f72579532223f412df0ad7ce9.jpg)

(a)  
![](images/6ebc4319f0061c5900970e287a061194526ed0897eeb6f6f7dbc97383e410083.jpg)  
(b)  
Figure 6: Polarity scores for n-gram representations within two example sentences. SST-2, MVMA-G.

Such visualization results show that our identified representations defined over the linguistic units as captured by RNNs can be highly interpretable.

## B.2 First-order Approximation

To examine how well the recurrence relation in Equation 7 can approximate the standard RNNs, we followed the method in the work of Maheswaranathan and Sussillo (2020) and compared the hidden state of the standard RNNs $\begin{array} { r l } { ( h _ { t } } & { { } = } \end{array}$ $R N N ( { \bf x } _ { t } , h _ { t - 1 } ) )$ at each time step to the corresponding context representations $( \hat { h } _ { t } = g ( \pmb { x } _ { t } ) +$ ${ \pmb A } ( { \pmb x } _ { t } ) { \pmb h } _ { t - 1 } )$ . The error at each time step is defined as

$$
\begin{array} { r } { \vert \vert \pmb { h } _ { t } - \hat { \pmb { h } } _ { t } \vert \vert _ { 2 } / \vert \vert \pmb { h } _ { t } \vert \vert _ { 2 } . } \end{array}\tag{15}
$$

We used the current standard hidden state to predict the next hidden state and the context representations on the SST-2 test set.

We noticed that the weight decaying coefficient has a remarkable impact on the error. Specifically, a larger coefficient can result in smaller errors. When the coefficient is 1e 5, the average errors on the Elman, GRU, and LSTM models were 26.2%, 21.7% and 46.6% and respectively. When the coefficient is 3e  4 the the average errors dropped to 17.1%, 15.1%, and 33.3% respectively. Note that since this is the single step error, the accumulated errors across many times steps can be large, particularly for LSTM, and thus the first-order approximation cannot fully replace standard RNNs. Despite this, the resulting context and n-gram representations can help us understand how RNNs process contextual information such as n-gram features.

![](images/3bc021464ce80541345d11570b11f775e66e5bff71321054987b082b4044299e.jpg)  
(a) MVMA-G

![](images/6918045e656c4fadf9e7995f9692226345718d30e2e8619ce518266d56faa92a.jpg)  
(b) MVM-G  
Figure 7: (a) and (b): T-sne visualization of the context representation for phrases (<30 tokens) from the AGnews dataset with four topics.

## C T-sne Visualization

We visualized the context representations from the MVMA-G model using t-sne (van der Maaten and Hinton, 2008), which provides us with an intuitive understanding on the efficacy of our identified representations. We automatically extracted 2,188 phrases with less than 30 tokens from AG-news with 4 topics<sup>17</sup> and projected their context representations to a two-dimension space. Figures 7a and 7b show there exist four major clusters corresponding to the four topics, indicating those representations can generally learn the topic information and explain the differences. Similar to the previous analysis, the MVM-G model is able to learn the topic information with the n-gram representations.

## D Results on Transformer

We have also run the Transformer model on the sentiment analysis datasets and the results are listed in Table 8.

<table><tr><td colspan="2">SST-2</td><td colspan="2">AG-news</td><td colspan="2">IMDB</td></tr><tr><td>dev</td><td>test</td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr><tr><td>83.4±0.4</td><td>82.0±0.1</td><td>90.9±0.5</td><td>90.5±0.4</td><td>88.4±0.2</td><td>88.1±0.2</td></tr></table>

Table 8: Accuracy on sentiment analysis tasks. Transformer

## E Implementation Details

## E.1 Sentiment Analysis

Settings For the SST-2, AG-news, and IMDB datasets, we used the cross-entropy as the loss function to train the models. Embeddings were randomly initialized and trainable during training. For the SST-5 dataset, we treated the classification as a regression problem as the labels are polarity intensity. The mean-squared error was used as the loss function during training. Note that we initialized embeddings with pre-trained GloVe (Pennington et al., 2014) and fixed them during training on SST-5 for the analysis of both the negation and intensification phenomena.

Furthermore, for the MM model, each token was represented as a matrix and the matrix size was set as 32 32. For the other models, the embedding and hidden sizes were both set as 300.

Polarity Adjectives We automatically extracted adjectives with polarity (examples shown in Table 9) from SST-2 in two steps. In the first step, following the method of Sun and Lu (2020), we calculated a frequency ratio for each token (in the vocabulary) between the frequencies of the token seen in the positive and negative instances respectively. If a token has a frequency ratio either larger than 3 or less than 1/3, it will be extracted as an positive token or an negative token. In the second step, we used the textblob package <sup>18</sup> to detect positive and negative adjectives from those positive tokens and negative tokens respectively.

## E.2 Relation Classification

Following the work of Gupta and Schütze (2018), we examined the RNN, baseline, MVMA and MVM models on SemEval 2010 Task 8 (Hendrickx et al., 2010) which has 9 directed relationships and an undirected other type. We used the final hidden states of the standard RNNs (or context representations of the MVMA, MVM and baseline models) as the instance representations for classification. The cross-entropy loss was employed during training.

## E.3 Named Entity Recognition

At each time step, we concatenated the context representations (or hidden states) from both directions in a bidirectional model, fed them to a projection layer and then to a linear CRF layer. More details about the architecture can be referred to the biLSTM-CRF model in the work of Lample et al. (2016). We also referred to the code at https://github.com/allanj/pytorch\_neural\_crf for the implementation of the linear CRF layer.

CoNLL-2003 contains four types of entities: persons (PER), organizations (ORG), locations (LOC) and miscellaneous names (MISC). The original dataset was labeled with the BIO (Beginning-Inside-Outside) format. For example, “United Arab Emirates” are labeled as “B-LOC I-LOC I-LOC”. We transform the tags into the IOBES format where two prefixes “E-” and “S-” are added. Specifically, “E-” is used to label the last token of an entity span. The “S-” prefix is used for a single-token span. For example, “United Arab Emirates” are labeled as “B-LOC I-LOC E-LOC” in this format. There are 20 categories of tags in total including the starting, ending and padding tags. We trained the models to predict each entity.

The embedding size and hidden size were set to 300 and 200 respectively. The SGD optimizer was used to learn parameters.

## E.4 Language Modeling

The embedding size and hidden size were both 512 for PTB and Wiki2, and 256 and 512 respectively for Wiki103. The cross-entropy loss was used during training. For PTB and Wiki2, the output of the final fully-connected layer was fed to a softmax function while the Adaptive softmax (Joulin et al., 2017) was used for Wiki103 (because of its large vocabulary size). We only considered the word-level models. We trained each model for 50 epochs, chose the model which had the best performance on the development set as the final model and evaluated the final model on the test set.

## F Jacobian matrix of LSTM

Unlike GRU and Elman RNN, LSTM has a memory cell apart from a hidden state. Here, we describe how to get their Jacobian matrices. An

<table><tr><td>Type</td><td>Adjectives</td><td>Size</td></tr><tr><td>Pos</td><td>outstanding, ecological, inventive, comfortable, nice, authentic, spontaneous, sympathetic, lovable, unadulterated, controversial, suitable, grand, happy, enthusiastic, adventurous, successful, noble, true, detailed, sophisticated, sensational, exotic, fantastic, remarkable, impressive, charismatic, good, effective, rich, popular, unforgettable, famous, comical, energetic, ingenious, extraordinary, ...</td><td>73</td></tr><tr><td>Neg</td><td>bad, tedious, miserable, psychotic, didactic, inexplicable, feeble, sloppy, disastrous, stupid, amateurish, false, cynical, farcical, terrible, unhappy, horrible, atrocious, idiotic, wrong, pathetic, angry, uninspired, vicious, unfocused, unnecessary, artificial, troubled, questionable, arduous, stereotypical, ..</td><td>47</td></tr></table>

Table 9: Examples of the extracted adjectives from the SST-2 dataset. “Pos” refers to positive adjectives and “Neg” refers to negative adjectives.

LSTM cell can be written as

$$
\begin{array} { r l r } & { i _ { t } = \sigma ( W _ { i i } x _ { t } + W _ { h i } h _ { t - 1 } ) , } & \\ & { f _ { t } = \sigma ( W _ { i f } x _ { t } + W _ { h f } h _ { t - 1 } ) , } & \\ & { o _ { t } = \sigma ( W _ { i o } x _ { t } + W _ { h o } h _ { t - 1 } ) , } & \\ & { c _ { t } ^ { m } = \operatorname { t a n h } ( W _ { i c } x _ { t } + W _ { h c } h _ { t - 1 } ) , } & \\ & { c _ { t } = f _ { t } \odot c _ { t - 1 } + i _ { t } \odot c _ { t } ^ { m } , h _ { t } = o _ { t } \odot \operatorname { t a n h } ( c _ { t } ) , } & \end{array}
$$

where $i _ { t } , f _ { t } , o _ { t } \in \mathbb { R } ^ { d }$ are the input gate, forget gate and output gate respectively. $\bar { \mathbf { } { c } _ { t } ^ { m } } \in \mathbb { R } ^ { d }$ is the new memory, and $c _ { t }$ is the final memory.

Let us expand the memory state and hidden state at time step t as

$$
\begin{array} { r l } & { { \pmb { c } } _ { t } = { \pmb { g } } _ { c } ( { \pmb { x } } _ { t } ) + B ( { \pmb { x } } _ { t } ) { \pmb { c } } _ { t - 1 } } \\ & { ~ + D ( { \pmb { x } } _ { t } ) { \pmb { h } } _ { t - 1 } + o _ { c } ( { \pmb { c } } _ { t - 1 } , { \pmb { h } } _ { t - 1 } ) , } \\ & { { \pmb { h } } _ { t } = { \pmb { g } } _ { h } ( { \pmb { x } } _ { t } ) + E ( { \pmb { x } } _ { t } ) { \pmb { c } } _ { t - 1 } } \\ & { ~ + F ( { \pmb { x } } _ { t } ) { \pmb { h } } _ { t - 1 } + o _ { h } ( { \pmb { c } } _ { t - 1 } , { \pmb { h } } _ { t - 1 } ) , } \end{array}\tag{17}
$$

where B, D, E and $F \in \mathbb { R } ^ { d \times d }$ are all Jacobian matrices. $\pmb { o } _ { c } ( \pmb { h } _ { t - 1 } )$ and ${ \pmb O } _ { h } ( { \pmb h } _ { t - 1 } )$ are remainder terms of the Taylor expansion.

We concatenate the memory state and hidden state and view the concatenation as an “extended hidden state”. The context representation for the “extended hidden state” at time step t (assuming of zero vectors as initial states) will be written as:

$$
\left[ \hat { \pmb { c } } _ { t } \right] = \sum _ { i = 1 } ^ { t } \left[ \pmb { v } _ { i : t } ^ { c } \right] = \sum _ { i = 1 } ^ { t } \left[ \prod _ { k = t } ^ { i + 1 } A ( \pmb { x } _ { k } ) \right] \left[ \pmb { g } _ { h } ( \pmb { x } _ { i } ) \right] ,\tag{18}
$$

where $\hat { \ b { c } } _ { t }$ and $\hat { h } _ { t }$ refer to the context representations corresponding to the memory state and hidden state respectively. $g _ { c } , g _ { h } \in \mathbb { R } ^ { d }$ , and $A \in \mathbb { R } ^ { 2 d \times 2 d }$ are all functions of inputs. $A ( x _ { k } )$ contains many interaction terms resulting from the gating mechanism, which may result in a strong expressive power. As the hidden state $h _ { t }$ is commonly used for downstream tasks, we will only consider $\pmb { v } _ { i : t } ^ { h }$ as the ngram representation on our tasks, and the context representation will be $\textstyle \sum _ { i = 1 } ^ { t } { \boldsymbol { v } } _ { i : t } ^ { h }$