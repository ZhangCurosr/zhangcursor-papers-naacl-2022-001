# Seed-Guided Topic Discovery with Out-of-Vocabulary Seeds

Yu Zhang<sup>1</sup>, Yu Meng<sup>1</sup>, Xuan Wang<sup>1</sup>, Sheng Wang<sup>2</sup>, Jiawei Han<sup>1</sup>

<sup>1</sup>University of Illinois at Urbana-Champaign, IL, USA

<sup>2</sup>University of Washington, Seattle, WA, USA

{yuz9,yumeng5,xwang174,hanj}@illinois.edu swang@cs.washington.edu

## Abstract

Discovering latent topics from text corpora has been studied for decades. Many existing topic models adopt a fully unsupervised setting, and their discovered topics may not cater to users particular interests due to their inability of leveraging user guidance. Although there exist seed-guided topic discovery approaches that leverage user-provided seeds to discover topicrepresentative terms, they are less concerned with two factors: (1) the existence of out-ofvocabulary seeds and (2) the power of pretrained language models (PLMs). In this paper, we generalize the task of seed-guided topic discovery to allow out-of-vocabulary seeds. We propose a novel framework, named SEE-TOPIC, wherein the general knowledge of PLMs and the local semantics learned from the input corpus can mutually benefit each other. Experiments on three real datasets from different domains demonstrate the effectiveness of SEETOPIC in terms of topic coherence, accuracy, and diversity.<sup>1</sup>

## 1 Introduction

Automatically discovering informative and coherent topics from massive text corpora is central to text analysis through helping users efficiently digest a large collection of documents (Griffiths and Steyvers, 2004) and advancing downstream applications such as summarization (Wang et al., 2009, 2022), classification (Chen et al., 2015; Meng et al., 2020b), and generation (Liu et al., 2021).

Unsupervised topic models have been the mainstream approach to topic discovery since the proposal of pLSA (Hofmann, 1999) and LDA (Blei et al., 2003). Despite their encouraging performance in finding informative latent topics, these topics may not reflect user preferences well, mainly due to their unsupervised nature. For example, given a collection of product reviews, a user may be specifically interested in product categories (e.g., “books”, “electronics”), but unsupervised topic models may generate topics containing different sentiments (e.g., “good”, “bad”). To consider users’ interests and needs, seed-guided topic discovery approaches (Jagarlamudi et al., 2012; Gallagher et al., 2017; Meng et al., 2020a) have been proposed to find representative terms for each category based on user-provided seeds or category names.<sup>2</sup> However, there are still two less concerned factors in these approaches.

Table 1: Three datasets (Cohan et al., 2020; McAuley and Leskovec, 2013; Zhang et al., 2017) from different domains and their topic categories (i.e., seeds). Red: Seeds never seen in the corpus (i.e., out-of-vocabulary). In all three datasets, a large proportion of seeds are outof-vocabulary.
<table><tr><td>Dataset</td><td colspan="2">Category Names (Seeds)</td></tr><tr><td>SciDocs (Scientific Papers)</td><td>cardiovascular diseases chronic kidney disease chronic respiratory diseases diabetes mellitus digestive diseases hiv/aids</td><td>hepatitis a/b/c/e mental disorders musculoskeletal disorders neoplasms (cancer) neurological disorders</td></tr><tr><td>Amazon (Product Reviews)</td><td>apps for android books cds and vinyl clothing, shoes and jewelry electronics</td><td>health and personal care home and kitchen movies and tv sports and outdoors video games</td></tr><tr><td>Twitter (Social Media Posts)</td><td>food shop and service travel and transport college and university nightlife spot</td><td>residence outdoors and recreation arts and entertainment professional and other places</td></tr></table>

The Existence of Out-of-Vocabulary Seeds. Previous studies (Jagarlamudi et al., 2012; Gallagher et al., 2017; Meng et al., 2020a) assume that all user-provided seeds must be in-vocabulary (i.e., appear at least once in the input corpus), so that they can utilize the occurrence statistics or Skip-Gram embedding methods (Mikolov et al., 2013) to model seed semantics. However, user-interested categories can have specific or composite descriptions, which may never appear in the corpus. Table 1 shows three datasets from different domains: scientific papers, product reviews, and social media posts. In each dataset, documents can belong to one or more categories, and we list the category names provided by the dataset collectors. These seeds should reflect their particular interests. In all three datasets, we have a large proportion of seeds (45% in SciDocs, 60% in Amazon, and 78% in Twitter) that never appear in the corpus. Some category names are too specific (e.g., “chronic respiratory diseases”, “nightlife spot”) to be exactly matched, others are the composition of multiple entities (e.g., “hepatitis a/b/c/e”, “neoplasms (cancer)”, “clothing, shoes andjewelry”).<sup>3</sup>

The Power of Pre-trained Language Models. Techniques used in previous studies are mainly based on LDA variants (Jagarlamudi et al., 2012) or context-free embeddings (Meng et al., 2020a). Recently, pre-trained language models (PLMs) such as BERT (Devlin et al., 2019) have achieved significant improvement in a wide range of text mining tasks. In topic discovery, the generic representation power of PLMs learned from web-scale corpora (e.g., Wikipedia or PubMed) may complement the information a model can obtain from the input corpus. Moreover, out-of-vocabulary seeds usually have meaningful in-vocabulary components (e.g., “night” and “life” in “nightlife spot”, “health” and “care” in “health and personal care”). The optimized tokenization strategy of PLMs (Sennrich et al., 2016; Wu et al., 2016) can help segment the seeds into such meaningful components (e.g., “nightlife” → “night” and “##life”), and the contextualization power of PLMs can help infer the correct meaning of each component (e.g., “##life” and “care”) in the category name. Therefore, PLMs are much needed in handling out-of-vocabulary seeds and effectively learning their semantics.

Contributions. Being aware of these two factors, in this paper, we study seed-guided topic discovery in the presence of out-of-vocabulary seeds. Our proposed SEETOPIC framework consists of two modules: (1) The general representation module uses a PLM to derive the representation of each term (including out-of-vocabulary seeds) based on the general linguistic knowledge acquired through pre-training. (2) The seed-guided local representation module learns in-vocabulary term embeddings specific to the input corpus and the given seeds. In order to optimize the learned representations for topic coherence, which is commonly reflected by pointwise mutual information (PMI) (Newman et al., 2010), our objective implicitly maximizes the PMI between each word and its context, the documents it appears, as well as the category it belongs to. The learning of the two modules is connected through an iterative ensemble ranking process, in which the general knowledge of PLMs and the term representations specifically learned from the target corpus conditioned on the seeds can complement each other.

To summarize, this study makes three contributions. (1) Task: we propose to study seedguided topic discovery in the presence of out-ofvocabulary seeds. (2) Framework: we design a unified framework that jointly models general knowledge through PLMs and local corpus statistics through embedding learning. (3) Experiment: extensive experiments on three datasets demonstrate the effectiveness of SEETOPIC in terms of topic coherence, accuracy, and diversity.

## 2 Problem Definition

As shown in Table 1, we assume a seed can be either a single word or a phrase. Given a corpus ${ \mathcal { D } } ,$ we use $\nu _ { D }$ to denote the set of terms appearing in D. In accordance with the assumption of category names, each term can also be a single word or a phrase. In practice, given a raw corpus, one can use existing phrase chunking tools (Manning et al., 2014; Shang et al., 2018) to detect phrases in it. After phrase chunking, if a category name is still not in $\gamma _ { D }$ , we define it as out-of-vocabulary.

Problem Definition. Given a corpus $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ d _ { 1 } , . . . , d _ { | \mathcal { D } | } \}$ and a set of category names ${ \mathcal { C } } =$ $\{ c _ { 1 } , . . . , c _ { | C | } \}$ where some category names are outof-vocabulary, the task is to find a set of invocabulary terms $S _ { i } = \{ w _ { 1 } , . . . , w _ { S } \} \subseteq { \mathcal { V } } _ { D }$ for each category $c _ { i }$ such that each term in $S _ { i }$ is semantically close to $c _ { i }$ andfarfrom other categories $c _ { j } ~ ( \forall j \neq i )$

## 3 The SEETOPIC Framework

In this section, we first introduce how we model general and local text semantics using a PLM mod-

ule and a seed-guided embedding learning module, respectively. Then, we present the iterative ensemble ranking process and our overall framework.

## 3.1 Modeling General Text Semantics using a PLM

PLMs such as BERT (Devlin et al., 2019) aim to learn generic language representations from webscale corpora (e.g., Wikipedia or PubMed) that can be applied to a wide variety of text-related applications. To transfer such general knowledge to our topic discovery task, we employ a PLM to encode each category name and each in-vocabulary term to a vector. To be specific, given a term w ∈ ${ \mathcal { C } } \cup { \mathcal { V } } _ { D }$ , we input the sequence “[CLS] w [SEP]” into the PLM. Here, w can be a phrase containing multiple words, and each word can be out of the PLM’s vocabulary. To deal with this, most PLMs use a pre-trained tokenizer (Sennrich et al., 2016; Wu et al., 2016) to segment each unseen word into frequent subwords. Then, the contextualization power of PLMs will help infer the correct meaning of each word/subword, so as to provide a more precise representation of the whole category.

After LM encoding, following (Sia et al., 2020; Thompson and Mimno, 2020; Li et al., 2020), we take the output of all tokens from the last layer and average them to get the term embedding $e _ { w }$ . In this way, even if a seed $c _ { i }$ is out-of-vocabulary, we can still obtain its representation $e _ { c _ { i } }$

## 3.2 Modeling Local Text Semantics in the Input Corpus

The motivation of topic discovery is to discover latent topic structures from the input corpus. Therefore, purely relying on general knowledge in the PLM is insufficient because topic discovery results should adapt to the input corpus D. Now, we introduce how we learn another set of embeddings $\{ \boldsymbol { u } _ { w } | w \in \mathcal { V } _ { D } \}$ from D.

Previous studies on embedding learning assume that the semantic of a term is similar to its local context (Mikolov et al., 2013), the document it appears (Tang et al., 2015; Xun et al., 2017a), and the category it belongs to (Meng et al., 2020a). Inspired by these studies, we propose the following embedding learning objective.

$$
\begin{array} { r l } & { \mathcal { I } = \displaystyle \sum _ { \underbrace { d \in \mathcal { D } } } \displaystyle \sum _ { w _ { i } \in d } \sum _ { w _ { j } \in \mathcal { C } ( w _ { i } , h ) } p ( w _ { j } | w _ { i } ) } \\ & { \quad \quad + \underbrace { \displaystyle \sum _ { d \in \mathcal { D } } \displaystyle \sum _ { w \in d } p ( d | w ) } _ { \mathrm { d o c u m e n t } } + \underbrace { \sum _ { i \in \mathcal { C } } \displaystyle \sum _ { w \in \mathcal { S } _ { i } } p ( c _ { i } | w ) } _ { \mathrm { c a t e g o r y } } , } \end{array}\tag{1}
$$

where

$$
p ( z | w ) = \frac { \exp ( { \pmb u } _ { w } ^ { T } { \pmb v } _ { z } ) } { \sum _ { z ^ { \prime } } \exp ( { \pmb u } _ { w } ^ { T } { \pmb v } _ { z ^ { \prime } } ) } , ( z \mathrm { c a n b e } w _ { j } , d , \mathrm { o r } c _ { i } ) .\tag{2}
$$

In this objective, $\mathbf { \Delta } \mathbf { u } _ { w _ { i } }$ (and ${ \pmb v } _ { w _ { j } } ) , { \pmb v } _ { d } , { \pmb v } _ { c _ { i } }$ are the embedding vectors of terms, documents, and categories, respectively. $\mathcal { C } ( w _ { i } , h )$ is the set of context terms of $w _ { i }$ in d. Specifically, if $d = w _ { 1 } w _ { 2 } . . . w _ { L }$ then $\mathcal { C } ( w _ { i } , h ) = \{ w _ { j } | i - h \le j \le i + h , j \neq i \}$ where h is the context window size.

Note that the last term in Eq. (1) encourages the similarity between each category $c _ { i }$ and its representative terms $s _ { i }$ . Here, we adopt an iterative process to gradually update category-representative terms. Initially, $s _ { i }$ consists of just a few invocabulary terms similar to $c _ { i }$ according to the PLM. At each iteration, the size of $s _ { i }$ will increase to contain more category-discriminative terms (the selection criterion of these terms will be introduced in the next section), and we need to encourage their proximity with $c _ { i }$ in the next iteration.

Directly optimizing the full softmax in Eq. (2) is costly. Therefore, we adopt the negative sampling strategy (Mikolov et al., 2013) for efficient approximation.

Interpreting the Objective. In topic modeling studies, pointwise mutual information (PMI) (Newman et al., 2010) is a standard evaluation metric for topic coherence (Lau et al., 2014; Röder et al., 2015). Levy and Goldberg (2014) prove that the Skip-Gram embedding model is implicitly factorizing the PMI matrix. Following their proof, we can show that maximizing Eq. (1) is implicitly doing the following factorization:

$$
\mathbf { U } _ { w } ^ { T } [ \mathbf { V } _ { w } ; \mathbf { V } _ { d } ; \mathbf { V } _ { c } ] = [ \mathbf { X } _ { w w } ; \mathbf { X } _ { w d } ; \mathbf { X } _ { w c } ] ,\tag{3}
$$

where the columns of $\mathbf { U } _ { w } , \mathbf { V } _ { w } , \mathbf { V } _ { d } , \mathbf { V } _ { c }$ are $\mathbf { \Delta } \mathbf { u } _ { w _ { i } }$ ${ \pmb v } _ { w _ { j } } , { \pmb v } _ { d } , { \pmb v } _ { c _ { i } }$ , respectively $( w _ { i } , w _ { j } \in \mathcal { V } _ { D } , d \in \mathcal { D }$ $c _ { i } \in \mathcal { C } ) ; \mathbf { X } _ { w w } , \mathbf { X } _ { w d } .$ , and $\mathbf { X } _ { w c }$ are PMI matrices.

$$
\begin{array} { r l } & { \mathbf { X } _ { w w } = [ \log \Big ( \frac { \# p ( w _ { i } , w _ { j } ) \cdot \lambda _ { \mathcal { D } } } { \# p ( w _ { i } ) \cdot \# p ( w _ { j } ) \cdot b } \Big ) ] _ { w _ { i } , w _ { j } \in \mathcal { V } _ { \mathcal { D } } } , } \\ & { \mathbf { X } _ { w d } = [ \log \Big ( \frac { \# d ( w ) \cdot \lambda _ { \mathcal { D } } } { \# p ( w ) \cdot \lambda _ { d } \cdot b } \Big ) ] _ { w \in \mathcal { V } _ { \mathcal { D } } , \ d \in \mathcal { D } } , } \\ & { \mathbf { X } _ { w c } = [ x _ { w , c _ { i } } ] _ { w \in \mathcal { V } _ { \mathcal { D } } , \ c _ { i } \in \mathcal { C } } , \quad \mathrm { w h e r e } } \\ & { \qquad x _ { w , c _ { i } } = \{ \log \frac { \log | \mathbf { \Psi } _ { b } | } { b } , \quad \mathrm { i f } \ w \in \mathcal { S } _ { i } , \ } \\ & { \qquad \quad - \infty , \quad \mathrm { i f } \ w \in \mathcal { S } _ { j } \ ( \forall j \neq i ) . } \end{array}\tag{4}
$$

Here, $\# _ { \mathcal { D } } ( w _ { i } , w _ { j } )$ denotes the number of cooccurrences of $w _ { i }$ and $w _ { j }$ in a context window in $\mathcal { D } ; \# _ { \mathcal { D } } ( w )$ denotes the number of occurrences of w in $\mathcal { D } ; \lambda _ { \mathcal { D } }$ is the total number of terms in $\mathcal { D } ; \# d ( w )$ denotes the number of times w occurs in d; $\lambda _ { d }$ is the total number of terms in d; b is the number of negative samples. (For the derivation of Eq. (3), please refer to Appendix A.)

To summarize, the learned local representations $\mathbf { \Delta } \mathbf { u } _ { w }$ are implicitly optimized for topic coherence, where term co-occurrences are measured in context, document, and category levels.

## 3.3 Ensemble Ranking

We have obtained two sets of term embeddings that model text semantics from different angles: $\{ e _ { w } | w \in \mathcal { C } \cup \mathcal { V } _ { D } \}$ carries the PLM’s knowledge, while $\{ \boldsymbol { u } _ { w } | \boldsymbol { w } \in \mathcal { V } _ { D } \}$ models the input corpus as well as user-provided seeds. We now propose an ensemble ranking method to leverage information from both sides to grab more discriminative terms for each category.

Given a category $c _ { i }$ and its current term set $s _ { i }$ we first calculate the scores of each term $w \in \mathcal { V } _ { D }$

$$
\begin{array} { l } { { \mathrm { s c o r e } _ { G } ( w | S _ { i } ) = \displaystyle \frac { 1 } { | S _ { i } | } \sum _ { w ^ { \prime } \in S _ { i } } \cos ( e _ { w } , e _ { w ^ { \prime } } ) , } } \\ { { \mathrm { s c o r e } _ { L } ( w | S _ { i } ) = \displaystyle \frac { 1 } { | S _ { i } | } \sum _ { w ^ { \prime } \in S _ { i } } \cos ( { \pmb u } _ { w } , { \pmb u } _ { w ^ { \prime } } ) . } } \end{array}\tag{5}
$$

The subscript $^ { 6 6 } G ^ { , }$ here means “general”, while $" L "$ means “local”. Then, we sort all terms by these two scores, respectively. Each term w will hence get two rank positions ran $\ k \omega _ { } ^ { }$ and $\mathrm { r a n k } _ { L } ( w )$ We propose the following ensemble score based on the reciprocal rank:

$$
\mathrm { s c o r e } ( w | S _ { i } ) = \bigg ( \frac { 1 } { 2 } \Big ( \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } \Big ) ^ { \rho } + \frac { 1 } { 2 } \Big ( \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } \Big ) ^ { \rho } \bigg ) ^ { 1 / \rho } .\tag{6}
$$

Here, $0 < \rho \le 1$ is a constant. In practice, instead of ranking all terms in the vocabulary, we only check the top-M results in the two ranking lists. If a term w is not among the top-M according to score<sub>G</sub>(w) (resp., score<sub>L</sub>(w)), we set $\operatorname { r a n k } _ { G } ( w ) = + \infty$ (resp., ran $\mathrm { k } _ { L } ( w ) = + \infty )$ . In fact, when $\rho ~ = ~ 1$ , Eq. (6) becomes the arithmetic mean of the two reciprocal ranks $\frac { 1 } { \mathrm { r a n k } _ { G } ( w ) }$ and $\frac { 1 } { \operatorname { r a n k } _ { L } ( w ) }$ . This is essentially the mean reciprocal rank (MRR) commonly used in ensemble ranking, where a high position in one ranking list can largely compensate a low position in the other. In contrast, when $\rho  0 ,$ , Eq. (6) becomes the geometric mean of the two reciprocal ranks (see Appendix B), where two ranking lists both have the “veto power” (i.e., a term needs to be ranked as top-M in both ranking lists to obtain a non-zero ensemble score). In experiment, we set $\rho = 0 . 1$ and show it outperforms MRR $( \mathrm { i . e . , } \rho = 1 )$ in our topic discovery task.

Algorithm 1: SEETOPIC   
Input: A text corpus $\mathcal { D } = \{ d _ { 1 } , . . . , d _ { | \mathcal { D } | } \} ,$ a set of   
seeds ${ \mathcal { C } } = { \bar { \{ } c _ { 1 } , . . . , c _ { | { \mathcal { C } } | } \} } ,$ and a PLM.   
Output: $( S _ { 1 } , . . . , \tilde { S _ { | C | } } ) \dot { }$ , where each $\boldsymbol { S } _ { i }$ is a set of   
category-discriminative terms for c<sub>i</sub>.   
1 Compute $\{ e _ { w } | w \in \mathcal { C } \cup \mathcal { V } _ { D } \}$ using the PLM;   
2 // Initialize S<sub>i</sub>;   
3 $\begin{array} { r } { S _ { 1 } , . . . , S _ { | { \mathcal C } | }  \emptyset ; } \end{array}$   
4 for n ← 1 to N do   
5 for $i \gets 1 \ t o \ | \mathcal { C } |$ do   
6 $w _ { n } \gets$ arg max cos $( e _ { w } , e _ { c _ { i } } ) ;$   
w∈V<sub>D</sub>\(S<sub>1</sub>∪...∪S<sub>|C|</sub>)   
7 $S _ { i } \gets S _ { i } \cup \{ w _ { n } \} ;$   
8 // Update $S _ { i }$ for T iterations;   
9 for $\bar { t }  1$ to T do   
10 Learn $\{ \boldsymbol { u } _ { w } | \boldsymbol { w } \in \mathcal { V } _ { D } \}$ from the input corpus D   
and the up-to-date representative terms   
$\boldsymbol { S _ { 1 } } , . . . , \boldsymbol { \bar { S _ { | c | } } }$ according to $\operatorname { E q . } \left( 1 \right) ;$   
11 score<sub>G</sub> $\mathbf { \nabla } \cdot \left( w \vert \dot { S } _ { i } \right)$ and score<sub>L</sub> $. ( \hat { w } | S _ { i } ) \gets \mathrm { E q . } ( 5 ) ;$   
12 score $( \dot { w } | \dot { S } _ { i } ) \gets \mathrm { E q . ~ } ( 6 ) ;$   
13 $\begin{array} { r } { S _ { 1 } , . . . , { \dot { S _ { | \mathcal { C } | } } }  \emptyset ; } \end{array}$   
14 for $n \gets \bar { 1 } ^ { \prime } t o \left( t + 1 \right) N$ do   
15 for $i \gets 1$ to |C| do   
16 $S _ { i } \gets \mathrm { E q . ~ } ( 7 ) ;$   
17 Return $( S _ { 1 } , . . . , S _ { | c | } ) ;$

After computing the ensemble score score $( w | S _ { i } )$ for each $w ,$ we update $s _ { i }$ . To guarantee that each $s _ { i }$ is category-discriminative, we do not allow any term to belong to more than one category. Therefore, we gradually expand each $s _ { i }$ by turns. At the beginning, we reset $\begin{array} { r } { S _ { 1 } = \dots = S _ { | { \mathcal { C } } | } = \emptyset } \end{array}$ . When it is $s _ { i } \mathrm { ' s }$ turn, we add one term $s _ { i }$ according to the following criterion:

$$
\mathcal { S } _ { i }  \mathcal { S } _ { i } \cup \{ \underset { w \in \mathcal { V } _ { \mathcal { D } } \backslash ( \mathcal { S } _ { 1 } \cup \ldots \cup \mathcal { S } _ { | \mathcal { C } | } ) } { \arg \operatorname* { m a x } } \mathrm { s c o r e } ( w | \mathcal { S } _ { i } ) \} .\tag{7}
$$

## 3.4 Overall Framework

We summarize the entire SEETOPIC framework in Algorithm 1. To deal with out-of-vocabulary category names, we first utilize a PLM to find their nearest in-vocabulary terms as the initial categorydiscriminative term set $s _ { i }$ (Lines 1-7). After initialization, $| S _ { i } | = N \left( \forall 1 \leq i \leq | \mathcal { C } | \right)$ . Note that for an in-vocabulary category name $c _ { i } \in \mathcal { V } _ { D }$ , itself will be added to the initial $S _ { i }$ as the top-1 similar in-vocabulary term.

After getting the initial $s _ { i }$ , we update it by T iterations (Lines 8-16). At each iteration, according to the up-to-date $\boldsymbol { S } _ { 1 } , \boldsymbol { S } _ { 2 } , . . . , \boldsymbol { S } _ { | \mathcal { C } | }$ , we relearn embeddings $\boldsymbol { u } _ { w } , \boldsymbol { v } _ { w } , \boldsymbol { v } _ { d }$ , and ${ \pmb v } _ { c _ { i } }$ using Eq. (1) (Line 10). The two set of embeddings, $\{ e _ { w } | w \in \mathcal { C } \cup \mathcal { V } _ { D } \}$ (computed at Line 1) and $\{ { \pmb u } _ { w } | w ~ \in ~ \mathcal { V } _ { D } \}$ (updated at Line 10), are then leveraged to perform ensemble ranking (Lines 11-12). Based on the ensemble score score $( w | S _ { i } )$ , we update $S _ { i }$ using Eq. (7) (Lines 13-16). After the t-th iteration, $| S _ { i } | = ( t + 1 ) N \left( \forall 1 \leq i \leq | \mathcal { C } | \right)$

Complexity Analysis. The time complexity of using the PLM is $\mathcal { O } ( ( | \mathcal { C } | + | \mathcal { V } _ { D } | ) \alpha _ { \mathrm { P L M } } )$ , where α<sub>PLM</sub> is the complexity of encoding one term via the PLM. The total complexity of local embedding is $\mathcal { O } ( T \lambda _ { \mathcal { D } } ( h + \vert \mathcal { C } \vert ) b )$ because in each iteration $1 \leq t \leq T$ , every $w \in \mathcal { D }$ interacts with every other term in the context window of size h, its belonging document, and each category $c _ { i } \in \mathcal { C }$ , and each update involves b negative samples. The total complexity of ensemble ranking is $\mathcal { O } ( T | \mathcal { V } _ { D } | | \mathcal { C } | | \mathcal { S } _ { i } | )$ as in each iteration $1 \leq t \leq T$ , we compute scores between each $w \in \mathcal { V } _ { D }$ and each $w ^ { \prime } \in S _ { i } \ ( \forall c _ { i } \in \mathcal { C } )$

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We conduct experiments on three public datasets from different domains: (1) SciDocs (Cohan et al., $2 0 2 0 ) ^ { 4 }$ is a large collection of scientific papers supporting diverse evaluation tasks. For the MeSH classification task (Coletti and Bleich, 2001), about 23K medical papers are collected, each of which is assigned to one of the 11 common disease categories derived from the MeSH vocabulary. We use the title and abstract of each paper as documents and the 11 category names as seeds. (2) Amazon (McAuley and Leskovec, $2 0 1 3 ) ^ { 5 }$ contains product reviews from May 1996 to July 2014. Each Amazon review belongs to one or more product categories. We use the subset sampled by Zhang et al. (2020, 2022), which contains 10 categories and 100K reviews. (3) Twitter (Zhang et al., 2017)<sup>6</sup> is a crawl of geo-tagged tweets in New York City from August 2014 to November 2014. The dataset collectors link these tweets with Foursquare’s POI database and assign them to 9 POI categories. We take these category names as input seeds.

Seeds used in the three datasets are shown in Table 1. Dataset statistics are summarized in Table 2. For all three datasets, we use AutoPhrase (Shang et al., $2 0 1 8 ) ^ { 7 }$ to perform phrase chunking in the corpus, and we remove words and phrases occurring less than 3 times.

Previous studies (Jagarlamudi et al., 2012; Meng et al., 2020a) have tried some other datasets (e.g., RCV1, 20 Newsgroups, NYT, and Yelp). However, the category names they use in these datasets are all picked from in-vocabulary terms. Therefore, we do not consider these datasets for evaluation in our task settings.

Table 2: Dataset Statistics.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>SciDocs</td><td rowspan=1 colspan=1>Amazon</td><td rowspan=1 colspan=1>Twitter</td></tr><tr><td rowspan=1 colspan=1>#Documents</td><td rowspan=1 colspan=1>23,473</td><td rowspan=1 colspan=1>100,000</td><td rowspan=1 colspan=1>135,529</td></tr><tr><td rowspan=1 colspan=1>#In-vocabulary Terms(After Phrase Chunking)</td><td rowspan=1 colspan=1>55,897</td><td rowspan=1 colspan=1>56,942</td><td rowspan=1 colspan=1>17,577</td></tr><tr><td rowspan=1 colspan=1>Avg Doc Length</td><td rowspan=1 colspan=1>239.8</td><td rowspan=1 colspan=1>119.0</td><td rowspan=1 colspan=1>6.7</td></tr><tr><td rowspan=1 colspan=1>#Seeds</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>#Out-of-vocabulary Seeds(After Phrase Chunking)</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td></tr></table>

Following (Sia et al., 2020), we adopt a 60-40 train-test split for all three datasets. The training set is used as the input corpus D, and the testing set is used for calculating topic coherence metrics (see evaluation metrics for details).

Compared Methods. We compare our SEETOPIC framework with the following methods, including seed-guided topic modeling methods, seedguided embedding learning methods, and PLMs. (1) SeededLDA (Jagarlamudi et al., $2 0 1 2 ) ^ { 8 }$ is a seed-guided topic modeling method. It improves LDA by biasing topics to produce input seeds and by biasing documents to select topics relevant to the seeds they contain. (2) Anchored CorEx (Gallagher et al., $2 0 1 7 ) ^ { 9 }$ is a seed-guided topic modeling method. It incorporates user-provided seeds by balancing between compressing the input corpus and preserving seed-related information. (3) Labeled ETM (Dieng et al., 2020)<sup>10</sup> is an embedding-based topic modeling method. It incorporates distributed representation of each term. Following (Meng et al., 2020a), we retrieve representative terms according to their embedding similarity with the category name. (4) CatE (Meng et al., 2020a)<sup>11</sup> is a seed-guided embedding learning method for discriminative topic discovery. It takes category names as input and jointly learns term embedding and specificity from the input corpus. Category-discriminative terms are then selected based on both embedding similarity with the category and specificity. (5) BERT (Devlin et al., 2019)<sup>12</sup> is a PLM. Following Lines 1-7 in Algorithm 1, we use BERT to encode each input category name and each term to a vector, and then perform similarity search to directly find all representative terms. (6) BioBERT (Lee et al., 2020)<sup>13</sup> is a PLM. It is used in the same way as BERT. Since BioBERT is specifically trained for biomedical text mining tasks, we report its performance on the SciDocs dataset only. (7) SEETOPIC-NoIter is a variant of our SEETOPIC framework. In Algorithm 1, after initialization (Lines 1-7), it executes Lines 9-16 only once $( \mathrm { i } . \mathrm { e } . , T = 1 )$ to find all representative terms.

Table 3: NPMI, LCP, MACC, and Diversity of compared algorithms on three datasets. NPMI and LCP measure topic coherence; MACC measures term accuracy; Diversity (abbreviated to Div.) measures topic diversity. Bold: the highest score. Underline: the second highest score. <sup>∗</sup>: significantly worse than SEETOPIC (p-value < 0.05). <sup>∗∗</sup>: significantly worse than SEETOPIC (p-value < 0.01).
<table><tr><td rowspan="2">Methods</td><td colspan="4">SciDocs</td><td colspan="4">Amazon</td><td colspan="4">Twitter</td></tr><tr><td>NPMI</td><td>LCP</td><td>MACC</td><td>Div.</td><td>NPMI</td><td>LCP</td><td>MACC</td><td>Div.</td><td>NPMI</td><td>LCP</td><td>MACC</td><td>Div.</td></tr><tr><td>SeededLDA</td><td>0.111**</td><td>-1.232</td><td>0.156**</td><td>0.451**</td><td>0.140**</td><td>-1.505</td><td>0.147**</td><td>0.393**</td><td>0.026**</td><td>-4.508**</td><td>0.195**</td><td>0.696**</td></tr><tr><td>Anchored CorEx</td><td>0.213**</td><td> $- 2 . 1 8 0 ^ { * * }$ </td><td> $0 . 2 6 4 ^ { * * }$ </td><td>1.000</td><td> $0 . 2 6 8 ^ { * * }$ </td><td>-1.963*</td><td> $0 . 3 3 3 ^ { \ast \ast }$ </td><td>1.000</td><td> $0 . 1 8 0 ^ { * * }$ </td><td> $- 4 . 3 8 4 ^ { * * }$ </td><td>0.233**</td><td>1.000</td></tr><tr><td>Labeled ETM</td><td>0.669*</td><td> $- 1 . 5 4 9 ^ { * * }$ </td><td> $0 . 4 5 8 ^ { * * }$ </td><td>0.961*</td><td> $0 . 6 1 6 ^ { * * }$ </td><td> $- 2 . 1 0 3 ^ { * * }$ </td><td> $0 . 5 8 5 ^ { * * }$ </td><td>1.000</td><td>0.610*</td><td> $- 2 . 1 9 7 ^ { * * }$ </td><td>0.268**</td><td>0.989</td></tr><tr><td>CatE</td><td>0.691*</td><td> $- 1 . 4 5 1 ^ { * * }$ </td><td> $0 . 6 3 3 ^ { * * }$ </td><td>1.000</td><td> $0 . 6 3 3 ^ { * * }$ </td><td> $- 1 . 6 8 8 ^ { * * }$ </td><td>0.856*</td><td>1.000</td><td>0.711</td><td>-1.653</td><td>0.483**</td><td>1.000</td></tr><tr><td>BERT</td><td>0.626**</td><td>-1.682**</td><td>0.740**</td><td>0.891**</td><td>0.588**</td><td>-2.186**</td><td>0.832**</td><td>1.000</td><td>0.626**</td><td>-2.088**</td><td>0.627</td><td>0.944**</td></tr><tr><td>BioBERT</td><td>0.618**</td><td>-1.704**</td><td>0.938</td><td>0.982**</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td></tr><tr><td>SEETOPIC-NoIter</td><td>0.681**</td><td>-1.535**</td><td>0.887</td><td>1.000</td><td>0.643**</td><td>-1.971**</td><td>0.892</td><td>1.000</td><td>0.636</td><td>-2.008**</td><td>0.618</td><td>1.000</td></tr><tr><td>SEETOPIC</td><td>0.717</td><td>-1.267</td><td>0.909</td><td>1.000</td><td>0.684</td><td>-1.391</td><td>0.904</td><td>1.000</td><td>0.640</td><td>-1.813</td><td>0.633</td><td>1.000</td></tr></table>

Here, all seed-guided topic modeling and embedding baselines (i.e., SeededLDA, Anchored CorEx, CatE, and Labeled ETM) can only take in-vocabulary seeds as input. For a fair comparison, we run Lines 1-7 in Algorithm 1 to get the initial representative in-vocabulary terms for each category, and input these terms as seeds into the baselines. In other words, all compared methods use BERT/BioBERT to initialize their term sets.

Evaluation Metrics. We evaluate topic discovery results from three different angles: topic coherence, term accuracy, and topic diversity.

(1) NPMI (Lau et al., 2014) is a standard metric in topic modeling to measure topic coherence. Within each topic, it calculates the normalized pointwise mutual information for each pair of terms in $s _ { i }$

$$
\mathrm { N P M I } = \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \frac { 1 } { \binom { | S _ { i } | } { 2 } } \sum _ { \substack { w _ { j } , w _ { k } \in \mathcal { S } _ { i } } } \frac { \log \frac { P ( w _ { j } , w _ { k } ) } { P ( w _ { j } ) P ( w _ { k } ) } } { - \log P ( w _ { j } , w _ { k } ) } ,\tag{8}
$$

where $P ( w _ { j } , w _ { k } )$ is the probability that $w _ { j }$ and w<sub>k</sub> co-occur in a document; $P ( w _ { j } )$ is the marginal probability of w<sub>j</sub>.<sup>14</sup> $w _ { j }$

(2) LCP (Mimno et al., 2011) is another standard metric to measure topic coherence. It calculates the pairwise log conditional probability of top-ranked

terms.

$$
\mathrm { L C P } = \frac { 1 } { | \mathcal { C } | } \sum _ { i = 1 } ^ { | \mathcal { C } | } \frac { 1 } { \binom { | \mathcal { S } _ { i } | } { 2 } } \sum _ { \substack { w _ { j } , w _ { k } \in S _ { i } } } \log \frac { P ( w _ { j } , w _ { k } ) } { P ( w _ { j } ) } .\tag{9}
$$

Note that PMI (Newman et al., 2010) is also a standard metric for topic coherence. We do observe that SEETOPIC outperforms baselines in terms of PMI in most cases. However, since our local embedding step is implicitly optimizing a PMI-like objective, we no longer use it as our evaluation metric.

(3) MACC (Meng et al., 2020a) measures term accuracy. It is defined as the proportion of retrieved terms that actually belong to the corresponding category according to the category name.

$$
\mathrm { M A C C } = \frac { 1 } { | \mathcal C | } \sum _ { i = 1 } ^ { | \mathcal C | } \frac { 1 } { | S _ { i } | } \sum _ { w _ { j } \in S _ { i } } \mathbf { 1 } ( w _ { j } \in c _ { i } ) ,\tag{10}
$$

where $\mathbf { 1 } ( w _ { j } ~ \in ~ c _ { i } )$ is the indicator function of whether $w _ { j }$ is relevant to category $c _ { i }$ . MACC requires human evaluation, so we invite five annotators to perform independent annotation. The reported MACC score is the average MACC of the five annotators. A high inter-annotator agreement is observed, with Fleiss’ kappa (Fleiss, 1971) being 0.856, 0.844, and 0.771 on SciDocs, Amazon, and Twitter, respectively.

(4) Diversity (Dieng et al., 2020) measures the mutual exclusivity of discovered topics. It is the percentage of unique terms in all topics, which corresponds to our task requirement that each retrieved term is discriminatively close to one category and far from the others.

$$
{ \mathrm { D i v e r s i t y } } = { \frac { | \bigcup _ { i = 1 } ^ { | { \mathcal { C } } | } S _ { i } | } { \sum _ { i = 1 } ^ { | { \mathcal { C } } | } | S _ { i } | } } .\tag{11}
$$

Experiment Settings. We use BioBERT as the

PLM on SciDocs, and BERT-base-uncased as the PLM on Amazon and Twitter. The embedding dimension of $\mathbf { \Delta } \mathbf { u } _ { w }$ is 768 (the same as $e _ { w } )$ ; the number of negative samples $b \ = \ 5$ . In ensemble ranking, the length of the general/local ranking list $M = 1 0 0$ ; the hyperparameter $\rho$ in Eq. (6) is set as 0.1; the number of iterations $T = 4 ;$ after each iteration, we increase the size of $s _ { i }$ by $N = 3$ We use the top-10 ranked terms in each topic for final evaluation (i.e., $| S _ { i } | = 1 0$ in Eqs. (8)-(11)). Experiments are run on Intel Xeon E5-2680 v2 @ 2.80GHz and one NVIDIA GeForce GTX 1080.

## 4.2 Performance Comparison

Table 3 shows the performance of all methods. We run each experiment 3 times with the average score reported. To show statistical significance, we conduct a two-tailed unpaired t-test to compare SEE-TOPIC and each baseline. (The performance of BERT and BioBERT is deterministic according to our usage. When comparing SEETOPIC with them, we conduct a two-tailed Z-test instead.) The significance level is also marked in Table 3.

We have the following observations from Table 3. (1) Our SEETOPIC model performs consistently well. In fact, it achieves the highest score in 8 columns and the second highest in the remaining 4 columns. (2) Classical seed-guided topic modeling baselines (i.e., SeededLDA and Anchored CorEx) perform not well in respect of NPMI (topic coherence) and MACC (term accuracy). Embeddingbased topic discovery approaches (i.e., Labeled ETM and CatE) make some progress, but they still significantly underperform the PLM-empowered SEETOPIC model on SciDocs and Amazon. (3) SEETOPIC consistently performs better than SEE-TOPIC-NoIter on all three datasets, indicating the positive contribution of the proposed iterative process. (4) SEETOPIC guarantees the mutual exclusivity of $\boldsymbol { S } _ { 1 } , . . . , \boldsymbol { S } _ { | \mathcal { C } | }$ . In comparison, SeededLDA, Labeled ETM, and BERT cannot guarantee such mutual exclusivity.

In-vocabulary vs. Out-of-vocabulary. Figure 1 compares the MACC scores of different seedguided topic discovery methods on in-vocabulary categories and out-of-vocabulary categories. We find that the performance improvement of SEE-TOPIC upon baselines on out-of-vocabulary categories is larger than that on in-vocabulary ones. For example, on Amazon, SEETOPIC underperforms CatE on in-vocabulary categories but outperforms CatE on out-of-vocabulary ones; on Twitter, the gap between SEETOPIC and baselines becomes much more evident on out-of-vocabulary categories. Note that all baselines in Figure 1 do not utilize the power of PLMs, so this observation validates our claim that PLMs are helpful in tackling out-of-vocabulary seeds.

![](images/92e86aed7b38d5ed712e1afa42c79883447edfb1f3bb5deeacd8ad99059635ef.jpg)  
(a) SciDocs

![](images/1fcb13ee34dda65f9736628a88458980b87e759dc488b76cc9c30c4883f21766.jpg)  
(b) Amazon

![](images/98e3d324c80e01e3060c79806db8e428d9891d89b6706fd00ed2689c6e2f8318.jpg)  
(c) Twitter

Figure 1: MACC of seed-guided topic discovery methods on in-vocabulary categories and out-of-vocabulary categories.  
![](images/67579f4b1b00992667f951534d835ddafec90d2304d4afd427b2d0245f3a9da6.jpg)  
(a) Effect of ρ on NPMI

![](images/6451bd4d798822e306c4480c337331c2721b0a1acc55f57137110ffe85eb3a7a.jpg)  
(b) Effect of ρ on LCP

![](images/1e77d4aac529e2e5e30e3b5702d3d06fbf1b936b0b50721c0de0a1565c7c0679.jpg)

![](images/25639f684ba5b2145b9e682ed52b93b4269fdd180799dedcd9d9ddaf37fecc98.jpg)  
(c) Effect of T on NPMI  
(d) Effect of T on LCP  
Figure 2: Parameter study of SEETOPIC measured by topic coherence.

## 4.3 Parameter Study

We study the effect of two important hyperparameters: $\rho$ (the hyperparameter in ensemble ranking) and $T$ (the number of iterations). We vary the value of $\rho$ in $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 , 1 \}$ (SEETOPIC uses $\rho = 0 . 1$ by default) and the value of T in {1, 2, 3, 4, 5} (SEETOPIC uses $T = 4$ by default, and SEETOPIC-NoIter is the case when $T = 1 )$ Figure 2 shows the change of model performance measured by NPMI and LCP.

Table 4: Top-5 representative terms retrieved by different algorithms for three out-of-vocabulary categories from SciDocs, Amazon, and Twitter. ✓: at least 3 of the 5 annotators judge the term as relevant to the seed. ✗: at most 2 of the 5 annotators judge the term as relevant to the seed.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Top-5 Representative Terms</td></tr><tr><td rowspan=1 colspan=2>Dataset: SciDocs, Category Name: hepatitis a/b/c/e</td></tr><tr><td rowspan=1 colspan=1>SeededLDA</td><td rowspan=1 colspan=1>patients (X), treatment (X), placebo (X), study (X), group (X)</td></tr><tr><td rowspan=1 colspan=1>Anchored CorEx</td><td rowspan=1 colspan=1>expression (X), gene (X), cells (X), genes (X), genetic (X)</td></tr><tr><td rowspan=1 colspan=1>Labeled ETM</td><td rowspan=1 colspan=1>hepatitis b virus hbv dna (√), serum hbv dna (√), serum alanine aminotransferase (X),alanine aminotransferase alt (X), below detection limit (X)</td></tr><tr><td rowspan=1 colspan=1>CatE</td><td rowspan=1 colspan=1>chronic hepatitis b virus hbv infection (√), hepatitis b e antigen hbeag (√), hepatitis b virus hbv dna (√),normal alanine aminotransferase (X), hbeag-negative chronic hepatitis b (√)</td></tr><tr><td rowspan=1 colspan=1>BioBERT</td><td rowspan=1 colspan=1>hepatitis b virus hbv dna (), chronic hepatitis b virus hbv infection (√), hepatitis b e antigen hbeag (),hepatitis b virus hbv infection (√), chronic hepatitis c virus hcv (√)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC-NoIter</td><td rowspan=1 colspan=1>hepatitis b virus hbv dna (√), hepatitis b e antigen hbeag (√), chronic hepatitis b virus hbv infection (√),hepatitis b surface antigen hbsag (√), hbeag-negative chronic hepatitis b (√)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC</td><td rowspan=1 colspan=1>chronic hepatitis b virus hbv infection (√), hbeag-negative chronic hepatitis b (√), hepatitis c virus hcv-infected (√),hepatitis b virus hbv dna (√), chronic hepatitis c virus hcv (√)</td></tr><tr><td rowspan=1 colspan=2>Dataset: Amazon,Category Name: sports and outdoors</td></tr><tr><td rowspan=1 colspan=1>SeededLDA</td><td rowspan=1 colspan=1>use (X), good (X), one (X), product (X), like (X)</td></tr><tr><td rowspan=1 colspan=1>Anchored CorEx</td><td rowspan=1 colspan=1>sports (√), use (X), size (X), wear (X), fit (√)</td></tr><tr><td rowspan=1 colspan=1>Labeled ETM</td><td rowspan=1 colspan=1>cars and tracks (√), tracks and cars (√), search options (X), championships (X), cool bosses (X)</td></tr><tr><td rowspan=1 colspan=1>CatE</td><td rowspan=1 colspan=1>outdoorsmen (√), outdoor activities (√), cars and tracks (√), foot support (√), offers plenty (X)</td></tr><tr><td rowspan=1 colspan=1>BERT</td><td rowspan=1 colspan=1>cars and tracks (√), outdoor activities (√), outdoorsmen (√), sports (√), sporting events (√)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC-NoIter</td><td rowspan=1 colspan=1>outdoorsmen (√), outdoor activities (√), cars and tracks (√), indoor soccer (√), bike riding (√)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC</td><td rowspan=1 colspan=1>canoeing (√), picnics (√), bike rides (√), bike riding (√), rafting (√)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Dataset: Twitter, Category Name: travel and transport</td></tr><tr><td rowspan=1 colspan=1>SeededLDA</td><td rowspan=1 colspan=1>nyc (X), new york (X), line (√), high (X), time square (√)</td></tr><tr><td rowspan=1 colspan=1>Anchored CorEx</td><td rowspan=1 colspan=1>new york (X), post photo (√), new (X), day (X), today (X)</td></tr><tr><td rowspan=1 colspan=1>Labeled ETM</td><td rowspan=1 colspan=1>tourism $\overline { { ( \checkmark ) , } }$ theview (√), file (X), morning view (√), gma (X)</td></tr><tr><td rowspan=1 colspan=1>CatE</td><td rowspan=1 colspan=1>maritime (√), tourism (√), natural history (X), scenery (√), elevate (X)</td></tr><tr><td rowspan=1 colspan=1>BERT</td><td rowspan=1 colspan=1>maritime (√), tourism (√), natural history (X), olive oil (X), baggage claim (√)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC-NoIter</td><td rowspan=1 colspan=1>maritime (√), tourism (√), natural history (X), scenery (√), navy (X)</td></tr><tr><td rowspan=1 colspan=1>SEETOPIC</td><td rowspan=1 colspan=1>wildlife (√), scenery (√), maritime (√), highlinepark (X), aquarium (√)</td></tr></table>

According to Figures 2(a) and 2(b), in most cases, the performance of SEETOPIC deteriorates as ρ increases from 0.1 to 0.9. Thus, setting $\rho = 0 . 1$ always leads to competitive NPMI and LCP scores on the three datasets. Although $\rho = 1$ is better than $\rho = 0 . 9$ , its performance is still suboptimal in comparison with $\rho = 0 . 1$ . This finding indicates that replacing the mean reciprocal rank (i.e., ρ = 1) with our proposed Eq. (6) is reasonable. According to Figures 2(c) and 2(d), SEETOPIC usually performs better when there are more iterations. On SciDocs and Twitter, the scores start to converge after $T = 4$ . Besides, more iterations will result in longer running time. Overall, we believe setting T = 4 strikes a good balance.

## 4.4 Case Study

Finally, we show the terms retrieved by different methods as a case study. From each of the three datasets, we select an out-of-vocabulary category and show its topic discovery results in Table 4. We mark a retrieved term as correct (✓) if at least 3 of the 5 annotators judge the term as relevant to the seed. Otherwise, we mark the term as incorrect (✗).

For the category “hepatitis a/b/c/e” from Sci-

Docs, SeededLDA and Anchored CorEx can only find very general medical terms, which are relevant to all seeds in SciDocs and thus inaccurate; Labeled ETM and CatE find terms about “alanine aminotransferase”, whose elevation suggest not only hepatitis but also other diseases like diabetes and heart failure, thus not discriminative either; BioBERT and SEETOPIC, with the power of a PLM, can accurately pick terms relevant to “hepatitis b” and “hepatitis c”. For the category “sports and outdoors” from Amazon, SeededLDA and Anchored CorEx again find very general terms, most of which are not category-discriminative; Labeled ETM and CatE are able to pick more specific terms such as “cars and tracks”, but they still make mistakes; BERT, as a PLM, can accurately find terms that have lexical overlap with the category name (e.g., “outdoorsmen”, “sporting events”), meanwhile such terms are less diverse; SEETOPIC-NoIter starts to discover more concrete terms than BERT (e.g., “indoor soccer”, “bike riding”) by leveraging local text semantics; the full SEETOPIC model, with an iterative updating process, can find more specific and informative terms (e.g., “canoeing”, “picnics”, and “rafting”). For the category “travel and transport” from Twitter, both BERT and CatE make mistakes by including the term “natural history”; SEETOPIC-NoIter, without an iterative update process, also includes this error; the full SEETOPIC model finally excludes this error and achieves the highest accuracy in the retrieved top-5 terms among all compared methods.

## 5 Related Work

Seed-Guided Topic Discovery. Seed-guided topic models aim to leverage user-provided seeds to discover underlying topics according to users’ interests. Early studies take LDA (Blei et al., 2003) as the backbone and incorporate seeds into model learning. For example, Andrzejewski et al. (2009) consider must-link and cannot-link constraints among seeds as priors. SeededLDA (Jagarlamudi et al., 2012) encourages topics to contain more seeds and encourages documents to select topics relevant to the seeds they contain. Anchored CorEx (Gallagher et al., 2017) extracts maximally informative topics by jointly compressing the corpus and preserving seed relevant information. Recent studies start to utilize embedding techniques to learn better word semantics. For example, CatE (Meng et al., 2020a) explicitly encourages distinction among retrieved topics via category-name guided embedding learning. However, all these models require the provided seeds to be in-vocabulary, mainly because they focus on the input corpus only and are not equipped with general knowledge of PLMs.

Embedding-Based Topic Discovery. A number of studies extend LDA to involve word embedding. The common strategy is to adapt distributions in LDA to generate real-valued data (e.g., Gaussian LDA (Das et al., 2015), LFTM (Nguyen et al., 2015), Spherical HDP (Batmanghelich et al., 2016), and CGTM (Xun et al., 2017b)). Some other studies think out of the LDA backbone. For example, TWE (Liu et al., 2015) uses topic structures to jointly learn topic embeddings and improve word embeddings. CLM (Xun et al., 2017a) collaboratively improves topic modeling and word embedding by coordinating global and local contexts. ETM (Dieng et al., 2020) models word-topic correlations via word embeddings to improve the expressiveness of topic models. More recently, Sia et al. (2020) show that directly clustering word embeddings (e.g., word2vec or BERT) also generates good topics; Thompson and Mimno (2020) further find that BERT and GPT-2 discover high-quality topics, but RoBERTa does not. These models are unsupervised and hard to be applied to seed-guided settings. In contrast, our SEETOPIC framework joint leverages PLMs, word embeddings, and seed information.

## 6 Conclusions and Future Work

In this paper, we study seed-guided topic discovery in the presence of out-of-vocabulary seeds. To understand and make use of in-vocabulary components in each seed, we utilize the tokenization and contextualization power of PLMs. We propose a seed-guided embedding learning framework inspired by the goal of maximizing PMI in topic modeling, and an iterative ensemble ranking process to jointly leverage general knowledge of the PLM and local signals learned from the input corpus. Experimental results show that SEETOPIC outperforms seed-guided topic discovery baselines and PLMs in terms of topic coherence, term accuracy, and topic diversity. A parameter study and a case study further validate some design choices in SEETOPIC.

In the future, it would be interesting to extend SEETOPIC to seed-guided hierarchical topic discovery, where parent and child information in the input category hierarchy may help infer the meaning of out-of-vocabulary nodes.

## Acknowledgments

We thank anonymous reviewers for their valuable and insightful feedback. Research was supported in part by US DARPA KAIROS Program No. FA8750-19-2-1004, SocialSim Program No. W911NF-17-C-0099, and INCAS Program No. HR001121C0165, National Science Foundation IIS-19-56151, IIS-17-41317, and IIS 17-04532, and the Molecule Maker Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897, and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE) by NSF under Award No. 2118329. Any opinions, findings, and conclusions or recommendations expressed herein are those of the authors and do not necessarily represent the views, either expressed or implied, of DARPA or the U.S. Government.

## References

David Andrzejewski, Xiaojin Zhu, and Mark Craven. 2009. Incorporating domain knowledge into topic modeling via dirichlet forest priors. In ICML’09, pages 25–32.

Kayhan Batmanghelich, Ardavan Saeedi, Karthik Narasimhan, and Sam Gershman. 2016. Nonparametric spherical topic modeling with word embeddings. In ACL’16, pages 537–542.

David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. JMLR, 3:993– 1022.

Xingyuan Chen, Yunqing Xia, Peng Jin, and John Carroll. 2015. Dataless text classification with descriptive lda. In AAAI’15, pages 2224–2231.

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. 2020. Specter: Document-level representation learning using citation-informed transformers. In ACL’20, pages 2270–2282.

Margaret H Coletti and Howard L Bleich. 2001. Medical subject headings used to search the biomedical literature. JAMIA, 8(4):317–323.

Rajarshi Das, Manzil Zaheer, and Chris Dyer. 2015. Gaussian lda for topic models with word embeddings. In ACL’15, pages 795–804.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT’19, pages 4171–4186.

Adji B Dieng, Francisco JR Ruiz, and David M Blei. 2020. Topic modeling in embedding spaces. TACL, 8:439–453.

Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378.

Ryan J Gallagher, Kyle Reing, David Kale, and Greg Ver Steeg. 2017. Anchored correlation explanation: Topic modeling with minimal domain knowledge. TACL, 5:529–542.

Thomas L Griffiths and Mark Steyvers. 2004. Finding scientific topics. PNAS, 101(suppl 1):5228–5235.

Thomas Hofmann. 1999. Probabilistic latent semantic indexing. In SIGIR’99, pages 50–57.

Jagadeesh Jagarlamudi, Hal Daumé, and Raghavendra Udupa. 2012. Incorporating lexical priors into topic models. In EACL’12, pages 204–213.

Jey Han Lau, David Newman, and Timothy Baldwin. 2014. Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality. In EACL’14, pages 530–539.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240.

Omer Levy and Yoav Goldberg. 2014. Neural word embedding as implicit matrix factorization. NIPS’14, pages 2177–2185.

Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. 2020. On the sentence embeddings from pre-trained language models. In EMNLP’20, pages 9119–9130.

Yang Liu, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. 2015. Topical word embeddings. In AAAI’15, pages 2418–2424.

Zequn Liu, Shukai Wang, Yiyang Gu, Ruiyi Zhang, Ming Zhang, and Sheng Wang. 2021. Graphine: A dataset for graph-aware terminology definition generation. In EMNLP’21, pages 3453–3463.

Christopher D Manning, Mihai Surdeanu, John Bauer, Jenny Rose Finkel, Steven Bethard, and David Mc-Closky. 2014. The stanford corenlp natural language processing toolkit. In ACL’14, System Demonstrations, pages 55–60.

Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In RecSys’13, pages 165– 172.

Yu Meng, Jiaxin Huang, Guangyuan Wang, Zihan Wang, Chao Zhang, Yu Zhang, and Jiawei Han. 2020a. Discriminative topic mining via categoryname guided text embedding. In WWW’20, pages 2121–2132.

Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, and Jiawei Han. 2020b. Text classification using label names only: A language model self-training approach. In EMNLP’20, pages 9006–9017.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In NIPS’13, pages 3111–3119.

David Mimno, Hanna Wallach, Edmund Talley, Miriam Leenders, and Andrew McCallum. 2011. Optimizing semantic coherence in topic models. In EMNLP’11, pages 262–272.

David Newman, Jey Han Lau, Karl Grieser, and Timothy Baldwin. 2010. Automatic evaluation of topic coherence. In NAACL-HLT’10, pages 100–108.

Dat Quoc Nguyen, Richard Billingsley, Lan Du, and Mark Johnson. 2015. Improving topic models with latent feature word representations. TACL, 3:299– 313.

Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. 2018. Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec. In WSDM’18, pages 459–467.

Michael Röder, Andreas Both, and Alexander Hinneburg. 2015. Exploring the space of topic coherence measures. In WSDM’15, pages 399–408.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In ACL’16, pages 1715–1725.

Jingbo Shang, Jialu Liu, Meng Jiang, Xiang Ren, Clare R Voss, and Jiawei Han. 2018. Automated phrase mining from massive text corpora. IEEE TKDE, 30(10):1825–1837.

Suzanna Sia, Ayush Dalmia, and Sabrina J Mielke. 2020. Tired of topic models? clusters of pretrained word embeddings make for fast and good topics too! In EMNLP’20, pages 1728–1736.

Jian Tang, Meng Qu, and Qiaozhu Mei. 2015. Pte: Predictive text embedding through large-scale heterogeneous text networks. In KDD’15, pages 1165–1174.

Laure Thompson and David Mimno. 2020. Topic modeling with contextualized word representation clusters. arXiv preprint arXiv:2010.12626.

Dingding Wang, Shenghuo Zhu, Tao Li, and Yihong Gong. 2009. Multi-document summarization using sentence-based topic models. In ACL’09, pages 297–300.

Mu-Chun Wang, Zixuan Liu, and Sheng Wang. 2022. Textomics: A dataset for genomics data summary generation. In ACL’22.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

Guangxu Xun, Yaliang Li, Jing Gao, and Aidong Zhang. 2017a. Collaboratively improving topic discovery and word embeddings by coordinating global and local contexts. In KDD’17, pages 535–543.

Guangxu Xun, Yaliang Li, Wayne Xin Zhao, Jing Gao, and Aidong Zhang. 2017b. A correlated topic model using word embeddings. In IJCAI’17, pages 4207– 4213.

Chao Zhang, Keyang Zhang, Quan Yuan, Fangbo Tao, Luming Zhang, Tim Hanratty, and Jiawei Han. 2017. React: Online multimodal embedding for recencyaware spatiotemporal activity modeling. In SI-GIR’17, pages 245–254.

Yu Zhang, Shweta Garg, Yu Meng, Xiusi Chen, and Jiawei Han. 2022. Motifclass: Weakly supervised text classification with higher-order metadata information. In WSDM’22, pages 1357–1367.

Yu Zhang, Yu Meng, Jiaxin Huang, Frank F Xu, Xuan Wang, and Jiawei Han. 2020. Minimally supervised categorization of text with metadata. In SIGIR’20, pages 1231–1240.

## A The Embedding Learning Objective

In Section 3.2, we propose the following embedding learning objective:

$$
\mathcal { I } = \underbrace { \sum _ { d \in \mathcal { P } } \sum _ { v _ { i } \in d } \sum _ { w _ { j } \in C ( w _ { i } , h ) } \frac { \exp ( u _ { w _ { i } } ^ { T } v _ { w _ { j } } ) } { \sum _ { \substack { \mathcal { I } \in \partial v _ { i } \in \mathcal { I } _ { D } } } } } _ { \mathcal { I } _ { \mathrm { c o r t e a l } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \exp ( u _ { w _ { i } v } ^ { T } v _ { w ^ { \prime } } ) } _ { \mathcal { I } _ { \mathrm { c o r t e a l } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \exp ( u _ { w _ { i } v } ^ { T } v _ { w ^ { \prime } } ) } _ { \mathcal { I } _ { \mathrm { c o r t e a l } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \sum _ { w _ { i } \in \mathcal { I } _ { D } } \exp ( u _ { w _ { i } v } ^ { T } v _ { w ^ { \prime } } ) } _ { \mathcal { I } _ { \mathrm { d o s s s u m e d } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \sum _ { w _ { i } \in \mathcal { I } _ { D } } \exp ( w _ { i } ^ { T } v _ { w ^ { \prime } } ) } _ { \mathcal { I } _ { \mathrm { d o s s u m e d } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \sum _ { w _ { i } \in \mathcal { I } _ { D } } \exp ( w _ { i } ^ { T } v _ { w ^ { \prime } } ) } _ { \mathcal { I } _ { \mathrm { d o s s u m e d } } } + \underbrace { \sum _ { \substack { w _ { i } \in \mathcal { I } _ { D } } } \sum _ { w _ { i } \in \mathcal { I } _ { D } } \exp ( w _ { i } ^ { T } v _ { w ^ { \prime } } ) } _  \mathcal { I } _  \mathrm  d o s s u m\tag{12}
$$

Now we prove that maximizing $\mathcal { I }$ is implicitly performing the factorization in Eq. (3).

Levy and Goldberg (2014) have proved that maximizing $\mathcal { I } _ { \mathrm { c o n t e x t } }$ is implicitly doing the following factorization.

$$
\begin{array} { r } { \pmb { u } _ { w _ { i } } ^ { T } \pmb { v } _ { w _ { j } } = \log \Big ( \frac { \# _ { \mathcal { D } } \left( w _ { i } , w _ { j } \right) \cdot \lambda _ { \mathcal { D } } } { \# _ { \mathcal { D } } \left( w _ { i } \right) \cdot \# _ { \mathcal { D } } \left( w _ { j } \right) \cdot b } \Big ) , } \\ { \mathrm { i . e . , } \ \mathbf { U } _ { w } ^ { T } \mathbf { V } _ { w } = \mathbf { X } _ { w w } . } \end{array}\tag{13}
$$

We follow their approach to consider the other two terms $\mathcal { T } _ { \mathrm { d o c u m e n t } }$ and $\mathcal { I } _ { \mathrm { c a t e g o r y } }$ in Eq. (12). Using the negative sampling strategy to rewrite J<sub>document</sub>, we get

$$
\begin{array} { r } { \displaystyle \sum _ { w \in \mathcal { V } _ { \mathcal { D } } } \displaystyle \sum _ { d \in \mathcal { D } } \# _ { d } ( w ) \Big ( \log \sigma ( \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d } ) + b \mathbb { E } _ { d ^ { \prime } } \left[ \log \sigma ( - \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d ^ { \prime } } ) \right] \Big ) , } \end{array}\tag{14}
$$

where $\sigma ( \cdot )$ is the sigmoid function. Following (Levy and Goldberg, 2014; Qiu et al., 2018), we assume the negative sampling distribution $\propto \lambda _ { d } . ^ { 1 5 }$ Then, the objective becomes

$$
\begin{array} { r l } & { \displaystyle \sum _ { w \in \mathcal { V } _ { \mathcal { D } } } \displaystyle \sum _ { d \in \mathcal { D } } \# _ { d } ( w ) \log \sigma ( \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d } ) + } \\ & { \displaystyle \sum _ { w \in \mathcal { V } _ { \mathcal { D } } } \# _ { \mathcal { D } } ( w ) \sum _ { d ^ { \prime } \in \mathcal { D } } \frac { b \cdot \lambda _ { d ^ { \prime } } } { \lambda _ { \mathcal { D } } } \log \sigma ( - \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d ^ { \prime } } ) . } \end{array}\tag{15}
$$

For a specific term-document pair $( w , d )$ , we consider its effect in the objective:

$$
\mathcal { I } _ { w , d } = \# _ { d } ( w ) \log \sigma ( \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d } ) + \# _ { \mathcal { D } } ( w ) \frac { \boldsymbol { b } \cdot \lambda _ { d } } { \lambda _ { \mathcal { D } } } \log \sigma ( - \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { d } ) .\tag{16}
$$

Let $x _ { w , d } = \pmb { u } _ { w } ^ { T } \pmb { v } _ { d }$ . To maximize ${ \mathcal { I } } _ { w , d } ,$ we should have

$$
0 = \frac { \partial \mathcal { T } _ { w , d } } { \partial x _ { w , d } } = \# _ { d } ( w ) \sigma ( - x _ { w , d } ) - \frac { \# _ { \mathcal { D } } ( w ) \cdot b \cdot \lambda _ { d } } { \lambda _ { \mathcal { D } } } \sigma ( x _ { w , d } ) .\tag{17}
$$

That is,

$$
e ^ { 2 x _ { w , d } } - \Big ( \frac { \# _ { d } ( w ) \cdot \lambda _ { \mathcal { D } } } { \# _ { \mathcal { D } } ( w ) \cdot b \cdot \lambda _ { d } } - 1 \Big ) e ^ { x _ { w , d } } - \frac { \# _ { d } ( w ) \cdot \lambda _ { \mathcal { D } } } { \# _ { \mathcal { D } } ( w ) \cdot b \cdot \lambda _ { d } } = 0 .\tag{18}
$$

Therefore, $e ^ { x _ { w , d } } ~ = ~ - 1$ (which is invalid) or $\begin{array} { r } { e ^ { x _ { w , d } } = \frac { \# d ( w ) \cdot \lambda _ { \mathcal { D } } } { \# _ { \mathcal { D } } ( w ) \cdot b \cdot \lambda _ { d } } } \end{array}$ . In other words,

$$
{ \pmb u } _ { w } ^ { T } { \pmb v } _ { d } = x _ { w , d } = \log \Big ( \frac { \# _ { d } ( w ) \cdot \lambda _ { \mathcal { D } } } { \# _ { \mathcal { D } } ( w ) \cdot b \cdot \lambda _ { d } } \Big ) ,
$$

$$
\mathrm { i . e . , ~ } \mathbf { U } _ { w } ^ { T } \mathbf { V } _ { d } = \mathbf { X } _ { w d } .\tag{19}
$$

Similarly, for $\mathcal { I } _ { \mathrm { c a t e g o r y } }$ , the objective can be rewritten as

$$
\begin{array} { r l } & { \displaystyle \sum _ { w \in \mathcal { V } _ { \mathcal { D } } } \sum _ { c _ { i } \in \mathcal { C } } \mathbf { 1 } _ { w \in \mathcal { S } _ { i } } \log \sigma ( \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { c _ { i } } ) + } \\ & { \displaystyle \sum _ { w \in \mathcal { V } _ { \mathcal { D } } } \mathbf { 1 } _ { w \in \mathcal { S } _ { 1 } \cup \ldots \cup \mathcal { S } _ { | \mathcal { C } | } } \sum _ { c ^ { \prime } \in \mathcal { C } } \frac { b } { | \mathcal { C } | } \log \sigma ( - \boldsymbol { u } _ { w } ^ { T } \boldsymbol { v } _ { c ^ { \prime } } ) . } \end{array}\tag{20}
$$

Following the derivation of $\mathcal { T } _ { \mathrm { d o c u m e n t } }$ , we get

$$
\begin{array} { r } { \begin{array} { r } { \pmb { u } _ { w } ^ { T } \pmb { v } _ { c _ { i } } = x _ { w , c _ { i } } = \log \Big ( \frac { \mathbf { 1 } _ { w \in S _ { i } } | \mathcal { C } | } { \mathbf { 1 } _ { w \in S _ { 1 } \cup \dots \cup S _ { | \mathcal { C } | } } \cdot b } \Big ) , } \\ { \mathrm { i . e . , ~ } \mathbf { U } _ { w } ^ { T } \mathbf { V } _ { c _ { i } } = \mathbf { X } _ { w c } . } \end{array} } \end{array}\tag{21}
$$

Putting Eqs. (13), (19), and (21) together gives us Eq. (3).

## B The Ensemble Ranking Function

In Section 3.3, we propose the following ensemble ranking function:

$$
\mathrm { s c o r e } ( w | S _ { i } ) = \bigg ( \frac { 1 } { 2 } \Big ( \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } \Big ) ^ { \rho } + \frac { 1 } { 2 } \Big ( \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } \Big ) ^ { \rho } \bigg ) ^ { 1 / \rho } .\tag{22}
$$

Now we prove this ranking function is a generalization of the arithmetic mean reciprocal rank (i.e., MRR) and the geometric mean reciprocal rank:

$$
\begin{array} { r l } & { \underset { \rho \to 1 } { \operatorname* { l i m } } \operatorname { s c o r e } ( w | S _ { i } ) = \frac { 1 } { 2 } \Big ( \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } + \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } \Big ) ; } \\ & { \underset { \rho \to 0 } { \operatorname* { l i m } } \operatorname { s c o r e } ( w | S _ { i } ) = \Big ( \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } \cdot \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } \Big ) ^ { 1 / 2 } . } \end{array}\tag{23}
$$

The case of $\rho  1$ is trivial. When $\rho \to 0$ , we aim to show that

$$
\operatorname* { l i m } _ { \rho \to 0 } \log \operatorname { s c o r e } ( w | S _ { i } ) = \log \Big ( \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } \cdot \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } \Big ) ^ { 1 / 2 } .\tag{24}
$$

In fact, let $\begin{array} { r } { r _ { G } = \frac { 1 } { \mathrm { r a n k } _ { G } ( w ) } } \end{array}$ and $\begin{array} { r } { r _ { L } = \frac { 1 } { \mathrm { r a n k } _ { L } ( w ) } . } \end{array}$

$$
\begin{array} { r l } { \underset { \rho  0 } { \operatorname* { l i m } } \log \operatorname* { s c o u r } ( w | S _ { i } ) = } & { \underset { \rho  0 } { \operatorname* { l i m } } \log ( \frac { 1 } { 2 } r _ { G } ^ { \rho } + \frac { 1 } { 2 } r _ { L } ^ { \rho } ) ^ { 1 / \rho } } \\ & { = \underset { \rho  0 } { \operatorname* { l i m } } \frac { \log ( \frac { 1 } { 2 } r _ { G } ^ { \rho } + \frac { 1 } { 2 } r _ { L } ^ { \rho } ) } { \rho } } \\ & { = \underset { \rho  0 } { \operatorname* { l i m } } \frac { \frac { 1 } { 2 } r _ { G } ^ { \rho } \log r _ { L } r _ { G } } { \frac { 1 } { 2 } r _ { G } ^ { \rho } + \frac { 1 } { 2 } r _ { L } ^ { \rho } } } \\ & { = \frac { \operatorname* { l i m } _ { \rho  0 } } { \operatorname* { l i m } } \frac { ( r _ { G } ^ { \rho } \log r _ { G } + r _ { L } ^ { \rho } ) } { 1 1 \operatorname* { m a x } ( r _ { G } ^ { \rho } + r _ { L } ^ { \rho } ) } } \\ & { = \frac { \operatorname* { l i m } _ { \rho  0 } ( r _ { G } ^ { \rho } + \log r _ { L } ) } { \operatorname* { l i m } _ { \rho  0 } ( r _ { G } ^ { \rho } + r _ { L } ^ { \rho } ) } } \\ & { = \frac { \log r _ { G } + \log r _ { L } } { 2 } } \\ & { = \log ( r _ { G } \cdot r _ { L } ) ^ { 1 / 2 } . } \end{array}\tag{25}
$$

The third line is obtained by applying L’Hopital’s rule.