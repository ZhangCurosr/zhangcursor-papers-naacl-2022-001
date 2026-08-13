# Towards a Progression-Aware Autonomous Dialogue Agent

Abraham Sanders<sup>1</sup>, Tomek Strzalkowski<sup>1</sup>, Mei Si<sup>1</sup>, Albert Chang<sup>1</sup>, Deepanshu Dey<sup>1</sup>, Jonas Braasch<sup>1</sup>, Dakuo Wang<sup>2</sup>

<sup>1</sup>Rensselaer Polytechnic Institute, Troy, NY, USA, <sup>2</sup> IBM Research, USA {sandea5,tomek,sim,changa4,deyd,braasj}@rpi.edu dakuo.wang@ibm.com

## Abstract

Recent advances in large-scale language modeling and generation have enabled the creation of dialogue agents that exhibit human-like responses in a wide range of conversational scenarios spanning a diverse set of tasks, from general chit-chat to focused goal-oriented discourse. While these agents excel at generating high-quality responses that are relevant to prior context, they suffer from a lack of awareness of the overall direction in which the conversation is headed, and the likelihood of task success inherent therein. Thus, we propose a framework in which dialogue agents can evaluate the progression of a conversation toward or away from desired outcomes, and use this signal to inform planning for subsequent responses. Our framework is composed of three key elements: (1) the notion of a "global" dialogue state (GDS) space, (2) a task-specific progression function (PF) computed in terms of a conversation’s trajectory through this space, and (3) a planning mechanism based on dialogue rollouts by which an agent may use progression signals to select its next response.

## 1 Introduction

All human conversation serves some purpose. These may range from negotiating an agreement to explaining a topic to maintaining a social relationship. People are generally capable of forming an assessment, sometimes subconsciously, whether a conversation is going well or not and adjusting their behavior accordingly. Such assessment, which underlies most human conversation, is essential in continuous awareness of the direction where the interaction is heading and whether the parties are in sync or not, e.g., Bernieri and Rosenthal (1991). In a task-oriented interaction, the participants assess if progress towards a successful outcome is being made. In a negotiation, parties assess if an agreement is likely. Even in a casual conversation, people intuitively sense when to continue, when to change the subject, or when to stop. Based on such (subjective) assessment, participants adjust what to say next: whether to push forward, make a concession, soften the tone, digress, or say goodbye. A wide range of research in conversation and discourse analysis is devoted to these and related issues including (Beebe and Masterson, 2000; Cassell et al., 2007; Friedman, 2004; Gremler and Gwinner, 2008; Langewitz et al., 2003); however, recent efforts in Dialogue State Tracking (DST) have been primarily focused on collecting fine-grained details (e.g., slot-value pairs for travel booking or restaurant reservation) (Balaraman et al., 2021) without concern for the overall direction and quality of the conversation, even though the latter is critical for achieving human-level dialogue interaction.

![](images/cb58fd57eec05e1649009bc986d76b95737c223d1e35e846a7e948bdbdc4f9d5.jpg)  
Figure 1: Our framework applied to the charity solicitation task in Persuasion For Good (Wang et al., 2019). Given the dialogue history (center left), the system uses rollouts (Lewis et al., 2017) to simulate the outcome of two response candidates (bottom, in red). Each rollout is mapped as a path through the Global Dialogue State space (center right) where it can be compared with similar outcomes. The candidates are finally ranked using the Progression Function (top), and the best is selected.

As such, we approach dialogue state tracking at a higher level, focusing instead on what we call the Global Dialogue State (GDS). Given a conversational task (e.g., negotiation), the global state of a dialogue reflects the most likely outcome (e.g., a strong agreement or a stalemate) given the history of the dialogue up to the current turn. In contrast to traditional DST, the global state remains invariant to the specific details discussed at each turn (e.g., names, dates, quantities) that are typically the concern of slot-filling models. Rather, global dialogue states are influenced by the contexts in which these details occur (e.g., “I would love to donate \$5 to this charity!” vs. “I would never donate \$5 to this charity”). Thus, the global state of a dialogue can be measured in terms of its semantic similarity to other groups of dialogues for the same task, which can be naturally formulated as a cluster-assignment problem in the dialogue embedding space. That is, a dialogue which is assigned at the current turn to a cluster of highly successful outcomes may assume a high likelihood of success, and likewise a dialogue assigned to a cluster of unsuccessful outcomes may assume a low likelihood of success. It follows from this that the path of a dialogue through global state space can be used to derive a Progression Function (PF) to give turn-level estimates of task success, which can in turn be used by a dialogue agent to inform its next response.

The remainder of this paper is organized as follows: In Section 2 we review relevant literature pertaining to dialogue state tracking and response planning; in Section 3 we formally define the global dialogue state and progression function, propose supervised and unsupervised approaches for modeling them, and describe how they can be used to assess and select dialogue response candidates; in Section 4 we experimentally apply our framework to the charity solicitation task in the Persuasion For Good dataset (Wang et al., 2019), reporting results from automatic and manual evaluations; and in Sections 5 and 6 we conclude with a discussion of limitations and future directions. Code for our methods and experiments has been released, <sup>1</sup> and a listing of software packages we use can be found in Appendix A.

## 2 Related Work

Our work lies at the intersection of dialogue state tracking and response planning. As previously noted, we approach dialogue state at a much higher level than is typically seen in the DST literature. Our concept of global dialogue state is not mutually exclusive with traditional DST approaches, which we refer to from here on as local DST. Rather, an effective dialogue system might integrate local and global DST approaches to enable simultaneous tracking of user intents and slot-value pairs (needed for interfacing with external resources) and the overall likelihood of conversational success.

## 2.1 Dialogue State Tracking

Local DST approaches are used in task-oriented (also called goal-oriented) dialogue systems. Local DST is responsible for identifying user intent (e.g., search for restaurants) and extracting slot-value pairs (e.g., location, price range). Recent DST systems perform state tracking in a diverse set of domains, including food ordering (Lertvittayakumjorn et al., 2021), travel reservations (Qin et al., 2021), negotiations (He et al., 2018), and many others. Datasets such as MultiWOZ (Budzianowski et al., 2018; Eric et al., 2020; Zang et al., 2020) and SGD (Rastogi et al., 2020) provide large-scale testbeds for training single DST systems that generalize across many task domains. However, local DST is generally not deployed in open-domain end-to-end dialogue systems that focus on social interaction and user engagement, recent examples including DialoGPT (Zhang et al., 2020), Meena (Adiwardana et al., 2020), and BlenderBot (Roller et al., 2021; Xu et al., 2021). In open-domain models, the task is unconstrained and thus it makes little sense to employ traditional slot-based dialogue state trackers. Instead, these models track state implicitly in their latent representations of dialogue history. Unlike local DST, global state tracking is applicable in both the task-oriented and open-domain settings.

## 2.2 Dialogue Response Planning

Many approaches exist for planning in dialogue response generation. Planning helps a dialogue agent maintain coherence over multiple turns and stay on track to complete its goal. Lewis et al.

(2017) introduce Dialogue Rollouts, allowing a negotiation agent to simulate the remainder of a conversation based on each of multiple candidate responses and select the one which yields the best outcome. Yarats and Lewis (2018) follow up by separating semantic planning and surface realization for response generation by first producing a latent semantic representation of the dialogue plan and then conditioning on it during generation with rollouts. Similarly, Jiang et al. (2019) implement a look-ahead module to implicitly predict multiple future turns in an end-to-end encoder-decoder architecture, experimenting with negotiation and restaurant reservation settings. These works all experiment in task domains where goal achievement is explicitly measurable, which is not true in the general case. Thus we propose to combine such methods with our progression function which provides estimates of goal completion likelihood. Particularly, in this paper we demonstrate the use of rollouts with the PF as a reward signal.

## 3 Methods

The goal of our system is to construct a global dialogue state space for a task-specific dataset and learn a progression function to estimate how well an ongoing dialogue is progressing toward the desired outcome of the task. The quantity output by the progression function is an estimate of a dialogue-level attribute which indicates task success (e.g. satisfaction in a customer service task). In many task domains, the success of a conversation cannot be completely measured by a single attribute. For example, in the charity solicitation task we use in our experiments, donation amount is the primary success attribute. Here, there are cases where the conversation appears to go very well, but ultimately no donation is made for unexpected reasons such as the solicitee not being able to afford to donate. One could reasonably expect such an outcome to be “acceptable” in the context of a solicitation task since the solicitee has engaged with the solicitor and displayed interest, and we cannot reasonably expect the solicitor to force a donation out of someone who cannot afford it. Thus we introduce the “acceptability score”, a synthetic attribute that measures success by considering multiple factors $( \mathrm { e . g . }$ , donation amount and sentiment). For any dialogue dataset, the acceptability score combines multiple dialogue-level attributes in a way sensitive to their covariance with the primary

success attribute:

$$
\mathrm { A C C } _ { D } = \mathrm { p r i m } _ { D } + \sum _ { i = 1 } ^ { | \mathbf { v } _ { D } | } \mathrm { C o v } ( \mathrm { p r i m } , \mathrm { a t t r } _ { i } ) \cdot \mathbf { v } _ { D i }\tag{1}
$$

where prim $^ { 1 } D$ is the primary success attribute (e.g. donation amount) value for dialogue $D , \mathbf { v } _ { D }$ is the vector of all other attribute values (e.g., sentiment) for dialogue $D _ { : }$ , and Cov(prim, attr<sub>i</sub>) is the training set covariance between the primary success indicator and the i-th other attribute. We define the output of the progression function to be an estimate of the acceptability score.

To learn the progression function, dialogue-level attribute annotations must exist for use in this purpose. However, in many settings such annotations are not available in sufficient quantity to directly learn a progression model with sufficient generalization. Consequently, we propose supervised and unsupervised approaches for learning the global state and progression models.

## 3.1 Unsupervised Approach

## 3.1.1 Global Dialogue State

In the unsupervised approach, the GDS space is a dialogue embedding space where clusters of embeddings represent groups of dialogues with similar semantic content. For each complete dialogue D in the training set, all utterances are independently embedded and then pooled to create a dialoguelevel embedding ${ \bf u } _ { D } \in \mathbb { R } ^ { d }$ where $d$ is the embedding size. The GDS space is thus given as a matrix in $\mathbb { R } ^ { N \times d }$ where N is the number of complete dialogues. To embed utterances we take advantage of pre-trained sentence encoders exposed to largescale corpora. Specifically, we use a publicly available MPNet (Song et al., 2020) model fine-tuned for semantic textual similarity using a contrastive objective on over 1B training pairs from 32 distinct datasets. <sup>2</sup> To combine utterance embeddings into a dialogue-level embedding we use recencyweighted mean pooling. The recency weight $\beta$ determines how much emphasis is placed on more recent utterances, where $\beta = 0$ means all utterances are weighted evenly and $\beta > 0$ means that more emphasis is placed on more recent utterances. The motivation for recency weighting is to test the hypothesis that more recent developments in a conversation are more relevant for predicting current progression toward a goal. For example, a conversation may start out off-task with participants engaging in small talk, and then later re-focus.

![](images/3084300136c82b6a071962d24b9fd60f85b109a56866cb93c926bb05f718fd9a.jpg)  
Figure 2: Architecture of the supervised and unsupervised GDS and PF models (top). In GDS space (top right), each cluster is characterized by similar dialogue semantics, and is thus interpreted as the class of typical outcomes for dialogues within. GDS and PF can be used with rollouts (bottom) to allow a dialogue agent to plan ahead.

The embedding for dialogue D with $| D |$ utterances is thus formulated as ${ \bf u } _ { D } = U ^ { T }$ softmax(r) where U is the matrix of utterance vectors in $\mathbb { R } ^ { | D | \times d }$ and $\mathbf { r } \in \mathbb { R } ^ { | D | }$ is a vector of evenly spaced real numbers over the interval $[ 0 , \beta ]$ . The softmax ensures all recency weights sum to 1 and can be interpreted as probabilities as done with attention scores in (Bahdanau et al., 2015; Vaswani et al., 2017). As shown in Figure 3, each utterance is thus weighted by a monotonically increasing probability mass where higher values of $\beta$ cause more mass to be concentrated at the end of the dialogue.

![](images/bdf3f1fb068d5cd8b27518ac113afe33d7b4830e1d8bd977aea5d19b6e03c8d8.jpg)

![](images/c2ffe76ebe282ba448a57e685ad4ce4289cca8a48e4eaf7c819a52a944d40a0a.jpg)  
Figure 3: Recency weight β controls how much emphasis is placed on recent utterances when computing u<sub>D</sub>.

The unsupervised GDS model is a clustering of the dialogues in their embedding space. The dialogue embeddings are either clustered directly or after projection to a lower-dimensional space using Parametric UMAP (Sainburg et al., 2020; McInnes et al., 2018a). We experiment with kmeans and HDBSCAN (McInnes and Healy, 2017; Campello et al., 2013) to cluster the embeddings. For k-means, we choose the number of clusters k and train with 10 random initializations. For HDBSCAN, we choose the minimum cluster size and minimum samples hyperparameters, and the optimal number of clusters are selected automatically. Unlike k-means which simply partitions the embedding space, HDBSCAN classifies some embeddings as noise points. Clustering hyperparameters are selected by cross-validation on several metrics as described later in Section 4. The process of constructing the GDS model is illustrated in Figure 2.

The clusters output by this process can be interpreted as the equivalence classes of final global states possible for the task represented in the dialogue dataset. To estimate the global state of an ongoing dialogue $D ^ { \prime } ,$ it is embedded as $\mathbf { u } _ { D ^ { \prime } } \in \mathbb { R } ^ { d }$ in the same manner as the complete training dialogues, followed by optional dimensionality reduction. The trained k-means or HDBSCAN model is then used to assign $D ^ { \prime }$ to one of the existing clusters, or possibly as a noise point in the case of HDBSCAN.

Each cluster is assigned an aggregate acceptability score by taking an average of acceptability for each dialogue in the cluster. If k-means is used, we aggregate using a 10% trimmed mean across all dialogues in the cluster. If HDBSCAN is used, a probability is returned for each dialogue representing the likelihood that it is a member of its assigned cluster, so we compute the probability-weighted average across all dialogues in the cluster. Dialogues classified as noise points are ignored.

To visualize the GDS model, Parametric UMAP is used again to project the clustered dialogue embeddings into $\bar { \mathbb { R } ^ { 2 } }$ or $\mathbb { R } ^ { 3 }$ . As shown in Figure 1, the GDS model can be mapped as a scatter plot with each cluster labeled by its aggregate values. If k-means is used, the cluster centroids can be displayed as a bold point within each cluster. HDB-SCAN clusters do not have centroids, but they do have a number of representative points that are close to the cluster core. We average these points to simulate a centroid for display purposes, and likewise show it as a bold point within each cluster. To show how an ongoing dialogue $D ^ { \prime }$ traverses the GDS space over time, its embeddings at each turn t are projected onto the map and connected with line segments to form a path.

## 3.1.2 Computing Progression

Since each cluster in the GDS space is intended to represent a class of end-task global states, we compute the progression of an ongoing dialogue $D ^ { \prime }$ with respect to the likelihood that its final global state will rest in each individual cluster. Supposing there are k final clusters after running k-means or HDBSCAN, we compute a probability vector $\mathbf { p } _ { D ^ { \prime } } \in \mathbb { R } ^ { k }$ such that ${ \bf p } _ { D ^ { \prime } i } = P ( { \bf u } _ { D ^ { \prime } } \in C _ { i } )$ for $i \in$ $\{ 1 , \ldots k \}$ where $C _ { i }$ is cluster i. $\mathbf { p } _ { D ^ { \prime } }$ is computed differently for k-means and HDBSCAN. K-means does not produce a probabilistic soft clustering, so we define $\mathbf { p } _ { D ^ { \prime } }$ with respect to the proximity of ${ \bf u } _ { D ^ { \prime } }$ to the centroids of each cluster:

$$
\mathbf { p } _ { D ^ { \prime } } = \mathrm { s o f t m a x } \left( \frac { 1 } { | | \mathbf { u } _ { D ^ { \prime } } - \mathbf { c } _ { i } | | _ { 2 } } \ : \ i \in \{ 1 , \dots k \} \right)\tag{2}
$$

where $\mathbf { c } _ { i } \in \mathbb { R } ^ { d }$ is the centroid of cluster i. HDB-SCAN does produce a probabilistic soft clustering, so in that case p<sub>D</sub> is already computed.

We ultimately want the closest (or most probable) clusters for ongoing dialogue $D ^ { \prime }$ to have the most sway in estimating its progression at the current point in time. That is, if $D ^ { \prime }$ has moved into a cluster of high-success outcomes, its progression should increase. Likewise if $D ^ { \prime }$ has moved away from such a high-success cluster, either into a lower-success cluster or off-task into a noisy or unknown region of the GDS space, its progression should decrease. Thus, once ${ \bf u } _ { D ^ { \prime } }$ is computed, we estimate its progression as the probability-weighted average of the aggregate acceptability scores assigned to each cluster. This is formulated as

$$
\mathrm { P R O G } ( \mathbf { u } _ { D ^ { \prime } } ) = \frac { \mathbf { v } ^ { T } \mathbf { p } _ { D ^ { \prime } } } { \sum _ { i = 1 } ^ { k } \mathbf { p } _ { D ^ { \prime } i } }\tag{3}
$$

where $\mathbf { v } \in \mathbb { R } ^ { k }$ is a vector of the aggregate acceptability scores assigned to each cluster. The scaling factor in the denominator ensures that ongoing dialogue embeddings classified as noise points by HDBSCAN will not be assigned progression values close to zero as a consequence of not belonging to any cluster, which can cause significant fluctuation in the progression function as the dialogue traverses noisy regions of the GDS space. <sup>3</sup> Figure 2 illustrates how progression of an ongoing dialogue depends on its position in GDS space.

## 3.2 Supervised Approach

For the supervised approach, we simply fine-tune RoBERTa (Liu et al., 2019) to directly predict acceptability given the dialogue history text, where all utterances are concatenated into a single sequence. To construct the GDS space we obtain the dialogue level embedding $\mathbf { u } _ { D }$ directly from the CLS (<s>) token for each complete dialogue in the training set, and cluster them as in Section 3.1.1. Unlike the unsupervised approach where recency weighting is used to “attend” to more recent parts of the dialogue, the supervised fine-tuning process causes the CLS embedding to aggregate the parts of the dialogue most relevant to the task objective, which is more optimal than the recency heuristic. Also, unlike the unsupervised approach where progression for an ongoing dialogue is computed with respect to its embedding, here progression is directly predicted by RoBERTa. In our experiments we compare RoBERTa-base, RoBERTa-large, and RoBERTa-large-adapted, the latter receiving additional domain adaptation training for dialogue. Domain adaptation is done via Masked Language Modeling (MLM) on a self-generated version of the Gutenberg Dialogue Dataset (Csaky and Recski, 2021). Hyperparameters and model weights from domain adaptation training are provided with our code release.

## 3.3 Response Planning

To allow a dialogue agent to use the progression function as feedback for response planning, we adopt Dialogue Rollouts (Lewis et al., 2017) to simulate the outcomes of a set of response candidates. A rollout for a response candidate simulates the next N turns of the conversation (for both participants) given that candidate is used. At each turn of a negotiation task, Lewis et al. (2017) sample a set of c response candidates and s rollouts per candidate. They score each rollout by a deterministic reward (the value of the items “won” by the agent during negotiation), and rank each candidate by the average of its rollout scores. The highest ranking candidate is then selected by the agent. As shown in Figure 2, we generalize this process to any task for which a progression function can be learned, replacing the deterministic reward with the progression function value. To demonstrate this, we fine-tune the 762M parameter DialoGPT (Zhang et al., 2020) <sup>4</sup> as a dialogue response generator and use beam sampling to generate response candidates and rollouts. We select DialoGPT for this task as it is pre-trained on a large Reddit dialogue corpus.

## 4 Experiments

## 4.1 Dataset

We apply our framework to the Persuasion For Good dataset (Wang et al., 2019), which is a crowdsourced dialogue dataset where the task for an individual playing the role of persuader is to convince another individual playing the role of persuadee to make a donation to a well-known children’s charity. We selected this dataset since it has a clear task objective (to solicit donations), but a complex relationship between dialogue content and success. The dataset authors identify 10 distinct persuasion strategies used to solicit donations, where different strategies correlate with donation amount at different strengths. Additionally, participants in Persuasion For Good dialogues complete a pre-task psychological survey, yielding 23 attributes based on the Big-Five personality traits (Goldberg, 1992), the Moral Foundations endorsement (Graham et al.,

2011), the Schwartz Portrait Value (Cieciuch and Davidov, 2012), and the Decision-Making style (Hamilton et al., 2016) questionnaires for each individual. The dataset authors demonstrated varying degrees of correlation between these psychological attributes and the end-task donation amount. The complexity in measuring progression in this context, coupled with it being a relatively small dataset, makes Persuasion For Good an interesting and challenging testbed for our framework. Persuasion For Good contains 1017 dialogues, each with approximately 10 turns per speaker (20 utterances).

## 4.2 Progression Function Experiments

As the objective of the task is to solicit donations, we consider the end-dialogue persuadee donation amount to be the primary dialogue success indicator. We also augment the dataset by computing average dialogue sentiment. To compute sentiment we use a RoBERTa model<sup>5</sup> fine-tuned on the sentiment classification task of the TweetEval benchmark (Barbieri et al., 2020), which was publicly released by the benchmark authors. We score sentiment at the utterance level in the range [ 1, 1] by multiplying the sentiment class probabilities predicted by RoBERTa for negative, neutral and positive by 1, 0, 1 respectively and summing the result. We then average the utterance-level sentiment score for each dialogue.

We filter the dataset to eliminate dialogues with end-task donation amounts outside the allowed task parameters (between \$0 and \$2 USD), and use a regular expression to filter out dialogues where the persuadee fails to make a donation after promising a non-zero dollar amount in the conversation. After filtration we are left with 751 dialogues for our study. We split the dialogues into a training and test set, leaving 577 dialogues for training and 174 for testing. After splitting, we mean-center the dialogue values in the training set for each attribute and scale them to have unit variance. We apply the same transformation to the test set using the distribution parameters of the training set. Our final pre-processing step is to compute the acceptability score. To do this, we compute the covariance matrix of the dialogue-level attribute values in the training set, which include the donation amount and psychological attributes for both the persuader and persuadee from the original dataset, along with our computed sentiment scores. Since the values are all standardized, the covariances are equivalent to Pearson’s r. We select the covariances of all attributes with respect to the persuadee donation amount (see Figure 5 in Appendix B) and define the acceptability score of each dialogue D as defined in Section 3. We use the same covariances obtained from the training set to compute acceptability scores on the test set. After pre-processing, the training set has 52 total attributes. These include the persuadee/persuader donation amounts, psychological variables, sentiment, and the acceptability score.

## 4.2.1 Progression Model Training

We train four progression models as outlined in Sections 3.1 and 3.2: (1) Unsupervised, (2) RoBERTabase, (3) RoBERTa-large, and (4) RoBERTa-largeadapted. For each model, 10% of the training set is held out as a validation set (58 dialogues). For the unsupervised model, a grid search is run for the hyperparameters (e.g., # clusters, recency $\beta ,$ dim. reduction, etc.) over the validation set, and the final model is obtained by re-training over the full training set using the best hyperparameters. The final model uses k-means for clustering with $k = 2 1$ and recency weight $\beta \ : = \ : 0 . 3 .$ . A complete hyperparameter listing and details on the grid search can be found in Appendix F. For the supervised RoBERTa models, fine-tuning is done with AdamW (Loshchilov and Hutter, 2019) and an initial learning rate of $3 \times 1 0 ^ { - 5 }$ for a maximum of 30 epochs. Early stopping is used over the validation set with the checkpoint corresponding to the lowest validation loss selected as the final model.

## 4.2.2 Automatic Evaluation

We evaluate the progression models on the following automatic metrics: (1) Mean Absolute Error (MAE) in predicting dialogue acceptability, and (2) Pearson’s correlation (r) between overall PF slope and dialogue acceptability. With MAE we validate that the progression function is able to estimate success of a complete dialogue, while PF slope correlation validates that during an ongoing dialogue, progression increases over time for high-success dialogues and decreases over time for low-success dialogues. To measure PF slope correlation, we fit a least-squares regression line to the progression curve of each dialogue in the test set, and measure Pearson’s r between the regression slopes and their corresponding acceptability scores. For robustness we repeat this evaluation 33 times with varying initialization seeds for each model type (final hyperparameters stay constant) and report the means and standard deviations across runs in Table 1.

Table 1: Progression Function Auto Eval Results: Mean (SD) of MAE and Pearson’s r across runs.
<table><tr><td>Model</td><td>MAE</td><td>r</td></tr><tr><td>Unsupervised*</td><td> $1 . 3 7 \left( \pm 0 . 0 1 \right)$ </td><td> $0 . 4 0 \left( \pm 0 . 0 2 \right)$ </td></tr><tr><td>RoBERTa-base</td><td> $0 . 9 9 \left( \pm 0 . 0 6 \right)$ </td><td>0.50 (± 0.04)</td></tr><tr><td>RoBERTa-large</td><td> $0 . 9 7 \left( \pm 0 . 1 0 \right)$ </td><td> $0 . 5 4 \left( \pm 0 . 0 6 \right)$ </td></tr><tr><td>RoBERTa-large-adapted</td><td> ${ \bf 0 . 9 5 } \left( \pm 0 . 0 5 \right)$ </td><td>0.57 (± 0.03)</td></tr></table>

For Pearson’s r, p < 0.01 for all runs (two-tailed; H is non-correlation). \* Hyperparameters of the unsupervised model can be found in Appendix G.

Unsurprisingly, the supervised models outperform the unsupervised model on both metrics, which is expected since they directly optimize a mean squared error objective. Of the supervised models, the RoBERTa-large instances perform the best, with dialogue domain adaptation boosting each metric. Pearson’s r is significant at the $p < 0 . 0 1$ level for all runs (the null hypothesis is non-correlation).

## 4.2.3 Manual Evaluation

To obtain a more precise evaluation, we asked three annotators to estimate sentence-level progression for dialogues in our test set. Two graduate students and one postdoc in our lab served as our annotators. For each of twelve randomly selected test dialogues, each annotator rated all sentences on a scale of {-1, 0, 1} for progression, with -1 indicating regression from the task goal, 0 indicating neutral progression, and +1 indicating progression toward the task goal. Altogether, each annotator provided 431 sentence ratings across 244 utterances. After aggregating at the utterance level, average inter-annotator agreement is 0.57 (Cohen’s kappa). For each dialogue, the cumulative sum of the utterance-level manual ratings creates a groundtruth progression curve, as shown in Figure 4.

We evaluate the progression models against the ground-truth curves using Pearson’s correlation (r) since the PF output and cumulative manual ratings are continuous and on different scales. We report the following correlations: (1) between utterancelevel PF value and ground-truth value (utt), (2) between utterance-level PF slope and ground-truth slope (utt-sl), (3) between dialogue-level PF slope and ground-truth slope (dlg-sl), and (4) between dialogue-level PF slope and the final ground-truth progression value (dlg-sl-f). We repeat this evaluation for each of the 33 uniquely initialized model instances of each type from Section 4.2.2, averaging each metric across raters and then across runs. We report the aggregate means and standard deviations across runs in Table 2. A complete listing of results for each rater (averaged across runs) is provided in Appendix C, along with detailed explanations of each metric (Appendix D). Additionally, Figure 10 in Appendix E provides examples of disagreement between the PF and ground-truth progression curves which can be challenged despite high interannotator agreement, demonstrating difficulty in establishing ground-truth for this open-ended task.

Table 2: Progression Function Manual Eval Results: Mean (SD) of utt, utt-sl, dlg-sl, & dlg-sl-f across runs.
<table><tr><td>Model</td><td>utt</td><td>utt-sl</td><td>dlg-sl</td><td>dlg-sl-f</td></tr><tr><td>Unsuperv.</td><td>0.09 (± 0.13) 0.04 (± 0.01) -0.01 (± 0.07)</td><td></td><td></td><td> $- 0 . 0 7 \left( \pm 0 . 0 7 \right)$ </td></tr><tr><td>RoBERTa-b</td><td></td><td>0.39 (± 0.07) 0.18 (± 0.03) 0.26 (± 0.06)</td><td></td><td> $0 . 2 6 \left( \pm 0 . 0 6 \right)$ </td></tr><tr><td>RoBERTa-1</td><td></td><td>0.39 (± 0.08) 0.18 (± 0.06) 0.34 (± 0.09)</td><td></td><td> $0 . 3 5 \left( \pm 0 . 1 0 \right)$ </td></tr><tr><td></td><td>RoBERTa-1-a 0.48 (± 0.05) 0.24 (± 0.03) 0.43 (± 0.06)</td><td></td><td></td><td> ${ \bf 0 . 4 1 } \left( \pm 0 . 0 7 \right)$ </td></tr></table>

b = base; l = large; l-a = large-adapted.

![](images/2e020920afb66406555fad90118ecb9c51125045fb9a15ff4497daf3c73275a6.jpg)  
Figure 4: Left: ground-truth progression curve given by the cumulative sum of utterance-level manual ratings. Right: estimated curve from RoBERTa-large-adapted.

## 4.2.4 Benefit of Domain Adaptation

To verify the beneficial effect of domain adaptation we perform two-tailed paired t-tests to confirm the differences in means between RoBERTa-large and RoBERTa-large-adapted on all automatic and manual metrics. For each metric, we pair the results from both models for each run of the same seed, since their regression heads would have received identical initializations. We find that the means of utt, utt-sl, dlg-sl, and dlg-sl-f differ at the $p < 0 . 0 1$ significance level, and the means of the automatic Pearson’s r metric differ at the $p < 0 . 0 5$ level. This confirms our intuition that domain adaptation for dialogue prior to fine-tuning the regression objective aids generalization in this task.

## 4.3 Rollout Experiments

To demonstrate the ability of the PF to guide a dialogue agent, we use it to score rollouts generated with DialoGPT as described in Section 3.3. Specifically, we design a self-play experiment to automatically evaluate the effect of PF-guided rollouts on the success of the solicitation task in Persuasion For Good. The following summarizes the experimental setup, procedure, and results.

## 4.3.1 Exeperimental Setup

First, we fine-tune DialoGPT to generate responses on Persuasion For Good. We add speaker control tokens to the vocabulary so that the model can be conditioned to generate as the persuader or persuadee, respectively. Training is done with AdamW (initial $\mathrm { l r } { = } 5 \times 1 0 ^ { - 5 } )$ for 6 epochs with early stopping over a 10% validation set using perplexity. The final model checkpoint was selected after 3 epochs, achieving validation perplexity of 8.82.

We then select a progression model to use for all self-play runs. Since the supervised RoBERTalarge-adapted model achieved the best average scores across all automatic and manual evaluations, we randomly select one of the 33 RoBERTa-largeadapted instances from Section 4.2.2 to use for all runs. We use this instance for rollout scoring and to measure the progression of each self-play dialogue.

Finally, we train a binary classifier to identify if the persuadee has stated the intent to donate in a conversation, which we use to detect successful self-play dialogues. We fine-tune a RoBERTa model as a classifier using just the persuadee’s utterances as input and use binarized donation labels in Persuasion For Good as targets. Specifically, for each dialogue the label is 0 if the donated amount is \$0, otherwise it is 1. We use the manually verified intended donation labels from Persuasion For Good “ANNSET” for our validation and test splits and use the remaining end-task donation labels for training. Training is done with early stopping over the validation split using macro F1. The final model checkpoint achieved test F1 of 0.89 and test accuracy of 0.90. All three trained models used in this experiment are available to download (see our code release for instructions and hyperparameters).

## 4.3.2 Self-Play Procedure

From our test set of 174 dialogues, we manually filter out 41 in which the persuadee pledges a donation within the first 10 utterances, leaving 133 remaining conversations. For each of these, the response generator is given the first 10 utterances as context and tasked to complete a second set of 10 utterances, playing the role of both the persuader and persuadee. Since the task is solicitation, we allow the generator to use rollouts only when acting as the persuader. We perform the self-play experiment using three persuader modes: (1) with no rollouts (No RO), (2) with 2 response candidates, 2 rollouts per candidate, and 3 utterances per rollout (2x2x3), and (3) with 3 response candidates, 3 rollouts per candidate, and 5 utterances per rollout (3x3x5). For each utterance in each rollout, we use beam sampling with num\_beams=6, top\_k=50, top\_p=0.95, and temperature=1.5+0.002 T where T is the number of tokens in the dialogue history. After generation, we compute the following metrics for each dialogue: (1) progression using the selected RoBERTa-large-adapted instance (Prog.), (2) persuader and persuadee sentiment using the sentiment classifier from Section 4.2 (ER Sent. & EE Sent.), and (3) the percent of test dialogues where the persuadee pledges a donation amount or explicitly states intent to donate, as detected by the binary donation intent classifier (EE Don.%).

Table 3: Rollouts self-play results: Mean (SD) of progression, sentiment, and % donated across runs.
<table><tr><td>Mode</td><td>Prog.</td><td>ER Sent.</td><td>EE Sent.</td><td>EE Don. %</td></tr><tr><td>No RO</td><td>0.01 (± 0.24)</td><td>0.51 (± 0.03)</td><td>0.44 (± 0.06)</td><td>38% (± 6%)</td></tr><tr><td>2x2x3</td><td>0.69 (± 0.29)</td><td>0.62 (± 0.05)</td><td>0.49 (± 0.07)</td><td>45% (± 10%)†</td></tr><tr><td>3x3x5</td><td>0.95 (± 0.16)</td><td>0.66 (± 0.02)</td><td>0.52 (± 0.04)</td><td>47% (± 11%)</td></tr></table>

All 2x2x3 and 3x3x5 means significant at p < 0.01 (or 0.05 if marked †) when compared to No RO with two-tailed paired t-tests. ER and EE refer to the persuader and persuadee respectively.

## 4.3.3 Self-Play Results

For robustness we repeat this procedure 5 times with varying generation seeds for each persuader mode. In total, 1,995 self-play dialogues are completed (133 dialogues for each of 3 modes for each of 5 seeds). We average each metric across dialogues and then across runs, and report the aggregate means and standard deviations across runs. Additionally, to verify the benefit of rollouts, we perform two-tailed paired t-tests to confirm the differences in means between the rollout-enabled modes (2x2x3 and 3x3x5) and the baseline (No RO). For each metric, we average the results across runs and pair these averages from both modes for each dialogue. Results are shown in Table 3.

We observe that the mean progression increases significantly when rollouts are used. This is expected since response candidates with the highest average end-rollout progression are selected. We also observe that rollouts lead to higher average sentiment for both the persuader and persuadee, which makes sense due to the correlation between sentiment and the acceptability score (see Figure 5 in Appendix B). Finally, rollouts yield a higher percentage of dialogues with a pledged or intended donation. <sup>6</sup> All of these results are significant at the $p < 0 . 0 1$ level except for EE Don.% in 2x2x3 mode which is significant at $p < 0 . 0 5$

Although progression is noticeably higher for the 3x3x5 mode than for the 2x2x3 mode (0.95 vs 0.69), all other metrics are close between these modes with a small advantage in 3x3x5 mode. This suggests that scaling rollout simulations can be beneficial, but there may be diminishing returns for simulation size. Example self-play dialogues are provided in Tables 7, 8, and 9 in Appendix H.

## 5 Limitations & Future Direction

We recognize several limitations of our study which warrant follow-up investigation. This study focuses on a single task and dataset, and thus is subject to the assumptions and biases therein. Since we intend our framework to be general, it is prudent to perform additional studies to verify the efficacy of our methods on a variety of datasets spanning multiple dialogue domains and tasks. Also, although we provide automatic evaluation of the ability of rollouts to improve performance on a solicitation task, we cannot assume that humans would respond in the same way as DialoGPT. Thus, human evaluation is needed to further validate this approach.

## 6 Conclusion

In this work we introduced the concept of global dialogue state and proposed a framework with which a dialogue agent can gain awareness of where an ongoing conversation is headed, the likelihood of a successful outcome, and how its own response decisions impact the overall direction of the dialogue. We demonstrated that an unsupervised approach to modeling the GDS space and progression function is feasible, which is useful in sparsely-labeled settings. However, we showed that with domainadaptation pre-training for dialogue, supervised methods are preferable when labels are available. Finally, we demonstrated how using the PF as a feedback mechanism via dialogue rollouts allows an agent to improve outcomes on a solicitation task.

## Ethical Considerations

## Ethical Dialogue Systems

We acknowledge the potential risks inherent in the deployment of goal-oriented dialogue systems, and especially note that care must be taken to ensure persuasive dialogue systems are designed for beneficial use as discussed by Wang et al. (2019). Concretely, when applying our framework, care must be taken to ensure that the goal of the system (defined by the primary success attribute of the acceptability score) should be generally accepted as beneficial. For example, our basis for dialogue acceptability in this work is with respect to raising money for children’s charity. In general, the achievement of the system’s goal should not intentionally lead the user or any other party to harm. Additionally, the definition of acceptability, through its primary or any other correlated attributes, should not allow for discriminative responses, purposefully malicious discourse, or other violations of accepted ethical standards. For example, we include sentiment as secondary attributes in the acceptability score, which, when applied via dialogue rollouts, encourages the system to be courteous, polite, and respectful. It is possible with minimal effort to include further secondary attributes that identify bias, hate speech, and other indicators to help the system remain safe to use.

## Annotator Compensation

All manual annotators were recruited on a voluntary basis in an educational setting and did not receive or expect monetary compensation. Specifically, two graduate students and one postdoc in our lab served as our annotators.

## Environmental Impact

All training and inference in this work was done with two NVIDIA Quadro RTX 8000 GPUs. The most compute-intensive portion of the work was the additional domain adaptation pre-training for RoBERTa-large-adapted (see Section 3.2), which took approximately two weeks. After that the multiseed self-play evaluations took approximately four days, and all other operations (e.g., training and evaluating PF models, fine-tuning DialoGPT) took 24 hours or less.

## Acknowledgements

We would like to thank our manual annotators for their valuable contribution and the anonymous reviewers for their helpful feedback. This paper is based upon work supported in part by the United States Air Force under Contract No. FA8750-21- C-0075 and in part by the IBM Corporation under the Artificial Intelligence Research Collaboration Agreement No. W1771793 between IBM and Rensselaer. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of USAF or IBM Corporation.

## References

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.

Daniel Adiwardana, Minh-Thang Luong, David R So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, et al. 2020. Towards a human-like open-domain chatbot. arXiv preprint arXiv:2001.09977.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Vevake Balaraman, Seyedmostafa Sheikhalishahi, and Bernardo Magnini. 2021. Recent neural methods on dialogue state tracking for task-oriented dialogue systems: A survey. In Proceedings ofthe 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 239–251.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644–1650, Online. Association for Computational Linguistics.

S.A. Beebe and J.T. Masterson. 2000. Communicating in Small Groups: Principles and Practices. Longman.

F.J. Bernieri and R. Rosenthal. 1991. Interpersonal coordination: Behavior matching and interactional synchrony. In R.S. Feldman and B. Rime, editors, Fundamentals ofnonverbal behaviors. Studies in emotion and social interaction. Cambridge University Press.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018.´ MultiWOZ - a largescale multi-domain Wizard-of-Oz dataset for taskoriented dialogue modelling. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 5016–5026, Brussels, Belgium. Association for Computational Linguistics.

Ricardo JGB Campello, Davoud Moulavi, and Jörg Sander. 2013. Density-based clustering based on hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and data mining, pages 160–172. Springer.

Justine Cassell, Alastair Gill, and Paul Tepper. 2007. Coordination in conversation and rapport. In Proceedings of the Workshop on Embodied Language Processing, pages 41–50, Prague, Czech Republic. Association for Computational Linguistics.

Jan Cieciuch and Eldad Davidov. 2012. A comparison of the invariance properties of the pvq-40 and the pvq-21 to measure human values across german and polish samples. In Survey Research Methods, volume 6, pages 37–48.

Richard Csaky and Gábor Recski. 2021. The Gutenberg dialogue dataset. In Proceedings of the 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 138–159, Online. Association for Computational Linguistics.

Mihail Eric, Rahul Goel, Shachi Paul, Abhishek Sethi, Sanchit Agarwal, Shuyang Gao, Adarsh Kumar, Anuj Goyal, Peter Ku, and Dilek Hakkani-Tur. 2020. MultiWOZ 2.1: A consolidated multi-domain dialogue dataset with state corrections and state tracking baselines. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 422–428, Marseille, France. European Language Resources Association.

William Falcon and The PyTorch Lightning team. 2019. Pytorch lightning.

Ray Friedman. 2004. Studying negotiations in context: an ethnographic approach. Internat’l Negotiation.

Lewis R Goldberg. 1992. The development of markers for the big-five factor structure. Psychological assessment, 4(1):26.

Jesse Graham, Brian A Nosek, Jonathan Haidt, Ravi Iyer, Spassena Koleva, and Peter H Ditto. 2011. Mapping the moral domain. Journal of personality and social psychology, 101(2):366.

Dwayne D Gremler and Kevin P Gwinner. 2008. Rapport-building behaviors used by retail employees. Retailing.

Katherine Hamilton, Shin-I Shih, and Susan Mohammed. 2016. The development and validation of the rational and intuitive decision styles scale. Journal ofpersonality assessment, 98(5):523–535.

Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. 2020. Array programming with NumPy. Nature, 585(7825):357–362.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. 2018. Decoupling strategy and generation in negotiation dialogues. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2333–2343, Brussels, Belgium. Association for Computational Linguistics.

J. D. Hunter. 2007. Matplotlib: A 2d graphics environment. Computing in Science & Engineering, 9(3):90– 95.

Zhuoxuan Jiang, Xian-Ling Mao, Ziming Huang, Jie Ma, and Shaochun Li. 2019. Towards end-to-end learning for efficient dialogue agent by modeling looking-ahead ability. In Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue, pages 133–142, Stockholm, Sweden. Association for Computational Linguistics.

Wolf Langewitz, Matthias Nübling, and Heidemarie Weber. 2003. A theory-based approach to analysing conversation sequences. Epidemiologia e Psichiatria Sociale, 12(2):103–108.

Piyawat Lertvittayakumjorn, Daniele Bonadiman, and Saab Mansour. 2021. Knowledge-driven slot constraints for goal-oriented dialogue systems. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3407–3419, Online. Association for Computational Linguistics.

Mike Lewis, Denis Yarats, Yann Dauphin, Devi Parikh, and Dhruv Batra. 2017. Deal or no deal? end-toend learning of negotiation dialogues. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 2443–2453, Copenhagen, Denmark. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.

Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Leland McInnes and John Healy. 2017. Accelerated hierarchical density based clustering. In 2017 IEEE International Conference on Data Mining Workshops (ICDMW), pages 33–42. IEEE.

Leland McInnes, John Healy, and Steve Astels. 2017. hdbscan: Hierarchical density based clustering. The Journal of Open Source Software, 2(11):205.

Leland McInnes, John Healy, and James Melville. 2018a. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. 2018b. Umap: Uniform manifold approximation and projection. The Journal of Open Source Software, 3(29):861.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 8024–8035.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

plotly technologies inc. 2015. Collaborative data science.

Libo Qin, Fuxuan Wei, Tianbao Xie, Xiao Xu, Wanxiang Che, and Ting Liu. 2021. GL-GIN: Fast and accurate non-autoregressive model for joint multiple intent detection and slot filling. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 178–188, Online. Association for Computational Linguistics.

Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta, and Pranav Khaitan. 2020. Towards scalable multi-domain conversational agents:

The schema-guided dialogue dataset. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8689–8696.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, and Jason Weston. 2021. Recipes for building an open-domain chatbot. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 300–325, Online. Association for Computational Linguistics.

Tim Sainburg, Leland McInnes, and Timothy Q Gentner. 2020. Parametric umap: learning embeddings with deep neural networks for representation and semisupervised learning.

Ville Satopaa, Jeannie Albrecht, David Irwin, and Barath Raghavan. 2011. Finding a "kneedle" in a haystack: Detecting knee points in system behavior. In 2011 31st International Conference on Distributed Computing Systems Workshops, pages 166–171.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. Mpnet: Masked and permuted pretraining for language understanding. In Advances in Neural Information Processing Systems, volume 33, pages 16857–16867. Curran Associates, Inc.

the pandas development team. 2020. pandasdev/pandas: Pandas.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272.

Xuewei Wang, Weiyan Shi, Richard Kim, Yoojung Oh, Sijia Yang, Jingwen Zhang, and Zhou Yu. 2019. Persuasion for good: Towards a personalized persuasive dialogue system for social good. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5635–5649, Florence, Italy. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Jing Xu, Arthur Szlam, and Jason Weston. 2021. Beyond goldfish memory: Long-term open-domain conversation. arXiv preprint arXiv:2107.07567.

Denis Yarats and Mike Lewis. 2018. Hierarchical text generation and planning for strategic dialogue. In Proceedings ofthe 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings ofMachine Learning Research, pages 5587–5595. PMLR.

Xiaoxue Zang, Abhinav Rastogi, Srinivas Sunkara, Raghav Gupta, Jianguo Zhang, and Jindong Chen. 2020. MultiWOZ 2.2 : A dialogue dataset with additional annotation corrections and state tracking baselines. In Proceedings of the 2nd Workshop on Natural Language Processingfor Conversational AI, pages 109–117, Online. Association for Computational Linguistics.

Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2020. DIALOGPT : Large-scale generative pre-training for conversational response generation. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 270–278, Online. Association for Computational Linguistics.

## A Software Packages Used

Table 4: Software packages used in obtaining or presenting the results in this work
<table><tr><td>Package</td><td>Version</td><td>Citation</td><td>URL</td></tr><tr><td>hdbscan</td><td>0.8.27</td><td>(McInnes et al., 2017)</td><td>https://hdbscan.readthedocs.io/</td></tr><tr><td>kneed</td><td>0.7.0</td><td>(Satopaa et al., 2011)</td><td>https://kneed.readthedocs.io/</td></tr><tr><td>Matplotlib</td><td>3.3.4</td><td>(Hunter, 2007)</td><td>https://matplotlib.org/</td></tr><tr><td>NumPy</td><td>1.19.5</td><td>(Harris et al., 2020)</td><td>https://numpy.org/</td></tr><tr><td>Pandas</td><td>1.2.4</td><td>(the pandas development team, 2020)</td><td>https://pandas.pydata.org/</td></tr><tr><td>plotly</td><td>5.1.0</td><td>(plotly technologies inc., 2015)</td><td>https://plotly.com/python/</td></tr><tr><td>PyTorch</td><td>1.9.0</td><td>(Paszke et al., 2019)</td><td>https://pytorch.org/</td></tr><tr><td>PyTorch Lightning</td><td>1.5.8</td><td>(Falcon and team, 2019)</td><td>https://pytorchlightning.ai</td></tr><tr><td>scikit-learn</td><td>0.24.1</td><td>(Pedregosa et al., 2011)</td><td>https://scikit-learn.org/</td></tr><tr><td>SciPy</td><td>1.6.2</td><td>(Virtanen et al., 2020)</td><td>https://scipy.org/scipylib/index.html</td></tr><tr><td>Sentence-Transformers</td><td>N/A*</td><td>(Reimers and Gurevych, 2019)</td><td>https://sbert.net/</td></tr><tr><td>TensorFlow</td><td>2.5.1</td><td>(Abadi et al., 2015)</td><td>https://tensorflow.org/</td></tr><tr><td>Transformers</td><td>4.11.3</td><td>(Wolf et al., 2020)</td><td>https://huggingface.co/transformers/</td></tr><tr><td>umap-learn</td><td>0.5.1</td><td>(McInnes et al., 2018b)</td><td>https://umap-learn.readthedocs.io/</td></tr></table>

\* we use all-mpnet-base-v2 directly through Transformers, but it is part of the Sentence-Transformers model library. Additionally, we base parts of our sentence embedding implementation on that found in Sentence-Transformers.

## B Training Set Covariances For Acceptability Score

Covariances with respect to EE donation (equivalent to Pearson's r since values are standardized)  
![](images/16e3ad7da2329c95167528d0e40ff07794dd5b3b6821dfc69ef01ff670514039.jpg)  
Figure 5: The covariances of all other dialogue attributes with respect to the persuadee donation are used to weight the acceptability score. ER and EE refer to the persuader and persuadee respectively.

## C Full Manual Evaluation Results

Table 5: Progression Function Manual Eval Results (all annotators, averaged across all runs)
<table><tr><td>Model</td><td>utt (1/2/3)</td><td>utt-sl (1/2/3)</td><td>dlg-sl (1/2/3)</td><td>dlg-sl-f (1/2/3)</td></tr><tr><td>unsupervised</td><td>0.09 / 0.06 / 0.12</td><td>0.02 / 0.04 / 0.05</td><td>0.04 / -0.03 / -0.02</td><td>-0.07 / -0.09 / -0.05</td></tr><tr><td>RoBERTa-base</td><td>0.39±/0.30±/0.48</td><td>0.15†/0.17†/0.21‡</td><td>0.30 / 0.12 / 0.34</td><td>0.26 / 0.16 / 0.35</td></tr><tr><td>RoBERTa-large</td><td>0.39‡/0.30‡/0.50</td><td>0.16/0.18†/0.21†</td><td>0.41 / 0.17 / 0.46</td><td>0.36 / 0.21 / 0.47</td></tr><tr><td>RoBERTa-large-adapted</td><td>0.49‡/0.37‡/0.59</td><td>0.21‡/0.24‡/0.29</td><td>0.51 / 0.26 / 0.52</td><td>0.45 / 0.27 / 0.51</td></tr></table>

Average Pearson’s r p-value across runs: †: p < 0.05; ‡: p < 0.01; (two-tailed; H<sub>0</sub> is non-correlation).

## D Explanations of Manual Metrics

![](images/143874d8514f8aec3f3269c28051595f8f0b64a43f68bdea48c56fc2cefe51ff.jpg)

![](images/7add6e23e31c032563a3e50d4d3153271248c052a86110e15e21d5205bb49db2.jpg)

![](images/84049872d1c92858441da953e4dc9bef128dc3b786f5b07bdebf387e47567c65.jpg)  
Figure 6: utt: Pearson’s r (right) between utterance-level PF values (center, e.g., circled) and ground-truth values (left, e.g., circled) for all 244 utterances across 12 dialogues. Points shown on the right are from annotator 3. This metric is intended to measure if the PF and ground-truth progression curves assign similar values (relative to their respective scales) at each step of an ongoing dialogue.

![](images/ee971dd320bcf1a507c682d69cd24088dd4fd4f5b420e2916f210f8c77dd3852.jpg)

![](images/135d9776921ee3dff6677b71fe969b422bdfbe26a72dd40e5aa59ed300963b1b.jpg)

![](images/8f293078cdfbd6738d160368094a01b90e429e3ee8fa9bae4e65c670254717ff.jpg)  
Figure 7: utt-sl: Pearson’s r (right) between utterance-level PF slopes (center, e.g., see triangle) and ground-truth slopes (left, e.g., see triangle), for all 244 utterances across 12 dialogues. Utterance-level slopes are computed as the differences in the progression curves between two dialogue steps. Points shown on the right are from annotator 3. This metric is intended to measure if the PF and ground-truth progression curves move in the same direction at each step of an ongoing dialogue.

![](images/152f3d16eb26ab7d1a2e3c281be33ef0b633d00d4105493f4109950c0bc3c02f.jpg)

![](images/44933aad2610fd169dd43d5b1dc9d21362a4331226d5f95a35ca47c47a141599.jpg)

![](images/be406a855886e0e5a873b7515f83a09db95c6ca3b9b4ada2d7d3c9bce6afe8f4.jpg)  
Figure 8: dlg-sl: Pearson’s r (right) between dialogue-level PF slopes (center, e.g., see line) and ground-truth slopes (left, e.g., see line), for all 12 dialogues. Dialogue-level slopes are computed by fitting least-squares regression lines to the progression curves. Points shown on the right are from annotator 3. This metric is intended to measure the ability of the overall PF trend to approximate the ground-truth progression curve.

![](images/04af536b20ea5f028df038dfe2eec26d585b9154d4ac501620d89e4e971736ca.jpg)

![](images/6b9de1cf1ea8c9eb8fdec6aa4982d4012793e75783a69a4ff5d2243e008f4806.jpg)

![](images/38888ab74b9a4f8d9c5f0143895dbb3d0fdf859f16723bc4933e6aa979b05682.jpg)  
Figure 9: dlg-sl-f: Pearson’s r (right) between dialogue-level PF slopes (center, e.g., see line) and the final groundtruth progression value (left, e.g., circled), for all 12 dialogues. Dialogue-level slopes are computed by fitting least-squares regression lines to the progression curves. Points shown on the right are from annotator 3. This metric is intended to measure the ability of the overall PF trend to predict the end-task outcome.

## E PF Disagreement Examples

3. ER: It's an international organization that promotes children's rights, provides relief and helps support children in developing countries. does that sound like something you'd get behind?   
4. EE: No thank you.

5. ER: May I ask why? Your help could really make a difference in a child's life, think of a child who didn't have enough to eat, just a small bit of help may have saved them, are you sure you're not interested?

6. EE: I am already making a difference in many children lives.

7. ER: That's great, it sounds like it's an issue you really care about then, why not put some of that effort into a charity that's doing really great work like Save the Children?

8. EE: I have donated 15 years of my life to charity

9. ER: I'm not asking you to donate years of your life, just a couple of cents if possible, is that something you could possibly do tonight?

10. EE: No thank you

11. ER: You're probably on a computer right now, so you probably have quite a bit of money yourself, don't you think you could spare maybe 25 cents for some needy children around the worid?

12. EE: No thank you

13. ER: Do you have children yourself? What if your children were in a position these kids are in, wouldn't you want someone to help them?

14. EE: Where are those childrens parent?

15. ER: If only they had parental support, but in war-torn countries like Syria their parents are being killed in wars or are fighting for their country, they have no help and that's why they need your support, please find it in your heart to just give a little bit tonight, that's all I'm asking.

16. EE: My small change won't do a thing for those kids.What they need is to be extracted from those countries so that they can have a better life not 0.25.

17. ER: Well with our current president it's not likely they're going to be brought the US, so they need any help they can get, and 25 cents might not seem like a lot, but if 10000 people donated 25 cents then the impact will be huge. I understand it may not seem like a lot, but it can really make a difference. You can donate some or all of your payment from this task to Save the Children, i'd really appreciate it if you did, but I understand if you still don't want to. What do you say?

18. EE: If it makes you happy, go ahead and take .50

19. ER: Oh thank you so much, you have no idea how much that .50 will mean to those children. thank vou yery much and have a wonderful night you blessed soul.

1. ER: Hello, how are you today?

2. EE: I AM doing well how are you

3. ER: Wonderful! Thank you. Have you heard of a charity called Save the Children?

4. EE: I have not

5. ER: Save the Children is an international non-governmental organization that promotes children's rights. Would you like to know more?

6. EE: No thank vou. Most of those places have so much overhead not much actually goes to the cause

7. ER: Your individual donation will help individual children and their families

8. EE: I appreciate you telling me about this butI am not interested

9. ER: What if I were to tell you that 100% of your donation would go directly to helping one child?

10. EE: No thank you as there is no guarantee to this. l'd rather donate locally

11. ER: I agree donating locally is also important. However, there are so many children that could benefit from just pennies

12. EE: Again, no thank you

13. ER: It will go to a trustable fund. What if I match your donation?   
14. EE: I'm sorry I am to interested in donating to this.

15. ER: Your donation could be directly deducted from your task payment Anywhere from a penny to \$2   
16. EE: I will donate 0

17. ER: Just 77 cents could help a refugee family and I will match the donation.

18. EE: Again, I don't know how else I can say this. I will donate nothing to this. I prefer to do this locally

19. ER: Okay, then I have an option for you. You have the ability to sponsor at child in the US! We spend money on so many little things each day and we have the moral responsibility to help. So if you would like to help locally. Can you commit to doing that?

20. EE: No I am sorry again. I will only do this in my town

21. ER: This can go to sponsor a child in your town. Can you just commit to 10 cents?

20. EE: No problem have a good night

![](images/8e4320ab34e464f6fa8ac1786e643c9b26296e97cb37b79efad31e80fb8d31f0.jpg)

![](images/3ec4c807ab26954694fd471fc7c50e054c9f05dcf43bacfdc8496d33f26147ab.jpg)

![](images/65eec81d56334e7440aeca138c17bf91cb63e20f7c59b2c4e4236fa2fbf0c314.jpg)

![](images/8b6e366eef63e59a6ab398e75b192a800c92d770d40995e6075760d6ef3a5a1a.jpg)

![](images/23dcfd507807621bc647d080e1ed41e3cce10c181a10146873aabe8ab01151cf.jpg)

![](images/b1a57bd40c9d172c6a8377fa629790194d67ab8c4666045e0e5a0cc79f283d61.jpg)

![](images/e28629d464a2c83757381922551edc677ddf192dae542f0c5e0c344d284d50f2.jpg)

![](images/cea7494ae4a128f51b08f610145195d8c6c7dea4ea389fe82eab1ec8a2801a44.jpg)  
Figure 10: Two examples where the PF (RoBERTa-large-adapted) disagrees with multiple annotators. We note that the PF correctly decreases (Right) and stays negative (Left) given the turns boxed in red showing poor progression.

```csv
Algorithm 1: Grid search for hyperparameter tuning of the unsupervised progression model on
the validation set. Descriptions for each hyperparameter are provided in Table 6.
for β 0.0, 0.1, . . . , 2.0 do
for d 2, 16, 32, 64, 128, 768 do
for normalize_embeddings True, False do
for distance_metric Cosine, Euclidean do
▷ k-means experiments
for k 2, 3, . . . , 30 do
for inverse_distance True, False do
for standardized_proximity True, False do
measure_PF_slope_r();
▷ HDBSCAN experiments
for min_cluster_size 10, 20, . . . , 100 do
for soft_value_aggregation True, False do
for prob_scaling  None, softmax, sum do
for standardized_proximity True, False do
measure_PF_slope_r();
```

Table 6: Hyperparameter Descriptions  
Hyperparameter Description   
β (recency weight) Controls how much emphasis is placed on recent tokens when computing dialogue embeddings.   
d (embedding size) The dimensionality of dialogue embeddings. Values < 768 reduced with Parametric UMAP.   
normalize\_embeddings If True, embeddings are normalized to have unit magnitude after dimensionality reduction.   
distance\_metric The distance metric used by Parametric UMAP and centroid proximity calculations.\*   
k (number of clusters) Number of clusters to use for k-means.   
inverse\_distance If True, Euclidean centroid proximity is computed as the inverse distance instead of negative distance.\*\*   
standardized\_proximity If True, centroid proximities are converted to z-scores before progression is computed. †   
min\_cluster\_size Minimum number of points in a HDBSCAN cluster. Clusters with fewer points get merged into larger ones.   
soft\_value\_aggregation If True, HDBSCAN cluster attribute aggregations are weighted with cluster membership probabilities.   
prob\_scaling The type of scaling applied to progression computed from HDBSCAN cluster membership probabilities.  
†: Applies only if k-means is used, or if HDBSCAN is used with softmax probability scaling.

## G Final Unsupervised Model Hyperparameters

The final unsupervised model uses k-means (k = 21), β = 0.3, d = 768, normalized embeddings, euclidean distance, and inverse distance for centroid proximity.

## H Examples from rollout self-play experiment

The following examples in Tables 7, 8, and 9 compare self-play between the baseline No RO mode and the two rollout modes 2x2x3 and 3x3x5. In each example, the utterance where the persuadee (EE) states intent to donate is highlighted in red. The complete dialogues from the self-play experiments are included with our code release.

Table 7: Rollout Self-Play Example 1: No RO vs. 2x2x3 (dialogue #7; generation seed: 247)
<table><tr><td>CONTEXT:</td></tr><tr><td>ER: Hello, Have you ever donated to a charity? EE: A few times, yes. ER: ow was your experience? EE: It was ok. I didn&#x27;t really think much of it. Most of the time its just a dollar or two.</td></tr><tr><td>ER: Do you donate all te time? EE: Not so much. ER: Have yu heard of Save the children? It is pretty well known around the world.</td></tr><tr><td>EE: I think so. ER: What do you think that they do?</td></tr><tr><td>EE: Give money to child. Support efforts in war torn countries. Healthcare. All that stuff.</td></tr><tr><td></td></tr><tr><td>NO ROLLOUTS (progression: -2.65; ER sentiment: 0.69; EE sentiment: 0.03)</td></tr><tr><td></td></tr><tr><td>ER: I. It is a worthy cause. You can donate all you want from it.</td></tr><tr><td></td></tr><tr><td>EE: I&#x27;m sorry. I&#x27;m broke right now. That&#x27;s why I&#x27;m asking if you do donate all the time.</td></tr><tr><td>ER: It&#x27;s okay. I understand and agree. I think it&#x27;s a good cause.</td></tr><tr><td>EE: So what is the charity that you donate to?</td></tr><tr><td>ER: I don&#x27;t have a specific charity, but I do think they are a decent charity.</td></tr><tr><td>EE: Are they a non-profit? What do they do?</td></tr><tr><td></td></tr><tr><td>ER: It&#x27;s a non profit. They help protect childrens rights to health and education.</td></tr><tr><td>EE: Well, I think they are a great charity. Do they provide help in other ways besides donating to them? ER: Yes they help with children in the US and other developing countries.</td></tr><tr><td>EE: So I&#x27;m guessing the money goes towards programs to help those children get a proper education?</td></tr><tr><td></td></tr><tr><td>2x2x3 ROLLOUTS (progression: 0.15; ER sentiment: 0.67; EE sentiment: 0.63)</td></tr><tr><td></td></tr><tr><td>ER: I, they do. I have donated to the Red Cross and the Salvation Army in the past.</td></tr><tr><td>EE: I donated a couple of months ago. Do you donate often? I donate when I can.</td></tr><tr><td>ER: Yes, I donate to them as well. How much do you like to donate?</td></tr><tr><td>EE: i donate $.50 each month. Do you do the same?</td></tr><tr><td>ER: Yeah, that sounds great. Any bit helps though. Do you have an amount in mind?</td></tr><tr><td>EE: I would like to donate half. Is that a good idea?</td></tr><tr><td>ER: Yes that would be great. The children really do need it. I donate to them as well.</td></tr><tr><td>EE: I&#x27;m glad we can agree on a generous donation. How much would you like me to donate?</td></tr><tr><td>ER: I think half would be fine, I&#x27;m not sure, but that is a very generous amount.</td></tr><tr><td></td></tr><tr><td>EE: Well, it is nice to think that I&#x27;m not getting hit by a bot, so that&#x27;s a good thing.</td></tr></table>

Table 8: Rollout Self-Play Example 2: No RO vs. 3x3x5 (dialogue #88; generation seed: 756)
<table><tr><td>CONTEXT:</td></tr><tr><td>ER: Hello therehow are you?</td></tr><tr><td>EE: I am good. How are you?</td></tr><tr><td>ER: I&#x27;m pretty good, just trying to get ready for work soon.</td></tr><tr><td>EE: I understand that. So tell me about this charity.</td></tr><tr><td>ER: Have you heard of Save the Children? They do work around health, education, etc.</td></tr><tr><td>EE: I&#x27;m not sure that I have.</td></tr><tr><td>ER: They work mostly in developing countries. I would appreciate it if some of the funds could be donated. The researchers will pass that donation on to the charity at the end.</td></tr><tr><td>EE: What kind of work do they do?</td></tr><tr><td>ER: They do emergency hunger relief, are engaged with Syrian children.</td></tr><tr><td>EE: Ok, that sounds good. How do I know it is a legitimate charity?</td></tr><tr><td></td></tr><tr><td>NO ROLLOUTS (progression: -0.43; ER sentiment: 0.71; EE sentiment: 0.50)</td></tr><tr><td>ER: They are a good non-profit that helps children all around the world</td></tr><tr><td>EE: Ah, that&#x27;s good to know. Do you know what the funding for this charity actually goes towards?</td></tr><tr><td>ER: They actually make a huge impact. Do you donate to charity often?</td></tr><tr><td>EE: No, I don&#x27;t think so, but I will look them up.</td></tr><tr><td>ER: They are a 501c, so it&#x27;s not like they&#x27;re going to ask you to give anything at this time. Every little bit helps. EE: Cool, I don&#x27;t think I&#x27;m ready to give anything up yet. How about you?</td></tr><tr><td>ER: They are a legit charity that I donate to regularly. Have you donated to a charity before?</td></tr><tr><td>EE: No, I&#x27;m usually a pretty cautious individual with donations, but I think that would be a good idea.</td></tr><tr><td>ER: They&#x27;re great with giving to charities but I wouldn&#x27;t be comfortable donating more than the 10.00, but I usually donate my entire earnings</td></tr><tr><td>from this task to help out others.</td></tr><tr><td>EE: Cool, that&#x27;s good to know. Have a great day!</td></tr><tr><td>3x3x5 ROLLOUTS (progression: 0.36; ER sentiment: 0.65; EE sentiment: 0.70)</td></tr><tr><td>ER: Well, it&#x27;s not an international non-governmental organization, but they help children in war zones like Syria. Children are our future and</td></tr><tr><td>we need to help out where we can.</td></tr><tr><td>EE: Ah, I see! I will look into this. Does it have any good ratings on Charity Navigator?</td></tr><tr><td>ER: I have no idea, but I would be willing to look into them. They&#x27;re very trustworthy.</td></tr><tr><td>EE: They are very reputable. I will research them later and donate. Thanks!</td></tr><tr><td>ER: I&#x27;m glad to hear that. This is something I would like to be passionate about.</td></tr><tr><td>EE: Yes, I really love helping out! I think we need to take care of our country&#x27;s children. And if you would like to know more about a charity</td></tr><tr><td>you might like to research more: URL</td></tr><tr><td>ER: I would like to hear a little more about this charity from you.</td></tr><tr><td>EE: Well, I will look that up. This is one I really enjoy working on. I&#x27;ve heard of some of the kids who are in need, but this is one I&#x27;m very</td></tr><tr><td>passionate about. Do you have any other charities you&#x27;ve worked on?</td></tr><tr><td>ER: I&#x27;m going to look at some websites right now. It&#x27;s called Childrens International. You can research them. They are great. They promote the</td></tr><tr><td>rights of children in these situations, especially those that are separated from their parents or families. Have you donated to them?</td></tr><tr><td>EE: Not yet, but I think I will sometime in the future.</td></tr></table>

Table 9: Rollout Self-Play Example 3: No RO vs. 3x3x5 (dialogue #98; generation seed: 589)
<table><tr><td>CONTEXT:</td></tr><tr><td>ER: Hi, do you usually donate to charities EE: I DO</td></tr><tr><td>ER: great! You&#x27;re such a nice person. What charities do you usually donate to?</td></tr><tr><td>EE: To church, I return tithes and I also do the offerings</td></tr><tr><td>ER: great! Well, have you ever heard about Save the Children charity?</td></tr><tr><td>EE: no, can you tell me more about it?</td></tr><tr><td>ER: it&#x27;s an international non-governemntal organization that promotes children&#x27;s rights, provides relief and helps support children in developing countries</td></tr><tr><td>EE: nice! I have a child so i&#x27;m all about helping kids</td></tr><tr><td>ER: would you donate to this charity? your donation will make such a huge impact for some children</td></tr><tr><td>EE: I would like to know more before i donate. How do i know if it&#x27;s a legit organization</td></tr><tr><td>NO ROLLOUTS (progression: -2.44; ER sentiment: 0.10; EE sentiment: 0.41)</td></tr><tr><td>ER: if you would like to donate, you can visit their site: URL</td></tr><tr><td>EE: i dont want to donate, if i did however, how much would i donate?</td></tr><tr><td>ER: if you have to donate, you would have to submit a donation agreement, i dont think you can donate right now</td></tr><tr><td>EE: ok, how do i submit it?</td></tr><tr><td>ER: You can&#x27;t donate until you check out their website.</td></tr><tr><td>EE: ok thanks for telling me, i will check it out</td></tr><tr><td>ER: It is a charity that focuses on children in the United States</td></tr><tr><td>EE: ok thanks for the information, if i ever get to a child&#x27;s situation, i will donate</td></tr><tr><td>ER: You&#x27;re very welcome</td></tr><tr><td>EE: ok, have a nice day</td></tr><tr><td>3x3x5 ROLLOUTS (progression: 1.14; ER sentiment: 0.36; EE sentiment: 0.52)</td></tr><tr><td>ER: I will send you a copy of the actual donation and you can make a donation from there.</td></tr><tr><td></td></tr><tr><td>EE: How do i donate?</td></tr><tr><td>ER: It&#x27;s in your task payment. You can do that right now. EE: How do i donate?</td></tr><tr><td></td></tr><tr><td>ER: You don&#x27;t need to donate at this point, but I&#x27;ll give you a heads up on the website: URL</td></tr><tr><td>EE: Ok, i&#x27;ll go check it out now. Thank you very much!</td></tr><tr><td>ER: You&#x27;re welcome, but I have to ask, Would you be willing to donate $2, to the Save the Children charity?</td></tr><tr><td>EE: Ok. I will probably just donate $2.</td></tr><tr><td>ER: That is great. Thank you so much for your donation!</td></tr><tr><td>EE: You&#x27;re very welcome. Thank you, have a nice one!</td></tr><tr><td></td></tr></table>