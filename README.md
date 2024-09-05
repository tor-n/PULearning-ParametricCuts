# PULearning-ParametricCuts
**Positive-unlabeled Learning using Pairwise Similarities and Parametric Minimum Cuts**

Positive-unlabeled (PU) learning is a binary classification problem where the labeled set contains only positive class samples. 

Most PU learning methods involve using a prior $\pi$ on the true fraction of positive samples. We propose here a method based on Hochbaum’s Normalized Cut (HNC), a network flow-based method, that partitions samples, both labeled and unlabeled, into two sets to achieve high intra-similarity and low inter-similarity, with a tradeoff parameter to balance these two goals. HNC is solved, for all tradeoff values, as a parametric minimum cut problem on an associated graph producing multiple optimal partitions, which are nested for increasing tradeoff values.

Our PU learning method, called {\em 2-HNC}, runs in two stages.
Stage 1 identifies optimal data partitions for all tradeoff values, using only positive labeled samples.
Stage 2 first ranks unlabeled samples by their likelihood of being negative, according to the sequential order of partitions from stage 1, and then uses the likely-negative along with positive samples to run HNC.
Among all generated partitions in both stages, the partition whose positive fraction is closest to the prior $\pi$ is selected. An experimental study demonstrates that {\em 2-HNC} is highly competitive compared to state-of-the-art methods.
