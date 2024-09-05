# Positive-unlabeled Learning using Pairwise Similarities and Parametric Minimum Cuts

*with* [Dorit S. Hochbaum](https://hochbaum.ieor.berkeley.edu/)</br>
*Accepted, to appear in* [KDIR 2024](https://kdir.scitevents.org/)
***

Positive-unlabeled (PU) learning is a binary classification problem where the labeled set contains only positive class samples. 

Most PU learning methods involve using a prior $\pi$ on the true fraction of positive samples. We propose here a method based on Hochbaum’s Normalized Cut (HNC), a network flow-based method, that partitions samples, both labeled and unlabeled, into two sets to achieve high intra-similarity and low inter-similarity, with a tradeoff parameter to balance these two goals. HNC is solved, for all tradeoff values, as a parametric minimum cut problem on an associated graph producing multiple optimal partitions, which are nested for increasing tradeoff values.

Our PU learning method, called *2-HNC*, runs in two stages.
Stage 1 identifies optimal data partitions for all tradeoff values, using only positive labeled samples.
Stage 2 first ranks unlabeled samples by their likelihood of being negative, according to the sequential order of partitions from stage 1, and then uses the likely-negative along with positive samples to run HNC.
Among all generated partitions in both stages, the partition whose positive fraction is closest to the prior $\pi$ is selected. An experimental study demonstrates that *2-HNC* is highly competitive compared to state-of-the-art methods.

***
The implementation of the parametric cut solver used in the original work can be found [here](https://riot.ieor.berkeley.edu/Applications/Pseudoflow/parametric.html).

The updated and faster implementation of the parametric cut solver (*bareHPF.c* in this directory) was developed recently by Alexander Irribarra Cortés (airribarra at inf.udec.cl) and Roberto Javier Asín Achá (roberto.asin at usm.cl).
***

Example:

To run the classifier on *mushroom* data and set the number of positive labeled samples to be 400, using the random seed of 0 to split the samples into labeled/unlabeled sets and the random seed of 1 in Stage 2 of the method to select reliable negative samples, run the following line:

python3 HNCPU.py -d mushroom -l 400 -S 0 -M 1;
