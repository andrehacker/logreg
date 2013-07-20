logreg
======

Parallel implementations of Logistic Regression.

Maven modules:

1) logreg-common: Common functionality regarding Machine Learning and Logistic Regression. Also contains some sequential implementations.

2) logreg-mapred: Hadoop Jobs for Logistic Regression
- Ensemble + SGD (mahout) Training
- Iterative Batch Gradient descent training
- Iterative L-BFGS training (uses Parallel Gradient-computation job)
- Forward Feature Selection using SFO (Single Feature optimization)

3) logreg-pact: Stratosphere Ozone Jobs for Logistic Regression

4) aim3-logreg-rcv1: Flexible preprocessing for rcv1-v2 (base on cuttlefish)
- Nice CLI
- contains v2 patch
- flexible time based splits
- multiple tf-idf weighting options
- ...

Developed for DIMA group at TU Berlin www.dima.tu-berlin.de