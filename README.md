logreg
======

Parallel implementations of Logistic Regression.

Maven modules:

1) logreg-common: Common functionality regarding Machine Learning and Logistic Regression.

2) logreg-mapred: Hadoop Jobs for LogReg
- Hadoop job for Ensemble + SGD (mahout)
- Hadoop job for Batch Gradient descent
- Hadoop job for L-BFGS (uses Parallel Gradient-computation job)
- ...

3) logreg-ozone: Stratosphere Ozone Jobs for LogReg

4) aim3-logreg-rcv1: Flexible preprocessing for rcv1-v2 (base on cuttlefish)
- contains v2 patch
- flexible time based splits
- multiple tf-idf weighting options
- ...

Developed for DIMA group at TU Berlin www.dima.tu-berlin.de