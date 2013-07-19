logreg
======

Parallel implementations of Logistic Regression.

Maven modules:

1) logreg: Common functionality regarding Machine Learning and Logistic Regression. Includes Hadoop jobs for LogReg
- Hadoop job for Ensemble + SGD (mahout)
- Hadoop job for Batch Gradient descent
- Hadoop job for L-BFGS (uses Parallel Gradient-computation job)

2) logreg-ozone: Jobs regarding Logistic regression for Stratosphere Ozone

3) aim3-logreg-rcv1: Flexible preprocessing for rcv1-v2 (base on cuttlefish)
- contains v2 patch
- flexible time based splits
- multiple tf-idf weighting options
- ...

Developed for DIMA group at TU Berlin www.dima.tu-berlin.de