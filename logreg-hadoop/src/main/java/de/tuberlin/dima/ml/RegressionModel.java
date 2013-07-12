package de.tuberlin.dima.ml;

import org.apache.mahout.math.Vector;

public interface RegressionModel {

  /**
   * Predict the outcome for input x
   * 
   * Either does not use bias/interception term or x and w both have a dimension for bias included
   */
  double predict(Vector x);

  /**
   * Predict the outcome for input x
   * 
   * Uses an explicite bias/interception term 
   */
  double predict(Vector x, double intercept);

}
