package com.andrehacker.ml;

import org.apache.mahout.math.Vector;

public interface RegressionModel {
  
  /**
   * Predict the outcome for input x using model described by w
   * 
   * Either does not use bias/interception term or x and w both have a dimension for bias included
   */
  double predict(Vector x, Vector w);
  
  /**
   * Predict the outcome for input x using model described by w
   * 
   * Uses an explicite bias/interception term 
   */
  double predict(Vector x, Vector w, double intercept);

}
