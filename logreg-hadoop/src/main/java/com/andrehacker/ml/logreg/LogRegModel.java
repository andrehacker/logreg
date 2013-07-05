package com.andrehacker.ml.logreg;

import org.apache.mahout.math.Vector;

import com.andrehacker.ml.ClassificationModel;
import com.andrehacker.ml.RegressionModel;

/**
 * Implements a simple model for Logistic Regression
 * It is defined by a weight (and an interception term) 
 * 
 * TODO Feature: Regularization L1 (see mahout sgd or paper with differential L1 approximation)
 */
public class LogRegModel implements RegressionModel, ClassificationModel {
  
  private Vector w;
  
  public LogRegModel(Vector w) {
    this.w = w;
  }
  
  @Override
  public double predict(Vector x) {
    return LogRegMath.predict(x, w);
  }

  @Override
  public double predict(Vector x, double intercept) {
    return LogRegMath.predict(x, w, intercept);
  }

  @Override
  public int classify(Vector x) {
    return LogRegMath.classify(x, w, 0);
  }
  
  public void setW(Vector w) {
    this.w = w;
  }
  
  public Vector getW() {
    return w;
  }

}
