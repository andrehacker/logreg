package com.andrehacker.ml.logreg;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.andrehacker.ml.util.MLUtils;

/**
 * Static methods for Logistic Regression
 * 
 * TODO Feature: L2-Regularization (simple)
 * TODO Feature: L1-Regularization (see mahout sgd or paper with differential L1 approximation)
 */
public class LogRegMath {
  
  private LogRegMath() { }

  public static double predict(Vector x, Vector w) {
    return logisticFunction(x.dot(w));
  }
  
  public static double predict(Vector x, Vector w, double intercept) {
    return logisticFunction(x.dot(w) + intercept);
  }
  
  /**
   * Classifies x using the given threshold (usually 0.5)
   * @return 1 = positive class, 0 = negative class
   */
  public static int classify(Vector x, Vector w, double intercept, double threshold) {
    double prediction = predict(x, w, intercept);
    return (int) Math.floor(prediction + threshold);
  }
  
  /**
   * Computes the prediction, using our current hypothesis (logistic function)
   * Detects overflow
   */
  public static double logisticFunction(double exp) {
    double negativeExpResult = Math.exp(-exp);
    if (exp != 0 && (negativeExpResult == 0 || Double.isInfinite(negativeExpResult))) {
      System.out.println(" - OVERFLOW? " + exp + "\t" + negativeExpResult);
    }
    return 1d / (1d + negativeExpResult);
  }

  /**
   * Computes the log-likelihood for a single data point
   * Uses the natural logarithm
   * 
   * See Hastie p.120 (equation 4.20) for explanation
   */
  public static double logLikelihood(double actual, double prediction) {
    return (actual * Math.log(prediction)) + ((1 - actual) * (1 - Math.log(prediction)));
  }

  public static double computeSqError(Vector x, Vector w, double y) {
    return Math.pow(predict(x, w) - y, 2);
  }

  /**
   * Compute the partial gradient of negative log-likelihood function regarding a single data point x
   * = ( h(x) - y) * x
   */
  public static Vector computePartialGradient(Vector x, Vector w, double y) {
    return x.times(predict(x, w) - y);
  }

  public static Matrix computeSecondPartialGradient(Vector x, Vector w, double y) {
    //Returns: x x^T h(x) (1-h(x))
    double predicted = LogRegMath.predict(x, w);
    Matrix productOfx = MLUtils.vectorToColumnMatrix(x).times(MLUtils.vectorToRowMatrix(x));
    
    Matrix result = productOfx.times(predicted * (1 - predicted));
    return result;
  }

}
