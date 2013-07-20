package de.tuberlin.dima.ml.logreg;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.util.MLUtils;

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

  public static int classify(Vector x, Vector w) {
    return classify(x, w, 0, 0.5);
  }
  
  /**
   * Computes the prediction, using our current hypothesis (logistic function)
   * 
   * Handles numerical issues
   * 
   * - Underflows: The real result is something very close to zero, but due to
   * the lack of precision we get 0.0. In the logistic function this does not
   * cause any problems so we just ignore it
   * 
   * - Overflows: We should get a very big number, but due to the lack of
   * precision we get Double.POSITIVE_INFINITY instead. In the logistic function
   * this does not cause any problems, because java will actually calculate with
   * INFINITY as we would expect it.
   * 
   * - NaN: This occurs only when dividing by zero, which is not the case here
   */
  public static double logisticFunction(double exp) {
    double negativeExpResult = Math.exp(-exp);
    // Overflow detection
    if (Double.isInfinite(negativeExpResult) || Double.isNaN(negativeExpResult)) {
      System.out.println("OVERFLOW (ignored): exp=" + exp + " e^(-exp)=" + negativeExpResult);
      // System.out.println(" result: " + 1d / (1d + negativeExpResult));
    }
    // Underflow detection (no problem in most cases, because just a lack of
    // precision)
    if (exp != 0 && negativeExpResult == 0) {
      System.out.println("UNDERFLOW (ignored): exp=" + exp + " e^(-exp)=" + negativeExpResult);
      // System.out.println(" result: " + 1d / (1d + negativeExpResult));
    }
    return 1d / (1d + negativeExpResult);
  }

  /**
   * Computes the log-likelihood for a single data point
   * Uses the natural logarithm
   * 
   * TODO Handle special numerical cases: 0.0, INFINITY, NaN
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
