package com.andrehacker.ml.logreg;

import org.apache.mahout.math.Vector;

/**
 * Static functions for Logistic Regression
 */
public class LogRegMath {
  
  private LogRegMath() { }

  public static double predict(Vector x, Vector w) {
    return predict(x, w, 0);
  }

  public static double predict(Vector x, Vector w, double intercept) {
    // Computes the prediction, using our current hypothesis (logistic function)
    // Overflow detection
    double xDotW = x.dot(w) + intercept;
    double negativeExp = Math.exp(-xDotW);
    if (xDotW != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + xDotW + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }
  
  public static int classify(Vector x, Vector w, double intercept) {
    double prediction = predict(x, w, intercept);
    return (int) Math.floor(prediction + 0.5d);
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
  
  public static double logisticFunction(double exponent) {
    // Computes the prediction, using our current hypothesis (logistic function)
    // Overflow detection
    double negativeExp = Math.exp(-exponent);
    if (exponent != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + exponent + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }

}
