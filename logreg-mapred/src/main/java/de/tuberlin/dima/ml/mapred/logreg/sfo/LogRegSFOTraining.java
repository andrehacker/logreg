package de.tuberlin.dima.ml.mapred.logreg.sfo;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import de.tuberlin.dima.ml.logreg.LogRegMath;

public class LogRegSFOTraining {
  
  public static Vector trainNewtonSFO(Matrix X, Vector y, Vector w, int maxIterations, double initialWeight, double penalty, boolean debug) {
    
    int rowCount = X.numRows();
    
//    this.w = w; // TODO Refactoring: Make this nicer
    
    double penaltyDivN = penalty/rowCount;
    
    int it = 0;
    while ((++it) <= maxIterations) {
      
      double batchGradient = 0;
      double batchGradientSecond = 0;
      double update = 0;
      double pi=0;  // Current prediction
      // Batch GD: Iterate over all x
      double debugSumPi=0;
      for (int i=0; i<rowCount; ++i) {
        Vector xi = X.viewRow(i);
        //batchGradient += computePartialGradientSFO(xn, w, y.get(n));
        pi = LogRegMath.predict(xi, w);
        debugSumPi += pi;
        batchGradient += computePartialGradientSFO(xi.getQuick(xi.size()-1), pi, y.get(i));
        batchGradientSecond += computeSecondPartialGradientSFO(xi.getQuick(xi.size()-1), pi);
      }
      
      // Apply Regularization to 1st derivation
      // using NG's derivation from http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
      // equal to Singhs definition, except the division by n
      // Apply only to the the dimension under training
      // TODO Avoid penalty on bias and apply real formula (also normalizes before)
      batchGradient += penaltyDivN * w.getQuick(w.size()-1);
      
      // Apply regularization to 2nd derivation
      batchGradientSecond += penaltyDivN;
      
      // Standard Newton Update
      update = batchGradient / batchGradientSecond;
      w.setQuick(w.size()-1, w.getQuick(w.size()-1) - update);
      
      if (debug) {
        System.out.println("- it " + it + ": grad: " + batchGradient + " gradSecond: " + batchGradientSecond + " new betad: " + w.getQuick(w.size()-1) + " sumPi: " + debugSumPi);
      }
    }
    return w;
  }

  /**
   * Convention: The new feature to be optimized is in last column
   */
  public static double computePartialGradientSFO(double xid, double pi, double y) {
    // Compute the partial gradient of negative log-likelihood function
    // regarding a single data point x and a single feature/dimension d
    // = ( h(x) - y) * x_d
    return (pi - y) * xid;
  }
  
  /**
   * Convention: The new feature to be optimized is in last column
   */
  public static double computeSecondPartialGradientSFO(double xid, double pi) {
    //Returns: (x_d)^2 h(x) (1-h(x))
    double xidSquared = Math.pow(xid, 2);
    
    return xidSquared * pi * (1 - pi);
  }

}
