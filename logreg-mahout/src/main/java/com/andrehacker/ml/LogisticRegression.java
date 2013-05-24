package com.andrehacker.ml;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

/**
 * TODO Feature: Cross-validation
 * TODO Feature: Stochastic GD
 * TODO Feature: Regularization L1
 */
public class LogisticRegression implements RegressionModel, ClassificationModel {
  
  private Vector w;
  
  public void trainNewton(Matrix data, Vector y, int maxIterations, double initialWeight, double penalty) {
    
    w = new DenseVector(data.numCols());
    w.assign(initialWeight);

    int rowCount = data.numRows();
    
    double penaltyDivN = penalty/rowCount;

    int it = 0;
    // Other termination criteria: small delta in weights or good quality reached
    while ((++it) <= maxIterations) {
      Vector batchGradient = new DenseVector(data.numCols());
      Matrix batchGradientSecond = new DenseMatrix(data.numCols(), data.numCols());
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        Vector xn = data.viewRow(n);
        Vector grad = computePartialGradient(xn, w, y.get(n));
        batchGradientSecond.assign(computeSecondPartialGradient(xn, w, y.get(n)), Functions.PLUS);
        batchGradient.assign(grad, Functions.PLUS);
      }
      // Add penalty to 1st derivation (using NG's derivation)
      Vector mask = MLUtils.ones(data.numCols()); // Avoid penalty on bias
      mask.set(0, 0);
      batchGradient.assign(w.times(mask).times(penaltyDivN), Functions.PLUS);
      
      // Add penalty to 2nd derivation
      batchGradientSecond = batchGradientSecond.plus(MLUtils.diag(mask).times(penaltyDivN));
      
      // Standard Newton Update
      batchGradient = MLUtils.inverse(batchGradientSecond).times(batchGradient);
      w.assign(batchGradient, Functions.MINUS);
    }
  }
  
  public Vector trainNewtonSFO(Matrix X, Vector y, Vector w, int maxIterations, double initialWeight, double penalty) {
    
    int rowCount = X.numRows();
    
    this.w = w; // TODO: Make this nicer
    
    double penaltyDivN = penalty/rowCount;
    
    int it = 0;
    while ((++it) <= maxIterations) {
      
      double batchGradient = 0d;
      double batchGradientSecond = 0d;
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        Vector xn = X.viewRow(n);
        double grad = computePartialGradientSFO(xn, w, y.get(n));
        batchGradientSecond += computeSecondPartialGradientSFO(xn, w, y.get(n));
        batchGradient += grad;
      }
      
      // Add penalty to 1st derivation (using NG's derivation)
      Vector mask = MLUtils.ones(X.numCols()); // Avoid penalty on bias
      mask.set(0, 0);
      batchGradient += penaltyDivN * w.getQuick(w.size()-1);
      
      // Add penalty to 2nd derivation
      batchGradientSecond += penaltyDivN;
      
      // Standard Newton Update
      batchGradient = batchGradient / batchGradientSecond;
      w.setQuick(w.size()-1, w.getQuick(w.size()-1) - batchGradient);
    }
    return w;
  }
  
  public void trainBatchGD(Matrix data, Vector y, int maxIterations, double learningRate, double initialWeight) {

    w = new DenseVector(data.numCols());
    w.assign(initialWeight);
    int rowCount = data.numRows();
    
    int it = 0;
    while ((++it) <= maxIterations) {
      
      Vector batchGradient = new DenseVector(data.numCols());
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        Vector grad = computePartialGradient(data.viewRow(n), w, y.get(n));
        batchGradient.assign(grad, Functions.PLUS);
      }
      // Weight update: w = w - 1/N * \gamma * grad
        w.assign(batchGradient.assign(Functions.MULT, learningRate / rowCount), Functions.MINUS);
    }
  }
  
  public double predict(Vector x, Vector w, boolean debug) {
    // Computes the prediction, using our current hypothesis (logistic function)
    if (debug) {
//      System.out.println("   - hyp: " + 1d / (1d + Math.exp( - x.dot(w))) );
    }
    // Overflow detection
    double xDotW = x.dot(w);
    double exp = Math.exp(xDotW);
    if (xDotW != 0 && (exp == 0 || Double.isInfinite(exp))) {
      System.out.println(" - OVERFLOW? " + xDotW + "\t" + exp);
    }
    return 1d / (1d + Math.exp( - x.dot(w)));
  }

  public int classify(Vector x, Vector w, boolean debug) {
    return (int) Math.round( predict(x, w, debug) );
  }

  private Vector computePartialGradient(Vector x, Vector w, double y) {
    // Compute the partial gradient of negative log-likelihood function regarding a single data point x
    // = ( h(x) - y) * x
    double diff = predict(x, w, false) - y;
    
    return x.times(diff);
  }

  private Matrix computeSecondPartialGradient(Vector x, Vector w, double y) {
    //Returns: x x^T h(x) (1-h(x))
    double predicted = predict(x, w, false);
    Matrix productOfx = MLUtils.vectorToColumnMatrix(x).times(MLUtils.vectorToRowMatrix(x));
    
    Matrix result = productOfx.times(predicted * (1 - predicted));
    return result;
  }
  
  /**
   * Convention: The new feature to be optimized is in last column
   */
  private double computePartialGradientSFO(Vector x, Vector w, double y) {
    // Compute the partial gradient of negative log-likelihood function
    // regarding a single data point x and a single feature/dimension d
    // = ( h(x) - y) * x_d
    double diff = predict(x, w, false) - y;
    
    return x.getQuick(x.size()-1) * diff;
  }
  
  /**
   * Convention: The new feature to be optimized is in last column
   */
  private double computeSecondPartialGradientSFO(Vector x, Vector w, double y) {
    //Returns: (x_d)^2 h(x) (1-h(x))
    double predicted = predict(x, w, false);
    double xdSquared = x.getQuick(x.size()-1);
    xdSquared = xdSquared * xdSquared;
    
    return xdSquared * predicted * (1 - predicted);
  }

  public Vector getWeight() {
    return w;
  }

}
