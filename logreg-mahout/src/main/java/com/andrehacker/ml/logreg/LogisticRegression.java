package com.andrehacker.ml.logreg;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import com.andrehacker.ml.ClassificationModel;
import com.andrehacker.ml.RegressionModel;
import com.andrehacker.ml.util.MLUtils;

/**
 * TODO Feature: Cross-validation
 * TODO Feature: Stochastic GD
 * TODO Feature: Regularization L1 (see mahout sgd or paper with differential L1 approximation)
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
    
    this.w = w; // TODO Refactoring: Make this nicer
    
    double penaltyDivN = penalty/rowCount;
    
    int it = 0;
    while ((++it) <= maxIterations) {
      
      double batchGradient = 0;
      double batchGradientSecond = 0;
      double update = 0;
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        Vector xn = X.viewRow(n);
        double grad = computePartialGradientSFO(xn, w, y.get(n));
        batchGradientSecond += computeSecondPartialGradientSFO(xn, w, y.get(n));
        batchGradient += grad;
      }
      
      // Add penalty to 1st derivation (using NG's derivation)
//      Vector mask = MLUtils.ones(X.numCols()); // Avoid penalty on bias
//      mask.set(0, 0);
      batchGradient += penaltyDivN * w.getQuick(w.size()-1);
      
      // Add penalty to 2nd derivation
      batchGradientSecond += penaltyDivN;
      
      // Standard Newton Update
      update = batchGradient / batchGradientSecond;
      w.setQuick(w.size()-1, w.getQuick(w.size()-1) - update);
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
  
  @Override
  public double predict(Vector x, Vector w) {
    // Computes the prediction, using our current hypothesis (logistic function)
    // Overflow detection
    double xDotW = x.dot(w);
    double negativeExp = Math.exp(-xDotW);
    if (xDotW != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + xDotW + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }

  @Override
  public double predict(Vector x, Vector w, double intercept) {
    // Computes the prediction, using our current hypothesis (logistic function)
    // Overflow detection
    double xDotW = x.dot(w) + intercept;
    double negativeExp = Math.exp(-xDotW);
    if (xDotW != 0 && (negativeExp == 0 || Double.isInfinite(negativeExp))) {
      System.out.println(" - OVERFLOW? " + xDotW + "\t" + negativeExp);
    }
    return 1d / (1d + negativeExp);
  }

  @Override
  public int classify(Vector x, Vector w) {
    return (int) Math.round( predict(x, w) );
  }

  private Vector computePartialGradient(Vector x, Vector w, double y) {
    // Compute the partial gradient of negative log-likelihood function regarding a single data point x
    // = ( h(x) - y) * x
    double diff = predict(x, w) - y;
    
    return x.times(diff);
  }

  private Matrix computeSecondPartialGradient(Vector x, Vector w, double y) {
    //Returns: x x^T h(x) (1-h(x))
    double predicted = predict(x, w);
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
    double diff = predict(x, w) - y;
    
    return x.getQuick(x.size()-1) * diff;
  }
  
  /**
   * Convention: The new feature to be optimized is in last column
   */
  private double computeSecondPartialGradientSFO(Vector x, Vector w, double y) {
    //Returns: (x_d)^2 h(x) (1-h(x))
    double predicted = predict(x, w);
    double xdSquared = x.getQuick(x.size()-1);
    xdSquared = xdSquared * xdSquared;
    
    return xdSquared * predicted * (1 - predicted);
  }

  public Vector getWeight() {
    return w;
  }

}
