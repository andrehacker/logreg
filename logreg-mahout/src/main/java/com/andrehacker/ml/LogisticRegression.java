package com.andrehacker.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

public class LogisticRegression implements RegressionModel, ClassificationModel {
  
  private Vector means;
  private Vector ranges;
  private Matrix data;
  private Vector w;
  private CsvReader csv;
  
  /**
   * TODO Regularization (L1 / L2 ?)
   * TODO Validation on Testdata
   * TODO Cross-validation
   * TODO Stochastic GD
   * TODO SFO
   * 
   * @param sampleFile
   * @param predictorNames
   * @throws IOException
   */
  public LogisticRegression(String sampleFile, List<String> predictorNames) throws IOException {

    // Read data into matrix
    BufferedReader reader = new BufferedReader(MLUtils.open(sampleFile));
    csv = new CsvReader();
    int rows = 40;
    csv.csvNumericToDenseMatrix(reader, rows, "color", predictorNames, true);
    data = csv.getData();
    
    // TODO: Adopt class labels
    csv.getY().assign(Functions.MINUS, 1);
    
    // Normalize data (std-dev 1 and mean 0)
    // Without normalizing, the dot-product (w*x) in the hypothesis could become too big
    normalize(data);
    
    // Empty model
    w = new DenseVector(predictorNames.size()+1);
  }
  
  public boolean trainNewton(int maxIterations, double initialWeight, double penalty) {
    return train(maxIterations, 1, initialWeight, true, penalty);
  }
  
  public boolean trainBatchGD(int maxIterations, double learningRate, double initialWeight) {
    return train(maxIterations, learningRate, initialWeight, false, 0);
  }

  private boolean train(int maxIterations, double learningRate, double initialWeight, boolean useNewton, double penalty) {

    int rowCount = data.numRows();
    Vector y = csv.getY();
    
    double penaltyDivN = penalty/rowCount;
    
    w.assign(initialWeight);
    int it = 1;
    // Other termination criteria: small delta in weights or good quality reached
    while (it <= maxIterations) {
      
      Vector batchGradient = new DenseVector(data.numCols());
      Matrix batchGradientSecond = new DenseMatrix(data.numCols(), data.numCols());
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        Vector grad = computePartialGradient(data.viewRow(n), w, y.get(n));
        if (useNewton) {
          batchGradientSecond.assign(computeSecondPartialGradient(data.viewRow(n), w, y.get(n)), Functions.PLUS);
        }
        batchGradient.assign(grad, Functions.PLUS);
      }
      if (useNewton) {
        // Add penalty to 1st derivation (using NG's derivation)
        Vector mask = MLUtils.ones(data.numCols()); // Avoid penalty on bias
        mask.set(0, 0);
        batchGradient.assign(w.times(mask).times(penaltyDivN), Functions.PLUS);
        // Add penalty to 2nd derivation
        batchGradientSecond = batchGradientSecond.plus(MLUtils.diag(mask).times(penaltyDivN));
        batchGradient = MLUtils.inverse(batchGradientSecond).times(batchGradient);
        w.assign(batchGradient, Functions.MINUS);
      } else {
        // Weight update: w = w - 1/N * \gamma * grad
        w.assign(batchGradient.assign(Functions.MULT, learningRate / rowCount), Functions.MINUS);
      }

      ++it;
    }
    
    return true;
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
  
  public void printLinearModel(Vector w) {
    System.out.println("LEARNED MODEL");
    for (int i=0; i<w.size(); ++i) {
      System.out.println(" - " + csv.getColumnName(i) + "\t" + w.get(i));
    }
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
  
  private void normalize(Matrix m) {
    means = MLUtils.meanByColumns(m);
    ranges = MLUtils.rangeByColumns(m);
    // TODO: Handle the case where range is 0
    
    for (int col=1; col<m.numCols(); ++col) {
      Vector newCol = m.viewColumn(col).assign(Functions.MINUS, means.get(col)).divide(ranges.get(col));
      m.assignColumn(col, newCol);
    }
  }

  public Vector getWeight() {
    return w;
  }

}
