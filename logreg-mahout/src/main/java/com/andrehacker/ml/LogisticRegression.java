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
  CsvReader csv;
  
  public boolean trainBatchGD(String sampleFile, List<String> predictorNames, int maxIterations, double learningRate, double initialWeight) throws IOException {
    
    // Read data
    BufferedReader reader = new BufferedReader(MLUtils.open(sampleFile));
    csv = new CsvReader();
    int rows = 40;
    csv.csvNumericToDenseMatrix(reader, rows, "color", predictorNames, true);
    data = csv.getData();
    Vector y = csv.getY();
    int rowCount = data.numRows();
    
    // TODO: Adopt class labels
    y.assign(Functions.MINUS, 1);
    
    // Normalize data (std-dev 1 and mean 0)
    // Without normalizing, the dot-product (w*x) in the hypothesis could become too big
    normalizeInput();
    
    // Start training
    Vector w = new DenseVector(predictorNames.size()+1);
    w.assign(initialWeight);
    int it = 0;
    // Other possible termination criteria: small delta in weights or good quality reached
    while (it <= maxIterations) {
      
      Vector batchGradient = new DenseVector(data.numCols());
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        // Compute partial gradient
        Vector grad = computePartialGradient(data.viewRow(n), w, y.get(n));
//        System.out.println("  - " + grad);
        
        batchGradient.assign(grad, Functions.PLUS);
      }
//      System.out.println("bgd: " + batchGradient);
      
      // Apply gradient to weights: w = w - \gamma * grad * 1/N
      w.assign(batchGradient.assign(Functions.MULT, learningRate / rowCount), Functions.MINUS);

      ++it;
    }

    // Normalize w
    //w.assign(w.normalize());  // This is BAD!
//    System.out.println("Final weight: " + w.toString());
    printLinearModel(w);
    
    Matrix confusion = new DenseMatrix(2,2);
    System.out.println();
    System.out.println("Mean Deviation:\t" + Validation.computeMeanDeviation(data, y, w, this));
    System.out.println("Success-rate:\t" + Validation.computeSuccessRate(data, y, w, this, confusion));
    System.out.println();
    System.out.println("CONFUSION");
    System.out.println("True Pos:\t" + (int)confusion.get(0, 0));
    System.out.println("True Neg:\t" + (int)confusion.get(1, 1));
    System.out.println("False Pos:\t" + (int)confusion.get(0, 1));
    System.out.println("False Neg:\t" + (int)confusion.get(1, 0));
    
    return true;
  }
  
  private void printLinearModel(Vector w) {
    System.out.println("LEARNED MODEL");
    for (int i=0; i<w.size(); ++i) {
      System.out.println(" - " + csv.getColumnName(i) + "\t" + w.get(i));
    }
  }

  private Vector computePartialGradient(Vector x, Vector w, double y) {
    // Compute the partial gradient of single data point x
    // = ( h(x) - y) * x
    double diff = predict(x, w, true) - y;
    
    return x.times(diff);
  }
  
  public double predict(Vector x, Vector w, boolean debug) {
    // Computes the prediction, using our current hypothesis (logistic function)
    if (debug) {
//      System.out.println("   - hyp: " + 1d / (1d + Math.exp( - x.dot(w))) );
    }
    return 1d / (1d + Math.exp( - x.dot(w)));
  }

  public int classify(Vector x, Vector w, boolean debug) {
    return (int) Math.round( predict(x, w, debug) );
  }
  
  private void normalizeInput() {
    means = MLUtils.meanByColumns(data);
    ranges = MLUtils.rangeByColumns(data);
    // TODO: Handle the case where range is 0
    
    for (int col=1; col<data.numCols(); ++col) {
      Vector newCol = data.viewColumn(col).assign(Functions.MINUS, means.get(col)).divide(ranges.get(col));
      data.assignColumn(col, newCol);
    }
  }

}
