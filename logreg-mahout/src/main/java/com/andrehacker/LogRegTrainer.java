package com.andrehacker;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import com.google.common.collect.Lists;

public class LogRegTrainer implements RegressionModel, ClassificationModel {
  
  /**
   * 
   * 
   * Other possible termination criteria: small delta in weights or good quality reached
   * 
   * @param maxIterations
   * @param learningRate
   * @param initialWeight
   * @return
   * @throws IOException 
   */
  public boolean trainBatchGD(String sampleFile, List<String> predictorNames, int maxIterations, double learningRate, double initialWeight) throws IOException {
    
    // Read data
    BufferedReader reader = new BufferedReader(IoUtils.open(sampleFile));
    CsvReader csv = new CsvReader();
    int rows = 40;
    csv.csvNumericToDenseMatrix(reader, rows, "color", predictorNames, true);
    Matrix data = csv.getData();
    Vector y = csv.getY();
    int rowCount = data.numRows();
    
    // Start training
    Vector w = new DenseVector(predictorNames.size()+1);
    w.assign(initialWeight);
    int it = 0;
    while (it <= maxIterations) {
      
      Vector batchGradient = new DenseVector(data.numCols());
      
      // Batch GD: Iterate over all x
      for (int n=0; n<rowCount; ++n) {
        // Compute partial gradient
        Vector grad = computePartialGradient(data.viewRow(n), w, y.get(n));
        
        batchGradient.assign(grad, Functions.PLUS);
      }
      
      // Apply gradient to weights: w = w - \gamma * grad * 1/N
      w.assign(batchGradient.assign(Functions.MULT, learningRate / rowCount), Functions.MINUS);

      System.out.println(w.toString());
      ++it;
    }

    // Normalize w
    w.assign(w.normalize());
    System.out.println(w.toString());
    
    System.out.println( Validation.computeSuccessRate(data, y, w, this) );
    
    return true;
  }
  
  private Vector computePartialGradient(Vector x, Vector w, double y) {
    
    // Compute the partial gradient of single data point x
    // = ( h(x) - y) * x
    double diff = hypothesis(x, w, false) - y;
    
    return x.times(diff);
  }
  
  public double hypothesis(Vector x, Vector w, boolean debug) {
    // Computes the prediction, using our current hypothesis (logistic function)
    if (debug) {
//      System.out.println(Math.exp( -1d * x.dot(w)));
    }
    return 1d / (1d + Math.exp( -1d * x.dot(w)));
  }

  public int classify(Vector x, Vector w, boolean debug) {
    return (int) Math.round( hypothesis(x, w, debug) );
  }

}
