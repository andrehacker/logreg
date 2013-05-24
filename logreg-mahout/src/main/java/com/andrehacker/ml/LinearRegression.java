package com.andrehacker.ml;

import java.io.BufferedReader;
import java.util.List;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class LinearRegression implements RegressionModel, ClassificationModel {
  
  public void train(String inputFile, List<String> predictorNames) throws Exception {
    BufferedReader reader = new BufferedReader(MLUtils.open(inputFile));
    
    // Read numeric csv into dense matrix
    CsvReader csv = new CsvReader();
    int rows = 40;
    csv.numericToDenseMatrix(reader, rows, "color", predictorNames, true);
    Matrix data = csv.getData();
    
    // Least squares solution:
    Vector w = MLUtils.pseudoInversebySVD(data).times(csv.getY());
    System.out.println("Learned weigths: " + w);

    Validation val = new Validation();
    val.computeMeanDeviation(data, csv.getY(), w, this);
    val.computeSuccessRate(data, csv.getY(), w, this);
    System.out.println("Mean Deviation: " + val.getMeanDeviation());
    System.out.println("Success-rate: " + val.getSuccessRate());
  }
  
  public double predict(Vector x, Vector w) {
    return x.dot(w);
  }

  public int classify(Vector x, Vector w) {
    return (int) Math.round( predict(x, w) );
  }
  
}
