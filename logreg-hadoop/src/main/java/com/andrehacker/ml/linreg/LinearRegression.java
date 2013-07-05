package com.andrehacker.ml.linreg;

import java.io.BufferedReader;
import java.util.List;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.andrehacker.ml.ClassificationModel;
import com.andrehacker.ml.RegressionModel;
import com.andrehacker.ml.util.CsvReader;
import com.andrehacker.ml.util.MLUtils;
import com.andrehacker.ml.validation.Validation;

public class LinearRegression implements RegressionModel, ClassificationModel {
  
  Vector w = null;
  
  public void train(String inputFile, List<String> predictorNames) throws Exception {
    BufferedReader reader = new BufferedReader(MLUtils.open(inputFile));
    
    // Read numeric csv into dense matrix
    CsvReader csv = new CsvReader();
    int rows = 40;
    csv.numericToDenseMatrix(reader, rows, "color", predictorNames, true);
    Matrix data = csv.getData();
    
    // Least squares solution:
    w = MLUtils.pseudoInversebySVD(data).times(csv.getY());
    System.out.println("Learned weigths: " + w);

    Validation val = new Validation();
    val.computeMeanDeviation(data, csv.getY(), this);
    val.computeAccuracy(data, csv.getY(), this);
    System.out.println("Mean Deviation: " + val.getMeanDeviation());
    System.out.println("Success-rate: " + val.getAccuracy());
  }
  
  public double predict(Vector x) {
    return x.dot(w);
  }
  
  public double predict(Vector x, double intercept) {
    return x.dot(w) + intercept;
  }

  public int classify(Vector x) {
    return (int) Math.round( predict(x) );
  }
  
}
