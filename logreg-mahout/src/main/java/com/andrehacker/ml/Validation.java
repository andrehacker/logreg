package com.andrehacker.ml;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

public class Validation {  
  
  Matrix confusion;
  double successRate;
  double meanDeviation;
  Matrix testData;
  Vector testY;
  
  public Validation(String testFile, List<String> predictorNames) {
    if (testFile != "") {
      // Read testdata
      BufferedReader reader;
      try {
        reader = new BufferedReader(MLUtils.open(testFile));
        CsvReader csvTest = new CsvReader();
        int rows = 40;
        csvTest.csvNumericToDenseMatrix(reader, rows, "color", predictorNames, true);
        testData = csvTest.getData();
        testY = csvTest.getY();
        csvTest.getY().assign(Functions.MINUS, 1);
        MLUtils.normalize(testData);
      } catch (IOException e) {
        e.printStackTrace();
      }
    } else {
      System.out.println("Error: No testfile specified!");
    }
  }

  public void computeMetrics(Vector w, RegressionModel regressionModel, ClassificationModel classificationModel) {
    // How many do we classify correctly?
    
    // TODO: Compute AUC
    confusion = new DenseMatrix(2,2);
    int truePos = 0;
    int trueNeg = 0;
    int falsePos = 0;
    int falseNeg = 0;
    double dev = 0;     // deviation
    for (int n=0; n<testData.numRows(); ++n) {
      //System.out.println(data.viewRow(n));
      double predictedClass = classificationModel.classify(testData.viewRow(n), w, true);
//      System.out.println("Is: " + prediction + " should: " + y.get(n));
      if (Math.round(predictedClass) == testY.get(n)) {
        if (testY.get(n) == 0)
          ++truePos;
        else
          ++trueNeg;
      } else {
        if (testY.get(n) == 0)
          ++falseNeg;
        else
          ++falsePos;
      }

      double prediction = regressionModel.predict(testData.viewRow(n), w, true);
      dev += Math.abs((testY.get(n) - prediction));
    }
    confusion.set(0, 0, truePos);
    confusion.set(0, 1, falseNeg);
    confusion.set(1, 0, falsePos);
    confusion.set(1, 1, trueNeg);
    successRate = ((double)truePos + trueNeg) / ((double)testData.numRows());
    meanDeviation = dev / testData.numRows();
  }
  
  public double computeMeanDeviation(Matrix data, Vector y, Vector w, RegressionModel model) {
    double dev = 0;
    for (int n=0; n<data.numRows(); ++n) {
      //System.out.println(data.viewRow(n));
      double prediction = model.predict(data.viewRow(n), w, true);
      dev += Math.abs((y.get(n) - prediction));
    
    //    System.out.println("Is: " + prediction + " should: " + y.get(n));
    }
    return dev / data.numRows();
  }

  public double computeSuccessRate(Matrix data, Vector y, Vector w, ClassificationModel model, Matrix confusionMatrix) {
    // How many do we classify correctly?
    // TODO: Compute AUC
    int truePos = 0;
    int trueNeg = 0;
    int falsePos = 0;
    int falseNeg = 0;
    for (int n=0; n<data.numRows(); ++n) {
      //System.out.println(data.viewRow(n));
      double prediction = model.classify(data.viewRow(n), w, true);
  //    System.out.println("Is: " + prediction + " should: " + y.get(n));
      if (Math.round(prediction) == y.get(n)) {
        if (y.get(n) == 0)
          ++truePos;
        else
          ++trueNeg;
      } else {
        if (y.get(n) == 0)
          ++falseNeg;
        else
          ++falsePos;
      }
    }
    confusionMatrix.set(0, 0, truePos);
    confusionMatrix.set(0, 1, falseNeg);
    confusionMatrix.set(1, 0, falsePos);
    confusionMatrix.set(1, 1, trueNeg);
    return ((double)truePos + trueNeg) / ((double)data.numRows());
  }
  
  public Matrix getConfusion() {
    return confusion;
  }
  
  public double getMeanDeviation() {
    return meanDeviation;
  }
  
  public double getSuccessRate() {
    return successRate;
  }

}
