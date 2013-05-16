package com.andrehacker;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class Validation {  
  
  public static double computeMeanDeviation(Matrix data, Vector y, Vector w, RegressionModel model) {
  // How many do we classify correctly?
  // TODO: Compute AUC
  double dev = 0;
  for (int n=0; n<data.numRows(); ++n) {
    //System.out.println(data.viewRow(n));
    double prediction = model.hypothesis(data.viewRow(n), w, true);
    //System.out.println("Is: " + prediction + " should: " + y.get(n));
    dev += Math.abs((y.get(n) - prediction));
  }
  return dev / data.numRows();
}

public static double computeSuccessRate(Matrix data, Vector y, Vector w, ClassificationModel model) {
  // How many do we classify correctly?
  // TODO: Compute AUC
  int correct = 0;
  for (int n=0; n<data.numRows(); ++n) {
    //System.out.println(data.viewRow(n));
    double prediction = model.classify(data.viewRow(n), w, true);
    //System.out.println("Is: " + prediction + " should: " + y.get(n));
    if (Math.round(prediction) == y.get(n)) {
      ++correct;
    }
  }
  return ((double)correct) / ((double)data.numRows());
}

}
