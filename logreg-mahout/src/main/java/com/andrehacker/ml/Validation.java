package com.andrehacker.ml;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public class Validation {  
  
  public static double computeMeanDeviation(Matrix data, Vector y, Vector w, RegressionModel model) {
  // How many do we classify correctly?
  double dev = 0;
  for (int n=0; n<data.numRows(); ++n) {
    //System.out.println(data.viewRow(n));
    double prediction = model.predict(data.viewRow(n), w, true);
    dev += Math.abs((y.get(n) - prediction));

//    System.out.println("Is: " + prediction + " should: " + y.get(n));
  }
  return dev / data.numRows();
}

public static double computeSuccessRate(Matrix data, Vector y, Vector w, ClassificationModel model, Matrix confusionMatrix) {
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

}
