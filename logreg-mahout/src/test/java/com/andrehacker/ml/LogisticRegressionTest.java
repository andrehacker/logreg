package com.andrehacker.ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.time.StopWatch;
import org.junit.Ignore;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.visualization.datasource.base.TypeMismatchException;
import com.google.visualization.datasource.datatable.ColumnDescription;
import com.google.visualization.datasource.datatable.DataTable;
import com.google.visualization.datasource.datatable.value.ValueType;
import com.google.visualization.datasource.render.JsonRenderer;

public class LogisticRegressionTest {
  
  private List<String> predictorNames = Lists.newArrayList(new String[] {
//      "x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c", "bias"
      "x", "y", "shape", "xx", "xy", "yy", "a", "b", "c"
//      "x", "y", "shape", "a", "b", "c"
//      "x", "y", "a", "b", "c"    // like Mahout page 252
//      "x", "c"    // Adding a or b is REALLY BAD. However x and y are good. Shape is okay.
     });
  
  private String testFile = "donut-test.csv";
  private String trainingFile = "donut.csv";

  @Test
  public void test() {
     
     double initialWeight = 0;
     try {
       LogisticRegression logReg = new LogisticRegression("donut.csv", predictorNames);

//       StopWatch sw = new StopWatch();
//       sw.start();
//       sw.stop();
       
       int iterations = 20;
       List<Double> penalties = Lists.newArrayList(new Double[] {0d, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1d, 2d, 5d, 10d, 20d, 50d, 100d, 500d, 1000d});

       DataTable data = new DataTable();
       ArrayList<ColumnDescription> cd = new ArrayList<ColumnDescription>();
       cd.add(new ColumnDescription("lambda", ValueType.NUMBER, "Regularization constraint"));
       cd.add(new ColumnDescription("Success_in", ValueType.NUMBER, "in-sample success"));
       cd.add(new ColumnDescription("Success_out", ValueType.NUMBER, "out-of-sample success"));
       data.addColumns(cd);
       for (double lambda : penalties) {
         logReg.trainNewton(iterations, initialWeight, lambda);   // breaks at 92 without regularization
//         logReg.printLinearModel(logReg.getWeight());
         Validation valTest = new Validation(testFile, predictorNames);
         Validation valTraining = new Validation(trainingFile, predictorNames);
         valTest.computeMetrics(logReg.getWeight(), logReg, logReg);
         valTraining.computeMetrics(logReg.getWeight(), logReg, logReg);
         System.out.println("Regularization: " + lambda);
         System.out.println("- Success-rate:\t\t" + valTest.getSuccessRate());
         System.out.println("- Mean Deviation:\t" + valTest.getMeanDeviation());

         try {
           data.addRowFromValues(lambda, valTraining.getSuccessRate(), valTest.getSuccessRate());
         } catch (TypeMismatchException e) {
           System.out.println("Invalid type!");
         }
       }
//       System.out.println("CONFUSION");
//       System.out.println("True Pos:\t" + (int)valTest.getConfusion().get(0, 0));
//       System.out.println("True Neg:\t" + (int)valTest.getConfusion().get(1, 1));
//       System.out.println("False Pos:\t" + (int)valTest.getConfusion().get(0, 1));
//       System.out.println("False Neg:\t" + (int)valTest.getConfusion().get(1, 0));
       
       String json = "result='" + JsonRenderer.renderDataTable(data, true, true, false).toString() + "'";
       MLUtils.writeUtf(json, "results.json");

     } catch (IOException e1) {
       e1.printStackTrace();
     }
  }

  @Ignore
  @Test
  public void testBatchGD() {
    double initialWeight = 0;
    try{
      LogisticRegression logReg = new LogisticRegression("donut.csv", predictorNames);
      StopWatch sw = new StopWatch();
      sw.start();
      logReg.trainBatchGD(200, 1, initialWeight);
      logReg.printLinearModel(logReg.getWeight());
//      logReg.printMetrics("donut-test.csv", true);

      Validation valTest = new Validation(testFile, predictorNames);
      Validation valTraining = new Validation(trainingFile, predictorNames);
      valTest.computeMetrics(logReg.getWeight(), logReg, logReg);
      valTraining.computeMetrics(logReg.getWeight(), logReg, logReg);
      System.out.println("Batch GD");
      System.out.println("- Success-rate:\t\t" + valTest.getSuccessRate());
      System.out.println("- Mean Deviation:\t" + valTest.getMeanDeviation());
    } catch (IOException e1) {
      e1.printStackTrace();
    }
  }

}
