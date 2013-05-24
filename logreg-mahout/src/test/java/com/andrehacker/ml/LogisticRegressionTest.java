package com.andrehacker.ml;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.time.StopWatch;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.visualization.datasource.base.TypeMismatchException;
import com.google.visualization.datasource.datatable.ColumnDescription;
import com.google.visualization.datasource.datatable.DataTable;
import com.google.visualization.datasource.datatable.value.ValueType;
import com.google.visualization.datasource.render.JsonRenderer;

public class LogisticRegressionTest {
  
  public static List<String> predictorNames;
  
  public static final String testFile = "donut-test.csv";
  public static final String trainingFile = "donut.csv";
  private static final double TARGET_POSITIVE = 2d;
  private static final double TARGET_NEGATIVE = 1d;
  private static final String TARGET_NAME = "color";
  
  private CsvReader csvTest;
  private CsvReader csvTrain;
  
  @Before
  public void before() throws Exception {
    predictorNames = Lists.newArrayList(new String[] {
//      "x", "y", "shape", "k", "k0", "xx", "xy", "yy", "a", "b", "c", "bias"
//      "x", "y", "shape", "xx", "xy", "yy", "a", "b", "c"    // nice chart
//      "x", "y", "shape", "a", "b", "c"
//      "x", "y", "a", "b", "c"    // like Mahout page 252
      "x", "c"    // Adding a or b is REALLY BAD. However x and y are good. Shape is okay.
     });
    
    csvTrain = MLUtils.readData(trainingFile, 40, predictorNames, TARGET_NAME);
    csvTrain.normalize();
    csvTrain.normalizeClassLabels(TARGET_POSITIVE, TARGET_NEGATIVE);
    csvTest = MLUtils.readData(testFile, 40, predictorNames, TARGET_NAME);
    csvTest.normalize(csvTrain.getMeans(), csvTrain.getRanges());
//    csvTest.normalize();
    csvTest.normalizeClassLabels(TARGET_POSITIVE, TARGET_NEGATIVE);
  }
  
  @Test
  public void test() {
     
     double initialWeight = 0;
//     LogisticRegression logReg = new LogisticRegression("donut.csv", predictorNames);
     LogisticRegression logReg = new LogisticRegression();

//       StopWatch sw = new StopWatch();
//       sw.start();
//       sw.stop();

     int iterations = 20;
     List<Double> penalties = Lists.newArrayList(new Double[] {0d, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1d, 2d, 5d, 10d, 20d, 50d, 100d, 500d, 1001d});
     
     DataTable data = new DataTable();
     ArrayList<ColumnDescription> cd = new ArrayList<ColumnDescription>();
     cd.add(new ColumnDescription("lambda", ValueType.NUMBER, "Regularization constraint"));
     cd.add(new ColumnDescription("Success_in", ValueType.NUMBER, "in-sample success"));
     cd.add(new ColumnDescription("Success_out", ValueType.NUMBER, "out-of-sample success"));
     data.addColumns(cd);
     for (double lambda : penalties) {
       logReg.trainNewton(csvTrain.getData(), csvTrain.getY(), iterations, initialWeight, lambda);   // breaks at 92 without regularization
//         logReg.printLinearModel(logReg.getWeight());
//         Validation valTest = new Validation(testFile, predictorNames, 2d, 1d);
//         Validation valTraining = new Validation(trainingFile, predictorNames, 2d, 1d);
       Validation valTest = new Validation();
       Validation valTrain = new Validation();
       valTest.computeMetrics(csvTest.getData(), csvTest.getY(), logReg.getWeight(), logReg, logReg);
       valTrain.computeMetrics(csvTrain.getData(), csvTrain.getY(), logReg.getWeight(), logReg, logReg);
       System.out.println("Regularization: " + lambda);
       System.out.println("- Success-rate:\t\t" + valTest.getSuccessRate());
       System.out.println("- Mean Deviation:\t" + valTest.getMeanDeviation());

       try {
         data.addRowFromValues(lambda, valTrain.getSuccessRate(), valTest.getSuccessRate());
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
  }

//  @Ignore
  @Test
  public void testBatchGD() {
    double initialWeight = 0;
//    LogisticRegression logReg = new LogisticRegression("donut.csv", predictorNames);
    LogisticRegression logReg = new LogisticRegression();
    StopWatch sw = new StopWatch();
    sw.start();
    logReg.trainBatchGD(csvTrain.getData(), csvTrain.getY(), 200, 1, initialWeight);

    Validation valTest = new Validation();
    Validation valTrain = new Validation();
    valTest.computeMetrics(csvTest.getData(), csvTest.getY(), logReg.getWeight(), logReg, logReg);
    valTrain.computeMetrics(csvTrain.getData(), csvTrain.getY(), logReg.getWeight(), logReg, logReg);
    
    System.out.println("\nBatch GD");
    MLUtils.printLinearModel(logReg.getWeight(), csvTrain);
    System.out.println("Evaluation");
    System.out.println("- Success-rate:\t\t" + valTest.getSuccessRate());
    System.out.println("- Mean Deviation:\t" + valTest.getMeanDeviation());
  }

}
