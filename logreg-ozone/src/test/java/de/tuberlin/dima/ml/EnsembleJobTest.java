package de.tuberlin.dima.ml;

import org.junit.Test;

import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.pact.logreg.ensemble.EnsembleJob;
import eu.stratosphere.pact.client.LocalExecutor;
import eu.stratosphere.pact.common.plan.Plan;

public class EnsembleJobTest {

  @Test
  public void test() throws Exception {
    
    String numberPartitions= "2";
//    String inputFile = "file:///home/andre/dev/datasets/donut/donut.csv";
//    String outputFile = "file:///home/andre/count";
    String inputFileTrain = "file:///home/andre/dev/datasets/libsvm-rcv1/rcv1_train_5000.binary";
    String inputFileTest = "file:///home/andre/dev/datasets/libsvm-rcv1/rcv1_test_5000.binary";
    String outputFile = "file:///home/andre/output-ensemble";
    String numFeatures = Long.toString(RCV1DatasetInfo.get().getNumFeatures());
    String runValidation = "1";
    
    // numSubTasks dataInput output
    String[] args = {numberPartitions, inputFileTrain, inputFileTest, outputFile, numFeatures, runValidation};
    Plan wordcount = (new EnsembleJob()).getPlan(args);
    
    LocalExecutor executor = new LocalExecutor();
    executor.start();
    executor.executePlan(wordcount);
    
    executor.stop();
  }

}
