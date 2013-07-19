package de.tuberlin.dima.ml.pact;

import org.junit.Test;

import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.pact.logreg.ensemble.EnsembleJob;
import de.tuberlin.dima.ml.util.IOUtils;
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
    boolean runLocal = false;
    
    // numSubTasks dataInput output
    String[] args = {numberPartitions, inputFileTrain, inputFileTest, outputFile, numFeatures, runValidation};

    if (runLocal) {
      
      Plan wordcount = (new EnsembleJob()).getPlan(args);

      LocalExecutor executor = new LocalExecutor();
      executor.start();
      executor.executePlan(wordcount);

      executor.stop();
      
    } else {
      
      System.out.println("CODE PATH: " + IOUtils.getDirectoryOfJarOrClass(EnsembleJob.class));
      String jarPath = IOUtils.getDirectoryOfJarOrClass(EnsembleJob.class)
          + "/logreg-pact-0.0.1-SNAPSHOT-job.jar";
      JobRunner.run(jarPath, args, "");
      System.out.println("Job completed");
      
    }
  }

}
