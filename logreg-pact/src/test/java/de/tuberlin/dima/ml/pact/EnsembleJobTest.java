package de.tuberlin.dima.ml.pact;

import org.junit.Test;

import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
import de.tuberlin.dima.ml.pact.logreg.ensemble.EnsembleJob;
import de.tuberlin.dima.ml.util.IOUtils;
import eu.stratosphere.pact.common.plan.Plan;

public class EnsembleJobTest {

  @Test
  public void test() throws Exception {
    
    String numberPartitions= "1";
//    String inputFile = "file:///home/andre/dev/datasets/donut/donut.csv";
//    String outputFile = "file:///home/andre/count";
    String inputFileTrain = "file:///home/andre/dev/datasets/libsvm-rcv1/rcv1_train.binary";
    String inputFileTest = "file:///home/andre/dev/datasets/libsvm-rcv1/rcv1_test_20000.binary";
    String outputFile = "file:///home/andre/output-ensemble";
    String numFeatures = Long.toString(RCV1DatasetInfo.get().getNumFeatures());
    String runValidation = "1";
    boolean runLocal = false;
    
    // numSubTasks dataInput output
    String[] jobArgs = {numberPartitions, inputFileTrain, inputFileTest, outputFile, numFeatures, runValidation};
    
    JobRunner runner = new JobRunner();

    if (runLocal) {
      
      Plan ensemblePlan = (new EnsembleJob()).getPlan(jobArgs);
      runner.runLocal(ensemblePlan);
      
    } else {

      String jarPath = IOUtils.getDirectoryOfJarOrClass(EnsembleJob.class)
          + "/logreg-pact-0.0.1-SNAPSHOT-job.jar";
      System.out.println("JAR PATH: " + jarPath);
      runner.run(jarPath, "de.tuberlin.dima.ml.pact.ensemble.EnsembleJob", jobArgs, "", true);
      
    }
    System.out.println("Job completed. Runtime=" + runner.getLastRuntime());
  }

}
