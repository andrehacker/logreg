package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFODriver;
import de.tuberlin.dima.ml.pact.JobRunner;
import eu.stratosphere.pact.common.plan.Plan;

public class SFODriverPact implements SFODriver {

  private String inputPathTrain;
  private String inputPathTest;
  private int labelIndex;
  private String outputPath;
  private int numSubTasks;
  private int numFeatures;

  private IncrementalModel baseModel;
  private List<FeatureGain> gains = Lists.newArrayList();
  
  public SFODriverPact(
      String inputPathTrain,
      String inputPathTest,
      int labelIndex,
      String outputPath,
      int numSubTasks,
      int numFeatures) {
    this.inputPathTrain = inputPathTrain;
    this.inputPathTest = inputPathTest;
    this.labelIndex = labelIndex;
    this.outputPath = outputPath;
    this.numSubTasks = numSubTasks;
    this.numFeatures = numFeatures;

    // Create empty model
    baseModel = new IncrementalModel(numFeatures);
  }


  @Override
  public void runSFO() throws Exception {
    boolean applyBest = true;
    Plan sfoPlan = new SFOPlanAssembler().createPlan(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest);
    
    JobRunner runner = new JobRunner();
    runner.runLocal(sfoPlan);
  }

  @Override
  public void addBestFeature() throws IOException {
    // TODO Auto-generated method stub
  }

  @Override
  public void addNBestFeatures(int n) {
    // TODO Auto-generated method stub
  }

  @Override
  public void retrainBaseModel() {
    // TODO Auto-generated method stub
  }

  @Override
  public List<FeatureGain> getGains() {
    return gains;
  }
  
  private void readGains() throws IOException {
    // Read results from hdfs into memory
    gains = SFOToolsPact.readEvalResult(outputPath);

    // Sort by gain
//    Collections.sort(gains, Collections.reverseOrder());
  }

  
  public static void main(String[] args) throws Exception {

    // CCAT=33, ECAT=59, GCAT=70, MCAT=102

    String inputPathTrain = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_train_10000.svm";
    String inputPathTest = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_train_10000.svm";
//    String inputPathTest = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_test_10000.svm";
    int labelIndex = 59;
    String outputPath = "file:///home/andre/output-sfo-pact";
    int numSubTasks = 2;
    int numFeatures = 47237;

    SFODriver driver = new SFODriverPact(inputPathTrain, inputPathTest, labelIndex, outputPath, numSubTasks, numFeatures);
    driver.runSFO();
  }

}
