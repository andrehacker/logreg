package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;

import de.tuberlin.dima.ml.datasets.DatasetInfo;
import de.tuberlin.dima.ml.datasets.RCV1DatasetInfo;
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
    this.baseModel = new IncrementalModel(numFeatures);
  }


  @Override
  public List<FeatureGain> computeGainsSFO() throws Exception {

    // TODO _SFO: Pass new base model, not the empty initial one!
    
    boolean applyBest = false;
    Plan sfoPlan = new SFOPlanAssembler().createPlan(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest, this.baseModel);
    
    JobRunner runner = new JobRunner();
    
    runner.runLocal(sfoPlan);
    
    // Read results from hdfs into memory
    this.gains = SFOToolsPact.readGainsAndCoefficients(outputPath);
    Collections.sort(this.gains, Collections.reverseOrder());
    
    return getGains();
  }

  @Override
  public void addBestFeature() throws IOException {
    addNBestFeatures(1);
  }

  @Override
  public void addNBestFeatures(int n) throws IOException {
 // Add best to base model
    for (int i=0; i<n; ++i) {
      int bestDimension = gains.get(i).getDimension();
      baseModel.addDimensionToModel(bestDimension,
          gains.get(i).getCoefficient());
      System.out
      .println("Added d=" + bestDimension
          + " to base model with c="
          + gains.get(i).getCoefficient());
    }

    System.out.println("- New base model: " + baseModel.getW().toString());
  }

  @Override
  public void retrainBaseModel() {
    System.out.println("Retraining base model not yet implemented");
  }

  @Override
  public List<FeatureGain> getGains() {
    return gains;
  }

  public static void main(String[] args) throws Exception {

    // CCAT=33, ECAT=59, GCAT=70, MCAT=102
    String predictorNamePath = "/home/andre/dev/datasets/RCV1-v2/stem.termid.idf.map.txt";
    RCV1DatasetInfo.readPredictorNames(predictorNamePath);

    String inputPathTrain = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_train_1000.svm";
    String inputPathTest = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_train_1000.svm";
//    String inputPathTest = "file:///home/andre/dev/datasets/libsvm-rcv1v2-topics/rcv1_topics_test_10000.svm";
    int labelIndex = 59;
    String outputPath = "file:///home/andre/output-sfo-pact";
    int numSubTasks = 1;
    int numFeatures = (int)RCV1DatasetInfo.get().getNumFeatures();

    SFODriver driver = new SFODriverPact(inputPathTrain, inputPathTest, labelIndex, outputPath, numSubTasks, numFeatures);
    driver.computeGainsSFO();
    
    printTopGains(driver.getGains(), RCV1DatasetInfo.get());
    
    driver.addBestFeature();

    driver.computeGainsSFO();
    
    printTopGains(driver.getGains(), RCV1DatasetInfo.get());
  }
  
  private static void printTopGains(List<FeatureGain> gains, DatasetInfo datasetInfo) {
    for (int i=0; i<10 && i<gains.size(); ++i) {
      System.out.println("d " + gains.get(i).getDimension() + 
          " (" + datasetInfo.getFeatureName(gains.get(i).getDimension()) 
          + ") gain: " + gains.get(i).getGain() + " coefficient: " + gains.get(i).getCoefficient());
    }
  }

}
