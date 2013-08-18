package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
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
  private int numFeatures;
  private boolean runLocal;
  private String confPath;
  private String jarPath;
  private String jobManagerAddress;
  private String jobManagerPort;

  private IncrementalModel baseModel;
  private List<FeatureGain> gains = Lists.newArrayList();
  
  public SFODriverPact(
      String inputPathTrain,
      String inputPathTest,
      int labelIndex,
      String outputPath,
      int numFeatures,
      boolean runLocal,
      String confPath,
      String jarPath,
      String jobManagerAddress,
      String jobManagerPort) {
    this.inputPathTrain = inputPathTrain;
    this.inputPathTest = inputPathTest;
    this.labelIndex = labelIndex;
    this.outputPath = outputPath;
    this.numFeatures = numFeatures;
    this.runLocal = runLocal;
    this.confPath = confPath;
    this.jarPath = jarPath;
    this.jobManagerAddress = jobManagerAddress;
    this.jobManagerPort = jobManagerPort;

    // Create empty model
    this.baseModel = new IncrementalModel(numFeatures);
  }


  @Override
  public List<FeatureGain> computeGainsSFO(int numSubTasks) throws Exception {

    // TODO _SFO: Pass new base model, not the empty initial one!
    
    boolean applyBest = false;
    
    JobRunner runner = new JobRunner();
    if (runLocal) {
      Plan sfoPlan = new SFOPlanAssembler().createPlan(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest, this.baseModel);
      runner.runLocal(sfoPlan);
    } else {
      // TODO _SFO Major: Basemodel does not get transmitted
      String[] jobArgs = SFOPlanAssembler.buildArgs(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest);
      //runner.run(jarPath, SFOPlanAssembler.class.getName(), jobArgs, "", jobManagerAddress, jobManagerPort, true);
      runner.run(jarPath, SFOPlanAssembler.class.getName(), jobArgs, confPath, "", "", true);
    }
    
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

}
