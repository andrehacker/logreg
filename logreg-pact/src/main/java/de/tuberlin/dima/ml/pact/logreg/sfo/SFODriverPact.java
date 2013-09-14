package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

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

  private IncrementalModel baseModel;
  private List<FeatureGain> gains = Lists.newArrayList();
  
  private Map<String, Long> counters = Maps.newHashMap();
  public static final String COUNTER_KEY_TOTAL_WALLCLOCK = "total-wall-clock";
  private static final String COUNTER_KEY_READ_RESULT = "read-result-gains-and-coefficients";
  
  public SFODriverPact(
      String inputPathTrain,
      String inputPathTest,
      int labelIndex,
      String outputPath,
      int numFeatures,
      boolean runLocal,
      String confPath,
      String jarPath) {
    this.inputPathTrain = inputPathTrain;
    this.inputPathTest = inputPathTest;
    this.labelIndex = labelIndex;
    this.outputPath = outputPath;
    this.numFeatures = numFeatures;
    this.runLocal = runLocal;
    this.confPath = confPath;
    this.jarPath = jarPath;

    // Create empty model
    this.baseModel = new IncrementalModel(numFeatures);
  }
  

  @Override
  public List<FeatureGain> computeGains(int numSubTasks) throws Exception {
    return forwardFeatureSelection(numSubTasks, 1, 1);
  }

  @Override
  public List<FeatureGain> forwardFeatureSelection(int numSubTasks, int iterations, int addPerIteration) throws Exception {

    final Stopwatch stopReadResults = new Stopwatch();

    // RUN
    JobRunner runner = new JobRunner();
    String[] jobArgs = SFOPlanAssembler.buildArgs(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, iterations, addPerIteration, this.baseModel);
    if (runLocal) {
      Plan sfoPlan = new SFOPlanAssembler().getPlan(jobArgs);
      runner.runLocal(sfoPlan);
    } else {
      runner.run(jarPath, SFOPlanAssembler.class.getName(), jobArgs, confPath, "", "", true);
    }
    counters.put(COUNTER_KEY_TOTAL_WALLCLOCK, runner.getLastWallClockRuntime());
    
    if (iterations <= 1) {
      // Read results from hdfs into memory
      stopReadResults.start();
      this.gains = SFOToolsPact.readGainsAndCoefficients(outputPath);
      stopReadResults.stop();
      counters.put(COUNTER_KEY_READ_RESULT, stopReadResults.elapsed(TimeUnit.MILLISECONDS));
      Collections.sort(this.gains, Collections.reverseOrder());
      return getGains();
    }
    return null;
  }

  @Override
  public void addBestFeatures(int n) throws IOException {
    // Add best to base model
    for (int i=0; i<n; ++i) {
      int bestDimension = gains.get(i).getDimension();
      baseModel.addDimensionToModel(bestDimension,
          gains.get(i).getCoefficient());
      System.out.println("Added d=" + bestDimension
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


  @Override
  public long getLastWallClockTime() {
    return counters.get(COUNTER_KEY_TOTAL_WALLCLOCK);
  }


  @Override
  public Map<String, Long> getAllCounters() {
    return counters;
  }

}
