package de.tuberlin.dima.ml.pact.logreg.sfo;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.codec.binary.Base64;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFODriver;
import de.tuberlin.dima.ml.pact.JobRunner;
import eu.stratosphere.pact.common.plan.Plan;
import eu.stratosphere.pact.common.type.Value;

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
  public List<FeatureGain> computeGainsSFO(int numSubTasks) throws Exception {

    // TODO _SFO: Pass new base model, not the empty initial one!
    
    final Stopwatch stopReadResults = new Stopwatch();

    boolean applyBest = false;
    
    // RUN
    JobRunner runner = new JobRunner();
    if (runLocal) {
      Plan sfoPlan = new SFOPlanAssembler().createPlan(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest, this.baseModel);
      runner.runLocal(sfoPlan);
    } else {
      // TODO _SFO Major: Basemodel does not get transmitted
      String[] jobArgs = SFOPlanAssembler.buildArgs(numSubTasks, inputPathTrain, inputPathTest, outputPath, numFeatures, labelIndex, applyBest);
      runner.run(jarPath, SFOPlanAssembler.class.getName(), jobArgs, confPath, "", "", true);
    }
    counters.put(COUNTER_KEY_TOTAL_WALLCLOCK, runner.getLastWallClockRuntime());
    
    // Read results from hdfs into memory
    stopReadResults.start();
    this.gains = SFOToolsPact.readGainsAndCoefficients(outputPath);
    stopReadResults.stop();
    counters.put(COUNTER_KEY_READ_RESULT, stopReadResults.elapsed(TimeUnit.MILLISECONDS));
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


  @Override
  public long getLastWallClockTime() {
    return counters.get(COUNTER_KEY_TOTAL_WALLCLOCK);
  }


  @Override
  public Map<String, Long> getAllCounters() {
    return counters;
  }
  
  public static void main(String[] args) {
    IncrementalModel model = new IncrementalModel(20);
    model.addDimensionToModel(1, 1.1);
    
    PactIncrementalModel pactModel = new PactIncrementalModel(model);
    String encoded = encodeValueAsBase64(pactModel);
    
    PactIncrementalModel pactModel2 = decodeValueFromBase64(encoded, PactIncrementalModel.class);
    
    System.out.println(encoded.length());
    System.out.println(pactModel.getValue().getUsedDimensions().size());
    System.out.println(pactModel2.getValue().getUsedDimensions().size());
    
  }
  
  private static String encodeValueAsBase64(Value object) {

    ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
    DataOutputStream outStream = new DataOutputStream(byteStream);
    try {
      object.write(outStream);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return Base64.encodeBase64String(byteStream.toByteArray());
  }
  
  private static <T extends Value> T decodeValueFromBase64(String encoded, Class<T> type) {
    
    byte[] bytes = Base64.decodeBase64(encoded);
    ByteArrayInputStream byteInStream = new ByteArrayInputStream(bytes);
    DataInputStream inStream = new DataInputStream(byteInStream);

    Value object = null;
    try {
      object = type.newInstance();
    } catch (InstantiationException e1) {
      e1.printStackTrace();
    } catch (IllegalAccessException e1) {
      e1.printStackTrace();
    }
    try {
      object.read(inStream);
    } catch (IOException e) {
      e.printStackTrace();
    }
    return (T)object;
  }

}
