package de.tuberlin.dima.ml.mapred.logreg.sfo;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import de.tuberlin.dima.ml.logreg.sfo.FeatureGain;
import de.tuberlin.dima.ml.logreg.sfo.IncrementalModel;
import de.tuberlin.dima.ml.logreg.sfo.SFODriver;
import de.tuberlin.dima.ml.mapred.util.HadoopUtils;

/**
 * @See {@link SFODriver} documentation
 */
public class SFODriverHadoop implements SFODriver {

  private String trainInputFile;
  private String testInputFile;
  private boolean isMultilabelInput;
  private int positiveClass;

  private String trainOutputPath;
  private String testOutputPath;
  
  private String hadoopConfDir;
  private String jarPath;   // might be empty

  private int numFeatures;
  private double newtonTolerance;
  private int newtonMaxIterations;
  private double regularization;

  private IncrementalModel baseModel;
  
  private List<FeatureGain> gains = Lists.newArrayList();
  
  private Map<String, Long> counters = Maps.newHashMap();
  public static final String COUNTER_KEY_TOTAL_WALLCLOCK = "total-wall-clock";
  public static final String COUNTER_KEY_TRAIN_TIME = "train-wall-clock";
  public static final String COUNTER_KEY_TEST_TIME = "test-wall-clock";
  public static final String COUNTER_KEY_READ_RESULT_GAINS = "read-result-gains";
  
  // Temporary path for base model (needed for distributed cache only)
  private static final String BASE_MODEL_PATH = "/sfo-base-model.seq";

  public SFODriverHadoop(
      String trainInputFile,
      String testInputFile,
      boolean isMultilabelInput,
      int positiveClass,
      String outputPathTrain,
      String outputPathTest,
      int numFeatures,
      double newtonTolerance,
      int newtonMaxIterations,
      double regularization,
      String hadoopConfDir,
      String jarPath) {
    this.trainInputFile = trainInputFile;
    this.testInputFile = testInputFile;
    this.trainOutputPath = outputPathTrain;
    this.testOutputPath = outputPathTest;
    this.isMultilabelInput = isMultilabelInput;
    this.positiveClass = positiveClass;
    this.numFeatures = numFeatures;
    this.newtonTolerance = newtonTolerance;
    this.newtonMaxIterations = newtonMaxIterations;
    this.regularization = regularization;
    this.hadoopConfDir = hadoopConfDir;
    this.jarPath = jarPath;

    // Create empty model
    baseModel = new IncrementalModel(numFeatures);
  }

  @Override
  public List<FeatureGain> forwardFeatureSelection(int numReduceTasks, int iterations, int addPerIteration) throws Exception {
    if (iterations > 1) {
      throw new UnsupportedOperationException("Hadoop does not have support for iterations. Currently not supported");
    } else {
      return computeGains(numReduceTasks);
    }
  }

  /**
   * Runs a single SFO-Iteration. Computes the gain in metric (e.g.
   * log-likelihood) for all possible models with one more feature added to the
   * current base model
   * 
   * Does not add the best feature to the base model, allows the client to
   * decide whether to add it or not.
   */
  @Override
  public List<FeatureGain> computeGains(int numReduceTasks) throws Exception {

    final Stopwatch stopTotal = new Stopwatch();
    final Stopwatch stopTrain = new Stopwatch();
    final Stopwatch stopTest = new Stopwatch();
    final Stopwatch stopReadResults = new Stopwatch();
    
    // Read configuration from config folder
    Configuration configuration = HadoopUtils.createConfigurationFromConfDir(hadoopConfDir);

    // Make base model available for train/test mappers.
    // Will be broadcasted via Distributed Cache when the job is started
    SFOToolsHadoop.writeBaseModel(baseModel, configuration, BASE_MODEL_PATH);

    // ----- TRAIN -----
    stopTotal.start();
    stopTrain.start();
    
    HadoopUtils.addJarToConfiguration(configuration, jarPath);

    ToolRunner.run(configuration, new SFOTrainJob(
          trainInputFile,
          isMultilabelInput,
          positiveClass,
          trainOutputPath,
          numFeatures,
          newtonTolerance,
          newtonMaxIterations,
          regularization,
          numReduceTasks,
          BASE_MODEL_PATH
        ), null);
    stopTrain.stop();
    counters.put(COUNTER_KEY_TRAIN_TIME, stopTrain.elapsed(TimeUnit.MILLISECONDS));
    System.out.println("Done training");

    // ----- TEST -----
    stopTest.start();
    ToolRunner.run(configuration, new SFOEvalJob(testInputFile, isMultilabelInput, positiveClass, testOutputPath, numReduceTasks,
        numFeatures, trainOutputPath, BASE_MODEL_PATH), null);
    stopTest.stop(); stopTotal.stop();
    counters.put(COUNTER_KEY_TEST_TIME, stopTest.elapsed(TimeUnit.MILLISECONDS));
    counters.put(COUNTER_KEY_TOTAL_WALLCLOCK, stopTotal.elapsed(TimeUnit.MILLISECONDS));
    System.out.println("Done validation");

    // ----- READ RESULTS -----
    stopReadResults.start();
    gains = SFOToolsHadoop.readEvalResult(testOutputPath, configuration);
    stopReadResults.stop();
    counters.put(COUNTER_KEY_READ_RESULT_GAINS, stopReadResults.elapsed(TimeUnit.MILLISECONDS));
    Collections.sort(gains, Collections.reverseOrder());
    
    return getGains();
  }
  
  /**
   * In Forward feature selection we can add multiple features in each
   * iteration.
   * @throws IOException 
   */
  @Override
  public void addBestFeatures(int n) throws IOException {
    
    // Read configuration from config folder
    Configuration configuration = HadoopUtils.createConfigurationFromConfDir(hadoopConfDir);
    
    // Read coefficients
//    Configuration conf = HadoopUtils.createConfiguration(hdfsAddress, jobTrackerAddress);
    List<Double> coefficients = SFOToolsHadoop.readTrainedCoefficients(configuration, trainOutputPath,
        numFeatures);

    // Add best to base model
    for (int i=0; i<n; ++i) {
      int bestDimension = gains.get(i).getDimension();
      baseModel.addDimensionToModel(bestDimension,
          coefficients.get(bestDimension));
      System.out
      .println("Added d=" + bestDimension
          + " to base model with c="
          + coefficients.get(bestDimension));
    }

    // Write updated model to hdfs
    SFOToolsHadoop.writeBaseModel(baseModel, configuration, BASE_MODEL_PATH);

    System.out.println("- New base model: " + baseModel.getW().toString());
  }

  /**
   * After adding n features to the base model (based on approximate computation
   * of the gain) we retrain the whole model.
   * 
   * "After evaluating them to choose the best feature (or group of features) to
   * add, we rerun the full logistic regression to produce an exact model that
   * includes the newly added feature(s). We begin with this exact model for the
   * next iteration of features selection, so the approximation error does not
   * add up."
   */
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

  @Override
  public void resetModel() {
    this.baseModel = new IncrementalModel(numFeatures);
  }

}
